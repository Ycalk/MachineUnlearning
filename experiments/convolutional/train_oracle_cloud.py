import argparse
import concurrent.futures as cf
import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import kagglehub
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn, optim
from torchvision import datasets
from tqdm import tqdm

INIT_SEEDS: tuple[int, ...] = (7, 42, 101, 2026)
TRAIN_SEEDS: tuple[int, ...] = (7, 42, 101, 2026)
CHECKPOINT_EPOCHS: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30)

EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 128

BASELINE_FORGET_CLASS: int = 0
NUM_CLASSES: int = 10

CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2023, 0.1994, 0.2010)

KAGGLE_MODEL_HANDLE: str = "ycalkk/convolutional-unlearning-cloud"
DATASET_HANDLE: str = "pankrzysiu/cifar10-python"
EXPORT_ROOT: Path = Path("./kaggle_export_cloud")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ConvolutionalModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features: nn.Module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head: nn.Module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(input))


def build_model(init_seed: int) -> ConvolutionalModel:
    """Создаёт модель на CPU с фиксированным `init_seed`."""
    set_seed(init_seed)
    return ConvolutionalModel()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """SHA-256 отпечаток float-тензоров весов модели (первые 16 символов)."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if value.is_floating_point():
            digest.update(value.detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class PreparedTrainData:
    """Преднормализованный и предпаддингованный (40x40) CIFAR-10 train в общей памяти."""

    padded_images: torch.Tensor  # (50000, 3, 40, 40), float32, shared memory
    targets: torch.Tensor  # (50000,), int64, shared memory


def load_prepared_train_data() -> PreparedTrainData:
    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
    train_raw = datasets.CIFAR10(root=dataset_path, train=True, download=False)

    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)

    # Нормализуем [0..255] -> float32 один раз, а паддинг 4 пикселя соответствует RGB=0 до нормализации
    images_u8 = torch.from_numpy(train_raw.data).permute(0, 3, 1, 2).contiguous()
    padded_u8 = F.pad(images_u8, (4, 4, 4, 4), value=0)
    padded_norm = (
        padded_u8.float().div_(255.0).sub_(mean).div_(std).contiguous().share_memory_()
    )
    targets = torch.tensor(train_raw.targets, dtype=torch.long).share_memory_()

    return PreparedTrainData(
        padded_images=padded_norm,
        targets=targets,
    )


def sample_augmented_batch(
    padded_batch: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Случайный кроп 32x32 из уже нормализованного 40x40 тензора + горизонтальное отражение."""
    batch_size = padded_batch.size(0)
    top = torch.randint(0, 9, (batch_size,), generator=generator)
    left = torch.randint(0, 9, (batch_size,), generator=generator)

    patches = padded_batch.unfold(2, 32, 1).unfold(3, 32, 1)
    cropped = patches[torch.arange(batch_size), :, top, left].contiguous()

    flip_mask = torch.rand(batch_size, generator=generator) < 0.5
    if bool(flip_mask.any()):
        cropped[flip_mask] = cropped[flip_mask].flip(-1)

    return cropped


class EpochMetrics(BaseModel):
    """Метрики эпохи, получаемые напрямую из батчей обучения (0 лишних проходов модели)."""

    epoch: int
    epoch_duration_seconds: float
    learning_rate: float
    train_loss: float
    train_accuracy: float
    train_entropy: float
    train_forget_class_probability: float
    train_accuracy_per_class: dict[int, float]


class TrainingRunRecord(BaseModel):
    model_type: Literal["full_baseline", "retain_oracle"]
    variation_name: str
    run_name: str
    forget_class: int | None
    metric_forget_class: int
    epochs: int
    checkpoint_epochs: list[int]
    learning_rate: float
    weight_decay: float
    batch_size: int
    init_seed: int
    train_seed: int
    init_sha256: str
    checkpoint_sha256: dict[int, str]
    total_duration_sec: float
    metrics: list[EpochMetrics]


class RunningEpochAccumulator:
    """Быстрый тензорный аккумулятор метрик по батчам обучения (без .numpy() в цикле)."""

    def __init__(self, device: torch.device) -> None:
        self.total_samples: int = 0
        self.loss_sum: torch.Tensor = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        self.correct_sum: torch.Tensor = torch.zeros(
            (), dtype=torch.int64, device=device
        )
        self.entropy_sum: torch.Tensor = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        self.forget_prob_sum: torch.Tensor = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        self.class_totals: torch.Tensor = torch.zeros(
            NUM_CLASSES, dtype=torch.int64, device=device
        )
        self.class_correct: torch.Tensor = torch.zeros(
            NUM_CLASSES, dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mean: torch.Tensor,
        forget_class: int,
    ) -> None:
        detached = logits.detach()
        batch_size = int(targets.size(0))
        self.total_samples += batch_size

        self.loss_sum += loss_mean.detach().to(torch.float64) * batch_size

        probs = F.softmax(detached, dim=-1)
        log_probs = F.log_softmax(detached, dim=-1)
        preds = detached.argmax(dim=-1)
        correct_mask = preds == targets

        self.correct_sum += correct_mask.sum()
        self.entropy_sum += -(probs * log_probs).sum(dim=-1).to(torch.float64).sum()
        self.forget_prob_sum += probs[:, forget_class].to(torch.float64).sum()

        self.class_totals += torch.bincount(targets, minlength=NUM_CLASSES)
        self.class_correct += torch.bincount(
            targets, weights=correct_mask.to(torch.int64), minlength=NUM_CLASSES
        ).to(torch.int64)

    def finalize(
        self, epoch: int, epoch_duration_sec: float, current_lr: float
    ) -> EpochMetrics:
        denom = max(self.total_samples, 1)
        totals_cpu = self.class_totals.cpu().tolist()
        correct_cpu = self.class_correct.cpu().tolist()
        per_class_acc = {
            int(cls): round(float(correct_cpu[cls] / totals_cpu[cls]), 6)
            for cls in range(NUM_CLASSES)
            if totals_cpu[cls] > 0
        }
        return EpochMetrics(
            epoch=epoch,
            epoch_duration_seconds=round(epoch_duration_sec, 3),
            learning_rate=round(current_lr, 8),
            train_loss=round(float(self.loss_sum.item() / denom), 6),
            train_accuracy=round(float(self.correct_sum.item() / denom), 6),
            train_entropy=round(float(self.entropy_sum.item() / denom), 6),
            train_forget_class_probability=round(
                float(self.forget_prob_sum.item() / denom), 6
            ),
            train_accuracy_per_class=per_class_acc,
        )


def save_checkpoint(model: nn.Module, run_dir: Path, epoch: int) -> str:
    """Сохраняет веса `weights_epoch_XX.pt` на CPU и возвращает их SHA-256 отпечаток."""
    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    sha = state_fingerprint(cpu_state)
    checkpoint_path = run_dir / f"weights_epoch_{epoch:02d}.pt"
    torch.save(cpu_state, checkpoint_path)
    return sha


def is_run_completed(run_dir: Path, checkpoint_epochs: tuple[int, ...]) -> bool:
    """Проверяет, что все чекпоинты эпох и `experiment_metadata.json` уже сохранены."""
    meta_path = run_dir / "experiment_metadata.json"
    if not meta_path.is_file():
        return False
    for epoch in checkpoint_epochs:
        if not (run_dir / f"weights_epoch_{epoch:02d}.pt").is_file():
            return False
    return True


def train_single_run(
    forget_class: int | None,
    init_seed: int,
    train_seed: int,
    data: PreparedTrainData,
    device: torch.device,
    export_root: Path,
    checkpoint_epochs: tuple[int, ...] = CHECKPOINT_EPOCHS,
    epochs: int = EPOCHS,
    force: bool = False,
    show_batch_pbar: bool = True,
) -> TrainingRunRecord:
    variation_name = (
        "full-baseline" if forget_class is None else f"retain-oracle-c{forget_class}"
    )
    run_name = f"init_{init_seed}_train_{train_seed}"
    run_dir = export_root / variation_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "experiment_metadata.json"
    if not force and is_run_completed(run_dir, checkpoint_epochs):
        return TrainingRunRecord.model_validate_json(
            meta_path.read_text(encoding="utf-8")
        )

    run_started = time.perf_counter()
    metric_forget_class = (
        BASELINE_FORGET_CLASS if forget_class is None else forget_class
    )

    # 1. Инициализация весов строго от init_seed
    model = build_model(init_seed)
    init_sha256 = state_fingerprint(model.state_dict())

    use_channels_last = device.type == "cpu"
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    # 2. Индексы обучающей выборки (50 000 для baseline или 45 000 для оракула)
    if forget_class is None:
        train_indices = torch.arange(data.targets.size(0), dtype=torch.long)
    else:
        train_indices = torch.nonzero(
            data.targets != forget_class, as_tuple=False
        ).squeeze(1)

    num_train = int(train_indices.size(0))
    num_batches = (num_train + BATCH_SIZE - 1) // BATCH_SIZE

    # 3. Изолированный генератор траектории обучения (шаффл батчей + кропы/флипы + dropout)
    train_gen = torch.Generator().manual_seed(train_seed)
    set_seed(train_seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_sha256: dict[int, str] = {}
    metrics: list[EpochMetrics] = []
    checkpoint_set = set(checkpoint_epochs)

    # Эпоха 0: мгновенное сохранение стартовых весов без прогона датасета
    if 0 in checkpoint_set:
        checkpoint_sha256[0] = save_checkpoint(model, run_dir, epoch=0)

    pbar = (
        tqdm(
            total=epochs * num_batches,
            desc=f"{variation_name}/{run_name}",
            unit="batch",
            leave=False,
        )
        if show_batch_pbar
        else None
    )

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        acc = RunningEpochAccumulator(device=device)
        current_lr = float(optimizer.param_groups[0]["lr"])

        perm = train_indices[torch.randperm(num_train, generator=train_gen)]
        for step, start in enumerate(range(0, num_train, BATCH_SIZE), start=1):
            batch_idx = perm[start : start + BATCH_SIZE]
            x = sample_augmented_batch(
                padded_batch=data.padded_images[batch_idx],
                generator=train_gen,
            ).to(device)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
            y = data.targets[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            acc.update(
                logits=logits,
                targets=y,
                loss_mean=loss,
                forget_class=metric_forget_class,
            )

            if pbar is not None:
                pbar.update(1)
                if step % 25 == 0 or step == num_batches:
                    done = max(acc.total_samples, 1)
                    pbar.set_postfix(
                        {
                            "ep": f"{epoch}/{epochs}",
                            "loss": f"{acc.loss_sum.item() / done:.3f}",
                            "acc": f"{acc.correct_sum.item() * 100 / done:.1f}%",
                        },
                        refresh=False,
                    )

        scheduler.step()

        if epoch in checkpoint_set:
            checkpoint_sha256[epoch] = save_checkpoint(model, run_dir, epoch=epoch)

        epoch_duration = time.perf_counter() - epoch_started
        metrics.append(
            acc.finalize(
                epoch=epoch,
                epoch_duration_sec=epoch_duration,
                current_lr=current_lr,
            )
        )

    if pbar is not None:
        pbar.close()

    total_duration = time.perf_counter() - run_started
    record = TrainingRunRecord(
        model_type="full_baseline" if forget_class is None else "retain_oracle",
        variation_name=variation_name,
        run_name=run_name,
        forget_class=forget_class,
        metric_forget_class=metric_forget_class,
        epochs=epochs,
        checkpoint_epochs=list(checkpoint_epochs),
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        init_seed=init_seed,
        train_seed=train_seed,
        init_sha256=init_sha256,
        checkpoint_sha256=checkpoint_sha256,
        total_duration_sec=round(total_duration, 2),
        metrics=metrics,
    )

    meta_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


_WORKER_DATA: PreparedTrainData | None = None
_WORKER_THREADS: int = 2


def _init_worker(data: PreparedTrainData, threads: int) -> None:
    global _WORKER_DATA, _WORKER_THREADS
    _WORKER_DATA = data
    _WORKER_THREADS = threads
    torch.set_num_threads(threads)


def _worker_train_task(
    task_args: tuple[int | None, int, int, Path, tuple[int, ...], int, bool],
) -> TrainingRunRecord:
    (
        forget_class,
        init_seed,
        train_seed,
        export_root,
        checkpoint_epochs,
        epochs,
        force,
    ) = task_args
    assert _WORKER_DATA is not None
    torch.set_num_threads(_WORKER_THREADS)
    return train_single_run(
        forget_class=forget_class,
        init_seed=init_seed,
        train_seed=train_seed,
        data=_WORKER_DATA,
        device=torch.device("cpu"),
        export_root=export_root,
        checkpoint_epochs=checkpoint_epochs,
        epochs=epochs,
        force=force,
        show_batch_pbar=False,
    )


def upload_variation_to_kaggle(
    variation_name: str,
    variation_dir: Path,
    records: list[TrainingRunRecord],
    model_handle: str = KAGGLE_MODEL_HANDLE,
) -> None:
    """Выгружает вариацию (со всеми 16 папками сидов внутри) одной версией в Kaggle."""
    init_hashes = sorted({r.init_sha256 for r in records})
    handle = f"{model_handle}/pyTorch/{variation_name}"
    notes = (
        f"Cloud 4x4 ({len(records)} runs): init_seeds={list(INIT_SEEDS)}, "
        f"train_seeds={list(TRAIN_SEEDS)}, checkpoints={list(CHECKPOINT_EPOCHS)}, "
        f"init_sha256={init_hashes}"
    )
    kagglehub.model_upload(
        handle=handle,
        local_model_dir=str(variation_dir),
        version_notes=notes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение облака базовых моделей и оракулов (11 вариаций × 16 сидов)."
    )
    parser.add_argument(
        "--variations",
        nargs="*",
        default=None,
        help="Список вариаций (например: full-baseline retain-oracle-c0). По умолчанию все 11.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Число эпох обучения (по умолчанию {EPOCHS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Число параллельных процессов обучения моделей на CPU (например, 2 на 5 ядрах).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Число CPU-потоков PyTorch на один процесс (на этом Xeon оптимум = 2).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=EXPORT_ROOT,
        help=f"Локальная директория экспорта (по умолчанию {EXPORT_ROOT}).",
    )
    parser.add_argument(
        "--kaggle-handle",
        type=str,
        default=KAGGLE_MODEL_HANDLE,
        help=f"Kaggle репозиторий модели (по умолчанию {KAGGLE_MODEL_HANDLE}).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Не выгружать в Kaggle (только локальное сохранение).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Переобучать заново даже если папка сидов уже завершена.",
    )
    return parser.parse_args()


def resolve_variations(selected: list[str] | None) -> list[int | None]:
    all_targets: dict[str, int | None] = {
        "full-baseline": None,
        **{f"retain-oracle-c{cls}": cls for cls in range(NUM_CLASSES)},
    }
    if not selected:
        return list(all_targets.values())
    unknown = [name for name in selected if name not in all_targets]
    if unknown:
        raise ValueError(
            f"Неизвестные вариации: {unknown}. Доступные: {list(all_targets)}"
        )
    return [all_targets[name] for name in selected]


def print_run_summary(record: TrainingRunRecord) -> None:
    final_metric = record.metrics[-1]
    print(
        f"  [{record.run_name}] init_sha={record.init_sha256[:8]} | "
        f"loss={final_metric.train_loss:.4f} | "
        f"acc={final_metric.train_accuracy * 100:.2f}% | "
        f"fgt_prob={final_metric.train_forget_class_probability:.4f} | "
        f"H={final_metric.train_entropy:.3f} | "
        f"{record.total_duration_sec:.1f}s ({record.total_duration_sec / 60:.1f}m)",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)

    checkpoint_epochs = tuple(ep for ep in CHECKPOINT_EPOCHS if ep <= args.epochs) or (
        0,
        args.epochs,
    )

    print(
        f"torch {torch.__version__} | device={device} | workers={args.workers} | "
        f"threads_per_worker={torch.get_num_threads()} | checkpoints={checkpoint_epochs}",
        flush=True,
    )

    t_load = time.perf_counter()
    data = load_prepared_train_data()
    print(
        f"Датасет подготовлен в памяти за {time.perf_counter() - t_load:.2f}s",
        flush=True,
    )

    variations = resolve_variations(args.variations)
    overall_started = time.perf_counter()
    all_records: list[TrainingRunRecord] = []

    use_pool = device.type == "cpu" and args.workers > 1
    mp_ctx = mp.get_context("fork") if use_pool else None

    for forget_class in variations:
        variation_name = (
            "full-baseline"
            if forget_class is None
            else f"retain-oracle-c{forget_class}"
        )
        variation_dir = args.export_dir / variation_name
        variation_started = time.perf_counter()
        variation_records: list[TrainingRunRecord] = []

        print(f"\n=== Вариация {variation_name} (16 комбинаций) ===", flush=True)
        seed_pairs = [
            (init_seed, train_seed)
            for init_seed in INIT_SEEDS
            for train_seed in TRAIN_SEEDS
        ]

        if use_pool and mp_ctx is not None:
            tasks = [
                (
                    forget_class,
                    init_seed,
                    train_seed,
                    args.export_dir,
                    checkpoint_epochs,
                    args.epochs,
                    args.force,
                )
                for init_seed, train_seed in seed_pairs
            ]
            with cf.ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp_ctx,
                initializer=_init_worker,
                initargs=(data, args.threads),
            ) as executor:
                for record in executor.map(_worker_train_task, tasks):
                    variation_records.append(record)
                    all_records.append(record)
                    print_run_summary(record)
        else:
            for init_seed, train_seed in seed_pairs:
                record = train_single_run(
                    forget_class=forget_class,
                    init_seed=init_seed,
                    train_seed=train_seed,
                    data=data,
                    device=device,
                    export_root=args.export_dir,
                    checkpoint_epochs=checkpoint_epochs,
                    epochs=args.epochs,
                    force=args.force,
                    show_batch_pbar=True,
                )
                variation_records.append(record)
                all_records.append(record)
                print_run_summary(record)

        if not args.skip_upload:
            print(
                f"Выгрузка вариации {variation_name} в {args.kaggle_handle}...",
                flush=True,
            )
            upload_variation_to_kaggle(
                variation_name=variation_name,
                variation_dir=variation_dir,
                records=variation_records,
                model_handle=args.kaggle_handle,
            )

        print(
            f"Вариация {variation_name} готова за "
            f"{(time.perf_counter() - variation_started) / 60:.1f} мин",
            flush=True,
        )

    by_init: dict[int, set[str]] = {}
    for rec in all_records:
        by_init.setdefault(rec.init_seed, set()).add(rec.init_sha256)
    print(
        f"\nГотово: {len(all_records)} моделей за "
        f"{(time.perf_counter() - overall_started) / 60:.1f} мин",
        flush=True,
    )
    for seed, hashes in sorted(by_init.items()):
        print(
            f"  init_seed={seed}: init_sha256={sorted(hashes)} (ожидается ровно 1 хэш)",
            flush=True,
        )


if __name__ == "__main__":
    main()
