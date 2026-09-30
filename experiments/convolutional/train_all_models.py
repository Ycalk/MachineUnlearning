"""Обучение всех 11 базовых моделей проекта: `full-baseline` + `retain-oracle-c0..c9`.

Скрипт повторяет `learning.ipynb`, но чинит воспроизводимость стартовых весов:

* `set_seed(SEED)` вызывается непосредственно перед созданием модели, поэтому init не
  зависит от того, сколько RNG было потрачено в сессии раньше. В исходном ноутбуке
  `set_seed()` вызывался один раз в начале сессии, а перед созданием очередной модели уже
  проходило обучение предыдущей — из-за этого в наборе оказались три разные группы стартов;
* порядок данных, аугментаций и dropout задаётся явно: `set_seed(TRAIN_SEED)` в начале
  `train_model` плюс собственный `torch.Generator` у каждого `DataLoader` (в torch 2.x от
  него же сеются worker-процессы), поэтому обучение не зависит от состояния глобального RNG;
* в метаданные пишутся `init_seed`, `init_sha256` и `train_seed` — по ним проверяется, что
  все модели набора вышли из одного старта;
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Literal, override

import kagglehub
import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel
from dataclasses import dataclass
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import trange

SEED: int = 7
EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 128

BASELINE_FORGET_CLASS: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ConvolutionalModel(nn.Module):
    def __init__(self, num_classes: int = 10):
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


def build_model() -> ConvolutionalModel:
    """Единственная точка создания модели.

    `set_seed(SEED)` стоит ровно перед конструктором (и модель создаётся на CPU, до
    `.to(device)`), поэтому стартовые веса не зависят ни от устройства обучения, ни от
    того, сколько RNG было потрачено в сессии до этого момента.
    """
    set_seed(SEED)
    return ConvolutionalModel()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Хэш float-тензоров состояния: доказывает, что модели стартовали из одного init."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if value.is_floating_point():
            digest.update(value.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()[:16]


def load_datasets() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    default_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    dataset_path = kagglehub.dataset_download("pankrzysiu/cifar10-python")
    train_raw = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *default_transform,
            ]
        ),
    )
    test_raw = datasets.CIFAR10(
        root=dataset_path,
        train=False,
        transform=transforms.Compose(default_transform),
    )
    return train_raw, test_raw


@dataclass
class AllDataLoaders:
    full_train: DataLoader[Any]
    retain_train: DataLoader[Any]
    forget_train: DataLoader[Any]
    full_test: DataLoader[Any]
    retain_test: DataLoader[Any]
    forget_test: DataLoader[Any]


def build_loaders(
    train_raw: datasets.CIFAR10,
    test_raw: datasets.CIFAR10,
    forget_class: int,
) -> AllDataLoaders:
    train_targets = np.array(train_raw.targets)
    test_targets = np.array(test_raw.targets)

    train_retain_idx = np.where(train_targets != forget_class)[0]
    test_retain_idx = np.where(test_targets != forget_class)[0]
    train_forget_idx = np.where(train_targets == forget_class)[0]
    test_forget_idx = np.where(test_targets == forget_class)[0]
    set_seed(SEED)
    return AllDataLoaders(
        full_train=DataLoader(
            train_raw,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(SEED),
        ),
        retain_train=DataLoader(
            Subset(train_raw, train_retain_idx.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(SEED),
        ),
        forget_train=DataLoader(
            Subset(train_raw, train_forget_idx.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(SEED),
        ),
        full_test=DataLoader(
            test_raw,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            generator=torch.Generator().manual_seed(SEED),
        ),
        retain_test=DataLoader(
            Subset(test_raw, test_retain_idx.tolist()),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            generator=torch.Generator().manual_seed(SEED),
        ),
        forget_test=DataLoader(
            Subset(test_raw, test_forget_idx.tolist()),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            generator=torch.Generator().manual_seed(SEED),
        ),
    )


class EpochMetrics(BaseModel):
    # loss
    full_test_loss: float
    retain_test_loss: float
    forget_test_loss: float

    full_train_loss: float
    retain_train_loss: float
    forget_train_loss: float

    # accuracy
    full_test_accuracy: float
    retain_test_accuracy: float
    forget_test_accuracy: float

    full_train_accuracy: float
    retain_train_accuracy: float
    forget_train_accuracy: float

    # forget class probability
    full_test_forget_class_probability: float
    retain_test_forget_class_probability: float
    forget_test_forget_class_probability: float
    forget_test_confusion_targets: dict[int, float]

    full_train_forget_class_probability: float
    retain_train_forget_class_probability: float
    forget_train_forget_class_probability: float
    forget_train_confusion_targets: dict[int, float]

    # entropy
    full_test_entropy: float
    retain_test_entropy: float
    forget_test_entropy: float

    full_train_entropy: float
    retain_train_entropy: float
    forget_train_entropy: float

    # accuracy per class
    full_test_accuracy_per_class: dict[int, float]
    retain_test_accuracy_per_class: dict[int, float]
    forget_test_accuracy_per_class: dict[int, float]

    full_train_accuracy_per_class: dict[int, float]
    forget_train_accuracy_per_class: dict[int, float]
    retain_train_accuracy_per_class: dict[int, float]


class TrainingRunRecord(BaseModel):
    model_type: Literal["full_baseline", "retain_oracle"]
    forget_class: int | None
    epochs: int
    learning_rate: float
    batch_size: int
    init_sha256: str
    seed: int
    metrics: list[EpochMetrics]


def compute_epoch_metrics(
    model: nn.Module,
    loaders: AllDataLoaders,
    device: torch.device,
    forget_class: int,
) -> EpochMetrics:
    model.eval()

    def evaluate_split(
        loader: DataLoader[Any],
    ) -> tuple[
        float, npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.float64]
    ]:
        losses, predictions, targets, probabilities = [], [], [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y, reduction="none")
                probability = nn.functional.softmax(logits, dim=-1)

                losses.extend(loss.cpu().numpy())
                predictions.extend(logits.argmax(dim=-1).cpu().numpy())
                targets.extend(y.cpu().numpy())
                probabilities.append(probability.cpu().numpy())

        return (
            float(np.mean(np.array(losses))),
            np.array(predictions),
            np.array(targets),
            np.vstack(probabilities),
        )

    metrics: dict[str, float | dict[int, float]] = {}
    for split_name in AllDataLoaders.__dataclass_fields__.keys():
        loss, predictions, targets, probabilities = evaluate_split(
            loaders.__getattribute__(split_name)
        )
        metrics[f"{split_name}_loss"] = loss

        metrics[f"{split_name}_accuracy"] = float(np.mean(predictions == targets))
        metrics[f"{split_name}_forget_class_probability"] = float(
            np.mean(probabilities[:, forget_class])
        )

        if split_name in ("forget_test", "forget_train"):
            unique, counts = np.unique(predictions, return_counts=True)
            total_forget = len(predictions)
            confusion_targets = {cls: 0.0 for cls in range(10)}
            for cls, count in zip(unique, counts):
                confusion_targets[int(cls)] = float(count / total_forget)

            metrics[f"{split_name}_confusion_targets"] = confusion_targets

        eps = 1e-12
        metrics[f"{split_name}_entropy"] = float(
            np.mean(-np.sum(probabilities * np.log(probabilities + eps), axis=1))
        )

        metrics[f"{split_name}_accuracy_per_class"] = {
            int(cls): float(np.mean(predictions[targets == cls] == cls))
            for cls in np.unique(targets)
        }

    return EpochMetrics.model_validate(metrics)


def train_model(
    model: nn.Module,
    training_loader: DataLoader[Any],
    all_loaders: AllDataLoaders,
    device: torch.device,
    forget_class: int,
) -> list[EpochMetrics]:
    set_seed(SEED)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    metrics: list[EpochMetrics] = []
    epoch_pbar = trange(1, EPOCHS + 1, desc="Training")

    for _ in epoch_pbar:
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in training_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        metrics.append(compute_epoch_metrics(model, all_loaders, device, forget_class))
        epoch_pbar.set_postfix(
            {
                "loss": f"{total_loss / total:.3f}",
                "acc": f"{correct * 100 / total:.1f}%",
            }
        )
    return metrics


def run_experiment(
    forget_class: int | None,
    train_raw: datasets.CIFAR10,
    test_raw: datasets.CIFAR10,
    device: torch.device,
) -> TrainingRunRecord:
    metric_forget_class = (
        BASELINE_FORGET_CLASS if forget_class is None else forget_class
    )
    data_loaders = build_loaders(
        train_raw=train_raw,
        test_raw=test_raw,
        forget_class=metric_forget_class,
    )
    model = build_model()
    init_sha256 = state_fingerprint(model.state_dict())

    metrics = train_model(
        model=model,
        training_loader=data_loaders.full_train
        if forget_class is None
        else data_loaders.retain_train,
        all_loaders=data_loaders,
        device=device,
        forget_class=metric_forget_class,
    )
    record = TrainingRunRecord(
        model_type="full_baseline" if forget_class is None else "retain_oracle",
        forget_class=forget_class,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        init_sha256=init_sha256,
        metrics=metrics,
    )

    model_name = (
        "full-baseline" if forget_class is None else f"retain-oracle-c{forget_class}"
    )
    save_dir = Path(f"./kaggle_export/{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), save_dir / "weights.pt")
    with open(save_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        f.write(record.model_dump_json(indent=2))
    kagglehub.model_upload(
        handle=f"ycalkk/convolutional-unlearning/pyTorch/{model_name}",
        local_model_dir=str(save_dir),
        version_notes=f"Починка воспроизводимости стартовых весов. seed={SEED}, init_sha256={record.init_sha256}",
    )
    return record


def main() -> None:
    print(
        f"torch {torch.__version__} (cuda {torch.cuda.is_available()}), "
        + f"init_seed={SEED}, эпох={EPOCHS}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw, test_raw = load_datasets()

    started = time.time()
    records = [
        run_experiment(None, train_raw, test_raw, device),
        *(run_experiment(cls, train_raw, test_raw, device) for cls in range(10)),
    ]

    fingerprints = {record.init_sha256 for record in records}
    print(f"Всего моделей: {len(records)} за {time.time() - started:.0f} с")
    print(
        f"init_sha256: {sorted(str(value) for value in fingerprints)} — "
        + "должен быть один и тот же у всех моделей"
    )


if __name__ == "__main__":
    main()
