r"""Unlearning через транспорт момент-геометрии класса (кастомный метод из `summary/README.md`).

Постановка
==========
Дана модель M_theta, обученная на D. Нужно удалить класс c так, чтобы модель стала
неотличима от оракула - модели, обученной на D без класса c.

Идея
====
Знание о классе c - это его геометрический след в пространстве признаков: центр mu_c и
форма ковариации Sigma_c эллипсоида класса, а не отдельные объекты. Чтобы класс забыть,
его облако переводится в геометрию смеси сохранённых классов.

Формулировка
============
Шаг 1. Целевое распределение сохранённых классов (K \ c):
    mu_r    = 1 / |K \ c| * sum_{k != c} mu_k
    Sigma_r = sum_{k != c} n_k * Sigma_k / sum_{j != c} n_j
Шаг 2. Матрица трансформации (обеление забытого класса + перекраска в retain):
    A = Sigma_r^{1/2} Sigma_c^{-1/2}
Шаг 3. Трансформация признака h~ = mu_r + A (h - mu_c) переписывается в терминах
линейного классификатора y = W h + b, стоящего сразу за эмбеддингом (`head[4]`):
    y = (W A) h + (b + W (mu_r - A mu_c))
    =>  W <- W A,  b <- b + W (mu_r - A mu_c)

Обучения и градиентов нет: всё считается по средним и ковариациям признаков, сама правка -
одна матрица 128x128, домноженная на веса головы. Данные forget-класса нужны только как
источник статистики Sigma_c, mu_c (аналог "запроса на удаление"), но не для оптимизации.

Метрики - те же, что в разделе "Кастомный метод Unlearning": средняя вероятность забытого
класса на его объектах, средняя энтропия ответов, macro-accuracy на 9 сохранённых классах,
распределение топ-ответов, и сравнение с retain-oracle-c{c}.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_all_models import ConvolutionalModel  # noqa: E402

MODEL_ROOT: str = "ycalkk/convolutional-unlearning/pyTorch"
DATASET_HANDLE: str = "pankrzysiu/cifar10-python"
BASELINE_HANDLE: str = f"{MODEL_ROOT}/full-baseline"

FORGET_CLASS: int = 0
NUM_CLASSES: int = 10
EMBEDDING_DIM: int = 128
BATCH_SIZE: int = 512
EIG_CLIP: float = 1e-6  # нижняя отсечка собственных чисел, в долях от lambda_max

CIFAR_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR_STD: tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
CLASS_NAMES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class GaussianStats:
    """Момент-геометрия облака класса: центр и ковариация эллипсоида."""

    mean: Tensor  # (D,)
    covariance: Tensor  # (D, D)
    count: int


@dataclass(frozen=True)
class Metrics:
    """Поведенческие метрики на объектах забытого класса и на сохранённых классах."""

    forget_probability: float  # средняя p(c) на объектах класса c
    forget_entropy: float  # средняя энтропия ответов на объектах класса c
    forget_accuracy: float  # точность на объектах класса c
    retain_macro_accuracy: float  # macro-accuracy на 9 сохранённых классах
    top_answers: list[tuple[str, float]]  # топ-4 предсказания на объектах класса c


def load_model(handle: str) -> ConvolutionalModel:
    data_dir = Path(kagglehub.model_download(handle))
    model = ConvolutionalModel()
    model.load_state_dict(
        torch.load(data_dir / "weights.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def load_cifar10() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Train/test с детерминированным transform-ом.

    Аугментация (RandomCrop/Flip) здесь не нужна: она размывает ковариацию класса, а
    статистики должны описывать сам класс, а не случайный пайплайн обучения.
    """
    root = kagglehub.dataset_download(DATASET_HANDLE)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    return (
        datasets.CIFAR10(root=root, train=True, transform=transform),
        datasets.CIFAR10(root=root, train=False, transform=transform),
    )


@torch.no_grad()
def embed(model: ConvolutionalModel, images: Tensor) -> Tensor:
    """Признак h, который видит финальный Linear: выход `head[1]` (128) после ReLU."""
    flattened = model.features(images).flatten(1)
    return torch.relu(model.head[1](flattened))


@torch.no_grad()
def class_embeddings(
    model: ConvolutionalModel,
    dataset: datasets.CIFAR10,
    batch_size: int = BATCH_SIZE,
) -> dict[int, Tensor]:
    """Эмбеддинги всего датасета, разложенные по классам: {класс: (N_c, 128)}."""
    buckets: dict[int, list[Tensor]] = {cls: [] for cls in range(NUM_CLASSES)}
    for images, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features = embed(model, images)
        for cls in targets.unique().tolist():
            buckets[int(cls)].append(features[targets == cls])
    return {cls: torch.cat(chunks) for cls, chunks in buckets.items() if chunks}


def gaussian_stats(features: Tensor) -> GaussianStats:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
    return GaussianStats(mean=mean, covariance=covariance, count=features.shape[0])


def retained_target(
    stats: dict[int, GaussianStats], forget_class: int
) -> GaussianStats:
    """mu_r и Sigma_r - геометрия смеси всех классов, кроме забываемого (шаг 1)."""
    kept = [cls for cls in sorted(stats) if cls != forget_class]
    mean = torch.stack([stats[cls].mean for cls in kept]).mean(dim=0)
    total = sum(stats[cls].count for cls in kept)
    covariance = torch.zeros_like(stats[kept[0]].covariance)
    for cls in kept:
        covariance += (stats[cls].count / total) * stats[cls].covariance
    return GaussianStats(
        mean=mean, covariance=covariance, count=sum(stats[cls].count for cls in kept)
    )


def matrix_power_symmetric(
    matrix: Tensor, exponent: float, clip: float = EIG_CLIP
) -> Tensor:
    """M^exponent для симметричной PSD-матрицы через eigh.

    `clip` защищает обеливание Sigma_c^{-1/2} от вырождённых направлений: собственные
    числа снизу ограничены clip * lambda_max, иначе шум в низкодисперсных направлениях
    раздувается обратной степенью.
    """
    values, vectors = torch.linalg.eigh(matrix)
    floor = clip * values.max().clamp(min=torch.finfo(values.dtype).eps)
    return (vectors * values.clamp(min=floor).pow(exponent)) @ vectors.T


def moment_transport(
    forget_stats: GaussianStats, retain_stats: GaussianStats
) -> tuple[Tensor, Tensor]:
    """Матрица A = Sigma_r^{1/2} Sigma_c^{-1/2} и сдвиг (mu_r - A mu_c) (шаг 2)."""
    whitening = matrix_power_symmetric(forget_stats.covariance, -0.5)
    coloring = matrix_power_symmetric(retain_stats.covariance, 0.5)
    transform = coloring @ whitening
    offset = retain_stats.mean - transform @ forget_stats.mean
    return transform, offset


def apply_moment_transport(
    model: ConvolutionalModel,
    forget_stats: GaussianStats,
    retain_stats: GaussianStats,
) -> tuple[ConvolutionalModel, Tensor]:
    """Перенос трансформации в веса головы (шаг 3). Возвращает копию модели и A.

    h~ = A h + offset подставляется в логиты y = W h + b, поэтому
    W <- W A, b <- b + W offset (bias правится исходным W, до домножения на A).
    """
    transform, offset = moment_transport(forget_stats, retain_stats)
    unlearned = copy.deepcopy(model)
    head = unlearned.head[4]
    weight = head.weight.detach().clone()
    with torch.no_grad():
        head.weight.copy_(weight @ transform)
        head.bias.copy_(head.bias.detach() + weight @ offset)
    return unlearned, transform


@torch.no_grad()
def evaluate(
    model: ConvolutionalModel,
    dataset: datasets.CIFAR10,
    forget_class: int = FORGET_CLASS,
) -> Metrics:
    probabilities: list[Tensor] = []
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for images, labels in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
        logits = model(images)
        probabilities.append(torch.softmax(logits, dim=-1))
        predictions.append(logits.argmax(dim=-1))
        targets.append(labels)

    probability = torch.cat(probabilities)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    forget_mask = target == forget_class

    forget_probability = probability[forget_mask, forget_class].mean().item()
    forget_entropy = (
        -(probability[forget_mask] * probability[forget_mask].clamp_min(1e-12).log())
        .sum(dim=-1)
        .mean()
        .item()
    )
    retain_macro_accuracy = float(
        torch.stack(
            [
                (prediction[target == cls] == cls).float().mean()
                for cls in range(NUM_CLASSES)
                if cls != forget_class
            ]
        ).mean()
    )
    top_answers = [
        (CLASS_NAMES[cls], count / int(forget_mask.sum()))
        for cls, count in zip(
            *torch.unique(prediction[forget_mask], return_counts=True), strict=True
        )
    ]
    top_answers.sort(key=lambda item: item[1], reverse=True)

    return Metrics(
        forget_probability=forget_probability,
        forget_entropy=forget_entropy,
        forget_accuracy=(prediction[forget_mask] == forget_class).float().mean().item(),
        retain_macro_accuracy=retain_macro_accuracy,
        top_answers=top_answers[:4],
    )


def format_metrics(label: str, metrics: Metrics, forget_class: int) -> str:
    answers = ", ".join(
        f"{name}:{share:.2f}" for name, share in metrics.top_answers[:3]
    )
    return (
        f"{label:<28} p({CLASS_NAMES[forget_class]}): {metrics.forget_probability:.3f}  "
        f"H: {metrics.forget_entropy:.3f}  "
        f"acc_retain: {metrics.retain_macro_accuracy:.4f}  "
        f"acc_forget: {metrics.forget_accuracy:.3f}  | {answers}"
    )


def main() -> None:
    forget_class = int(sys.argv[1]) if len(sys.argv) > 1 else FORGET_CLASS
    torch.manual_seed(7)

    train_raw, test_raw = load_cifar10()
    baseline = load_model(BASELINE_HANDLE)

    train_embeddings = class_embeddings(baseline, train_raw)
    stats = {
        cls: gaussian_stats(features) for cls, features in train_embeddings.items()
    }
    retain_stats = retained_target(stats, forget_class)

    unlearned, transform = apply_moment_transport(
        baseline, stats[forget_class], retain_stats
    )

    print(f"Забываемый класс: {forget_class} ({CLASS_NAMES[forget_class]})")
    print(
        f"A: {tuple(transform.shape)}, "
        f"||A||_F = {torch.linalg.matrix_norm(transform).item():.3f}, "
        f"det = {torch.linalg.det(transform).item():.3e}, "
        f"||A A^T - I||_F = "
        f"{torch.linalg.matrix_norm(transform @ transform.T - torch.eye(EMBEDDING_DIM)).item():.3f}"
    )
    print(
        format_metrics(
            "база, как есть", evaluate(baseline, test_raw, forget_class), forget_class
        )
    )
    print(
        format_metrics(
            "база + транспорт класса",
            evaluate(unlearned, test_raw, forget_class),
            forget_class,
        )
    )
    oracle = load_model(f"{MODEL_ROOT}/retain-oracle-c{forget_class}")
    print(
        format_metrics(
            f"oracle-c{forget_class} (эталон)",
            evaluate(oracle, test_raw, forget_class),
            forget_class,
        )
    )


if __name__ == "__main__":
    main()
