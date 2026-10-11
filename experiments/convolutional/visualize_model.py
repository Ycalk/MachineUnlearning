"""Графическая схема архитектуры `ConvolutionalModel`.

Рисунок строится из самого класса модели: состав слоёв читается из `model.features` /
`model.head`, формы тензоров — из реального прямого прохода (forward-hooks) на входе
`1×3×32×32`, а ширина плашек подбирается под фактическую ширину текста в шрифте.
Поэтому схема не расходится с кодом при правке архитектуры: изменился слой — изменилась
подпись, форма и размеры.

Запуск (из каталога `experiments/convolutional`):

    python visualize_model.py

Результат: `summary/convolutional_architecture.png` и `.svg`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch import nn

from train_all_models import ConvolutionalModel

INPUT_SHAPE: tuple[int, int, int, int] = (1, 3, 32, 32)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "summary" / "convolutional_architecture"
)

INK = "#22303c"
MUTED = "#5b6b7c"
EDGE = "#5b6b7c"
ARROW = "#3d5a73"

LABEL_SIZE = 9.0
SHAPE_SIZE = 8.0
CAPTION_SIZE = 8.5

COLORS: dict[str, str] = {
    "conv": "#cfe2f3",
    "bn": "#d9ead3",
    "relu": "#fff2cc",
    "pool": "#d9d2e9",
    "flatten": "#fce5cd",
    "linear": "#f4cccc",
    "dropout": "#eceff1",
    "output": "#9fc5e8",
}

POOL_TYPES: tuple[type[nn.Module], ...] = (nn.MaxPool2d, nn.AdaptiveAvgPool2d)


@dataclass(frozen=True)
class Chip:
    """Один слой на схеме: подпись, форма выхода и цветовая категория."""

    label: str
    shape: str
    kind: str


def describe(module: nn.Module) -> tuple[str, str]:
    """Подпись слоя и его цветовая категория."""
    match module:
        case nn.Conv2d():
            return (
                (
                    f"Conv2d({module.in_channels}\u2192{module.out_channels}, "
                    f"{module.kernel_size[0]}\u00d7{module.kernel_size[1]}, p={module.padding[0]})"
                ),
                "conv",
            )
        case nn.BatchNorm2d():
            return f"BatchNorm2d({module.num_features})", "bn"
        case nn.ReLU():
            return "ReLU()", "relu"
        case nn.MaxPool2d():
            return f"MaxPool2d({module.kernel_size})", "pool"
        case nn.AdaptiveAvgPool2d():
            height, width = module.output_size
            return f"AdaptiveAvgPool2d({height}\u00d7{width})", "pool"
        case nn.Flatten():
            return "Flatten()", "flatten"
        case nn.Linear():
            return f"Linear({module.in_features}\u2192{module.out_features})", "linear"
        case nn.Dropout():
            return f"Dropout(p={module.p})", "dropout"
        case _:
            raise TypeError(f"слой без описания: {type(module).__name__}")


def forward_shapes(
    model: nn.Module, input_shape: tuple[int, int, int, int]
) -> dict[int, tuple[int, ...]]:
    """Формы выходов всех слоёв на реальном проходе (без batch-размерности), ключ — `id` слоя."""
    shapes: dict[int, tuple[int, ...]] = {}
    handles = [
        module.register_forward_hook(
            lambda m, _i, out: shapes.__setitem__(id(m), tuple(out.shape)[1:])
        )
        for module in model.modules()
        if not any(True for _ in module.children())
    ]
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(*input_shape))
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def split_blocks(features: nn.Sequential) -> list[list[nn.Module]]:
    """Свёрточные блоки: каждый заканчивается слоем пулинга."""
    blocks: list[list[nn.Module]] = []
    current: list[nn.Module] = []
    for module in features.children():
        current.append(module)
        if isinstance(module, POOL_TYPES):
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def chips_of(layers: list[nn.Module], shapes: dict[int, tuple[int, ...]]) -> list[Chip]:
    chips: list[Chip] = []
    for module in layers:
        label, kind = describe(module)
        shape = shapes[id(module)]
        chips.append(
            Chip(label=label, shape="\u00d7".join(str(dim) for dim in shape), kind=kind)
        )
    return chips


class Canvas:
    """Примитивы рисования в единицах холста; ширина текста измеряется в этих же единицах."""

    def __init__(self, width: float, height: float, scale: float = 9.5) -> None:
        self.width = width
        self.height = height
        self.fig, self.ax = plt.subplots(figsize=(width / scale, height / scale))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")

    def text_widths(self, requests: list[tuple[str, float]]) -> list[float]:
        """Ширина строк в единицах холста (по фактическим метрикам шрифта)."""
        artists = [self.ax.text(0, 0, s, fontsize=size) for s, size in requests]
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inverse = self.ax.transData.inverted()
        widths: list[float] = []
        for artist in artists:
            extent = artist.get_window_extent(renderer=renderer)
            (x0, _), (x1, _) = inverse.transform([(extent.x0, 0), (extent.x1, 0)])
            widths.append(abs(x1 - x0))
            artist.remove()
        return widths

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        face: str,
        radius: float = 1.1,
        edge: str = EDGE,
        lw: float = 1.1,
        zorder: int = 2,
    ) -> None:
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
                zorder=zorder,
            )
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        lw: float = 1.6,
        color: str = ARROW,
        head: float = 13.0,
    ) -> None:
        self.ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=head,
                color=color,
                linewidth=lw,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        size: float = LABEL_SIZE,
        color: str = INK,
        ha: str = "center",
        va: str = "center",
        weight: str = "normal",
        rotation: float = 0.0,
    ) -> None:
        self.ax.text(
            x,
            y,
            s,
            fontsize=size,
            color=color,
            ha=ha,
            va=va,
            fontweight=weight,
            rotation=rotation,
            zorder=4,
        )

    def close(self) -> None:
        plt.close(self.fig)

    def save(self, path: Path) -> None:
        for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {})):
            self.fig.savefig(
                path.with_suffix(suffix),
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.15,
                **kwargs,
            )
        plt.close(self.fig)


class Column:
    """Столбик слоёв одного блока: ширина плашек, высота и место на холсте."""

    PAD = 1.7
    INNER = 1.4
    MIN_WIDTH = 20.0

    def __init__(
        self,
        chips: list[Chip],
        title: str,
        subtitle: str,
        width: float,
        chip_h: float,
        chip_gap: float,
    ) -> None:
        self.chips = chips
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.chip_h = chip_h
        self.chip_gap = chip_gap
        self.height = len(chips) * chip_h + (len(chips) - 1) * chip_gap

    def draw(self, canvas: Canvas, x: float, top: float) -> None:
        canvas.box(
            x - self.PAD,
            top - self.height - self.PAD,
            self.width + 2 * self.PAD,
            self.height + 2 * self.PAD,
            face="#fbfcfd",
            edge="#c3ced8",
            radius=1.6,
            zorder=1,
        )
        canvas.text(x + self.width / 2, top + 5.6, self.title, size=9.5, weight="bold")
        canvas.text(
            x + self.width / 2, top + 2.4, self.subtitle, size=SHAPE_SIZE, color=MUTED
        )
        y = top
        for chip in self.chips:
            y -= self.chip_h
            canvas.box(
                x, y, self.width, self.chip_h, face=COLORS[chip.kind], radius=0.9
            )
            canvas.text(
                x + self.PAD,
                y + self.chip_h / 2,
                chip.label,
                size=LABEL_SIZE,
                ha="left",
            )
            canvas.text(
                x + self.width - self.PAD,
                y + self.chip_h / 2,
                chip.shape,
                size=SHAPE_SIZE,
                color=MUTED,
                ha="right",
            )
            y -= self.chip_gap


def chip_column_width(canvas: Canvas, chips: list[Chip]) -> float:
    """Ширина столбика: подпись + форма + отступы, по фактическим метрикам шрифта."""
    requests = [
        item
        for chip in chips
        for item in ((chip.label, LABEL_SIZE), (chip.shape, SHAPE_SIZE))
    ]
    measured = canvas.text_widths(requests)
    fitted = [
        measured[2 * index] + measured[2 * index + 1] + 2 * Column.PAD + Column.INNER
        for index in range(len(chips))
    ]
    return max(Column.MIN_WIDTH, *fitted)


def draw_input(canvas: Canvas, x: float, cy: float, size: float = 12.0) -> float:
    """Стопка из трёх каналов изображения; возвращает правую границу."""
    offset = size * 0.15
    for index, face in enumerate(("#f6b26b", "#93c47d", "#6d9eeb")):
        shift = (2 - index) * offset
        canvas.box(
            x + shift,
            cy - size / 2 + shift,
            size,
            size,
            face=face,
            radius=0.6,
            edge=INK,
            lw=1.0,
            zorder=2 + index,
        )
    canvas.text(
        x + size / 2 + offset,
        cy - size / 2 - 3.6,
        "\u0412\u0445\u043e\u0434\n3\u00d732\u00d732",
        size=LABEL_SIZE,
    )
    return x + size + 2 * offset


def draw_flatten(
    canvas: Canvas, x: float, cy: float, height: float, width: float = 6.2
) -> float:
    canvas.box(x, cy - height / 2, width, height, face=COLORS["flatten"], radius=0.6)
    canvas.text(x + width / 2, cy, "Flatten", size=LABEL_SIZE, rotation=90)
    canvas.text(
        x + width / 2, cy - height / 2 - 3.0, "2048", size=SHAPE_SIZE, color=MUTED
    )
    return x + width


def draw_output(canvas: Canvas, x: float, cy: float, num_classes: int) -> None:
    canvas.box(x, cy - 2.8, 9.5, 5.6, face=COLORS["output"], radius=2.8)
    canvas.text(x + 4.75, cy, str(num_classes), size=10.0, weight="bold")
    canvas.text(
        x + 4.75,
        cy - 5.4,
        "\u043b\u043e\u0433\u0438\u0442\u044b",
        size=SHAPE_SIZE,
        color=MUTED,
    )


def build_figure(model: ConvolutionalModel, output: Path) -> None:
    shapes = forward_shapes(model, INPUT_SHAPE)
    # `Flatten` уже нарисован отдельной полосой перед классификатором, в столбике он лишний.
    head_layers = [
        module for module in model.head.children() if not isinstance(module, nn.Flatten)
    ]

    gap = 4.5
    chip_h, chip_gap = 7.4, 1.7
    left_margin, right_margin = 9.0, 9.0

    # Пробный холст для измерения текста: единицы холста не зависят от его размера
    # (обе оси масштабируются одним и тем же отношением), поэтому метрики переносимы.
    probe = Canvas(200.0, 60.0)
    columns: list[Column] = []
    for index, layers in enumerate(split_blocks(model.features), start=1):
        chips = chips_of(layers, shapes)
        columns.append(
            Column(
                chips,
                f"\u0411\u043b\u043e\u043a {index}",
                f"\u0432\u044b\u0445\u043e\u0434: {chips[-1].shape}",
                chip_column_width(probe, chips),
                chip_h,
                chip_gap,
            )
        )
    head_chips = chips_of(head_layers, shapes)
    columns.append(
        Column(
            head_chips,
            "\u041a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440",
            f"\u0432\u044b\u0445\u043e\u0434: {head_chips[-1].shape} \u043b\u043e\u0433\u0438\u0442\u043e\u0432",
            chip_column_width(probe, head_chips),
            chip_h,
            chip_gap,
        )
    )
    probe.close()

    flatten_width, output_width = 6.2, 11.0
    width = (
        left_margin
        + 13.0
        + gap
        + sum(column.width + gap for column in columns)
        + flatten_width
        + gap
        + output_width
        + right_margin
    )
    height = 54.0
    column_height = max(column.height for column in columns)
    top = 9.0 + column_height
    cy = 9.0 + column_height / 2

    canvas = Canvas(width, height)
    cursor = draw_input(canvas, left_margin, cy) + gap

    for column in columns[:-1]:
        column.draw(canvas, cursor, top)
        canvas.arrow(
            cursor + column.width + Column.PAD,
            cy,
            cursor + column.width + gap - 0.8,
            cy,
        )
        cursor += column.width + gap

    cursor = draw_flatten(canvas, cursor, cy, column_height, flatten_width) + gap

    head = columns[-1]
    head.draw(canvas, cursor, top)
    canvas.arrow(
        cursor + head.width + Column.PAD, cy, cursor + head.width + gap - 0.8, cy
    )
    cursor += head.width + gap

    draw_output(canvas, cursor, cy, model.head[-1].out_features)

    parameters = (
        f"{sum(parameter.numel() for parameter in model.parameters()):,}".replace(
            ",", " "
        )
    )
    canvas.text(
        width / 2,
        2.8,
        "\u0412\u0445\u043e\u0434 \u2014 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 CIFAR-10 3\u00d732\u00d732; "
        "\u0444\u043e\u0440\u043c\u044b \u0442\u0435\u043d\u0437\u043e\u0440\u043e\u0432 \u0441\u043d\u044f\u0442\u044b \u0432 "
        "\u043f\u0440\u044f\u043c\u043e\u043c \u043f\u0440\u043e\u0445\u043e\u0434\u0435; \u043d\u0430 \u0432\u044b\u0445\u043e\u0434\u0435 \u2014 "
        "\u043b\u043e\u0433\u0438\u0442\u044b \u0431\u0435\u0437 softmax; "
        f"\u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432: {parameters}",
        size=CAPTION_SIZE,
        color=MUTED,
    )

    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Схема архитектуры ConvolutionalModel")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="базовый путь без расширения (пишутся .png и .svg)",
    )
    args = parser.parse_args()

    model = ConvolutionalModel()
    build_figure(model, args.output)
    print(f"схема: {args.output.with_suffix('.png')} / .svg")


if __name__ == "__main__":
    main()
