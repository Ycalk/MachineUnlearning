"""Анимация схлопывания классов для игрушки 3 -> 2 (два класса, три признака).

Модель: Linear(2 -> 32) + ReLU -> Linear(32 -> 3) -> Linear(3 -> 2).
Вход: два нарочно несимметричных облака в 2D.

Две панели на кадр:
  1) признаки в 3D (ось x — вдоль отрезка между центрами),
  2) логиты в 2D.
Подпись одна — номер эпохи.
Оба облака в каждой панели нормированы на расстояние между центрами, поэтому
центры всегда стоят в +-0.5, а видно, как облака стягиваются к ним.

Запуск: poetry run python toy_3to2_animation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image
from torch import nn

OUT = Path(__file__).resolve().parent.parent / "summary"
OUT.mkdir(exist_ok=True)

EPOCHS, N_PER_CLASS, WD, LR, WIDE = 4000, 300, 1e-3, 3e-3, 32
CENTRES = np.array([[-2.9, 0.4], [2.5, -0.6]])
COLORS = ["#1f77b4", "#ff7f0e"]
# до 40-й эпохи — по одному кадру на эпоху, дальше реже
SNAP_AT = list(range(41)) + [45, 50, 55, 60, 70, 80, 100, 120, 150, 200,
                             300, 400, 600, 800, 1200, 2000, 3000, 4000]


def make_data() -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.vstack([CENTRES[c] + rng.normal(scale=0.5, size=(N_PER_CLASS, 2)) for c in range(2)])
    y = np.concatenate([np.full(N_PER_CLASS, c) for c in range(2)])
    return torch.tensor(x, dtype=torch.float32), y


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(2, WIDE)
        self.l2 = nn.Linear(WIDE, 3)
        self.l3 = nn.Linear(3, 2)

    def features(self, x):
        return self.l2(torch.relu(self.l1(x)))

    def forward(self, x):
        return self.l3(self.features(x))


def collapse(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Центрирование и нормировка на расстояние между центрами + within/between."""
    mus = np.stack([z[y == c].mean(0) for c in range(2)])
    dist = np.linalg.norm(mus[0] - mus[1])
    within = np.mean([np.sum((z[y == c] - mus[c]) ** 2) / (y == c).sum() for c in range(2)])
    tot = np.mean(np.sum((z - z.mean(0)) ** 2, axis=1))
    return (z - mus.mean(0)) / dist, float(within / max(tot - within, 1e-12))


def snapshot(model: Toy, x: torch.Tensor, y: np.ndarray, epoch: int) -> dict:
    with torch.no_grad():
        zf = model.features(x).numpy().astype(np.float64)
        zl = model(x).numpy().astype(np.float64)
    f_norm, f_ratio = collapse(zf, y)
    l_norm, l_ratio = collapse(zl, y)
    return {"epoch": epoch, "f": f_norm, "l": l_norm,
            "f_ratio": f_ratio, "l_ratio": l_ratio, "y": y}


def render(snaps: list[dict], out: Path) -> None:
    frames: list[Image.Image] = []
    # общий базис для 3D-панели: ось e1 — вдоль отрезка между центрами (из последнего кадра)
    last = snaps[-1]
    ml = np.stack([last["f"][last["y"] == c].mean(0) for c in range(2)])
    e1 = ml[0] - ml[1]
    e1 = e1 / np.linalg.norm(e1)
    _, _, vt = np.linalg.svd(last["f"] - last["f"].mean(0), full_matrices=False)
    e2 = vt[0] - (vt[0] @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    basis = np.stack([e1, e2, np.cross(e1, e2)])

    fig = plt.figure(figsize=(11.6, 6.0), dpi=100)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    class_handles = [Line2D([], [], marker="o", ls="", color=COLORS[c], label=f"класс {c + 1}")
                     for c in range(2)]
    suptitle = fig.suptitle("", fontsize=15)
    # раскладка фиксируется один раз: иначе панели «съезжают» между кадрами
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    pos3, pos2 = ax3.get_position(), ax2.get_position()

    for i, s in enumerate(snaps):
        ax3.clear()
        ax2.clear()
        yy = s["y"]

        z3 = s["f"] @ basis.T
        for c in range(2):
            ax3.scatter(*z3[yy == c].T, s=7, color=COLORS[c], alpha=0.45, linewidths=0)
        m3 = np.stack([z3[yy == c].mean(0) for c in range(2)])
        ax3.plot(*np.c_[m3[0], m3[1]], color="0.5", lw=1.0, ls="--")
        for c in range(2):
            ax3.scatter(*m3[c], s=110, color=COLORS[c], edgecolor="black", linewidths=0.9, zorder=5)
        ax3.set_xlim(-0.9, 0.9); ax3.set_ylim(-0.9, 0.9); ax3.set_zlim(-0.9, 0.9)
        ax3.set_box_aspect((1, 1, 1))
        ax3.view_init(elev=14, azim=-72)
        ax3.tick_params(labelsize=6)
        ax3.legend(handles=class_handles, loc="upper left", fontsize=8, frameon=False)

        z2 = s["l"]
        for c in range(2):
            ax2.scatter(*z2[yy == c].T, s=7, color=COLORS[c], alpha=0.45, linewidths=0)
        m2 = np.stack([z2[yy == c].mean(0) for c in range(2)])
        ax2.plot(*np.c_[m2[0], m2[1]], color="0.5", lw=1.0, ls="--")
        for c in range(2):
            ax2.scatter(*m2[c], s=110, color=COLORS[c], edgecolor="black", linewidths=0.9, zorder=5)
        ax2.set_xlim(-0.9, 0.9); ax2.set_ylim(-0.9, 0.9)
        ax2.set_aspect("equal")
        ax2.tick_params(labelsize=7)

        suptitle.set_text(f"эпоха {s['epoch']}")
        ax3.set_position(pos3)
        ax2.set_position(pos2)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(buf).quantize(colors=96, method=Image.MEDIANCUT,
                                                   dither=Image.NONE))
    plt.close(fig)

    durations = [130] * 41 + [220] * (len(frames) - 42) + [1500]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    print("saved", out, len(frames), "frames", frames[0].size)


def main() -> None:
    torch.manual_seed(0)
    x, y = make_data()
    model = Toy()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    yt = torch.tensor(y)
    snaps: list[dict] = []
    for epoch in range(EPOCHS + 1):
        if epoch in SNAP_AT:
            snaps.append(snapshot(model, x, y, epoch))
        if epoch == EPOCHS:
            break
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), yt).backward()
        opt.step()
    for s in snaps:
        print(f"эпоха {s['epoch']:5d}  признаки {s['f_ratio']:7.3f}   логиты {s['l_ratio']:7.3f}")
    render(snaps, OUT / "toy_3to2_collapse.gif")


if __name__ == "__main__":
    main()
