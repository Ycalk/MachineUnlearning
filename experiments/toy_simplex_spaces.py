"""Игрушечные модели: один симплекс в четырёх системах координат.

Вход всегда 2-мерный. Модель: Linear(2 -> 32) + ReLU -> Linear(32 -> H) -> Linear(H -> C).
Широкий ReLU-слой нужен, чтобы линейное отображение 2D-данных могло породить
H-мерную геометрию (просто Linear(2 -> H) даёт ранг не больше 2).

    H -> C | вход | признаки | логиты | softmax | фигура
    -------+------+----------+--------+---------+-----------------
      3 4  |  2D  |   3D     |   4D   |   4D    | тетраэдр (3D)
      3 2  |  2D  |   3D     |   2D   |   2D    | отрезок (1D)
      2 3  |  2D  |   2D     |   3D   |   3D    | треугольник (2D)

Панель рисуется в 2D, если пространство 2-мерное, и в 3D, если 3-мерное или
больше (тогда — проекция на 3 главные оси).

Запуск: poetry run python toy_simplex_spaces.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent.parent / "summary"
OUT.mkdir(exist_ok=True)

EPOCHS, N_PER_CLASS, WD, LR, WIDE = 4000, 300, 1e-3, 3e-3, 32
COLORS = plt.get_cmap("tab10").colors
CONFIGS = [(3, 4), (3, 2), (2, 3)]            # (ширина признаков, число классов)
# Вход НЕ красивый: классы расставлены нарочно несимметрично (никакого симплекса),
# чтобы симплекс появлялся только после обучения.
CENTRES = {2: np.array([[-2.9, 0.4], [2.5, -0.6]]),
           3: np.array([[-3.1, -1.2], [1.0, -2.7], [0.6, 3.3]]),
           4: np.array([[-3.3, -1.9], [2.7, -2.6], [3.1, 1.1], [-0.7, 3.2]])}


def make_data(C: int) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(0)
    centres = CENTRES[C]
    x = np.vstack([centres[c] + rng.normal(scale=0.5, size=(N_PER_CLASS, 2)) for c in range(C)])
    y = np.concatenate([np.full(N_PER_CLASS, c) for c in range(C)])
    return torch.tensor(x, dtype=torch.float32), y


class Toy(nn.Module):
    def __init__(self, hidden: int, classes: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(2, WIDE)
        self.l2 = nn.Linear(WIDE, hidden)
        self.l3 = nn.Linear(hidden, classes)

    def features(self, x):
        return self.l2(torch.relu(self.l1(x)))

    def forward(self, x):
        return self.l3(self.features(x))


def nc_stats(z: np.ndarray, y: np.ndarray, C: int) -> dict:
    mus = np.stack([z[y == c].mean(0) for c in range(C)])
    mc = mus - mus.mean(0)
    n = mc / np.linalg.norm(mc, axis=1, keepdims=True)
    off = (n @ n.T)[~np.eye(C, dtype=bool)]
    within = sum(np.sum((z[y == c] - mus[c]) ** 2) for c in range(C))
    tot = np.sum((z - z.mean(0)) ** 2)
    return {"cos": float(off.mean()), "std": float(off.std()),
            "cv": float(np.linalg.norm(mc, axis=1).std() / np.linalg.norm(mc, axis=1).mean()),
            "nc1": float(within / (tot - within)),
            "mus": mus}


def place(z: np.ndarray, mus: np.ndarray):
    """Центрирует и, если нужно, проецирует на 3 главные оси."""
    c = z.mean(0)
    zc, mc = z - c, mus - c
    if zc.shape[1] <= 3:
        return zc, mc, 1.0, ("2d" if zc.shape[1] <= 2 else "3d")
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    frac = float(np.sum(s[:3] ** 2) / np.sum(s ** 2))
    return zc @ vt[:3].T, mc @ vt[:3].T, frac, "3d"


def draw(ax, coords, m3, y, C, kind) -> None:
    lim = float(np.abs(m3).max()) * 2.2
    for c in range(C):
        ax.scatter(*coords[y == c].T, s=9, color=COLORS[c], alpha=0.4, linewidths=0)
    for i in range(C):
        for j in range(i + 1, C):
            ax.plot(*np.c_[m3[i], m3[j]], color="0.6", lw=0.9, ls="--")
    for c in range(C):
        ax.scatter(*m3[c], s=95, color=COLORS[c], edgecolor="black", linewidths=0.8, zorder=5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if kind == "2d":
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    else:
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=32)
        ax.tick_params(labelsize=6)


def build(hidden: int, C: int) -> None:
    x, y = make_data(C)
    model = Toy(hidden, C)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    yt = torch.tensor(y)
    for _ in range(EPOCHS):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), yt).backward()
        opt.step()

    with torch.no_grad():
        feats = model.features(x).numpy().astype(float)
        logits = model(x).numpy().astype(float)
    spaces = [
        ("вход (2D)", x.numpy().astype(float), ""),
        (f"признаки ({hidden}D, выход второго слоя)", feats, ""),
        (f"логиты ({C}D, выход третьего слоя)", logits, " · центрировано"),
        (f"softmax ({C}D, на логитах)", torch.softmax(torch.tensor(logits), 1).numpy(),
         " · центрировано"),
    ]

    fig = plt.figure(figsize=(13, 11), dpi=115)
    for i, (name, z, note) in enumerate(spaces, start=1):
        st = nc_stats(z, y, C)
        coords, m3, frac, kind = place(z, st["mus"])
        ax = fig.add_subplot(2, 2, i, projection="3d" if kind == "3d" else None)
        draw(ax, coords, m3, y, C, kind)
        extra = "" if frac == 1.0 else f"\n3 оси: {frac * 100:.0f}% дисперсии"
        ax.set_title(f"{name}{note}\ncos = {st['cos']:+.3f} ± {st['std']:.3f}   CV = {st['cv']:.2f}"
                     f"\nвнутри/между = {st['nc1']:.2f}", fontsize=9)
        print(f"{hidden}->{C} {name:44s} [{kind}] cos={st['cos']:+.3f}±{st['std']:.3f} "
              f"CV={st['cv']:.3f} NC1={st['nc1']:.3f}")

    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[c], label=f"класс {c + 1}")
               for c in range(C)]
    fig.legend(handles=handles, loc="lower center", ncol=C, fontsize=11, frameon=False)
    fig.suptitle(f"Модель {hidden} → {C}: симплекс с {C} вершинами "
                 f"в четырёх системах координат", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = OUT / f"toy_{hidden}to{C}.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def main() -> None:
    for hidden, C in CONFIGS:
        build(hidden, C)


if __name__ == "__main__":
    main()
