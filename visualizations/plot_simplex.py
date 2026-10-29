"""Иллюстрация правильных симплексов (simplex ETF) для C = 2, 3, 4.

Все вершины лежат на единичной сфере в (C-1)-мерном пространстве: радиус-векторы
имеют длину 1, все попарные расстояния равны. Тетраэдр вписан в единичную сферу.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
GREY = "#9aa0a6"
DARK = "#222222"


def project(points, az=55.0, el=22.0):
    az, el = np.deg2rad(az), np.deg2rad(el)
    u = -points[:, 0] * np.sin(az) + points[:, 1] * np.cos(az)
    v = -(points[:, 0] * np.cos(az) + points[:, 1] * np.sin(az)) * np.sin(el) + points[:, 2] * np.cos(el)
    return np.stack([u, v], axis=1)


def sphere_wireframe(ax, az=55.0, el=22.0):
    t = np.linspace(0, 2 * np.pi, 240)
    for z in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        r = np.sqrt(max(1 - z**2, 0.0))
        xy = project(np.stack([r * np.cos(t), r * np.sin(t), np.full_like(t, z)], 1), az, el)
        ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=0.7, alpha=0.35, zorder=1)
    for phi in np.deg2rad(np.arange(0, 180, 30)):
        xy = project(np.stack([np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t)], 1), az, el)
        ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=0.7, alpha=0.35, zorder=1)


def panel_1d(ax):
    ax.plot([-1.4, 1.4], [0, 0], color=GREY, lw=1.2, zorder=1)
    for x in (-1, 1):
        ax.plot([x, x], [-0.22, 0.22], color=GREY, lw=1.0, ls=(0, (3, 4)), zorder=1)
    for x, c in ((-1, C[0]), (1, C[1])):
        ax.annotate("", xy=(x, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.8, mutation_scale=17), zorder=4)
        ax.plot([x], [0], marker="o", ms=12, mfc="white", mec=c, mew=2.6, zorder=5)
    ax.plot([0], [0], marker="o", ms=4.5, color=DARK, zorder=5)
    for x, lab in ((-1, "-1"), (1, "+1")):
        ax.text(x, -0.36, lab, ha="center", va="top", fontsize=11.5, color=DARK)
    ax.text(0, -0.36, "0", ha="center", va="top", fontsize=11.5, color=DARK)
    ax.text(0, 0.30, r"$|v_1| = |v_2| = 1$", ha="center", fontsize=12, color=DARK)
    ax.text(0, -0.72, "вершины: две точки ±1\nрасстояние между ними = 2",
            ha="center", va="top", fontsize=10.5, color=GREY, linespacing=1.5)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.05, 0.62)


def panel_2d(ax):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=GREY, lw=1.0, ls=(0, (4, 4)), alpha=0.85, zorder=1)
    ang = np.deg2rad([90, 210, 330])
    V = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    for i in range(3):
        a, b = V[i], V[(i + 1) % 3]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=DARK, lw=1.7, zorder=3)
    for i, v in enumerate(V):
        ax.annotate("", xy=v * 0.97, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=C[i], lw=2.5, mutation_scale=16), zorder=4)
        ax.plot([v[0]], [v[1]], marker="o", ms=11, mfc="white", mec=C[i], mew=2.4, zorder=5)
    ax.plot([0], [0], marker="o", ms=4.5, color=DARK, zorder=5)
    ax.text(0.30, 0.34, "120°", ha="center", fontsize=12, color=DARK)
    ax.text(0, -1.42, r"сторона $= \sqrt{3} \approx 1.732$",
            ha="center", va="top", fontsize=10.5, color=GREY)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.62, 1.22)


def panel_3d(ax):
    az, el = 55.0, 22.0
    sphere_wireframe(ax, az, el)
    r = np.sqrt(8.0 / 9.0)
    V = np.array([
        [0.0, 0.0, 1.0],
        [0.0, r, -1.0 / 3.0],
        [r * np.cos(np.deg2rad(210)), r * np.sin(np.deg2rad(210)), -1.0 / 3.0],
        [r * np.cos(np.deg2rad(330)), r * np.sin(np.deg2rad(330)), -1.0 / 3.0],
    ])
    P = project(V, az, el)
    for i in range(4):
        for j in range(i + 1, 4):
            p, q = P[i], P[j]
            ax.plot([p[0], q[0]], [p[1], q[1]], color=DARK, lw=1.9, zorder=3)
    centre = project(np.zeros((1, 3)), az, el)[0]
    for i, p in enumerate(P):
        ax.plot([centre[0], p[0]], [centre[1], p[1]], color=C[i], lw=1.3,
                ls=(0, (3, 3)), alpha=0.95, zorder=4)
        ax.plot([p[0]], [p[1]], marker="o", ms=11, mfc="white", mec=C[i], mew=2.4, zorder=6)
    ax.plot([centre[0]], [centre[1]], marker="o", ms=3.5, color=DARK, zorder=7)
    ax.text(0, -1.52, r"ребро $= 2\sqrt{2/3} \approx 1.633$",
            ha="center", va="top", fontsize=10.5, color=GREY)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.78, 1.30)


positions = ([0.02, 0.13, 0.30, 0.70], [0.355, 0.13, 0.30, 0.70], [0.69, 0.13, 0.30, 0.70])
fig = plt.figure(figsize=(18.0, 6.6), dpi=150)
axes = [fig.add_axes(p) for p in positions]
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

panel_1d(axes[0])
panel_2d(axes[1])
panel_3d(axes[2])

titles = [
    "$C = 2$ — отрезок в 1D",
    "$C = 3$ — треугольник в 2D",
    "$C = 4$ — тетраэдр в 3D",
]
for x, title in zip((0.17, 0.505, 0.84), titles):
    fig.text(x, 0.905, title, ha="center", va="center", fontsize=14, color=DARK)

fig.text(0.5, 0.975,
         "Правильный симплекс (simplex ETF): C вершин на единичной сфере в (C−1)-мерном пространстве",
         ha="center", va="center", fontsize=15.5, color=DARK)
fig.text(0.5, 0.045,
         "все попарные углы равны arccos(−1/(C−1));   все радиус-векторы имеют длину 1;   "
         "сумма всех векторов равна нулю",
         ha="center", va="center", fontsize=11, color=GREY)

out = OUT_DIR / "simplex_etf.png"
fig.savefig(out, facecolor="white")
plt.close(fig)
print("saved", out)
