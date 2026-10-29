"""GIF-анимации с двумя облаками и полноценными легендами.

4. `4_two_clouds_orthogonal.gif` — $Y$ получается из $X$ поворотом: одно облако
   ложится на другое, форма сохраняется, невязка почти нулевая.
5. Три отдельных GIF на одних и тех же облаках:
   `5a_rotation_fails.gif`  — $Y$ связан с $X$ линейно, но не поворотом: поворот
                              не дотягивается ни при каком угле;
   `5b_lstsq_works.gif`     — те же облака, общая линейная матрица ложится точно,
                              но ценой формы облака;
   `5c_lstsq_fails.gif`     — $Y$ связан с $X$ нелинейно: не дотягивается даже
                              лучшая общая линейная матрица.
6. `6_active_vs_passive.gif` — активный и пассивный взгляд на одну и ту же
   смену базиса.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"

C_A = "#1f77b4"
C_B = "#ff7f0e"
C_E1 = "#d62728"
C_E2 = "#9467bd"
C_GREY = "#909090"
C_TGT_F = "#2ca02c"
C_TEXT = "#222222"   # единый цвет заголовков
C_BOX = "#666666"    # единая рамка инфо-блока

RNG = np.random.default_rng(7)


# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_clouds():
    a = np.array([-1.25, 0.85]) + RNG.normal(scale=0.30, size=(45, 2))
    b = np.array([1.15, -0.65]) + RNG.normal(scale=0.30, size=(45, 2))
    return a, b


def cov_ellipse(pts: np.ndarray, nstd: float = 1.6):
    mu = pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(pts.T))
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    return mu, 2 * nstd * np.sqrt(w[0]), 2 * nstd * np.sqrt(w[1]), angle


def residual(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def fit_procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ортогональная матрица R, минимизирующая ||X R^T - Y|| (используем X @ R.T)."""
    u, _, vt = np.linalg.svd(x.T @ y)
    return vt.T @ u.T


def fit_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Общая линейная матрица M, минимизирующая ||X M^T - Y||."""
    return np.linalg.lstsq(x, y, rcond=None)[0].T


def setup(ax, lim: float = 3.4) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#cfcfcf", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#cfcfcf", linestyle="--", linewidth=0.8)


def draw_basis(ax, e1, e2, prefix: str = "e", dashed: bool = False,
               c1: str = C_E1, c2: str = C_E2) -> None:
    """Сплошные стрелки — активный базис; штриховые стрелки с маркером на конце —
    неподвижный базис оракула."""
    common = dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=17)
    if dashed:
        style = dict(common, linestyle=(0, (4, 2.5)), lw=2.6)
    else:
        style = dict(common, lw=3.2)
    ax.annotate("", xy=e1, xytext=(0, 0), arrowprops=dict(style, color=c1))
    ax.annotate("", xy=e2, xytext=(0, 0), arrowprops=dict(style, color=c2))

    if dashed:
        for tip, col in ((e1, c1), (e2, c2)):
            ax.plot(*tip, marker="o", mfc="white", mec=col, ms=9, mew=2.0, zorder=6)

    kwargs = dict(fontsize=12, fontweight="bold")
    if dashed:
        kwargs["bbox"] = dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85)
    ax.text(e1[0] * 1.16, e1[1] * 1.16, rf"${prefix}_1$", color=c1, **kwargs)
    ax.text(e2[0] * 1.16, e2[1] * 1.16, rf"${prefix}_2$", color=c2, **kwargs)


def save_gif(fig, out, frames: int, draw_frame, fps: int = 11, colors: int = 128) -> None:
    """Явная сборка кадров: полный кадр на каждый шаг, disposal=2.

    Без промежуточной дельта-оптимизации и без дизеринга — иначе плеер
    накапливает предыдущие кадры (ghosting) и файл раздувается.
    """
    pil_frames = []
    for i in range(frames):
        draw_frame(i)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        pil_frames.append(
            Image.fromarray(buf).quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
        )
    pil_frames[0].save(
        out, save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / fps), loop=0, disposal=2, optimize=False,
    )
    plt.close(fig)
    print("saved", out, f"{len(pil_frames)} frames", pil_frames[0].size)


def legend_handles(extra=()) -> list:
    """Полный набор обозначений: обе цели, оба облака, оба эллипса, обе оси."""
    return [
        Line2D([], [], marker="o", ls="None", mfc="none", mec=C_A, ms=7,
               label="цель $Y$ (оракул), класс A"),
        Line2D([], [], marker="o", ls="None", mfc="none", mec=C_B, ms=7,
               label="цель $Y$ (оракул), класс B"),
        Line2D([], [], marker="o", ls="None", mfc=C_A, mec="black", mew=0.5, ms=6,
               label="$X$ (baseline), класс A"),
        Line2D([], [], marker="o", ls="None", mfc=C_B, mec="black", mew=0.5, ms=6,
               label="$X$ (baseline), класс B"),
        Line2D([], [], color=C_GREY, ls=":", lw=1.6,
               label="эллипс класса A: исходный"),
        Line2D([], [], color="#0b4f7a", ls="--", lw=1.9,
               label="эллипс класса A: текущий"),
        Line2D([], [], color=C_E1, lw=3.0, label="ось $e_1$"),
        Line2D([], [], color=C_E2, lw=3.0, label="ось $e_2$"),
        *extra,
    ]


def full_legend(ax, loc: str = "upper left", fontsize: float = 6.6, ncol: int = 2, extra=()) -> None:
    ax.legend(handles=legend_handles(extra), loc=loc, fontsize=fontsize, ncol=ncol,
              framealpha=0.93, borderpad=0.5, columnspacing=0.9, handletextpad=0.6)


def draw_cloud(ax, x: np.ndarray, alpha: float = 0.9, size: float = 38) -> None:
    ax.scatter(x[:45, 0], x[:45, 1], color=C_A, alpha=alpha, s=size, edgecolors="none", zorder=4)
    ax.scatter(x[45:, 0], x[45:, 1], color=C_B, alpha=alpha, s=size, edgecolors="none", zorder=4)


def draw_target(ax, y: np.ndarray, size: float = 40) -> None:
    ax.scatter(y[:45, 0], y[:45, 1], facecolors="none", edgecolors=C_A, s=size, lw=1.2, zorder=5)
    ax.scatter(y[45:, 0], y[45:, 1], facecolors="none", edgecolors=C_B, s=size, lw=1.2, zorder=5)


# -----------------------------------------------------------------------------
# 4. Два облака, ортогональная смена базиса
# -----------------------------------------------------------------------------
def gif_two_clouds_orthogonal():
    a, b = make_clouds()
    x = np.vstack([a, b])

    theta0 = np.deg2rad(38.0)
    r_true = rot(theta0)
    y = x @ r_true.T + RNG.normal(scale=0.025, size=x.shape)
    r_fit = fit_procrustes(x, y)
    eps_final = residual(x @ r_fit.T, y)

    ts = np.linspace(0.0, 1.0, 22)
    ts = np.concatenate([ts, ts[::-1]])

    fig, ax = plt.subplots(figsize=(7.8, 7.8), dpi=100)

    def update(frame):
        ax.clear()
        t = ts[frame]
        ang = t * np.arctan2(r_fit[1, 0], r_fit[0, 0])
        r = rot(ang)
        x_t = x @ r.T

        setup(ax)
        for radius in (1.0, 2.0):
            ax.add_patch(plt.Circle((0, 0), radius, color="#e6e6e6", fill=False, ls=":", lw=0.9))

        draw_target(ax, y)
        draw_cloud(ax, x_t)

        mu0, w0, h0, e0 = cov_ellipse(x[:45])
        mu, w, h, e = cov_ellipse(x_t[:45])
        ax.add_patch(Ellipse(mu0, w0, h0, angle=e0, fill=False, ls=":", lw=1.5, color=C_GREY, zorder=6))
        ax.add_patch(Ellipse(mu, w, h, angle=e, fill=False, ls="--", lw=1.9, color="#0b4f7a", zorder=7))

        draw_basis(ax, r[0], r[1], "e")
        draw_basis(ax, r_true[0], r_true[1], "f", dashed=True, c1="#8c564b", c2="#17becf")

        title = "\n".join([
            r"Два облака: $X$ (baseline) ложится на $Y$ (оракул)",
            rf"поворот базиса $e \to f$: ${abs(np.degrees(ang)):.0f}^\circ$",
            rf"$\|x\| = {np.linalg.norm(x_t[0]):.2f}$ (неизменно)   |   "
            rf"$\varepsilon = {residual(x_t, y):.3f}$ (предел ${eps_final:.3f}$)",
        ])
        ax.set_title(title, fontsize=9.8, pad=10, fontweight="bold", color=C_TEXT)

        full_legend(ax, fontsize=6.8, ncol=2, extra=[
            Line2D([], [], color="#8c564b", ls=(0, (4, 2.5)), lw=2.6, marker="o",
                   mfc="white", mec="#8c564b", mew=1.6, ms=7,
                   label="ось $f_1$ (оракул, штрих)"),
            Line2D([], [], color="#17becf", ls=(0, (4, 2.5)), lw=2.6, marker="o",
                   mfc="white", mec="#17becf", mew=1.6, ms=7,
                   label="ось $f_2$ (оракул, штрих)"),
        ])
        ax.text(0.03, 0.03,
                r"• $Y \approx X R^T$: одно облако ложится на другое" "\n"
                r"• длины, углы и расстояния между точками сохранены" "\n"
                r"• при $\varepsilon \approx 0$ геометрия двух моделей идентична",
                transform=ax.transAxes, fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C_BOX, lw=1.4))

    out = OUT_DIR / "4_two_clouds_orthogonal.gif"
    save_gif(fig, out, len(ts), update, fps=12)


# -----------------------------------------------------------------------------
# 5. Три случая: поворот не тянет / общая матрица тянет / не тянет никто
# -----------------------------------------------------------------------------
def case_data() -> dict:
    """Одни и те же облака для всех трёх случаев (RNG сбрасывается)."""
    global RNG
    RNG = np.random.default_rng(7)
    a, b = make_clouds()
    x = np.vstack([a, b])

    # линейная связь, но не поворот: растяжение + сжатие + перекос
    deform = np.array([[1.80, 0.30], [-0.45, 0.52]])
    y_lin = x @ deform.T + RNG.normal(scale=0.025, size=x.shape)

    # нелинейная связь: квадратичные члены
    warp = np.stack([
        0.80 * x[:, 0] + 0.35 * x[:, 1] ** 2,
        0.90 * x[:, 1] - 0.30 * x[:, 0] ** 2,
    ], axis=1)
    y_nl = warp + RNG.normal(scale=0.03, size=x.shape)

    return {
        "x": x,
        "y_lin": y_lin,
        "y_nl": y_nl,
        "r_lin": fit_procrustes(x, y_lin),
        "m_lin": fit_lstsq(x, y_lin),
        "m_nl": fit_lstsq(x, y_nl),
    }


def gif_single_case(out_name: str, d: dict, y_key: str, mat_key: str,
                    rotation: bool, title: str, box, fps: int = 11,
                    frames: int = 22) -> None:
    """Один случай — один GIF: движущееся облако X и неподвижная цель Y."""
    x, y, mat = d["x"], d[y_key], d[mat_key]
    ts = np.linspace(0.0, 1.0, frames)
    ts = np.concatenate([ts, ts[::-1]])

    mu0, w0, h0, e0 = cov_ellipse(x[:45])
    ratio0 = w0 / h0
    eps_final = residual(x @ mat.T, y)
    ang_final = np.arctan2(mat[1, 0], mat[0, 0])

    fig, ax = plt.subplots(figsize=(7.8, 7.9), dpi=100)

    def update(frame):
        ax.clear()
        t = ts[frame]
        cur = rot(t * ang_final) if rotation else (1.0 - t) * np.eye(2) + t * mat
        x_t = x @ cur.T

        setup(ax, lim=4.3)
        draw_target(ax, y)
        draw_cloud(ax, x_t, alpha=0.9, size=34)

        mu, w, h, e = cov_ellipse(x_t[:45])
        ax.add_patch(Ellipse(mu0, w0, h0, angle=e0, fill=False, ls=":", lw=1.5,
                             color=C_GREY, zorder=6))
        ax.add_patch(Ellipse(mu, w, h, angle=e, fill=False, ls="--", lw=1.9,
                             color="#0b4f7a", zorder=7))
        draw_basis(ax, cur[0], cur[1], "e")

        ax.set_title(
            "\n".join([title,
                       rf"$\varepsilon$ = {residual(x_t, y):.3f}   |   "
                       rf"предел $\varepsilon^*$ = {eps_final:.3f}"]),
            fontsize=9.8, fontweight="bold", color=C_TEXT, pad=10)

        full_legend(ax, fontsize=6.8, ncol=2)
        text = box(x_t) if callable(box) else box
        ax.text(0.03, 0.03, text, transform=ax.transAxes, fontsize=8.4,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C_BOX,
                          lw=1.4, alpha=0.95))

    save_gif(fig, OUT_DIR / out_name, len(ts), update, fps=fps)


def gif_rotation_fails():
    """Ортогональный поворот не справляется с теми же облаками."""
    gif_single_case(
        "5a_rotation_fails.gif", case_data(), "y_lin", "r_lin", rotation=True,
        title="Поворот не работает: связь линейна, но это не поворот",
        box=(
            r"• $R^T R = I$: форму менять нельзя" "\n"
            r"• облако не ложится на цель" "\n"
            r"• ни при каком угле поворота"
        ),
    )


def gif_lstsq_works():
    """Те же облака, но общая линейная матрица дотягивается."""
    d = case_data()
    mu_x, w_x, h_x, _ = cov_ellipse(d["x"][:45])
    ratio0 = w_x / h_x

    def box(x_t):
        mu, w, h, _ = cov_ellipse(x_t[:45])
        return (
            r"• $M^T M \neq I$: растяжение и перекос" "\n"
            rf"• соотношение осей эллипса: $\times {(w / h) / ratio0:.2f}$" "\n"
            r"• дотягивается ценой формы облака"
        )
    gif_single_case(
        "5b_lstsq_works.gif", d, "y_lin", "m_lin", rotation=False,
        title="МНК работает: те же облака, общая линейная матрица",
        box=box,
    )


def gif_lstsq_fails():
    """Нелинейная связь: не справляется даже общая линейная матрица."""
    gif_single_case(
        "5c_lstsq_fails.gif", case_data(), "y_nl", "m_nl", rotation=False,
        title="МНК не работает: связь между облаками нелинейна",
        box=(
            r"• лучшая из всех линейных матриц" "\n"
            r"• точки всё равно не совпадают" "\n"
            r"• остаток принципиальный, не шум"
        ),
    )


# -----------------------------------------------------------------------------
# 6. Активный vs пассивный взгляд
# -----------------------------------------------------------------------------
def gif_active_vs_passive():
    a, b = make_clouds()
    x = np.vstack([a, b])
    v = np.array([-0.9, 1.6])

    theta0 = np.deg2rad(40.0)
    ts = np.linspace(0.0, 1.0, 18)
    ts = np.concatenate([ts, ts[::-1]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 7.0), dpi=100)
    fig.subplots_adjust(top=0.86, bottom=0.04, wspace=0.14)

    def update(frame):
        t = ts[frame]
        ang = t * theta0
        r = rot(ang)
        ax1.clear()
        ax2.clear()
        setup(ax1)
        setup(ax2)

        # ---- активный взгляд: точки вращаются, оси неподвижны
        x_a = x @ r.T
        v_a = r @ v
        draw_cloud(ax1, x_a, size=34)
        draw_basis(ax1, np.array([1.0, 0.0]), np.array([0.0, 1.0]), "e")
        ax1.annotate("", xy=v_a, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#111111", lw=2.2))
        ax1.text(v_a[0] * 1.1, v_a[1] * 1.1,
                 rf"$Rx = ({v_a[0]:.2f}, {v_a[1]:.2f})$", fontsize=9, fontweight="bold")
        ax1.set_title(
            "Активный взгляд: точки двигаются, оси неподвижны\n"
            r"преобразование $x \mapsto Rx$",
            fontsize=9.8, fontweight="bold", pad=10, color=C_TEXT)
        ax1.legend(handles=[
            Line2D([], [], marker="o", ls="None", mfc=C_A, mec="black", mew=0.5, ms=6,
                   label="точки, класс A: $Rx$ (повернуты)"),
            Line2D([], [], marker="o", ls="None", mfc=C_B, mec="black", mew=0.5, ms=6,
                   label="точки, класс B: $Rx$ (повернуты)"),
            Line2D([], [], color=C_E1, lw=3.0, label="ось $e_1$ (неподвижна)"),
            Line2D([], [], color=C_E2, lw=3.0, label="ось $e_2$ (неподвижна)"),
            Line2D([], [], color="#111111", lw=2.2, label="вектор $v$: повернут в $Rv$"),
        ], loc="upper left", fontsize=7.4, ncol=1, framealpha=0.93)

        # ---- пассивный взгляд: точки неподвижны, оси повернуты на R^T
        e1_new = r.T @ np.array([1.0, 0.0])
        e2_new = r.T @ np.array([0.0, 1.0])
        draw_cloud(ax2, x, size=34)
        draw_basis(ax2, e1_new, e2_new, "e")
        ax2.annotate("", xy=v, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#111111", lw=2.2))
        coords = r @ v
        ax2.text(v[0] * 1.1, v[1] * 1.1,
                 rf"координаты $= ({coords[0]:.2f}, {coords[1]:.2f})$", fontsize=9, fontweight="bold")
        ax2.set_title(
            "Пассивный взгляд: точки неподвижны, оси повернуты на $R^T$\n"
            r"смена базиса: $x \mapsto R^{-1}x$",
            fontsize=9.8, fontweight="bold", pad=10, color=C_TEXT)
        ax2.legend(handles=[
            Line2D([], [], marker="o", ls="None", mfc=C_A, mec="black", mew=0.5, ms=6,
                   label="точки, класс A: $x$ (неподвижны)"),
            Line2D([], [], marker="o", ls="None", mfc=C_B, mec="black", mew=0.5, ms=6,
                   label="точки, класс B: $x$ (неподвижны)"),
            Line2D([], [], color=C_E1, lw=3.0, label=r"ось $e_1$: повернута на $R^T$"),
            Line2D([], [], color=C_E2, lw=3.0, label=r"ось $e_2$: повернута на $R^T$"),
            Line2D([], [], color="#111111", lw=2.2, label="вектор $v$: неподвижен"),
        ], loc="upper left", fontsize=7.4, ncol=1, framealpha=0.93)

        fig.suptitle(
            r"Это одно и то же: повернуть облако при неподвижных осях  =  оставить облако и повернуть базис",
            fontsize=11.0, fontweight="bold", y=0.975, color=C_TEXT)

    out = OUT_DIR / "6_active_vs_passive.gif"
    save_gif(fig, out, len(ts), update, fps=11)


if __name__ == "__main__":
    gif_two_clouds_orthogonal()
    gif_rotation_fails()
    gif_lstsq_works()
    gif_lstsq_fails()
    gif_active_vs_passive()
    print("готово")
