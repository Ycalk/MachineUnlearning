"""Скрипт генерации наглядных GIF-анимаций для объяснения:
1. Ортогональная смена базиса (чистый поворот, сохранение длин и расстояний).
2. Общая смена базиса (растяжение, сжатие, перекос осей, искажение формы).
3. Случаи, когда линейная смена базиса бессмысленна (нелинейные многообразия и независимый шум).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Стиль и эстетика
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

# -----------------------------------------------------------------------------
# Общие данные для анимаций 1 и 2
# -----------------------------------------------------------------------------
np.random.seed(42)
N_POINTS = 35

# Класс 1 (синий) и Класс 2 (оранжевый) в исходном базисе
c1_center = np.array([-1.2, 0.8])
c2_center = np.array([1.1, -0.6])
pts_c1 = c1_center + np.random.randn(N_POINTS, 2) * 0.35
pts_c2 = c2_center + np.random.randn(N_POINTS, 2) * 0.35

# Выделенные реперные векторы v1 и v2
v1_orig = np.array([-1.4, 1.4])
v2_orig = np.array([1.3, -1.1])
e1_orig = np.array([1.0, 0.0])
e2_orig = np.array([0.0, 1.0])


# =============================================================================
# 1. Ортогональная смена базиса (Orthogonal Basis Change)
# =============================================================================
def generate_orthogonal_gif():
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=110)
    
    # 40 кадров: плавный поворот туда и обратно
    angles = np.linspace(0, np.pi / 3, 25)
    angles = np.concatenate([angles, angles[::-1]])  # петля туда-обратно
    
    def update(frame):
        ax.clear()
        theta = angles[frame]
        
        # Ортогональная матрица поворота
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # Трансформация точек и векторов
        c1_rot = pts_c1 @ R.T
        c2_rot = pts_c2 @ R.T
        v1_rot = v1_orig @ R.T
        v2_rot = v2_orig @ R.T
        e1_rot = e1_orig @ R.T
        e2_rot = e2_orig @ R.T
        
        # Сетка и оси
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        ax.axhline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        ax.set_aspect("equal")
        
        # Опорные окружности инвариантности длины
        for r in [1.0, np.linalg.norm(v1_orig), np.linalg.norm(v2_orig)]:
            circle = plt.Circle((0, 0), r, color="#e0e0e0", fill=False, linestyle=":", linewidth=0.9)
            ax.add_patch(circle)
            
        # Облака классов
        ax.scatter(c1_rot[:, 0], c1_rot[:, 1], color="#1f77b4", alpha=0.75, s=45, label="Класс A (кошки)", edgecolors="black", linewidths=0.5)
        ax.scatter(c2_rot[:, 0], c2_rot[:, 1], color="#ff7f0e", alpha=0.75, s=45, label="Класс B (собаки)", edgecolors="black", linewidths=0.5)
        
        # Реперные векторы v1 и v2
        ax.annotate("", xy=v1_rot, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.4))
        ax.annotate("", xy=v2_rot, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2.4))
        ax.text(v1_rot[0]*1.1, v1_rot[1]*1.1, r"$v_1$", color="#1f77b4", fontsize=12, fontweight="bold")
        ax.text(v2_rot[0]*1.1, v2_rot[1]*1.1, r"$v_2$", color="#ff7f0e", fontsize=12, fontweight="bold")
        
        # Линия между v1 и v2 (расстояние)
        ax.plot([v1_rot[0], v2_rot[0]], [v1_rot[1], v2_rot[1]], color="#2ca02c", linestyle="--", linewidth=1.8, label="Расстояние d(v1, v2)")
        
        # Базисные векторы осей
        ax.annotate("", xy=e1_rot, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#d62728", lw=3.0))
        ax.annotate("", xy=e2_rot, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#9467bd", lw=3.0))
        ax.text(e1_rot[0]*1.15, e1_rot[1]*1.15, r"$e_1$", color="#d62728", fontsize=13, fontweight="bold")
        ax.text(e2_rot[0]*1.15, e2_rot[1]*1.15, r"$e_2$", color="#9467bd", fontsize=13, fontweight="bold")
        
        # Метрики инвариантности
        dist = np.linalg.norm(v1_rot - v2_rot)
        angle_deg = np.degrees(np.arccos(np.clip(np.dot(e1_rot, e2_rot), -1.0, 1.0)))
        
        title_text = (
            "1. ОРТОГОНАЛЬНАЯ СМЕНА БАЗИСА (Чистый жесткий поворот)\n"
            rf"$\angle(e_1, e_2) = {angle_deg:.1f}^\circ$ (строго $90^\circ$)  |  "
            rf"$\|e_1\| = {np.linalg.norm(e1_rot):.2f}$, $\|e_2\| = {np.linalg.norm(e2_rot):.2f}$  |  "
            rf"$d(v_1, v_2) = {dist:.3f}$ (неизменно!)"
        )
        ax.set_title(title_text, fontsize=10.5, pad=12, fontweight="bold", color="#111111")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)
        
        # Информационная плашка
        info = (
            r"• $R^T R = I$ (ортогональность)" "\n"
            r"• Длины векторов: $\|Rx\| = \|x\|$" "\n"
            r"• Углы и расстояния сохранены на 100%" "\n"
            r"• Облако вращается как твердое тело"
        )
        ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#2ca02c", lw=1.5))
        
    anim = animation.FuncAnimation(fig, update, frames=len(angles), interval=80)
    out_path = OUT_DIR / "1_orthogonal_basis_change.gif"
    anim.save(out_path, writer="pillow", fps=12)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# 2. Общая смена базиса (General Basis Change / Linear Deformation)
# =============================================================================
def generate_general_gif():
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=110)
    
    # Неортогональная целевая матрица: растяжение по x, сжатие по y, перекос (shear)
    # Матрица перехода: A = [[1.7, 0.9], [-0.4, 0.6]]
    A_target = np.array([
        [1.65, 0.85],
        [-0.45, 0.60]
    ])
    
    t_vals = np.linspace(0, 1.0, 25)
    t_vals = np.concatenate([t_vals, t_vals[::-1]])  # петля туда-обратно
    
    def update(frame):
        ax.clear()
        t = t_vals[frame]
        
        # Линейная интерполяция между I и A_target: M(t) = (1 - t) I + t A
        M = (1.0 - t) * np.eye(2) + t * A_target
        
        # Трансформация точек и векторов
        c1_trans = pts_c1 @ M.T
        c2_trans = pts_c2 @ M.T
        v1_trans = v1_orig @ M.T
        v2_trans = v2_orig @ M.T
        e1_trans = e1_orig @ M.T
        e2_trans = e2_orig @ M.T
        
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.axhline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        ax.set_aspect("equal")
        
        # Пунктиром показываем исходный единичный круг, трансформированный в эллипс
        phi = np.linspace(0, 2 * np.pi, 100)
        unit_circle = np.stack([np.cos(phi), np.sin(phi)], axis=1)
        ellipse = unit_circle @ M.T
        ax.plot(ellipse[:, 0], ellipse[:, 1], color="#e377c2", linestyle=":", linewidth=1.5, label="Искажение единичной сферы")
        
        # Облака классов
        ax.scatter(c1_trans[:, 0], c1_trans[:, 1], color="#1f77b4", alpha=0.75, s=45, label="Класс A (деформирован)", edgecolors="black", linewidths=0.5)
        ax.scatter(c2_trans[:, 0], c2_trans[:, 1], color="#ff7f0e", alpha=0.75, s=45, label="Класс B (деформирован)", edgecolors="black", linewidths=0.5)
        
        # Реперные векторы v1 и v2
        ax.annotate("", xy=v1_trans, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.4))
        ax.annotate("", xy=v2_trans, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2.4))
        ax.text(v1_trans[0]*1.1, v1_trans[1]*1.1, r"$v_1$", color="#1f77b4", fontsize=12, fontweight="bold")
        ax.text(v2_trans[0]*1.1, v2_trans[1]*1.1, r"$v_2$", color="#ff7f0e", fontsize=12, fontweight="bold")
        
        # Линия расстояния между v1 и v2
        ax.plot([v1_trans[0], v2_trans[0]], [v1_trans[1], v2_trans[1]], color="#d62728", linestyle="--", linewidth=1.8, label="Расстояние d(v1, v2)")
        
        # Базисные векторы осей
        ax.annotate("", xy=e1_trans, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#d62728", lw=3.0))
        ax.annotate("", xy=e2_trans, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#9467bd", lw=3.0))
        ax.text(e1_trans[0]*1.15, e1_trans[1]*1.15, r"$e_1$", color="#d62728", fontsize=13, fontweight="bold")
        ax.text(e2_trans[0]*1.15, e2_trans[1]*1.15, r"$e_2$", color="#9467bd", fontsize=13, fontweight="bold")
        
        # Метрики деформации
        dist = np.linalg.norm(v1_trans - v2_trans)
        len_e1 = np.linalg.norm(e1_trans)
        len_e2 = np.linalg.norm(e2_trans)
        cos_val = np.dot(e1_trans, e2_trans) / (len_e1 * len_e2)
        angle_deg = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
        
        title_text = (
            "2. ОБЩАЯ СМЕНА БАЗИСА (Растяжение, сжатие, перекос осей)\n"
            rf"$\angle(e_1, e_2): 90^\circ \to {angle_deg:.1f}^\circ$ (перекос)  |  "
            rf"$\|e_1\| = {len_e1:.2f}, \|e_2\| = {len_e2:.2f}$  |  "
            rf"$d(v_1, v_2): {np.linalg.norm(v1_orig - v2_orig):.2f} \to {dist:.2f}$"
        )
        ax.set_title(title_text, fontsize=10.5, pad=12, fontweight="bold", color="#111111")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)
        
        # Информационная плашка
        info = (
            r"• $M^T M \neq I$ (не ортогонально)" "\n"
            r"• Оси наклоняются друг к другу ($\angle \neq 90^\circ$)" "\n"
            r"• Длины осей изменяются ($\|e_1\| \neq \|e_2\|$)" "\n"
            r"• Форма облака сминается: круг $\to$ эллипс"
        )
        ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff5f5", edgecolor="#d62728", lw=1.5))
        
    anim = animation.FuncAnimation(fig, update, frames=len(t_vals), interval=80)
    out_path = OUT_DIR / "2_general_basis_change.gif"
    anim.save(out_path, writer="pillow", fps=12)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# 3. Когда линейная смена базиса бессмысленна (Nonlinear vs Uncorrelated Noise)
# =============================================================================
def generate_useless_linear_gif():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=110)
    
    # ---------------- Панель 1: Нелинейная структура (концентрические кольца)
    np.random.seed(10)
    n_ring = 80
    r_inner = np.random.uniform(0.4, 0.8, n_ring)
    phi_inner = np.random.uniform(0, 2 * np.pi, n_ring)
    ring_inner = np.stack([r_inner * np.cos(phi_inner), r_inner * np.sin(phi_inner)], axis=1)
    
    r_outer = np.random.uniform(1.4, 1.8, n_ring)
    phi_outer = np.random.uniform(0, 2 * np.pi, n_ring)
    ring_outer = np.stack([r_outer * np.cos(phi_outer), r_outer * np.sin(phi_outer)], axis=1)
    
    # Линейное преобразование пытается вытянуть кольца
    M_nonlin_target = np.array([[2.2, 0.4], [0.1, 0.5]])
    
    # ---------------- Панель 2: Независимый случайный шум (МНК стремится к 0)
    n_noise = 80
    noise_X = np.random.randn(n_noise, 2) * 1.0
    noise_Y = np.random.randn(n_noise, 2) * 1.0
    # МНК матрица для независимого шума
    G_noise = np.linalg.lstsq(noise_X, noise_Y, rcond=None)[0]
    # G_noise очень мала: ||G|| ~ 0
    
    t_vals = np.linspace(0, 1.0, 25)
    t_vals = np.concatenate([t_vals, t_vals[::-1]])
    
    def update(frame):
        t = t_vals[frame]
        
        # 1. Левая панель: Нелинейность
        ax1.clear()
        M1 = (1.0 - t) * np.eye(2) + t * M_nonlin_target
        r_in_t = ring_inner @ M1.T
        r_out_t = ring_outer @ M1.T
        
        ax1.set_xlim(-4.0, 4.0)
        ax1.set_ylim(-4.0, 4.0)
        ax1.set_aspect("equal")
        ax1.axhline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        ax1.axvline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        
        ax1.scatter(r_in_t[:, 0], r_in_t[:, 1], color="#1f77b4", s=40, alpha=0.8, label="Класс 1 (внутренний)")
        ax1.scatter(r_out_t[:, 0], r_out_t[:, 1], color="#d62728", s=40, alpha=0.8, label="Класс 2 (внешнее кольцо)")
        
        ax1.set_title(
            "Случай А: Нелинейная структура данных\n"
            "(Линейный МНК не способен распутать кольца)",
            fontsize=11, fontweight="bold", pad=10
        )
        ax1.legend(loc="upper right", fontsize=8.5)
        
        info1 = (
            "МНК может только сжать/повернуть кольцо в эллипс.\n"
            "Класс 1 навсегда заперт внутри Класса 2!\n"
            "Никакая смена базиса не разделит их линейно."
        )
        ax1.text(0.03, 0.03, info1, transform=ax1.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9db", edgecolor="#f59f00", lw=1.5))
        
        # 2. Правая панель: Независимый шум
        ax2.clear()
        # Линейный МНК отображает X -> X * (t * G_noise)
        # Поскольку G_noise ~ 0, при t -> 1 точки схлопываются в центр!
        noise_t = noise_X @ ((1.0 - t) * np.eye(2) + t * G_noise).T
        
        ax2.set_xlim(-3.0, 3.0)
        ax2.set_ylim(-3.0, 3.0)
        ax2.set_aspect("equal")
        ax2.axhline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        ax2.axvline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        
        ax2.scatter(noise_Y[:, 0], noise_Y[:, 1], color="#aaaaaa", s=25, alpha=0.4, label="Цель Y (чужой шум)")
        ax2.scatter(noise_t[:, 0], noise_t[:, 1], color="#9467bd", s=45, alpha=0.8, edgecolors="black", linewidths=0.5, label="Предсказание МНК: X @ G")
        
        ax2.set_title(
            "Случай Б: Независимый шум (нет корреляции)\n"
            "МНК минимизирует ошибку, схлопывая всё в центр",
            fontsize=11, fontweight="bold", pad=10
        )
        residual = np.linalg.norm(noise_t - noise_Y) / np.linalg.norm(noise_Y)
        info2 = (
            r"• Корреляция $X^T Y \approx 0 \to G \approx 0$" "\n"
            r"• МНК схлопывает точки в $(0, 0)$" "\n"
            rf"• Ошибка Residual $\approx {residual:.2f}$ ($100\%$ провал)" "\n"
            "МНК не может придумать связь из ничего."
        )
        ax2.text(0.03, 0.03, info2, transform=ax2.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5))
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.86, bottom=0.04)
    anim = animation.FuncAnimation(fig, update, frames=len(t_vals), interval=80)
    out_path = OUT_DIR / "3_when_linear_change_fails.gif"
    anim.save(out_path, writer="pillow", fps=12)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Генерация GIF 1...")
    generate_orthogonal_gif()
    print("Генерация GIF 2...")
    generate_general_gif()
    print("Генерация GIF 3...")
    generate_useless_linear_gif()
    print("Все GIF успешно созданы в папке visualizations/!")
