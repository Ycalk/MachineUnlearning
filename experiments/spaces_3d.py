"""Три пространства реальной модели в 3D: Flatten, S2_embedding, логиты.

Для каждого пространства:
  * PCA на 3 главные оси (SVD по центрированным данным), доля дисперсии;
  * NC-статистика в ИСХОДНОЙ размерности (cos, CV норм, спектр Грам);
  * 3D-скетч: точки (подвыборка) + центры классов.

Запуск: poetry run python spaces_3d.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent / "convolutional"))
from train_all_models import ConvolutionalModel as Model  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "summary"
OUT.mkdir(exist_ok=True)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
COLORS = plt.get_cmap("tab10").colors
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def load_model() -> Model:
    path = glob.glob(
        "/home/ycalk/.cache/kagglehub/models/ycalkk/convolutional-unlearning-cloud"
        "/pyTorch/full-baseline/1/init_7_train_7/weights_epoch_30.pt"
    )[0]
    model = Model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def load_test() -> tuple[torch.Tensor, np.ndarray]:
    root = Path(glob.glob("/home/ycalk/.cache/kagglehub/datasets/pankrzysiu/cifar10-python/versions/*")[0])
    test = datasets.CIFAR10(root=str(root), train=False)
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    x = (torch.from_numpy(test.data).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
    return x, np.asarray(test.targets)


@torch.no_grad()
def taps(model: Model, x: torch.Tensor) -> dict[str, np.ndarray]:
    flat, emb, logit = [], [], []
    for i in range(0, len(x), 1000):
        f = model.head[0](model.features(x[i:i + 1000]))
        e = torch.relu(model.head[1](f))
        flat.append(f)
        emb.append(e)
        logit.append(model.head[4](e))
    return {
        "Flatten (2048)": torch.cat(flat).numpy().astype(np.float64),
        "S2_embedding (128)": torch.cat(emb).numpy().astype(np.float64),
        "logits (10)": torch.cat(logit).numpy().astype(np.float64),
    }


def nc_stats(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mus = np.stack([z[y == c].mean(0) for c in range(10)])
    mc = mus - mus.mean(0)
    n = mc / np.linalg.norm(mc, axis=1, keepdims=True)
    off = (n @ n.T)[~np.eye(10, dtype=bool)]
    gram = np.linalg.eigvalsh(n @ n.T)[::-1]
    within = sum(np.sum((z[y == c] - mus[c]) ** 2) for c in range(10))
    tot = np.sum((z - z.mean(0)) ** 2)
    return {
        "cos_mean": float(off.mean()),
        "cos_std": float(off.std()),
        "norm_cv": float(np.linalg.norm(mc, axis=1).std() / np.linalg.norm(mc, axis=1).mean()),
        "nc1": float(within / (tot - within)),
        "gram": np.round(gram, 2).tolist(),
        "mus": mus,
    }


def panel(ax, z: np.ndarray, y: np.ndarray, name: str) -> None:
    st = nc_stats(z, y)
    zc = z - z.mean(0)
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    frac = float(np.sum(s[:3] ** 2) / np.sum(s ** 2))
    coords = zc @ vt[:3].T
    mug = st["mus"] - z.mean(0)
    mus3 = mug @ vt[:3].T
    k = np.random.default_rng(0).choice(len(z), 700, replace=False)
    for c in range(10):
        m = (y[k] == c)
        ax.scatter(coords[k][m, 0], coords[k][m, 1], coords[k][m, 2],
                   s=14, color=COLORS[c], alpha=0.5, linewidths=0)
    for c in range(10):
        ax.plot([0, mus3[c, 0]], [0, mus3[c, 1]], [0, mus3[c, 2]],
                color="0.55", lw=0.7, ls="--")
        ax.scatter(*mus3[c], s=110, color=COLORS[c], edgecolor="black", linewidths=0.9, zorder=5)
    lim = float(np.abs(mus3).max()) * 2.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(
        f"{name}\n"
        f"3 оси: {frac * 100:.0f}% дисперсии | cos {st['cos_mean']:+.3f} ± {st['cos_std']:.3f}"
        f" (идеал −0.111 ± 0)\n"
        f"CV норм {st['norm_cv']:.2f} | внутри/между {st['nc1']:.2f} | Грам {st['gram'][:3]}",
        fontsize=9,
    )
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_zlabel("PC3", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=20, azim=32)


def main() -> None:
    model, (x, y) = load_model(), load_test()
    spaces = taps(model, x)
    fig = plt.figure(figsize=(19, 7.2), dpi=110)
    for i, (name, z) in enumerate(spaces.items(), start=1):
        panel(fig.add_subplot(1, 3, i, projection="3d"), z, y, name)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[c], label=CLASS_NAMES[c])
               for c in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=10, frameon=False)
    fig.suptitle("Реальная модель: три пространства представлений, проекция на 3 главные оси",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = OUT / "spaces_3d.png"
    fig.savefig(out)
    print("saved", out)
    for name, z in spaces.items():
        st = nc_stats(z, y)
        print(f"{name:22s} dim={z.shape[1]:5d} cos={st['cos_mean']:+.3f}±{st['cos_std']:.3f} "
              f"CV={st['norm_cv']:.3f} NC1={st['nc1']:.3f} gram={st['gram']}")


if __name__ == "__main__":
    main()
