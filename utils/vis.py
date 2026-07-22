from typing import Dict, Optional
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_losses(loss_hist: Dict[str, list], save_dir: Optional[str] = None) -> None:
    """Plot loss curves on a log scale.

    Args:
        loss_hist: dict of lists with keys 'total','data','div','vort','ux','vy'
        save_dir: optional directory to save the figure
    """
    plt.figure(figsize=(10, 4))
    for k in ["total", "data", "div", "vort", "ux", "vy"]:
        if k in loss_hist and len(loss_hist[k]) > 0:
            plt.semilogy(loss_hist[k], label=k)
    plt.xlabel("Iterations (logged)")
    plt.ylabel("Loss (log)")
    plt.title("Loss history")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        ensure_dir(save_dir)
        plt.savefig(os.path.join(save_dir, "loss_history.png"), dpi=150)
    plt.show()

    split_keys = ["train_data", "val_data", "test_data"]
    if any(k in loss_hist and len(loss_hist[k]) > 0 for k in split_keys):
        plt.figure(figsize=(8, 4))
        for k in split_keys:
            values = np.asarray(loss_hist.get(k, []), dtype=np.float64)
            if values.size:
                finite = np.isfinite(values)
                if finite.any():
                    plt.semilogy(np.where(finite, values, np.nan), label=k)
        plt.xlabel("Iterations (logged)")
        plt.ylabel("Masked data MSE (log)")
        plt.title("Train / val / test data loss")
        plt.legend()
        plt.tight_layout()
        if save_dir:
            ensure_dir(save_dir)
            plt.savefig(os.path.join(save_dir, "split_data_loss_history.png"), dpi=150)
        plt.show()


def imshow3(A: np.ndarray, B: np.ndarray, C: np.ndarray, titles=("A", "B", "|A-B|"),
            fname: Optional[str] = None, extent=None, third_label: str = "Absolute error") -> None:
    """Show three maps, with a shared A/B scale and an explicit third-panel scale."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    ab_min = min(float(np.nanmin(A)), float(np.nanmin(B)))
    ab_max = max(float(np.nanmax(A)), float(np.nanmax(B)))
    c_min = 0.0 if np.nanmin(C) >= 0 else float(np.nanmin(C))
    c_max = float(np.nanmax(C))
    if c_max <= c_min:
        c_max = c_min + np.finfo(float).eps

    for ax, field, title in zip(axes, (A, B), titles[:2]):
        image = ax.imshow(field, origin="lower", extent=extent, vmin=ab_min, vmax=ab_max)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="Value")

    image = axes[2].imshow(C, origin="lower", extent=extent, vmin=c_min, vmax=c_max)
    axes[2].set_title(titles[2])
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(image, ax=axes[2], label=third_label)
    plt.tight_layout()
    if fname:
        ensure_dir(os.path.dirname(fname))
        plt.savefig(fname, dpi=150)
    plt.show()


def plot_vorticity(omega_true: np.ndarray, omega_pred: np.ndarray, fname: Optional[str] = None,
                   extent=None) -> None:
    vmin = min(omega_true.min(), omega_pred.min())
    vmax = max(omega_true.max(), omega_pred.max())
    err = np.abs(omega_true - omega_pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, field, title in zip(axes[:2], (omega_true, omega_pred), ("ω (true)", "ω (pred)")):
        image = ax.imshow(field, origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="Vorticity")
    err_max = max(float(np.nanmax(err)), np.finfo(float).eps)
    image = axes[2].imshow(err, origin="lower", extent=extent, vmin=0.0, vmax=err_max)
    axes[2].set_title("|ω error|"); axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    fig.colorbar(image, ax=axes[2], label="Absolute error")
    plt.tight_layout()
    if fname:
        ensure_dir(os.path.dirname(fname))
        plt.savefig(fname, dpi=150)
    plt.show()


def quiver_field(U: np.ndarray, V: np.ndarray, step: int = 4, title: str = "velocity", fname: Optional[str] = None) -> None:
    """Downsampled quiver plot for quick inspection."""
    h, w = U.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    plt.figure(figsize=(6, 6))
    plt.quiver(xx, yy, U[::step, ::step], V[::step, ::step])
    plt.xlim(-0.5, w - 0.5)
    plt.ylim(-0.5, h - 0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.tight_layout()
    if fname:
        ensure_dir(os.path.dirname(fname))
        plt.savefig(fname, dpi=150)
    plt.show()
