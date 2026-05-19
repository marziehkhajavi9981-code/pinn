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
            fname: Optional[str] = None) -> None:
    """Show three image-like arrays side-by-side."""
    plt.figure(figsize=(12, 4.2))
    plt.subplot(1, 3, 1); plt.imshow(A, origin="lower"); plt.title(titles[0]); plt.colorbar()
    plt.subplot(1, 3, 2); plt.imshow(B, origin="lower"); plt.title(titles[1]); plt.colorbar()
    plt.subplot(1, 3, 3); plt.imshow(C, origin="lower"); plt.title(titles[2]); plt.colorbar()
    plt.tight_layout()
    if fname:
        ensure_dir(os.path.dirname(fname))
        plt.savefig(fname, dpi=150)
    plt.show()


def plot_vorticity(omega_true: np.ndarray, omega_pred: np.ndarray, fname: Optional[str] = None) -> None:
    vmin = min(omega_true.min(), omega_pred.min())
    vmax = max(omega_true.max(), omega_pred.max())
    err = np.abs(omega_true - omega_pred)

    plt.figure(figsize=(12, 4.2))
    plt.subplot(1, 3, 1); plt.imshow(omega_true, origin="lower", vmin=vmin, vmax=vmax); plt.title("ω (true)"); plt.colorbar()
    plt.subplot(1, 3, 2); plt.imshow(omega_pred, origin="lower", vmin=vmin, vmax=vmax); plt.title("ω (pred)"); plt.colorbar()
    plt.subplot(1, 3, 3); plt.imshow(err, origin="lower"); plt.title("|ω error|"); plt.colorbar()
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
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    if fname:
        ensure_dir(os.path.dirname(fname))
        plt.savefig(fname, dpi=150)
    plt.show()
