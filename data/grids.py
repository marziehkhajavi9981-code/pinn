from typing import Dict, Tuple
import numpy as np


def make_space_time_grids(N: int, h: int, w: int, dx: float = 1.0, dy: float = 1.0, dt: float = 1.0):
    """Create 1D space/time arrays and a 2D XY meshgrid.

    Args:
        N: number of time steps
        h, w: spatial dimensions
        dx, dy, dt: grid spacings
    
    Returns:
        x: [w] x-coordinates
        y: [h] y-coordinates
        t: [N] time coordinates
        Xg: [h, w] x-meshgrid
        Yg: [h, w] y-meshgrid
    """
    x = np.arange(w, dtype=np.float32) * dx
    y = np.arange(h, dtype=np.float32) * dy
    t = np.arange(N, dtype=np.float32) * dt
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    return x, y, t, Xg, Yg


def initial_condition_samples(Xg: np.ndarray, Yg: np.ndarray, U0: np.ndarray, V0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build inputs (x,y,0) and velocity targets from frame 0.

    Args:
        Xg, Yg: meshgrid arrays [h, w]
        U0, V0: first-frame velocities [h, w]
    Returns:
        X_ic [h*w,3], uv_ic [h*w,2]
    """
    h, w = Xg.shape
    X_ic = np.stack([Xg.ravel(), Yg.ravel(), np.zeros(h * w, np.float32)], axis=1)
    uv_ic = np.stack([U0.ravel(), V0.ravel()], axis=1)
    return X_ic.astype(np.float32), uv_ic.astype(np.float32)


def boundary_condition_samples(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                               U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Collect boundary points (four edges) for each frame and their targets.

    Args:
        x, y, t: 1D coordinate arrays
        U, V: velocity stacks [N, h, w]
    Returns:
        X_bc: [(N*M), 3]
        uv_bc: [(N*M), 2]
        where M is the number of boundary points per frame
    """
    N, h, w = U.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1] = True, True, True, True
    by, bx = np.where(mask)

    X_bc_list, uv_bc_list = [], []
    for k in range(N):
        tk = np.full_like(bx, t[k], dtype=np.float32)
        X_bc_list.append(np.stack([x[bx], y[by], tk], axis=1))
        uv_bc_list.append(np.stack([U[k, by, bx], V[k, by, bx]], axis=1))

    X_bc = np.vstack(X_bc_list).astype(np.float32)
    uv_bc = np.vstack(uv_bc_list).astype(np.float32)
    return X_bc, uv_bc


def finite_differences(U: np.ndarray, V: np.ndarray, dy: float, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute spatial grads using centered differences along (y,x) axes.

    Args:
        U, V: arrays [N, h, w]
        dy, dx: spacings
    Returns:
        Ux, Uy, Vx, Vy: each [N, h, w]
    """
    Uy, Ux = np.gradient(U, dy, dx, axis=(1, 2))
    Vy, Vx = np.gradient(V, dy, dx, axis=(1, 2))
    return Ux.astype(np.float32), Uy.astype(np.float32), Vx.astype(np.float32), Vy.astype(np.float32)


def random_physics_samples(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                           Ux: np.ndarray, Uy: np.ndarray, Vx: np.ndarray, Vy: np.ndarray,
                           n_physics: int, seed: int = 0,
                           time_indices: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Sample random interior points (exclude boundary indices) and gather FD targets.

    Args:
        x, y, t: 1D coordinate arrays
        Ux, Uy, Vx, Vy: spatial gradient arrays [N, h, w]
        n_physics: number of random samples
        seed: random seed
        time_indices: optional source frame indices to sample from
    
    Returns:
        X_ph: [n_physics, 3]
        targets: dict with keys 'ux','uy','vx','vy' (each [n_physics, 1])
    """
    N, h, w = Ux.shape
    rng = np.random.default_rng(seed)
    if time_indices is None:
        k_idx = rng.integers(0, N, size=n_physics, dtype=np.int32)
    else:
        time_indices = np.asarray(time_indices, dtype=np.int32)
        if time_indices.size == 0:
            raise ValueError("time_indices must not be empty")
        if time_indices.min() < 0 or time_indices.max() >= N:
            raise ValueError(f"time_indices must be within [0, {N - 1}]")
        k_idx = rng.choice(time_indices, size=n_physics, replace=True).astype(np.int32)
    y_idx = rng.integers(1, h - 1, size=n_physics, dtype=np.int32)
    x_idx = rng.integers(1, w - 1, size=n_physics, dtype=np.int32)

    X_ph = np.stack([x[x_idx], y[y_idx], t[k_idx]], axis=1).astype(np.float32)
    ux_t = Ux[k_idx, y_idx, x_idx][:, None]
    uy_t = Uy[k_idx, y_idx, x_idx][:, None]
    vx_t = Vx[k_idx, y_idx, x_idx][:, None]
    vy_t = Vy[k_idx, y_idx, x_idx][:, None]
    targets = {"ux": ux_t, "uy": uy_t, "vx": vx_t, "vy": vy_t}
    return X_ph, targets
