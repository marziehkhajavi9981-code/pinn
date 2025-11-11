
# ---------------------------------------------
# FILE: train.py
# ---------------------------------------------
"""End-to-end training & evaluation script.

Usage:
    python train.py --u u_stack.npz --v v_stack.npz \
                    --crop_h 64 --crop_w 64 --device cuda

This will:
  1) Load U,V stacks and build IC/BC + physics samples
  2) Train with Adam warmup then LBFGS polish
  3) Evaluate a selected frame, compute metrics, and save debug figures
"""
import argparse
import os
import time
import numpy as np
import torch

from config import TrainConfig
from data.grids import (crop_stacks, make_space_time_grids, initial_condition_samples,
                        boundary_condition_samples, finite_differences, random_physics_samples)
from pinn.pinn import PhysicsInformedNN
from utils.vis import plot_losses, imshow3, plot_vorticity, quiver_field, ensure_dir


def r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + eps
    return 1.0 - ss_res / ss_tot


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig()
    # CLI overrides
    if args.device: cfg.device = args.device
    if args.hidden: cfg.hidden = tuple(map(int, args.hidden.split(',')))
    if args.double is not None: cfg.use_double_precision = bool(args.double)
    if args.save_dir: cfg.save_dir = args.save_dir

    ensure_dir(cfg.save_dir)

    # Precision default
    if cfg.use_double_precision:
        torch.set_default_dtype(torch.float64)

    # --------------- Load data ---------------
    U = np.load(args.u)["U"].astype(np.float32)  # [N,H,W]
    V = np.load(args.v)["V"].astype(np.float32)
    N, H, W = U.shape
    print(f"Loaded U,V with shape: {U.shape} {V.shape}")

    Uc, Vc = crop_stacks(U, V, args.crop_h, args.crop_w)
    N, h, w = Uc.shape
    print(f"Cropped to: {Uc.shape}")

    # --------------- Coordinates ---------------
    dx = dy = dt = 1.0
    x, y, t, Xg, Yg = make_space_time_grids(N, h, w, dx, dy, dt)

    # --------------- IC/BC supervised samples ---------------
    X_ic, uv_ic = initial_condition_samples(Xg, Yg, Uc[0], Vc[0])
    X_bc, uv_bc = boundary_condition_samples(x, y, t, Uc, Vc)
    X_u = np.vstack([X_ic, X_bc]).astype(np.float32)
    uv = np.vstack([uv_ic, uv_bc]).astype(np.float32)

    # --------------- Physics targets (FD) ---------------
    Ux, Uy, Vx, Vy = finite_differences(Uc, Vc, dy=dy, dx=dx)
    X_ph, ph_targets_np = random_physics_samples(x, y, t, Ux, Uy, Vx, Vy, cfg.n_physics, seed=0)

    # --------------- Bounds for normalization ---------------
    lb = np.array([x.min(), y.min(), t.min()], dtype=np.float32)
    ub = np.array([x.max(), y.max(), t.max()], dtype=np.float32)

    # --------------- Torch tensors ---------------
    device = cfg.device
    dtype = torch.float64 if cfg.use_double_precision else torch.float32
    X_u_t = torch.as_tensor(X_u, dtype=dtype)
    uv_t = torch.as_tensor(uv, dtype=dtype)
    X_ph_t = torch.as_tensor(X_ph, dtype=dtype)
    ph_targets_t = {k: torch.as_tensor(v, dtype=dtype) for k, v in ph_targets_np.items()}
    lb_t = torch.as_tensor(lb, dtype=dtype)
    ub_t = torch.as_tensor(ub, dtype=dtype)

    # --------------- Model ---------------
    weights = {"data": cfg.w_data, "div": cfg.w_div, "vort": cfg.w_vort, "ux": cfg.w_ux, "vy": cfg.w_vy}
    model = PhysicsInformedNN(X_u=X_u_t, uv=uv_t, X_ph=X_ph_t, ph_targets=ph_targets_t,
                              hidden_layers=cfg.hidden, lb=lb_t, ub=ub_t,
                              weights=weights, device=device, use_double=cfg.use_double_precision)

    # --------------- Train ---------------
    start = time.time()
    print("\n[Stage 1] Adam warmup...")
    model.train_adam(steps=cfg.adam_steps, lr=cfg.adam_lr, physics_mb=cfg.physics_minibatch, log_every=cfg.log_every)

    print("\n[Stage 2] LBFGS polish...")
    model.train_lbfgs(maxiter=cfg.lbfgs_maxiter, history_size=cfg.lbfgs_history,
                      use_strong_wolfe=cfg.lbfgs_use_strong_wolfe, physics_mb=cfg.physics_minibatch,
                      log_every=cfg.log_every)
    print(f"Total training time: {time.time() - start:.2f}s")

    # --------------- Loss curves ---------------
    plot_losses(model.loss_hist, save_dir=cfg.save_dir)

    # --------------- Evaluate on a frame ---------------
    frame_eval = min(120, N - 1)
    X_star = np.stack([Xg.ravel(), Yg.ravel(), np.full(h * w, t[frame_eval], np.float32)], axis=1)
    uv_pred = model.predict(X_star)
    U_pred = uv_pred[:, 0].reshape(h, w)
    V_pred = uv_pred[:, 1].reshape(h, w)

    u_true = Uc[frame_eval]
    v_true = Vc[frame_eval]

    print(f"RelL2(u): {np.linalg.norm(u_true - U_pred) / (np.linalg.norm(u_true) + 1e-12):.3e}")
    print(f"MSE(u)={mse(u_true, U_pred):.4e}, R2(u)={r2(u_true, U_pred):.4f}")
    print(f"MSE(v)={mse(v_true, V_pred):.4e}, R2(v)={r2(v_true, V_pred):.4f}")

    # --------------- Visual debug ---------------
    imshow3(u_true, U_pred, np.abs(u_true - U_pred), titles=("u true", "u pred", "|error|"),
            fname=os.path.join(cfg.save_dir, "u_triptych.png"))
    imshow3(v_true, V_pred, np.abs(v_true - V_pred), titles=("v true", "v pred", "|error|"),
            fname=os.path.join(cfg.save_dir, "v_triptych.png"))

    # vorticity maps
    dU_dy, dU_dx = np.gradient(u_true, 1.0, 1.0)
    dV_dy, dV_dx = np.gradient(v_true, 1.0, 1.0)
    omega_true = dV_dx - dU_dy

    dUpred_dy, dUpred_dx = np.gradient(U_pred, 1.0, 1.0)
    dVpred_dy, dVpred_dx = np.gradient(V_pred, 1.0, 1.0)
    omega_pred = dVpred_dx - dUpred_dy

    plot_vorticity(omega_true, omega_pred, fname=os.path.join(cfg.save_dir, "omega_triptych.png"))

    # Optional: quiver plots (downsample)
    quiver_field(u_true, v_true, step=4, title="velocity true", fname=os.path.join(cfg.save_dir, "quiver_true.png"))
    quiver_field(U_pred, V_pred, step=4, title="velocity pred", fname=os.path.join(cfg.save_dir, "quiver_pred.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN for 2D velocity with kinematic constraints.")
    parser.add_argument("--u", type=str, required=True, help="Path to u_stack.npz with key 'U'")
    parser.add_argument("--v", type=str, required=True, help="Path to v_stack.npz with key 'V'")
    parser.add_argument("--crop_h", type=int, default=64, help="Crop height")
    parser.add_argument("--crop_w", type=int, default=64, help="Crop width")
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu (overrides config)")
    parser.add_argument("--hidden", type=str, default=None, help="Comma-separated hidden sizes, e.g., 64,64,64,64")
    parser.add_argument("--double", type=int, default=None, help="1 to use float64, 0 for float32")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for figures")
    main(parser.parse_args())
