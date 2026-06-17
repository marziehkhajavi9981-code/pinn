
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
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from config import TrainConfig
from data.grids import (crop_stacks, make_space_time_grids,
                        boundary_condition_samples, finite_differences, random_physics_samples)
from pinn.pinn import PhysicsInformedNN
from utils.vis import plot_losses, imshow3, plot_vorticity, quiver_field, ensure_dir

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


def masked_mse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or not np.any(mask):
        return float("nan")
    return mse(y_true[mask], y_pred[mask])


def load_split_masks(mask_npz: str, expected_shape: Tuple[int, int, int], stride: int) -> Optional[Dict[str, np.ndarray]]:
    if not mask_npz:
        return None
    data = np.load(mask_npz, allow_pickle=False)
    required = ["obs_mask", "train_mask", "val_mask", "test_mask"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Mask artifact {mask_npz} is missing keys: {missing}")
    masks = {k: data[k].astype(bool) for k in required}
    for name, mask in masks.items():
        if mask.shape != expected_shape:
            raise ValueError(f"{name} shape {mask.shape} does not match cropped data shape {expected_shape}")
    if stride > 1:
        masks = {k: v[::stride].copy() for k, v in masks.items()}
    return masks


def choose_eval_frame(test_mask: Optional[np.ndarray], fallback: int, stride: int) -> int:
    if test_mask is None or not np.any(test_mask):
        return fallback
    counts = test_mask.reshape(test_mask.shape[0], -1).sum(axis=1)
    local_idx = int(np.argmax(counts))
    return local_idx * stride


def all_point_samples(Xg: np.ndarray, Yg: np.ndarray, t: np.ndarray,
                      Uc: np.ndarray, Vc: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Build supervision from every (x,y,t) lattice point in the cropped stacks.
    
    Args:
        Xg, Yg: meshgrids [h, w]
        t: time array [N]
        Uc, Vc: velocity stacks [N, h, w]
        stride: sample every stride-th timestep (default=1 for all frames)
    
    Returns:
        X_all: [N', h, w, 3] coordinates (x, y, t), where N' = N // stride
        uv_all: [N', h, w, 2] velocities (u, v)
    """
    # Apply stride to temporal dimension
    t_sampled = t[::stride]
    Uc_sampled = Uc[::stride]
    Vc_sampled = Vc[::stride]
    
    N_sampled, h, w = Uc_sampled.shape
    
    # Build spatial coordinates [h, w, 2]
    xy = np.stack([Xg, Yg], axis=-1).astype(np.float32)  # [h, w, 2]
    
    # Broadcast to [N', h, w, 2]
    xy_full = np.broadcast_to(xy[None, :, :, :], (N_sampled, h, w, 2)).copy()
    
    # Build time coordinates [N', h, w, 1]
    t_full = t_sampled[:, None, None, None].astype(np.float32)  # [N', 1, 1, 1]
    t_full = np.broadcast_to(t_full, (N_sampled, h, w, 1)).copy()  # [N', h, w, 1]
    
    # Concatenate to get [N', h, w, 3]
    X_all = np.concatenate([xy_full, t_full], axis=-1)
    
    # Stack velocities to get [N', h, w, 2]
    uv_all = np.stack([Uc_sampled, Vc_sampled], axis=-1).astype(np.float32)
    
    return X_all, uv_all


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig()
    # CLI overrides
    if args.device is not None: cfg.device = args.device
    if args.hidden: cfg.hidden = tuple(map(int, args.hidden.split(',')))
    if args.double is not None: cfg.use_double_precision = bool(args.double)
    if args.save_dir: cfg.save_dir = args.save_dir
    if args.time_stride is not None: cfg.time_stride = args.time_stride
    if args.model_type is not None: cfg.model_type = args.model_type.lower()
    if args.conv_channels: cfg.conv_channels = tuple(map(int, args.conv_channels.split(',')))
    if args.mask_npz: cfg.mask_npz = args.mask_npz
    if args.n_physics is not None: cfg.n_physics = args.n_physics
    if args.adam_steps is not None: cfg.adam_steps = args.adam_steps
    if args.lbfgs_maxiter is not None: cfg.lbfgs_maxiter = args.lbfgs_maxiter
    if args.log_every is not None: cfg.log_every = args.log_every

    ensure_dir(cfg.save_dir)

    # Precision default
    if cfg.use_double_precision:
        torch.set_default_dtype(torch.float64)

    # --------------- Load data ---------------
    print("Loading data...")
    U = np.load(args.u)["U"].astype(np.float32)  # [N,H,W]
    V = np.load(args.v)["V"].astype(np.float32)
    N, H, W = U.shape
    print(f"✓ Loaded U,V with shape: {U.shape} {V.shape}")

    print(f"Cropping to {args.crop_h}x{args.crop_w}...")
    Uc, Vc = crop_stacks(U, V, args.crop_h, args.crop_w)
    N, h, w = Uc.shape
    print(f"✓ Cropped to: {Uc.shape}")

    # --------------- Coordinates ---------------
    print("Building coordinate grids...")
    dx = dy = dt = 1.0
    x, y, t, Xg, Yg = make_space_time_grids(N, h, w, dx, dy, dt)
    print(f"✓ Created grids: x={x.shape}, y={y.shape}, t={t.shape}")

    # --------------- Supervised samples ---------------
    if cfg.time_stride > 1:
        n_frames_sampled = len(range(0, N, cfg.time_stride))
        print(f"Building supervised samples with time stride={cfg.time_stride} ({n_frames_sampled}/{N} frames, {n_frames_sampled*h*w} points)...")
    else:
        print(f"Building supervised samples (all {N*h*w} spacetime points)...")
    
    X_u, uv = all_point_samples(Xg, Yg, t, Uc, Vc, stride=cfg.time_stride)  # [N', h, w, 3], [N', h, w, 2]
    print(f"✓ X_u shape: {X_u.shape}, uv shape: {uv.shape}")
    sampled_time_indices = np.arange(N, dtype=np.int32)[::cfg.time_stride]

    split_masks = load_split_masks(cfg.mask_npz, (N, h, w), cfg.time_stride)
    if split_masks is not None:
        print(f"✓ Loaded split masks from {cfg.mask_npz}")
        print(f"  - train observed points: {int(split_masks['train_mask'].sum()):,}")
        print(f"  - val observed points:   {int(split_masks['val_mask'].sum()):,}")
        print(f"  - test observed points:  {int(split_masks['test_mask'].sum()):,}")
        for split_name in ("train_mask", "val_mask", "test_mask"):
            if int(split_masks[split_name].sum()) == 0:
                print(f"  ! Warning: {split_name} has zero points after applying time_stride={cfg.time_stride}")
        if cfg.use_boundary_conditions:
            raise ValueError("Boundary condition append mode is not supported together with mask_npz yet.")
    
    if cfg.use_boundary_conditions:
        print("Adding boundary condition samples...")
        X_bc, uv_bc = boundary_condition_samples(x, y, t, Uc, Vc)  # [M, 3], [M, 2]
        # Flatten all-points and append BC
        X_u = np.vstack([X_u.reshape(-1, 3), X_bc]).astype(np.float32)  # [N*h*w+M, 3]
        uv = np.vstack([uv.reshape(-1, 2), uv_bc]).astype(np.float32)   # [N*h*w+M, 2]
        print(f"✓ Added {X_bc.shape[0]} BC samples. Total: {X_u.shape[0]} points")

    # --------------- Physics targets (FD) ---------------
    print(f"Computing finite differences and sampling {cfg.n_physics} physics points...")
    Ux, Uy, Vx, Vy = finite_differences(Uc, Vc, dy=dy, dx=dx)  # each [N, h, w]
    physics_time_indices = sampled_time_indices if cfg.model_type in ("conv2d", "conv3d") else None
    X_ph, ph_targets_np = random_physics_samples(
        x, y, t, Ux, Uy, Vx, Vy, cfg.n_physics, seed=0, time_indices=physics_time_indices
    )  # [n_physics, 3], dict of [n_physics, 1]
    print(f"✓ Physics samples: {X_ph.shape}")
    if physics_time_indices is not None:
        print(f"  - Conv physics sampled from {len(physics_time_indices)}/{N} strided frames")

    # --------------- Bounds for normalization ---------------
    lb = np.array([x.min(), y.min(), t.min()], dtype=np.float32)
    ub = np.array([x.max(), y.max(), t.max()], dtype=np.float32)

    # --------------- Torch tensors ---------------
    print("Converting to PyTorch tensors...")
    device = cfg.device
    dtype = torch.float64 if cfg.use_double_precision else torch.float32
    
    # Convert to tensors, keeping original shapes
    X_u_t = torch.as_tensor(X_u, dtype=dtype)  # [N, h, w, 3] or [M, 3] if BC added
    uv_t = torch.as_tensor(uv, dtype=dtype)    # [N, h, w, 2] or [M, 2] if BC added
    train_mask_t = torch.as_tensor(split_masks["train_mask"], dtype=torch.bool) if split_masks else None
    val_mask_t = torch.as_tensor(split_masks["val_mask"], dtype=torch.bool) if split_masks else None
    test_mask_t = torch.as_tensor(split_masks["test_mask"], dtype=torch.bool) if split_masks else None
    X_ph_t = torch.as_tensor(X_ph, dtype=dtype)  # [n_physics, 3]
    ph_targets_t = {k: torch.as_tensor(v, dtype=dtype) for k, v in ph_targets_np.items()}  # each [n_physics, 1]
    lb_t = torch.as_tensor(lb, dtype=dtype)  # [3]
    ub_t = torch.as_tensor(ub, dtype=dtype)  # [3]
    print(f"✓ Converted to torch tensors on device: {device}")

    # --------------- Model ---------------
    if cfg.model_type == "mlp":
        print(f"Building PINN (MLP) with {len(cfg.hidden)} hidden layers: {cfg.hidden}...")
    elif cfg.model_type == "conv2d":
        print(f"Building PINN (Conv2D) with channels: {cfg.conv_channels}...")
    elif cfg.model_type == "conv3d":
        print(f"Building PINN (Conv3D) with channels: {cfg.conv_channels}...")
    
    weights = {"data": cfg.w_data, "div": cfg.w_div, "vort": cfg.w_vort, "ux": cfg.w_ux, "vy": cfg.w_vy}
    model = PhysicsInformedNN(X_u=X_u_t, uv=uv_t, X_ph=X_ph_t, ph_targets=ph_targets_t,
                              hidden_layers=cfg.hidden, lb=lb_t, ub=ub_t,
                              weights=weights, device=device, use_double=cfg.use_double_precision,
                              model_type=cfg.model_type, conv_channels=cfg.conv_channels,
                              data_mask=train_mask_t, val_mask=val_mask_t, test_mask=test_mask_t)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params:,} parameters")

    # --------------- Train ---------------
    print(f"\n{'='*60}")
    print(f"Starting training on device: {device}")
    n_data = int(train_mask_t.sum()) if train_mask_t is not None else int(np.prod(model.Xu.shape[:-1]))
    print(f"  - Training data points: {n_data:,}")
    print(f"  - Physics points: {model.Xph.shape[0]:,}")
    print(f"  - Loss weights: data={cfg.w_data}, div={cfg.w_div}, vort={cfg.w_vort}")
    print(f"{'='*60}\n")
    
    start = time.time()
    print("[Stage 1] Adam warmup...")
    model.train_adam(steps=cfg.adam_steps, lr=cfg.adam_lr, log_every=cfg.log_every)

    print("\n[Stage 2] LBFGS polish...")
    model.train_lbfgs(maxiter=cfg.lbfgs_maxiter, history_size=cfg.lbfgs_history,
                      use_strong_wolfe=cfg.lbfgs_use_strong_wolfe, log_every=cfg.log_every)
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"✓ Training complete! Total time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # --------------- Loss curves ---------------
    plot_losses(model.loss_hist, save_dir=cfg.save_dir)
    split_eval = model.data_split_losses()
    if split_masks is not None:
        print("\nFinal masked data losses:")
        print(f"  Train MSE: {split_eval['train_data']:.4e}")
        print(f"  Val MSE:   {split_eval['val_data']:.4e}")
        print(f"  Test MSE:  {split_eval['test_data']:.4e}")

    # --------------- Evaluate on a frame ---------------
    print("Evaluating model on test frame...")
    fallback_frame = min(120, N - 1)
    frame_eval = choose_eval_frame(split_masks["test_mask"] if split_masks else None, fallback_frame, cfg.time_stride)
    frame_eval = min(frame_eval, N - 1)
    X_star = np.stack([Xg.ravel(), Yg.ravel(), np.full(h * w, t[frame_eval], np.float32)], axis=1)
    uv_pred = model.predict(X_star)
    U_pred = uv_pred[:, 0].reshape(h, w)
    V_pred = uv_pred[:, 1].reshape(h, w)

    u_true = Uc[frame_eval]
    v_true = Vc[frame_eval]

    print(f"\nEvaluation metrics (frame {frame_eval}):")
    print(f"  RelL2(u): {np.linalg.norm(u_true - U_pred) / (np.linalg.norm(u_true) + 1e-12):.3e}")
    print(f"  MSE(u)={mse(u_true, U_pred):.4e}, R2(u)={r2(u_true, U_pred):.4f}")
    print(f"  MSE(v)={mse(v_true, V_pred):.4e}, R2(v)={r2(v_true, V_pred):.4f}")
    if split_masks is not None:
        local_eval = frame_eval // cfg.time_stride
        test_mask_frame = split_masks["test_mask"][local_eval] if local_eval < split_masks["test_mask"].shape[0] else None
        print(f"  Test-mask MSE(u)={masked_mse(u_true, U_pred, test_mask_frame):.4e}")
        print(f"  Test-mask MSE(v)={masked_mse(v_true, V_pred, test_mask_frame):.4e}")

    # --------------- Visual debug ---------------
    print(f"\nSaving visualizations to {cfg.save_dir}/...")
    imshow3(u_true, U_pred, np.abs(u_true - U_pred), titles=("u true", "u pred", "|error|"),
            fname=os.path.join(cfg.save_dir, "u_triptych.png"))
    imshow3(v_true, V_pred, np.abs(v_true - V_pred), titles=("v true", "v pred", "|error|"),
            fname=os.path.join(cfg.save_dir, "v_triptych.png"))
    print("  ✓ u_triptych.png, v_triptych.png")
    if split_masks is not None:
        local_eval = frame_eval // cfg.time_stride
        if local_eval < split_masks["test_mask"].shape[0]:
            test_mask_vis = split_masks["test_mask"][local_eval]
        else:
            test_mask_vis = np.zeros((h, w), dtype=bool)
        imshow3(u_true, U_pred, test_mask_vis.astype(np.float32),
                titles=("u true (unmasked)", "u pred", "test mask"),
                fname=os.path.join(cfg.save_dir, "u_test_unmasked_triptych.png"))
        imshow3(v_true, V_pred, test_mask_vis.astype(np.float32),
                titles=("v true (unmasked)", "v pred", "test mask"),
                fname=os.path.join(cfg.save_dir, "v_test_unmasked_triptych.png"))
        print("  ✓ u_test_unmasked_triptych.png, v_test_unmasked_triptych.png")

    # vorticity maps
    dU_dy, dU_dx = np.gradient(u_true, 1.0, 1.0)
    dV_dy, dV_dx = np.gradient(v_true, 1.0, 1.0)
    omega_true = dV_dx - dU_dy

    dUpred_dy, dUpred_dx = np.gradient(U_pred, 1.0, 1.0)
    dVpred_dy, dVpred_dx = np.gradient(V_pred, 1.0, 1.0)
    omega_pred = dVpred_dx - dUpred_dy

    plot_vorticity(omega_true, omega_pred, fname=os.path.join(cfg.save_dir, "omega_triptych.png"))
    print("  ✓ omega_triptych.png")

    # Optional: quiver plots (downsample)
    quiver_field(u_true, v_true, step=4, title="velocity true", fname=os.path.join(cfg.save_dir, "quiver_true.png"))
    quiver_field(U_pred, V_pred, step=4, title="velocity pred", fname=os.path.join(cfg.save_dir, "quiver_pred.png"))
    print("  ✓ quiver_true.png, quiver_pred.png")
    print("\n All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN for 2D velocity with kinematic constraints.")
    parser.add_argument("--u", type=str, required=True, help="Path to u_stack.npz with key 'U'")
    parser.add_argument("--v", type=str, required=True, help="Path to v_stack.npz with key 'V'")
    parser.add_argument("--crop_h", type=int, default=64, help="Crop height")
    parser.add_argument("--crop_w", type=int, default=64, help="Crop width")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (overrides config)")
    parser.add_argument("--model_type", type=str, default=None, choices=["mlp", "conv2d", "conv3d"],
                        help="Model architecture: mlp (default), conv2d, or conv3d")
    parser.add_argument("--hidden", type=str, default=None, help="Comma-separated hidden sizes for MLP, e.g., 64,64,64,64")
    parser.add_argument("--conv_channels", type=str, default=None, help="Comma-separated channels for Conv2D/Conv3D, e.g., 32,64,64,32")
    parser.add_argument("--double", type=int, default=None, help="1 to use float64, 0 for float32")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--time_stride", type=int, default=None, help="Sample every N-th timestep (1=all frames, 2=every other, etc.)")
    parser.add_argument("--mask_npz", type=str, default=None, help="Optional masks/splits .npz generated by scripts/make_masks_splits.py")
    parser.add_argument("--n_physics", type=int, default=None, help="Number of sampled physics points")
    parser.add_argument("--adam_steps", type=int, default=None, help="Adam warmup steps")
    parser.add_argument("--lbfgs_maxiter", type=int, default=None, help="LBFGS max iterations")
    parser.add_argument("--log_every", type=int, default=None, help="Log every N optimizer iterations")
    main(parser.parse_args())
