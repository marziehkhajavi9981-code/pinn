from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Model / training
    model_type: str = "conv2d"                     # "mlp", "conv2d", or "conv3d"
    hidden: tuple = (64, 64, 64)                # for MLP (~8,706 params)
    conv_channels: tuple = (16, 16, 16)          # for Conv2D (~8,705 params) / Conv3D uses (11, 11) for ~8,054 params
    use_double_precision: bool = True           # float64 usually stabilizes PINNs
    device: str = "cuda"                        # "cuda" or "cpu"
    use_boundary_conditions: bool = False       # optionally include BC supervision
    
    # Data sampling
    time_stride: int = 20                       # sample every N-th timestep (1 = all frames, 2 = every other frame, etc.)

    # Loss weights
    w_data: float = 2.0
    w_div: float = 0.0
    w_vort: float = 1.0
    w_ux: float = 0.5
    w_vy: float = 0.5

    # Physics sampling
    n_physics: int = 10000                      # number of interior physics points per epoch

    # Optimizers
    adam_lr: float = 1e-3
    adam_steps: int = 1000
    lbfgs_maxiter: int = 1000
    lbfgs_history: int = 50
    lbfgs_use_strong_wolfe: bool = False        # safer off on CUDA

    # Logging / plotting
    log_every: int = 5                          # print detailed logs every N iterations
    save_dir: str = "outputs"                   # figures & checkpoints

