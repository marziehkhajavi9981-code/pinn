from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Model / training
    hidden: tuple = (128, 128, 128, 128)
    use_double_precision: bool = True           # float64 usually stabilizes PINNs
    device: str = "cpu"                        # "cuda" or "cpu"

    # Loss weights
    w_data: float = 2.0
    w_div: float = 1.0
    w_vort: float = 1.0
    w_ux: float = 0.5
    w_vy: float = 0.5

    # Physics sampling
    n_physics: int = 10000                      # number of interior physics points per epoch
    physics_minibatch: int = 2048               # minibatch size for physics points

    # Optimizers
    adam_lr: float = 1e-3
    adam_steps: int = 2000
    lbfgs_maxiter: int = 2000
    lbfgs_history: int = 50
    lbfgs_use_strong_wolfe: bool = False        # safer off on CUDA

    # Logging / plotting
    log_every: int = 10
    save_dir: str = "outputs"                  # figures & checkpoints

