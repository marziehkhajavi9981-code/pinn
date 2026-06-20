## Physics-Informed Velocity Field Reconstruction (Guide)

This guide explains the physical problem we are solving and how this codebase implements a Physics-Informed Neural Network (PINN) to reconstruct a 2D time-varying velocity field from data while enforcing kinematic constraints.

### What problem are we solving?
- We want a function that maps space-time coordinates to a 2D velocity:
  - Inputs: position and time, (x, y, t)
  - Outputs: velocity components, (u, v)
- We assume the flow is approximately incompressible (divergence-free) and reason about its vorticity (rotation).
- We have data: stacks of u and v on a regular grid over time:
  - U: [N, H, W] (time frames × height × width) for u
  - V: [N, H, W] for v
- Our goal: learn a continuous model (u, v) = f(x, y, t) that matches known values on the initial condition and boundaries, and whose spatial derivatives match what we compute from the data (finite differences) inside the domain.

### Physical background (2D kinematics)
- Incompressibility (divergence-free):
  - div v = ∂u/∂x + ∂v/∂y = 0
- Vorticity (scalar in 2D):
  - ω = ∂v/∂x − ∂u/∂y
- In this code, we don’t directly solve the full Navier–Stokes PDE. Instead, we use kinematic constraints (divergence, vorticity) and supervised derivative targets computed from the data. This is a “physics-guided supervised” approach: we guide the network using physically meaningful quantities derived from measurements.

### What the PINN does here
- A small multi-layer perceptron (MLP) takes (x, y, t) and outputs (u, v).
- Training enforces:
  - Data consistency on initial condition (t = t0) and on spatial boundaries (IC/BC).
  - Kinematic consistency in the interior: divergence and vorticity match values derived from data, and optionally ∂u/∂x and ∂v/∂y match finite-difference targets.
- Autograd (PyTorch) computes spatial derivatives of the network outputs with respect to inputs to form the physics losses.

---

## Code Walkthrough

### Data preparation and sampling
- Files expected:
  - u_stack.npz with key `U` shaped [N, H, W]
  - v_stack.npz with key `V` shaped [N, H, W]
- We use the complete stacks and define uniform spacings dx = dy = dt = 1 for simplicity.
- We generate:
  - IC supervised samples: coordinates and velocities for the initial time slice.
  - BC (Boundary Conditions): **no-slip walls** where u = 0, v = 0 (physical constraint, not data fitting).
  - Physics samples in the interior: random points across space-time where we enforce incompressibility.
  - Finite differences (FD): compute ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y on the grid for vorticity matching (optional).

Key entry-point:

```12:33:/home/fardin/pinn/train.py
    U = np.load(args.u)["U"].astype(np.float32)  # [N,H,W]
    V = np.load(args.v)["V"].astype(np.float32)
    ...
    X_ic, uv_ic = initial_condition_samples(Xg, Yg, U[0], V[0])
    X_bc, uv_bc = boundary_condition_samples(x, y, t, U, V)
    ...
    Ux, Uy, Vx, Vy = finite_differences(U, V, dy=dy, dx=dx)
    X_ph, ph_targets_np = random_physics_samples(x, y, t, Ux, Uy, Vx, Vy, cfg.n_physics, seed=0)
```

Helpers (sampling, FD):

```48:90:/home/fardin/pinn/data/grids.py
def boundary_condition_samples(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                               U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
def finite_differences(U: np.ndarray, V: np.ndarray, dy: float, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
def random_physics_samples(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                           Ux: np.ndarray, Uy: np.ndarray, Vx: np.ndarray, Vy: np.ndarray,
                           n_physics: int, seed: int = 0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ...
```

### The model (PINN)
- `PhysicsInformedNN` wraps an `MLP` and defines the forward mapping and loss functions.
- Normalization: inputs are affinely mapped to [-1, 1] per dimension for stable MLP training.
- Gradients: autograd computes ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y by differentiating outputs w.r.t. inputs.

Model and gradients:

```55:99:/home/fardin/pinn/pinn/pinn.py
        self.model = MLP(in_dim=3, hidden=hidden_layers, out_dim=2).to(self.device)
        ...
    def _norm(self, X: torch.Tensor) -> torch.Tensor:
        return 2.0 * (X - self.lb) / (self.ub - self.lb + 1e-12) - 1.0
    def forward_uv(self, X: torch.Tensor) -> torch.Tensor:
        Xn = self._norm(X)
        return self.model(Xn)
    def grads_xy(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X = X.clone().detach().requires_grad_(True)
        uv = self.forward_uv(X)
        u, v = uv[:, 0:1], uv[:, 1:2]
        du_dX = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                    retain_graph=True, create_graph=True)[0]
        dv_dX = torch.autograd.grad(v, X, grad_outputs=torch.ones_like(v),
                                    retain_graph=True, create_graph=True)[0]
        u_x, u_y = du_dX[:, 0:1], du_dX[:, 1:2]
        v_x, v_y = dv_dX[:, 0:1], dv_dX[:, 1:2]
        return u_x, u_y, v_x, v_y
```

MLP:

```6:33:/home/fardin/pinn/models/mlp.py
class MLP(nn.Module):
    ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### Losses
- Data loss (IC+BC):
  - IC: MSE between predicted and true velocities at initial time (t=0)
  - BC: MSE enforcing u=0, v=0 at boundaries (no-slip walls)
- Physics losses (interior):
  - **Divergence: MSE((u_x + v_y)_model, (Ux + Vy)_actual)** — matches the finite-difference divergence from data
  - Vorticity: MSE(v_x − u_y, Vx − Uy) — matches vorticity from data (optional regularization)
  - Direct derivative matches: MSE(u_x, Ux) and MSE(v_y, Vy) — smoothness regularization
- Weighted sum with configuration weights.
- Interior points can be evaluated in minibatches to reduce memory. Loss is averaged by number of samples (not just chunk count) to avoid bias.

Loss definition:

```100:146:/home/fardin/pinn/pinn/pinn.py
    def loss_fn(self, physics_minibatch: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        uv_pred = self.forward_uv(self.Xu)
        L_data = self.mse(uv_pred, self.uv)
        if physics_minibatch is None or physics_minibatch <= 0:
            u_x, u_y, v_x, v_y = self.grads_xy(self.Xph)
            L_div = self.mse(u_x + v_y, self.ux_t + self.vy_t)
            L_vort = self.mse(v_x - u_y, self.vx_t - self.uy_t)
            L_ux = self.mse(u_x, self.ux_t)
            L_vy = self.mse(v_y, self.vy_t)
        else:
            n = self.Xph.shape[0]
            L_div = L_vort = L_ux = L_vy = 0.0
            n_chunks = math.ceil(n / physics_minibatch)
            for i in range(n_chunks):
                s = slice(i * physics_minibatch, min((i + 1) * physics_minibatch, n))
                m = s.stop - s.start
                u_x, u_y, v_x, v_y = self.grads_xy(self.Xph[s])
                L_div = L_div + self.mse(u_x + v_y, self.ux_t[s] + self.vy_t[s]) * m
                L_vort = L_vort + self.mse(v_x - u_y, self.vx_t[s] - self.uy_t[s]) * m
                L_ux = L_ux + self.mse(u_x, self.ux_t[s]) * m
                L_vy = L_vy + self.mse(v_y, self.vy_t[s]) * m
            L_div /= n; L_vort /= n; L_ux /= n; L_vy /= n
        L_ph = self.w_div * L_div + self.w_vort * L_vort + self.w_ux * L_ux + self.w_vy * L_vy
        L_tot = self.w_data * L_data + L_ph
        ...
        return L_tot, scalars
```

### Training
- Two-stage optimization:
  1) Adam “warmup” to reach a good basin quickly.
  2) LBFGS “polish” to refine the solution with a second-order method.
- Logging tracks per-term losses over iterations. Visualization utilities save plots and field comparisons.

Training calls:

```101:110:/home/fardin/pinn/train.py
    print("\n[Stage 1] Adam warmup...")
    model.train_adam(steps=cfg.adam_steps, lr=cfg.adam_lr, physics_mb=cfg.physics_minibatch, log_every=cfg.log_every)
    print("\n[Stage 2] LBFGS polish...")
    model.train_lbfgs(maxiter=cfg.lbfgs_maxiter, history_size=cfg.lbfgs_history,
                      use_strong_wolfe=cfg.lbfgs_use_strong_wolfe, physics_mb=cfg.physics_minibatch,
                      log_every=cfg.log_every)
```

Adam loop:

```155:165:/home/fardin/pinn/pinn/pinn.py
    def train_adam(self, steps: int, lr: float, physics_mb: Optional[int] = None, log_every: int = 10) -> None:
        opt = optim.Adam(self.parameters(), lr=lr)
        for it in range(1, steps + 1):
            opt.zero_grad()
            L, parts = self.loss_fn(physics_minibatch=physics_mb)
            L.backward()
            opt.step()
            self._log(parts, float(L.detach().cpu()), log_every)
            if it % (log_every * 10) == 0:
                print(f"[Adam {it:05d}] total={float(L.detach().cpu()):.4e} data={parts['data']:.3e} div={parts['div']:.3e} vort={parts['vort']:.3e} ux={parts['ux']:.3e} vy={parts['vy']:.3e}")
```

LBFGS with closure:

```166:185:/home/fardin/pinn/pinn/pinn.py
    def train_lbfgs(self, maxiter: int, history_size: int, use_strong_wolfe: bool = False,
                     physics_mb: Optional[int] = None, log_every: int = 10) -> None:
        ...
        def closure():
            optimizer.zero_grad()
            L, parts = self.loss_fn(physics_minibatch=physics_mb)
            L.backward()
            self._log(parts, float(L.detach().cpu()), log_every)
            return L
        optimizer.step(closure)
```

---

## Configuration and Running

### Configuration (`config.py`)
- Hidden sizes, precision, and device.
- Loss weights for data/div/vort/ux/vy.
- Number of physics points and minibatch size.
- Optimizer hyperparameters (Adam steps and LR; LBFGS settings).
- Logging cadence and output directory.

```1:31:/home/fardin/pinn/config.py
@dataclass
class TrainConfig:
    hidden: tuple = (64, 64, 64, 64)
    use_double_precision: bool = True
    device: str = "cpu"
    w_data: float = 2.0
    w_div: float = 1.0
    w_vort: float = 1.0
    w_ux: float = 0.5
    w_vy: float = 0.5
    n_physics: int = 10000
    physics_minibatch: int = 2048
    adam_lr: float = 1e-3
    adam_steps: int = 2000
    lbfgs_maxiter: int = 600
    lbfgs_history: int = 50
    lbfgs_use_strong_wolfe: bool = False
    log_every: int = 10
    save_dir: str = "outputs"
```

### Running

```bash
python train.py --u u_stack.npz --v v_stack.npz --device cuda
```

CLI notes:
- If `--device` is omitted, the config’s default is used.
- Use `--double 1` to force float64 if needed.

### Outputs
- Loss curves saved to `save_dir` (default `outputs`).
- Triptychs comparing ground truth vs prediction vs error for u and v.
- Vorticity comparison and optional quiver plots.

---

## Troubleshooting and tips
- Backprop through graph error:
  - Fixed in `grads_xy` by retaining the autograd graph across both gradient computations.
- CUDA printing error (`float(Tensor)` on GPU):
  - Logging converts to `float(L.detach().cpu())` before formatting.
- Memory pressure:
  - Lower `physics_minibatch`.
  - Consider computing both u and v gradients in a single autograd call to reduce graph retention.
- Stability:
  - Start with double precision (`use_double_precision=True`), then try float32 once stable.
  - Tune loss weights; you can anneal physics weights after initial data fit.

---

## Extending the approach
- Residual-based PINN:
  - Replace derivative supervision with PDE residuals (e.g., full Navier–Stokes momentum + continuity) computed by autograd. This removes dependence on FD targets and enforces the governing equations directly.
- Non-uniform grids:
  - Pass actual dx, dy, dt where appropriate and reflect in normalization and FD/physics sampling.
- Additional constraints:
  - Add boundary type variants (Dirichlet/Neumann) or time-dependent BCs as needed.

---

## File map
- `train.py`: end-to-end script (load, sample, train, evaluate, visualize).
- `pinn/pinn.py`: PINN model, losses, training loops, and prediction.
- `models/mlp.py`: MLP architecture used for f(x, y, t) → (u, v).
- `data/grids.py`: sampling utilities (IC/BC, interior), finite differences.
- `utils/vis.py`: plotting utilities for losses and field comparisons.

---

## Glossary
- IC (Initial Condition): data at the first time frame (t = t0).
- BC (Boundary Condition): data along the spatial domain boundaries.
- Divergence: ∂u/∂x + ∂v/∂y. The implemented loss matches model divergence to finite-difference divergence from data.
- Vorticity (2D): ∂v/∂x − ∂u/∂y, scalar measure of local rotation.
- PINN: Physics-Informed Neural Network, a model trained using data and physics constraints.


