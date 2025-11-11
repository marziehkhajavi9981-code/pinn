from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.mlp import MLP


class PhysicsInformedNN(nn.Module):
    """PINN that learns (u,v)(x,y,t) with kinematic constraints.

    Args:
        X_u: supervised points [Nu,3]
        uv: supervised targets [Nu,2]
        X_ph: physics points [Nph,3]
        ph_targets: dict containing 'ux','uy','vx','vy' (each [Nph,1])
        hidden_layers: hidden sizes for MLP
        lb, ub: lower/upper bounds for input normalization (3,)
        weights: dict of loss weights
        device: 'cpu' or 'cuda'
        use_double: if True, converts module parameters and tensors to float64
    """

    def __init__(self,
                 X_u: torch.Tensor,
                 uv: torch.Tensor,
                 X_ph: torch.Tensor,
                 ph_targets: Dict[str, torch.Tensor],
                 hidden_layers: Tuple[int, ...],
                 lb: torch.Tensor,
                 ub: torch.Tensor,
                 weights: Dict[str, float],
                 device: str = "cuda",
                 use_double: bool = True) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.use_double = use_double

        # Bounds for normalization
        self.lb = lb.to(self.device)
        self.ub = ub.to(self.device)

        # Supervised data (IC+BC)
        self.Xu = X_u.to(self.device)
        self.uv = uv.to(self.device)

        # Physics points + targets
        self.Xph = X_ph.to(self.device)
        self.ux_t = ph_targets['ux'].to(self.device)
        self.uy_t = ph_targets['uy'].to(self.device)
        self.vx_t = ph_targets['vx'].to(self.device)
        self.vy_t = ph_targets['vy'].to(self.device)

        # Model
        self.model = MLP(in_dim=3, hidden=hidden_layers, out_dim=2).to(self.device)
        self.mse = nn.MSELoss(reduction='mean')

        # Weights
        self.w_data = float(weights.get('data', 2.0))
        self.w_div = float(weights.get('div', 1.0))
        self.w_vort = float(weights.get('vort', 1.0))
        self.w_ux = float(weights.get('ux', 0.5))
        self.w_vy = float(weights.get('vy', 0.5))

        # Logging buffers
        self.loss_hist = {k: [] for k in ["total", "data", "div", "vort", "ux", "vy"]}
        self._log_every = 10
        self._iter_count = 0

        # Precision
        if self.use_double:
            self.double()

    # ----------------------- core ops -----------------------
    def _norm(self, X: torch.Tensor) -> torch.Tensor:
        """Affine map inputs to [-1,1] per-dimension (stable for MLP).
        Note: Derivatives w.r.t. original X are handled by autograd (chain rule).
        """
        return 2.0 * (X - self.lb) / (self.ub - self.lb + 1e-12) - 1.0

    def forward_uv(self, X: torch.Tensor) -> torch.Tensor:
        Xn = self._norm(X)
        return self.model(Xn)

    def grads_xy(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute first partials (u_x, u_y, v_x, v_y) w.r.t. (x,y) via autograd."""
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

    # ----------------------- losses ------------------------
    def loss_fn(self, physics_minibatch: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with optional physics minibatching.

        Args:
            physics_minibatch: if set, splits X_ph (& targets) into chunks.
        Returns:
            total loss tensor, and a dict of float scalars for logging
        """
        # Data loss (IC+BC)
        uv_pred = self.forward_uv(self.Xu)
        L_data = self.mse(uv_pred, self.uv)

        # Physics loss
        if physics_minibatch is None or physics_minibatch <= 0:
            u_x, u_y, v_x, v_y = self.grads_xy(self.Xph)
            L_div = self.mse(u_x + v_y, torch.zeros_like(u_x))
            L_vort = self.mse(v_x - u_y, self.vx_t - self.uy_t)
            L_ux = self.mse(u_x, self.ux_t)
            L_vy = self.mse(v_y, self.vy_t)
        else:
            # Chunked evaluation to save memory
            n = self.Xph.shape[0]
            L_div = L_vort = L_ux = L_vy = 0.0
            n_chunks = math.ceil(n / physics_minibatch)
            for i in range(n_chunks):
                s = slice(i * physics_minibatch, min((i + 1) * physics_minibatch, n))
                m = s.stop - s.start
                u_x, u_y, v_x, v_y = self.grads_xy(self.Xph[s])
                L_div = L_div + self.mse(u_x + v_y, torch.zeros_like(u_x)) * m
                L_vort = L_vort + self.mse(v_x - u_y, self.vx_t[s] - self.uy_t[s]) * m
                L_ux = L_ux + self.mse(u_x, self.ux_t[s]) * m
                L_vy = L_vy + self.mse(v_y, self.vy_t[s]) * m
            # average over samples (not chunks)
            L_div /= n; L_vort /= n; L_ux /= n; L_vy /= n

        L_ph = self.w_div * L_div + self.w_vort * L_vort + self.w_ux * L_ux + self.w_vy * L_vy
        L_tot = self.w_data * L_data + L_ph

        scalars = {
            "data": float(L_data.detach().cpu()),
            "div": float(L_div.detach().cpu()),
            "vort": float(L_vort.detach().cpu()),
            "ux": float(L_ux.detach().cpu()),
            "vy": float(L_vy.detach().cpu()),
        }
        return L_tot, scalars

    # ----------------------- training ---------------------
    def _log(self, scalars: Dict[str, float], total: float, log_every: int) -> None:
        self._iter_count += 1
        if self._iter_count % log_every == 0:
            self.loss_hist["total"].append(total)
            for k, v in scalars.items():
                self.loss_hist[k].append(v)

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

    def train_lbfgs(self, maxiter: int, history_size: int, use_strong_wolfe: bool = False,
                     physics_mb: Optional[int] = None, log_every: int = 10) -> None:
        if use_strong_wolfe:
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, max_iter=maxiter,
                                    history_size=history_size, line_search_fn='strong_wolfe',
                                    tolerance_change=1e-9, tolerance_grad=1e-9)
        else:
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, max_iter=maxiter,
                                    history_size=history_size, tolerance_change=1e-9, tolerance_grad=1e-9)

        def closure():
            optimizer.zero_grad()
            L, parts = self.loss_fn(physics_minibatch=physics_mb)
            L.backward()
            # throttle logging (LBFGS may call closure many times)
            self._log(parts, float(L.detach().cpu()), log_every)
            return L

        optimizer.step(closure)

    # ----------------------- inference -------------------
    @torch.no_grad()
    def predict(self, X_star: np.ndarray) -> np.ndarray:
        Xs = torch.as_tensor(X_star, dtype=torch.float64 if self.use_double else torch.float32, device=self.device)
        uv = self.forward_uv(Xs).detach().cpu().numpy()
        return uv

