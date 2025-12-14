from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models.mlp import MLP
from models.conv2d import Conv2DNet
from models.conv3d import Conv3DNet


class PhysicsInformedNN(nn.Module):
    """PINN that learns (u,v)(x,y,t) with kinematic constraints.

    Args:
        X_u: supervised points [N, h, w, 3] or [Nu, 3] (flattened)
        uv: supervised targets [N, h, w, 2] or [Nu, 2] (flattened)
        X_ph: physics points [Nph, 3]
        ph_targets: dict containing 'ux','uy','vx','vy' (each [Nph, 1])
        hidden_layers: hidden sizes for MLP
        lb, ub: lower/upper bounds for input normalization [3]
        weights: dict of loss weights
        device: 'cpu' or 'cuda'
        use_double: if True, converts module parameters and tensors to float64
        model_type: 'mlp', 'conv2d', or 'conv3d'
        conv_channels: channel sizes for convolutional models
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
                 use_double: bool = True,
                 model_type: str = "mlp",
                 conv_channels: Tuple[int, ...] = (7, 22, 14)) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.use_double = use_double
        self.model_type = model_type.lower()

        # Bounds for normalization
        self.lb = lb.to(self.device)  # [3]
        self.ub = ub.to(self.device)  # [3]

        # Store grid shape for Conv models
        self.grid_shape = None  # Will be (N, h, w) for Conv models
        
        # Supervised data: handle differently based on model type
        if self.model_type in ["conv2d", "conv3d"]:
            # Conv models need 4D input [N, h, w, 3]
            if X_u.ndim == 4:
                self.Xu = X_u.to(self.device)  # [N, h, w, 3]
                self.uv = uv.to(self.device)   # [N, h, w, 2]
                self.grid_shape = X_u.shape[:3]  # (N, h, w)
            else:
                raise ValueError(f"{self.model_type} requires 4D input [N, h, w, 3], got shape {X_u.shape}")
        else:
            # MLP: flatten if needed [N,h,w,3] -> [N*h*w,3]
            if X_u.ndim == 4:
                self.Xu = X_u.reshape(-1, X_u.shape[-1]).to(self.device)  # [N*h*w, 3]
                self.uv = uv.reshape(-1, uv.shape[-1]).to(self.device)    # [N*h*w, 2]
                self.grid_shape = X_u.shape[:3]  # Store for potential use
            else:
                self.Xu = X_u.to(self.device)  # [Nu, 3]
                self.uv = uv.to(self.device)   # [Nu, 2]

        # Physics points + targets (always flattened)
        self.Xph = X_ph.to(self.device)  # [Nph, 3]
        self.ux_t = ph_targets['ux'].to(self.device)  # [Nph, 1]
        self.uy_t = ph_targets['uy'].to(self.device)  # [Nph, 1]
        self.vx_t = ph_targets['vx'].to(self.device)  # [Nph, 1]
        self.vy_t = ph_targets['vy'].to(self.device)  # [Nph, 1]

        # Model selection
        if self.model_type == "mlp":
            self.model = MLP(in_dim=3, hidden=hidden_layers, out_dim=2).to(self.device)
        elif self.model_type == "conv2d":
            self.model = Conv2DNet(channels=conv_channels, in_channels=3, out_channels=2).to(self.device)
        elif self.model_type == "conv3d":
            # Use optimized Conv3D channels for similar parameter count
            conv3d_channels = (11, 11) if conv_channels == (7, 22, 14) else conv_channels
            self.model = Conv3DNet(channels=conv3d_channels, in_channels=3, out_channels=2).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'mlp', 'conv2d', or 'conv3d'.")
        
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
        """Forward pass through the model.
        
        Args:
            X: [batch, 3] for MLP or [N, h, w, 3] for Conv models
        Returns:
            [batch, 2] for MLP or [N, h, w, 2] for Conv models
        """
        Xn = self._norm(X)
        return self.model(Xn)

    def grads_xy(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute first partials (u_x, u_y, v_x, v_y) w.r.t. (x,y) via autograd.
        
        Args:
            X: [batch, 3] for MLP or [N, h, w, 3] for Conv models
        Returns:
            Four tensors of gradients, same shape as input (minus last dim)
        """
        X = X.clone().detach().requires_grad_(True)
        uv = self.forward_uv(X)
        
        if self.model_type in ["conv2d", "conv3d"]:
            # Conv models: [N, h, w, 2]
            u, v = uv[..., 0:1], uv[..., 1:2]  # [N, h, w, 1]
            
            du_dX = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                        retain_graph=True, create_graph=True)[0]
            dv_dX = torch.autograd.grad(v, X, grad_outputs=torch.ones_like(v),
                                        retain_graph=True, create_graph=True)[0]
            u_x, u_y = du_dX[..., 0:1], du_dX[..., 1:2]  # [N, h, w, 1]
            v_x, v_y = dv_dX[..., 0:1], dv_dX[..., 1:2]  # [N, h, w, 1]
        else:
            # MLP: [batch, 2]
            u, v = uv[:, 0:1], uv[:, 1:2]  # [batch, 1]
            
            du_dX = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                        retain_graph=True, create_graph=True)[0]
            dv_dX = torch.autograd.grad(v, X, grad_outputs=torch.ones_like(v),
                                        retain_graph=True, create_graph=True)[0]
            u_x, u_y = du_dX[:, 0:1], du_dX[:, 1:2]  # [batch, 1]
            v_x, v_y = dv_dX[:, 0:1], dv_dX[:, 1:2]  # [batch, 1]
        
        return u_x, u_y, v_x, v_y
    
    

    def _sample_physics_from_grid(self, 
                                   u_x_grid: torch.Tensor, 
                                   u_y_grid: torch.Tensor,
                                   v_x_grid: torch.Tensor, 
                                   v_y_grid: torch.Tensor,
                                   X_points: torch.Tensor,
                                   X_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample physics gradients from precomputed grid (VECTORIZED).
        
        Args:
            u_x_grid, u_y_grid, v_x_grid, v_y_grid: [N, h, w, 1] gradients on full grid
            X_points: [n_physics, 3] physics point coordinates
            X_grid: [N, h, w, 3] grid coordinates
        
        Returns:
            Sampled gradients [n_physics, 1] for each component
        """
        N, h, w = self.grid_shape
        batch_size = X_points.shape[0]
        
        # Extract grid coordinates (1D arrays)
        x_coords = X_grid[0, 0, :, 0]  # [w] - x coordinates
        y_coords = X_grid[0, :, 0, 1]  # [h] - y coordinates
        t_coords = X_grid[:, 0, 0, 2]  # [N] - t coordinates
        
        # Extract physics point coordinates [batch_size]
        x_vals = X_points[:, 0]  # [batch_size]
        y_vals = X_points[:, 1]  # [batch_size]
        t_vals = X_points[:, 2]  # [batch_size]
        
        # VECTORIZED: Find nearest indices for ALL points at once
        # Compute distances: [batch_size, N/h/w]
        t_dists = torch.abs(t_vals.unsqueeze(1) - t_coords.unsqueeze(0))  # [batch_size, N]
        y_dists = torch.abs(y_vals.unsqueeze(1) - y_coords.unsqueeze(0))  # [batch_size, h]
        x_dists = torch.abs(x_vals.unsqueeze(1) - x_coords.unsqueeze(0))  # [batch_size, w]
        
        # Find nearest indices: [batch_size]
        n_indices = torch.argmin(t_dists, dim=1)  # [batch_size]
        i_indices = torch.argmin(y_dists, dim=1)  # [batch_size]
        j_indices = torch.argmin(x_dists, dim=1)  # [batch_size]
        
        # VECTORIZED: Sample all gradients at once using advanced indexing
        # u_x_grid[n, i, j, :] for each point
        u_x_sampled = u_x_grid[n_indices, i_indices, j_indices, :]  # [batch_size, 1]
        u_y_sampled = u_y_grid[n_indices, i_indices, j_indices, :]  # [batch_size, 1]
        v_x_sampled = v_x_grid[n_indices, i_indices, j_indices, :]  # [batch_size, 1]
        v_y_sampled = v_y_grid[n_indices, i_indices, j_indices, :]  # [batch_size, 1]
        
        return u_x_sampled, u_y_sampled, v_x_sampled, v_y_sampled

    # ----------------------- losses ------------------------
    def loss_fn(self, physics_minibatch: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with optional physics minibatching.

        Args:
            physics_minibatch: if set, splits X_ph (& targets) into chunks.
        Returns:
            total loss tensor, and a dict of float scalars for logging
        """
        # OPTIMIZED: For Conv models with grid data, compute full grid once
        if self.model_type in ["conv2d", "conv3d"] and self.grid_shape is not None:
            # Single forward pass on full grid (reused for both losses!)
            X_grid = self.Xu.clone().detach().requires_grad_(True)
            uv_grid = self.forward_uv(X_grid)
            
            # Data loss from grid
            L_data = self.mse(uv_grid, self.uv)
            
            # Physics loss: compute gradients on full grid, then sample
            u_x_grid, u_y_grid, v_x_grid, v_y_grid = self.grads_xy(X_grid)
            
            # Sample physics gradients from grid
            u_x, u_y, v_x, v_y = self._sample_physics_from_grid(
                u_x_grid, u_y_grid, v_x_grid, v_y_grid, self.Xph, X_grid
            )
            
            L_div = self.mse(u_x + v_y, torch.zeros_like(u_x))
            L_vort = self.mse(v_x - u_y, self.vx_t - self.uy_t)
            L_ux = self.mse(u_x, self.ux_t)
            L_vy = self.mse(v_y, self.vy_t)
        else:
            # Original approach for MLP or when no grid available
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
        pbar = tqdm(range(1, steps + 1), desc="Adam", unit="step")
        for it in pbar:
            opt.zero_grad()
            L, parts = self.loss_fn(physics_minibatch=physics_mb)
            L.backward()
            opt.step()
            
            loss_val = float(L.detach().cpu())
            self._log(parts, loss_val, log_every)
            
            # Update progress bar with loss values
            pbar.set_postfix({
                'total': f'{loss_val:.3e}',
                'data': f'{parts["data"]:.2e}',
                'div': f'{parts["div"]:.2e}',
                'vort': f'{parts["vort"]:.2e}'
            })
            
            # Print detailed log more frequently
            if it % log_every == 0:
                tqdm.write(f"[Adam {it:05d}] total={loss_val:.4e} data={parts['data']:.3e} div={parts['div']:.3e} vort={parts['vort']:.3e} ux={parts['ux']:.3e} vy={parts['vy']:.3e}")

    def train_lbfgs(self, maxiter: int, history_size: int, use_strong_wolfe: bool = False,
                     physics_mb: Optional[int] = None, log_every: int = 10) -> None:
        if use_strong_wolfe:
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, max_iter=maxiter,
                                    history_size=history_size, line_search_fn='strong_wolfe',
                                    tolerance_change=1e-9, tolerance_grad=1e-9)
        else:
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, max_iter=maxiter,
                                    history_size=history_size, tolerance_change=1e-9, tolerance_grad=1e-9)

        # Counter for closure calls
        self._lbfgs_iter = [0]
        pbar = tqdm(total=maxiter, desc="LBFGS", unit="iter")
        
        def closure():
            optimizer.zero_grad()
            L, parts = self.loss_fn(physics_minibatch=physics_mb)
            L.backward()
            
            loss_val = float(L.detach().cpu())
            self._log(parts, loss_val, log_every)
            
            # Update progress bar
            self._lbfgs_iter[0] += 1
            if self._lbfgs_iter[0] <= maxiter:
                pbar.update(1)
                pbar.set_postfix({
                    'total': f'{loss_val:.3e}',
                    'data': f'{parts["data"]:.2e}',
                    'div': f'{parts["div"]:.2e}',
                    'vort': f'{parts["vort"]:.2e}'
                })
            
            # Print detailed log more frequently
            if self._lbfgs_iter[0] % log_every == 0:
                tqdm.write(f"[LBFGS {self._lbfgs_iter[0]:05d}] total={loss_val:.4e} data={parts['data']:.3e} div={parts['div']:.3e} vort={parts['vort']:.3e} ux={parts['ux']:.3e} vy={parts['vy']:.3e}")
            
            return L

        optimizer.step(closure)
        pbar.close()

    # ----------------------- inference -------------------
    @torch.no_grad()
    def predict(self, X_star: np.ndarray) -> np.ndarray:
        """Predict velocities at given points.
        
        Args:
            X_star: [N, 3] array of (x, y, t) coordinates
        
        Returns:
            [N, 2] array of (u, v) predictions
        """
        Xs = torch.as_tensor(X_star, dtype=torch.float64 if self.use_double else torch.float32, device=self.device)
        
        # For Conv models, we need to reshape to grid format if possible
        if self.model_type in ["conv2d", "conv3d"] and self.grid_shape is not None:
            # Check if we're predicting on a single time slice
            t_vals = X_star[:, 2]
            unique_t = np.unique(t_vals)
            
            if len(unique_t) == 1:
                # Single time slice - reshape to [1, h, w, 3]
                N_grid, h_grid, w_grid = self.grid_shape
                n_points = X_star.shape[0]
                
                # Check if the number of points matches h*w
                if n_points == h_grid * w_grid:
                    # Reshape to grid format [1, h, w, 3]
                    Xs_grid = Xs.reshape(1, h_grid, w_grid, 3)
                    uv = self.forward_uv(Xs_grid).detach().cpu().numpy()
                    # Flatten back to [N, 2]
                    return uv.reshape(-1, 2)
            
            # For multiple time slices or non-matching dimensions, fall back to MLP-style
            # We need to evaluate point by point or handle it differently
            # For now, raise an error with a helpful message
            if self.model_type in ["conv2d", "conv3d"]:
                raise ValueError(
                    f"Conv models require grid-structured input for prediction.\n"
                    f"Expected single time slice with {self.grid_shape[1]*self.grid_shape[2]} points, "
                    f"got {X_star.shape[0]} points with {len(unique_t)} unique time values.\n"
                    f"Hint: For Conv models, predict on complete spatial grids at single time slices."
                )
        
        # MLP path: direct prediction
        uv = self.forward_uv(Xs).detach().cpu().numpy()
        return uv

