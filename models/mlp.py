from typing import Sequence
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple fully-connected network with Tanh activations.

    Args:
        in_dim: input dimension (here 3 for x,y,t)
        hidden: sequence of hidden layer sizes
        out_dim: output dimension (here 2 for u,v)
    """

    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int) -> None:
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())  # smooth, bounded, PINN-friendly
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)