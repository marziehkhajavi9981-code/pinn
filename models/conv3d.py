from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DNet(nn.Module):
    """3D Convolutional network that processes spatiotemporal (N, h, w) data.
    
    Takes input [N, h, w, 3] and outputs [N, h, w, 2].
    Uses reflection padding to avoid boundary artifacts.
    
    Args:
        channels: sequence of channel sizes for conv layers
        in_channels: input channels (default 3 for x,y,t)
        out_channels: output channels (default 2 for u,v)
    """

    def __init__(self, 
                 channels: Sequence[int] = (32, 64, 64, 32), 
                 in_channels: int = 3,
                 out_channels: int = 2) -> None:
        super().__init__()
        
        # Build encoder - use padding=0, we'll add reflection padding manually
        self.encoder_convs = nn.ModuleList()
        curr_channels = in_channels
        for i, ch in enumerate(channels):
            self.encoder_convs.append(nn.Conv3d(curr_channels, ch, kernel_size=3, padding=0))
            curr_channels = ch
        
        # Build decoder
        self.decoder_convs = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder_convs.append(nn.Conv3d(channels[i], channels[i-1], kernel_size=3, padding=0))
        
        # Final output layer (no activation on output)
        self.output_conv = nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=0)
        
        # Activation function
        self.activation = nn.Tanh()
        
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, h, w, 3]
        Returns:
            [N, h, w, 2]
        """
        if x.ndim != 4:
            raise ValueError(f"Conv3DNet requires 4D input [N, h, w, 3], got shape {x.shape}")
        
        # Input: [N, h, w, 3] -> [1, 3, N, h, w] (batch=1, channels-first for Conv3d)
        # Treat N as temporal dimension
        x = x.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, N, h, w]
        
        # Encoder with reflection padding (avoids boundary artifacts)
        for conv in self.encoder_convs:
            x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')  # 3D padding
            x = conv(x)
            x = self.activation(x)
        
        # Decoder with reflection padding
        for conv in self.decoder_convs:
            x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')  # 3D padding
            x = conv(x)
            x = self.activation(x)
        
        # Output layer (no activation on final layer)
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')  # 3D padding
        x = self.output_conv(x)
        
        # Output: [1, 2, N, h, w] -> [N, h, w, 2]
        x = x.squeeze(0).permute(1, 2, 3, 0)
        
        return x

