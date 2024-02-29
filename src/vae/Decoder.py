import torch
import torch.nn as nn
from torch.nn import functional as F
from .AttentionBlock import AttentionBlock
from .ResidualBlock import ResidualBlock

class Decoder(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Conv2d(4, 4, kernel_size=1, padding=0),   # (b 4 h/8 w/8)
                nn.Conv2d(4, 512, kernel_size=3, padding=1), # (b 512 h/8 w/8)
                ResidualBlock (512, 512), 
                AttentionBlock(512), 
                ResidualBlock (512, 512), 
                ResidualBlock (512, 512), 
                ResidualBlock (512, 512), 
                ResidualBlock (512, 512), 

                nn.Upsample(scale_factor=2),                  # (b 512 h/4 w/4)
                nn.Conv2d(512, 512, kernel_size=3, padding=1), 
                ResidualBlock(512, 512), 
                ResidualBlock(512, 512), 
                ResidualBlock(512, 512), 

                nn.Upsample(scale_factor=2),                   # (b 512 h/2 w/2)
                nn.Conv2d(512, 512, kernel_size=3, padding=1), 
                ResidualBlock(512, 256),                       # (b 256 h/2 w/2)
                ResidualBlock(256, 256), 
                ResidualBlock(256, 256), 

                nn.Upsample(scale_factor=2),                   # (b 256 h w)
                nn.Conv2d(256, 256, kernel_size=3, padding=1), 
                ResidualBlock(256, 128),                       # (b 128 h w)   
                ResidualBlock(128, 128), 
                ResidualBlock(128, 128), 
                nn.GroupNorm(32, 128), 
                nn.SiLU(), 
                nn.Conv2d(128, 3, kernel_size=3, padding=1), 
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (b 4 h/8 w/8)
            x = x / 0.18215
            return super().forward(x) # b 3 h w
