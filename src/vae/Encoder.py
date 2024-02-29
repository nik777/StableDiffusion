import torch 
import torch.nn as nn
from torch.nn import functional as F

from .ResidualBlock import ResidualBlock
from .AttentionBlock import AttentionBlock


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # b 128 h w

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), # b 128 h/2 w/2
            
            ResidualBlock(128, 256),                                 # b 256 h/2 w/2
            ResidualBlock(256, 256), 
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), # b 256 h/4 w/4
            
            ResidualBlock(256, 512),                                 # b 512 h/4 w/4 
            ResidualBlock(512, 512), 
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), # b 512 h/8 w/8
            
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            
            ResidualBlock(512, 512), 
            AttentionBlock(512), 
            
            ResidualBlock(512, 512), 
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 
            nn.Conv2d(512, 8, kernel_size=3, padding=1),             # b 8 h/8 w/8
            nn.Conv2d(8,   8, kernel_size=1, padding=0), 
        ])

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (b 3 h w)
        # n: (b 4 h/8 w/8)  Sample from N(0,1)

        for layer in self.layers:
            if getattr(layer, 'stride', None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            x = layer(x)

        mean, log_var = x.chunk(2, dim=1)
        std = log_var.clamp(-30, 20).exp().sqrt()
        out = mean + std * z
        out = out * 0.18215
        return out

        
