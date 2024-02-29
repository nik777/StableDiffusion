import torch 
import torch.nn as nn
import torch.nn.functional as F
import einops

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.x_trans   = nn.Sequential(nn.GroupNorm(32, in_channels),
                                       nn.SiLU(),
                                       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.t_trans   = nn.Sequential(nn.SiLU(),
                                       nn.Linear(n_time, out_channels))
        self.out_trans = nn.Sequential(nn.GroupNorm(32, out_channels),
                                       nn.SiLU(),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        
        self.resize  = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (b in_c h w)
        # t: (1, n_time)
        r = self.resize(x)      # (b out_c h w)
        x = self.x_trans(x)     # (b out_c h w)
        t = einops.rearrange(self.t_trans(t), '1 d -> 1 d 1 1') # (1 out_c 1 1)
        x = self.out_trans(x+t) # (b out_c h w)
        return x + r