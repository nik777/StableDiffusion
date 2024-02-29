import torch 
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def  __init__(self, d_in: int, d_out: int|None = None, scale_factor = 2):
        super().__init__()
        self.scale_factor = scale_factor
        d_out = d_out or d_in
        self.conv = nn.Conv2d(d_in, d_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b c h w)
        return self.conv(F.interpolate(x, scale_factor=self.scale_factor, mode='nearest'))