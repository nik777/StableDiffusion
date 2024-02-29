import torch 
import einops 
import torch.nn as nn
from torch.nn import functional as F
from ..attention import SelfAttention

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b c h w)
        b, c, h, w = x.shape
        r = x
        x = einops.rearrange(self.norm(x), 'b  c h  w -> b (h w) c')
        x = einops.rearrange(self.attn(x), 'b (h w) c -> b  c h  w', h=h)
        x = x + r
        return x