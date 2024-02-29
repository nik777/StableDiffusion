import torch 
import einops
import torch.nn as nn
import torch.nn.functional as F
from ..attention import SelfAttentionFlash as SelfAttention
from ..attention import CrossAttentionFlash as CrossAttention
from .GEGLU import GEGLU

class AttentionBlock(nn.Module):
    def __init__(self, h: int, hd: int, d_cont: int=768):
        super().__init__()
        d = h*hd
        self.norm_conv1    = nn.Sequential(nn.GroupNorm(32, d),nn.Conv2d(d, d, kernel_size=1, padding=0))
        self.norm_self_att = nn.Sequential(nn.LayerNorm(d),    SelfAttention(h, d, in_proj_bias=False))
        self.norm          = nn.LayerNorm(d)    
        self.cross_att     = CrossAttention(h, d, d_cont, in_proj_bias=False)
        self.geglu         = nn.Sequential(nn.LayerNorm(d),    GEGLU(d, d))
        self.conv_out      = nn.Conv2d(d, d, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (b c h w)
        # c: (b s d)
        _, _, h, _ = x.shape
        r = x
        x = self.norm_conv1(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.norm_self_att(x)        
        x = x + self.cross_att(self.norm(x), c)
        x = x + self.geglu(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h)   
        return self.conv_out(x) + r