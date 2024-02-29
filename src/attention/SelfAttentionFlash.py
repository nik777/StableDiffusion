import torch 
import einops 
import math

import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func


class SelfAttentionFlash(nn.Module):
    def __init__(self, h: int, d: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.w_qkv = nn.Linear(d, 3*d, bias=in_proj_bias)
        self.w_o   = nn.Linear(d,   d, bias=out_proj_bias)
        self.h     = h       # heads
        self.hd    = d // h  # head dim 

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        qkv = einops.rearrange(self.w_qkv(x), 'b s (n h d) -> b s n h d', h=self.h, n=3)
        out = einops.rearrange(flash_attn_qkvpacked_func(qkv, causal=causal), 'b s h d -> b s (h d)')
        return self.w_o(out)




