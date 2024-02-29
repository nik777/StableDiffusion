import torch 
import einops 
import math

import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func


class CrossAttentionFlash(nn.Module):
    def __init__(self, h: int, d: int, cross_d: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.w_q  = nn.Linear(d,       d, bias=in_proj_bias)
        self.w_k  = nn.Linear(cross_d, d, bias=in_proj_bias)
        self.w_v  = nn.Linear(cross_d, d, bias=in_proj_bias)
        self.w_o  = nn.Linear(d,       d, bias=out_proj_bias)
        self.h    = h       # heads
        self.hd   = d // h  # head dim 

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = einops.rearrange(self.w_q(x),       'b s (h d) -> b s h d', h=self.h)
        k = einops.rearrange(self.w_k(context), 'b s (h d) -> b s h d', h=self.h)
        v = einops.rearrange(self.w_v(context), 'b s (h d) -> b s h d', h=self.h)
        out = einops.rearrange(flash_attn_func(q, k, v), 'b s h d -> b s (h d)')
        return self.w_o(out)




