import torch 
import einops 
import math

import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, h: int, d: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.w_qkv = nn.Linear(d, 3*d, bias=in_proj_bias)
        self.w_o   = nn.Linear(d,   d, bias=out_proj_bias)
        self.h     = h       # heads
        self.hd    = d // h  # head dim 

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        q,k,v = self.w_qkv(x).chunk(3, dim=-1)
        q = einops.rearrange(q, 'b s (h d) -> b s h d', h=self.h)
        k = einops.rearrange(k, 'b s (h d) -> b s h d', h=self.h)
        v = einops.rearrange(v, 'b s (h d) -> b s h d', h=self.h)
                             
        scores = einops.einsum(q, k, 'b s1 h d, b s2 h d -> b h s1 s2')/math.sqrt(self.hd)
        scores = scores.softmax(dim=-1)
        out = einops.einsum(scores, v, 'b h s s2, b s2 h d -> b s h d')
        out = einops.rearrange(out, 'b s h d -> b s (h d)')

        return self.w_o(out)