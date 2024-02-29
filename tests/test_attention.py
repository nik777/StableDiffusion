import pytest 
import torch 
import random 
import einops 
import math

from src.attention import SelfAttentionFlash 
from src.attention import CrossAttentionFlash 
from src.attention import SelfAttention

rtol, atol = (3e-3, 1e-2)

def test_SelfAttentionFlash():
    batch_size = random.randint(1, 10)
    seq_len = random.randint(1, 256)
    head_dim = random.randint(1, 256)
    n_heads = random.randint(1, 8)
    d = head_dim * n_heads
    x = torch.rand((batch_size, seq_len, d), dtype=torch.bfloat16, device='cuda')

    attn = SelfAttentionFlash(n_heads, d).to(torch.bfloat16).to('cuda')
    out = attn(x)

    assert out.shape == (batch_size, seq_len, d)

    k, q, v = attn.w_qkv(x).chunk(3, dim=-1)
    assert k.shape == q.shape == v.shape == (batch_size, seq_len, d)

    q = einops.rearrange(q, 'b s (h hd) -> b s h hd', h=n_heads)
    k = einops.rearrange(k, 'b s (h hd) -> b s h hd', h=n_heads)
    v = einops.rearrange(v, 'b s (h hd) -> b s h hd', h=n_heads)

    scores = einops.einsum(q, k, 'b s1 h d, b s2 h d -> b h s1 s2')/math.sqrt(head_dim) 
    scores = scores.softmax(dim=-1) 
    s_out = einops.einsum(scores, v, 'b h s1 s2, b s2 h d -> b s1 h d')
    s_out = einops.rearrange(s_out, 'b s h hd -> b s (h hd)')
    s_out = attn.w_o(s_out)

    assert out.shape == s_out.shape
    assert torch.allclose(out, s_out, atol=atol, rtol=rtol)


def test_SelfAttention():
    batch_size = random.randint(1, 10)
    seq_len = random.randint(1, 256)
    head_dim = random.randint(1, 256)
    n_heads = random.randint(1, 8)
    d = head_dim * n_heads
    x = torch.rand((batch_size, seq_len, d), dtype=torch.bfloat16, device='cuda')

    attn = SelfAttention(n_heads, d).to(torch.bfloat16).to('cuda')
    out = attn(x)

    assert out.shape == (batch_size, seq_len, d)

    k, q, v = attn.w_qkv(x).chunk(3, dim=-1)
    assert k.shape == q.shape == v.shape == (batch_size, seq_len, d)

    q = einops.rearrange(q, 'b s (h hd) -> b s h hd', h=n_heads)
    k = einops.rearrange(k, 'b s (h hd) -> b s h hd', h=n_heads)
    v = einops.rearrange(v, 'b s (h hd) -> b s h hd', h=n_heads)

    scores = einops.einsum(q, k, 'b s1 h d, b s2 h d -> b h s1 s2')/math.sqrt(head_dim) 
    scores = scores.softmax(dim=-1) 
    s_out = einops.einsum(scores, v, 'b h s1 s2, b s2 h d -> b s1 h d')
    s_out = einops.rearrange(s_out, 'b s h hd -> b s (h hd)')
    s_out = attn.w_o(s_out)

    assert out.shape == s_out.shape
    assert torch.allclose(out, s_out, atol=atol, rtol=rtol)    

def test_CrossAttentionFlash():
    batch_size  = random.randint(1, 10)
    seq_len     = random.randint(1, 256)
    cross_d     = random.randint(1, 256)
    h           = random.randint(1, 32)
    hd          = random.randint(1, 256) 
    d           = h * hd
    
    

    x = torch.rand((batch_size, seq_len, d),  dtype=torch.bfloat16, device='cuda')
    c = torch.rand((batch_size, seq_len, cross_d), dtype=torch.bfloat16, device='cuda')

    cros_att = CrossAttentionFlash(h, d, cross_d).to(torch.bfloat16).to('cuda')
    cros_out = cros_att(x, c)

    assert cros_out.shape == (batch_size, seq_len, d)

    q = cros_att.w_q(x)
    k = cros_att.w_k(c)
    v = cros_att.w_v(c)

    q = einops.rearrange(q, 'b s (h d) -> b h s d', h=h)
    k = einops.rearrange(k, 'b s (h d) -> b h s d', h=h)
    v = einops.rearrange(v, 'b s (h d) -> b h s d', h=h)

    scores = einops.einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(hd)

    scores= scores.softmax(dim=-1)
    out = einops.einsum(scores, v, 'b h s s1, b h s1 d -> b h s d')
    out = einops.rearrange(out, 'b h s d -> b s (h d)')
    out = cros_att.w_o(out)

    assert out.shape == cros_out.shape
    assert torch.allclose(out, cros_out, atol=atol, rtol=rtol)

    x = torch.rand((batch_size, seq_len, d), dtype=torch.bfloat16, device='cuda')
    
