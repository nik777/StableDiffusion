import pytest
import torch
from src.unet import TimeEmbedding
from src.unet import ResidualBlock
from src.unet import AttentionBlock
from src.unet import SwitchSequential
from src.unet import UNET
from src.unet import Upsample
from src.unet import GEGLU


def test_TimeEmbedding():
    d = 64
    e = TimeEmbedding(d).to('cuda')
    t = torch.rand(1, d).to('cuda')
    assert e(t).shape == (1, 4*d)

def test_ResidualBlock():
    b, h, w  = 1, 32, 32
    in_c     = 64
    out_c    = 128
    n_time   = 640

    r = ResidualBlock(in_c, out_c, n_time).to('cuda')
    x = torch.rand(b, in_c, h, w).to('cuda')
    t = torch.rand(1, n_time).to('cuda')
    assert r(x, t).shape == (b, out_c, h, w)

def test_GEGLU():
    d_in, d_out = 64, 128
    g = GEGLU(d_in, d_out).to('cuda')
    x = torch.rand(1, d_in).to('cuda')
    assert g(x).shape == (1, d_out)

def test_AttentionBlock():
        b,h,w = 8, 32, 32
        n_heads = 16
        head_d  = 64
        c       = n_heads*head_d
        s       = 128
        d_cont  = 256
        
        a = AttentionBlock(n_heads, head_d, d_cont).to(torch.bfloat16).to('cuda')
        x = torch.rand(b, c, h, w).to(torch.bfloat16).to('cuda')
        context = torch.rand(b, s, d_cont).to(torch.bfloat16).to('cuda')
        assert a(x, context).shape == (b, c, h, w)

def test_SwitchSequential():
    b, h, w  = 8, 32, 32
    in_c     = 64
    out_c    = 128
    n_time   = 640
    # out after residual block is (b, out_c, h, w) 
    n_heads  = 4
    head_d  = out_c // n_heads
    s       = 128
    d_cont  = 256

    x = torch.rand(b, in_c, h, w).to(torch.bfloat16).to('cuda')
    t = torch.rand(1, n_time).to(torch.bfloat16).to('cuda')
    context = torch.rand(b, s, d_cont).to(torch.bfloat16).to('cuda')

    r = ResidualBlock(in_c, out_c, n_time)
    a = AttentionBlock(n_heads, head_d, d_cont)

    l = SwitchSequential(r, a).to(torch.bfloat16).to('cuda')
    assert l(x, context, t).shape == (b, out_c, h, w)


def test_Upsample():
    b, c, h, w = 1, 64, 32, 32
    scale_factor = 2
    u = Upsample(c, scale_factor=scale_factor).to('cuda')
    x = torch.rand(b, c, h, w).to('cuda')
    assert u(x).shape == (b, c, h*scale_factor, w*scale_factor)


def test_UNET():
    device = 'cuda:2'
    # x: (b 4 h/8 w/8)
    # c: (b (h/8 * w/8) 320)
    # t: (1 1280)
    b, c, h, w = 8, 4, 64, 64
    s       = 256
    d_cont  = 768
    x = torch.rand(b, c, h, w).to(torch.bfloat16).to(device)
    context = torch.rand(b, s, d_cont).to(torch.bfloat16).to(device)
    t = torch.rand(1, 1280).to(torch.bfloat16).to(device)

    unet = UNET().to(torch.bfloat16).to(device)

    assert unet(x, context, t).shape == (b, c, h, w)


