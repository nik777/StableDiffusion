import pytest 
import torch
import torch.nn as nn
from src.diffusion import Diffusion

def test_Diffusion():
    device = 'cuda:2'
    model = Diffusion().to(torch.bfloat16).to(device)
        
    # x: (b 4 h/8 w/8)
    # c: (b (h/8 * w/8) 320)
    # t: (1 1280)
    b, c, h, w = 8, 4, 64, 64
    s       = 8
    d_cont  = 768
    x = torch.rand(b, c, h, w).to(torch.bfloat16).to(device)
    context = torch.rand(b, s, d_cont).to(torch.bfloat16).to(device)
    t = torch.rand(1, 320).to(torch.bfloat16).to(device)
    assert model(x, context, t).shape == (b, c, h, w)