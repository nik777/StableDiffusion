import pytest 
import torch 

from src.vae import AttentionBlock
from src.vae import ResidualBlock
from src.vae import Decoder
from src.vae import Encoder


def test_AttentionBlock():
    block = AttentionBlock(64).to(torch.bfloat16).to('cuda')

    x = torch.rand((4, 64, 32, 32), dtype=torch.bfloat16, device='cuda')
    out = block(x)

    assert out.shape == (4, 64, 32, 32)


def test_ResidualBlock():
    block = ResidualBlock(64, 256).to(torch.bfloat16).to('cuda')

    x = torch.rand((4, 64, 32, 32), dtype=torch.bfloat16, device='cuda')
    out = block(x)

    assert out.shape == (4, 256, 32, 32)


def test_Decoder():
    decoder = Decoder().to(torch.bfloat16).to('cuda')

    x = torch.rand((4, 4, 32, 32), dtype=torch.bfloat16, device='cuda')
    out = decoder(x)

    assert out.shape == (4, 3, 32*8, 32*8)


def test_Encoder():
    encoder = Encoder().to(torch.bfloat16).to('cuda')
    x = torch.rand((4, 3, 32*8, 32*8), dtype=torch.bfloat16, device='cuda')
    z = torch.rand((4, 4, 32, 32),     dtype=torch.bfloat16, device='cuda')                                                     

    out = encoder(x, z)
    assert out.shape == (4, 4, 32, 32)
    

