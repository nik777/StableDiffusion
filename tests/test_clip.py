import pytest 
import torch 


from src.clip import CLIP

def test_CLIP():
    clip = CLIP().to(torch.bfloat16).to('cuda')

    tokens = torch.randint(0, 49408, (4, 77), dtype=torch.long).to('cuda')
    out = clip(tokens)

    assert out.shape == (4, 77, 768)

