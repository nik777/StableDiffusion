import torch
import torch.nn as nn
import torch.nn.functional as F
from ..unet import UNET
from ..unet import TimeEmbedding

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNET()
        self.time_embedding = TimeEmbedding(320)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (b 4 h/8 w/8)
        # c: (b s d)
        # t: (1 320)
        t = self.time_embedding(t)
        return self.unet(x, c, t)      