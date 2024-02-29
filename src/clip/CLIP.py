import torch 
import torch.nn as nn
from .Layer import Layer
from .Embedding import Embedding

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = Embedding(49408, 768, 77)
        self.layers = nn.ModuleList([Layer(12, 768) for _ in range(12)])
        self.norm = nn.LayerNorm(768)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # x: (b s)
        x = self.e(x) # (b s d)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
