import torch 
import torch.nn as nn
from ..attention import SelfAttentionFlash as SelfAttention

class Layer(nn.Module):
    def __init__(self, h: int, d:int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn = SelfAttention(h, d)
        self.fc1 = nn.Linear(d, d*4)
        self.fc2 = nn.Linear(d*4, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b s d)
        r = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + r

        r = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = x*torch.sigmoid(1.702*x)
        x = self.fc2(x)
        x = x + r
        return x
