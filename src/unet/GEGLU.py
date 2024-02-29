import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in,   4*d_in*2)
        self.fc2 = nn.Linear(4*d_in, d_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b s d)
        x, gates = self.fc1(x).chunk(2, dim=-1)
        x = x * F.gelu(gates)
        return self.fc2(x)