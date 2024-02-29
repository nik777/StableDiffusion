import torch 
import torch.nn as nn
import torch.nn.functional as F
from .AttentionBlock import AttentionBlock
from .ResidualBlock import ResidualBlock

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, ResidualBlock):
                x = module(x, t)
            elif isinstance(module, AttentionBlock):
                x = module(x, c)
            else:
                x = module(x)
        return x
    

class SwitchSeqAddSkip(SwitchSequential):
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, ResidualBlock):
                x = module(x, t)
            elif isinstance(module, AttentionBlock):
                x = module(x, c)
            else:
                x = module(x)
        return x + t