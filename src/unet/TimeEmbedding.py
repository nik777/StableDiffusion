import torch 
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Sequential):
    def __init__(self, d:int):
        super().__init__(nn.Linear(d, 4*d),
                         nn.SiLU(),
                         nn.Linear(4*d, 4*d))
