import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d: int, n_tokens: int):
        super().__init__()
        self.t_emb = nn.Embedding(vocab_size, d)
        self.p_emb = nn.Parameter(torch.zeros(n_tokens, d))

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (b s)
        return self.t_emb(x) + self.p_emb
