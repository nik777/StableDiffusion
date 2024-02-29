import torch 
import torch.nn as nn
import torch.nn.functional as F
from .SwitchSequential import SwitchSequential
from .ResidualBlock    import ResidualBlock
from .AttentionBlock   import AttentionBlock
from .Upsample         import Upsample

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), # b 320 h/8 w/8
                                      SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
                                      SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
                                      SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)), # b 320 h/16 w/16
                                      SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),          # b 640 h/16 w/16 
                                      SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
                                      SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)), # b 640 h/32 w/32
                                      SwitchSequential(ResidualBlock(640, 1280),  AttentionBlock(8, 160)),        # b 1280 h/32 w/32
                                      SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
                                      SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)), # b 1280 h/64 w/64
                                      SwitchSequential(ResidualBlock(1280, 1280)),
                                      SwitchSequential(ResidualBlock(1280, 1280))])

        self.bottleneck = SwitchSequential(ResidualBlock(1280, 1280), 
                                           AttentionBlock(8, 160), 
                                           ResidualBlock(1280, 1280))
        
        self.decoder = nn.ModuleList([SwitchSequential(ResidualBlock(2560, 1280)),
                                     SwitchSequential(ResidualBlock(2560, 1280)),
                                     SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),         # b 1280 h/32 w/32
                                     SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
                                     SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
                                     SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)), # b 1280 h/16 w/16
                                     SwitchSequential(ResidualBlock(1920, 640),  AttentionBlock(8, 80)),                  # b 640 h/16 w/16
                                     SwitchSequential(ResidualBlock(1280, 640),  AttentionBlock(8, 80)),                
                                     SwitchSequential(ResidualBlock(960, 640),   AttentionBlock(8, 80), Upsample(640)), # b 640 h/8 w/8
                                     SwitchSequential(ResidualBlock(960, 320),   AttentionBlock(8, 40)),                # b 320 h/8 w/8
                                     SwitchSequential(ResidualBlock(640, 320),   AttentionBlock(8, 40)),
                                     SwitchSequential(ResidualBlock(640, 320),   AttentionBlock(8, 40))])
        
        self.out = nn.Sequential(nn.GroupNorm(32, 320), nn.SiLU(), nn.Conv2d(320, 4, kernel_size=3, padding=1))
        
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (b 4 h/8 w/8)
        # c: (b (h/8 * w/8) 320)
        # t: (1 1280)
        skips = []

        for m in self.encoder:
            x = m(x, c, t)
            skips.append(x)
        
        x = self.bottleneck(x, c, t)

        for m in self.decoder:
            x = torch.cat([x, skips.pop()], dim=1)
            x = m(x, c, t)

        return self.out(x)
        
        

        