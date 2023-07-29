import torch
import torch.nn as nn
from .p2t import p2t_base


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder,self).__init__()

        self.encoder = p2t_base()
        self.encoder.load_state_dict(torch.load(opt.p2t_path, map_location='cpu'),
                                     strict=False)

    def forward(self, x):
        out = self.encoder(x)
        return out[::-1] 