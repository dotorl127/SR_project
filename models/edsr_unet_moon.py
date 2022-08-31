import torch.nn as nn
import models
from models import register


@register('edsr-unet-moon')
class edsr_unet_moon(nn.Module):

    def __init__(self, encoder_spec, backbone_spec):
        super().__init__()

        self.encoder = models.make(encoder_spec)
        self.backbone = models.make(backbone_spec)

    def forward(self, inp):
        x = self.encoder(inp)
        x = self.backbone(x)
        return x