import torch
import torch.nn as nn
from models import register


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, dropout=0.0):
        super(Down, self).__init__()
        down = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if norm:
            down.append(nn.InstanceNorm2d(out_channels))
        down.append(nn.LeakyReLU(0.2))
        if dropout:
            down.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*down)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Up, self).__init__()
        up = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
        ]
        if dropout:
            up.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*up)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, flag=True):
        super(OutConv, self).__init__()
        if flag:
            self.conv = nn.Sequential(
                nn.MaxPool2d((4, 1)),  # for add model
                nn.Upsample(scale_factor=2),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(in_channels, out_channels, 4, padding=1),
                nn.ReLU(True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(in_channels, out_channels, 4, padding=1),
                nn.ReLU(True),
            )

    def forward(self, x):
        return self.conv(x)


@register('pix2pix-unet')
class Pix2Pix_Unet(nn.Module):
    def __init__(self, n_colors):
        super(Pix2Pix_Unet, self).__init__()
        self.upsample = nn.Upsample(size=(256, 1024), mode='bilinear', align_corners=True)
        self.down1 = Down(n_colors, 64, norm=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512, dropout=0.5)
        self.down5 = Down(512, 512, dropout=0.5)
        self.down6 = Down(512, 512, dropout=0.5)
        self.down7 = Down(512, 512, dropout=0.5)
        self.down8 = Down(512, 512, norm=False, dropout=0.5)
        self.up1 = Up(512, 512, dropout=0.5)
        self.up2 = Up(1024, 512, dropout=0.5)
        self.up3 = Up(1024, 512, dropout=0.5)
        self.up4 = Up(1024, 512, dropout=0.5)
        self.up5 = Up(1024, 256)
        self.up6 = Up(512, 128)
        self.up7 = Up(256, 64)
        self.outc = OutConv(128, n_colors, True)

    def forward(self, x):
        x = self.upsample(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        x = self.outc(u7)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('pix2pix-unet-del')
class Pix2Pix_Unet_del(nn.Module):
    def __init__(self, n_colors):
        super(Pix2Pix_Unet_del, self).__init__()
        self.down1 = Down(n_colors, 64, norm=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512, dropout=0.5)
        self.down5 = Down(512, 512, dropout=0.5)
        self.down6 = Down(512, 512, norm=False, dropout=0.5)
        self.up1 = Up(512, 512, dropout=0.5)
        self.up2 = Up(1024, 512, dropout=0.5)
        self.up3 = Up(1024, 256)
        self.up4 = Up(512, 128)
        self.up5 = Up(256, 64)
        self.outc = OutConv(128, n_colors, False)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        x = self.outc(u5)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
