import torch.nn as nn

from .common import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        if n_ch <= 0 or n_cls <= 0 or hidden_size <= 0:
            raise ValueError("n_ch, n_cls, hidden_size must be positive")
        self.n_ch = n_ch
        self.n_cls = n_cls
        self.hidden_size = hidden_size
        self.bilinear = bilinear
        h = hidden_size
        chs = {
            'enc1': max(1, h),
            'enc2': max(1, h * 2),
            'enc3': max(1, h * 4),
            'enc4': max(1, h * 8),
            'bottle': max(1, h * 16),
        }
        self.inc = DoubleConv(max(1, n_ch), chs['enc1'])
        self.down1 = Down(chs['enc1'], chs['enc2'])
        self.down2 = Down(chs['enc2'], chs['enc3'])
        self.down3 = Down(chs['enc3'], chs['enc4'])
        self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear)
        self.up2 = Up(chs['enc4'], chs['enc3'], chs['enc3'], bilinear)
        self.up3 = Up(chs['enc3'], chs['enc2'], chs['enc2'], bilinear)
        self.up4 = Up(chs['enc2'], chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], max(1, n_cls))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
