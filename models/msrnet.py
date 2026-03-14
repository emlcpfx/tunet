import torch
import torch.nn as nn

from .blocks import OutConv


class RecurrentConv(nn.Module):
    def __init__(self, out_channels, t=2):
        super().__init__()
        self.t = t
        num_groups = min(8, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        for _ in range(1, self.t):
            out = self.conv(x + out)
        return out


class RRCNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.adapt = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.rcnn1 = RecurrentConv(out_channels, t=t)
        self.rcnn2 = RecurrentConv(out_channels, t=t)

    def forward(self, x):
        residual = self.adapt(x)
        out = self.rcnn1(residual)
        out = self.rcnn2(out)
        return out + residual


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        num_groups_int = min(8, F_int)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups_int, F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups_int, F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MSRNDown(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.rrcnn = RRCNN_block(in_channels, out_channels, t=t)

    def forward(self, x):
        return self.rrcnn(self.pool(x))


class AttentionUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, t=2):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
            gate_channels = in_channels
        else:
            up_out_channels = in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2)
            conv_in_channels = up_out_channels + skip_channels
            gate_channels = up_out_channels
        F_int = max(1, skip_channels // 2)
        self.attn = AttentionGate(F_g=gate_channels, F_l=skip_channels, F_int=F_int)
        self.rrcnn = RRCNN_block(max(1, conv_in_channels), max(1, out_channels), t=t)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            pad_left = diffX // 2
            pad_right = diffX - pad_left
            pad_top = diffY // 2
            pad_bottom = diffY - pad_top
            x1 = nn.functional.pad(x1, [pad_left, pad_right, pad_top, pad_bottom])
        x2 = self.attn(g=x1, x=x2)
        return self.rrcnn(torch.cat([x2, x1], dim=1))


class MSRNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True, t=2):
        super().__init__()
        if n_ch <= 0 or n_cls <= 0 or hidden_size <= 0:
            raise ValueError("n_ch, n_cls, hidden_size must be positive")
        self.n_ch = n_ch
        self.n_cls = n_cls
        self.hidden_size = hidden_size
        self.bilinear = bilinear
        self.t = t
        h = hidden_size
        chs = {
            'enc1': max(1, h),
            'enc2': max(1, h * 2),
            'enc3': max(1, h * 4),
            'enc4': max(1, h * 8),
            'bottle': max(1, h * 16),
        }
        self.inc = RRCNN_block(n_ch, chs['enc1'], t=t)
        self.down1 = MSRNDown(chs['enc1'], chs['enc2'], t=t)
        self.down2 = MSRNDown(chs['enc2'], chs['enc3'], t=t)
        self.down3 = MSRNDown(chs['enc3'], chs['enc4'], t=t)
        self.down4 = MSRNDown(chs['enc4'], chs['bottle'], t=t)
        self.up1 = AttentionUp(chs['bottle'], chs['enc4'], chs['enc4'], bilinear, t=t)
        self.up2 = AttentionUp(chs['enc4'], chs['enc3'], chs['enc3'], bilinear, t=t)
        self.up3 = AttentionUp(chs['enc3'], chs['enc2'], chs['enc2'], bilinear, t=t)
        self.up4 = AttentionUp(chs['enc2'], chs['enc1'], chs['enc1'], bilinear, t=t)
        self.outc = OutConv(chs['enc1'], n_cls)

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
