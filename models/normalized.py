import torch
import torch.nn as nn


class NormalizedUNet(nn.Module):
    """Wrapper that applies input normalization and output denormalization.

    sRGB mode:   Accepts [0,1] input → [-1,1] → UNet → [0,1] output (clamped).
    Linear mode: Accepts [0,+inf) linear input → log1p → [-1,1] → UNet
                 → denorm to log-space → expm1 → linear output (no upper clamp).
    """

    def __init__(self, unet_model, use_sigmoid=False, color_space='srgb'):
        super().__init__()
        self.unet = unet_model
        self.use_sigmoid = use_sigmoid
        self.color_space = color_space
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.color_space == 'linear':
            x = torch.log1p(x.clamp(min=0.0))
        normalized_x = (x - self.mean) / self.std
        unet_output = self.unet(normalized_x)
        if self.use_sigmoid:
            return torch.sigmoid(unet_output)
        denormalized = (unet_output * self.std) + self.mean
        if self.color_space == 'linear':
            return torch.expm1(denormalized.clamp(min=0.0))
        return torch.clamp(denormalized, 0.0, 1.0)
