import torch
import torch.nn as nn


class NormalizedUNet(nn.Module):
    """Wrapper that applies input normalization and output denormalization.

    Accepts [0,1] input, normalizes to [-1,1] for the inner model,
    then denormalizes the output back to [0,1].
    """

    def __init__(self, unet_model, use_sigmoid=False):
        super().__init__()
        self.unet = unet_model
        self.use_sigmoid = use_sigmoid
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        normalized_x = (x - self.mean) / self.std
        unet_output = self.unet(normalized_x)
        if self.use_sigmoid:
            return torch.sigmoid(unet_output)
        else:
            denormalized_output = (unet_output * self.std) + self.mean
            return torch.clamp(denormalized_output, 0.0, 1.0)
