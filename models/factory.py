from .unet import UNet
from .msrnet import MSRNet


def create_model(model_type='unet', n_ch=3, n_cls=3, hidden_size=64, bilinear=True, t=2):
    if model_type == 'msrn':
        return MSRNet(n_ch=n_ch, n_cls=n_cls, hidden_size=hidden_size, bilinear=bilinear, t=t)
    elif model_type == 'unet':
        return UNet(n_ch=n_ch, n_cls=n_cls, hidden_size=hidden_size, bilinear=bilinear)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Must be 'unet' or 'msrn'.")
