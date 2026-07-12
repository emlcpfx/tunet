"""Architectural residual helpers: out = src + delta.

When predict_residual is on, the network outputs an edit delta in normalized
space and the plate is added back. Untouched pixels fall out as identity
(delta → 0) instead of requiring the net to re-encode the full RGB image.
"""

import torch
import torch.nn as nn


def apply_residual(delta: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
    """Compose residual prediction: src_rgb + delta.

    model_input may be 3ch (RGB) or 4ch (RGB + mask); only the first 3
    channels are used as the plate.
    """
    return model_input[:, :3] + delta


def zero_init_output_layer(model: nn.Module) -> bool:
    """Zero the final OutConv so a residual model starts as identity.

    Returns True if an outc layer was found and zeroed.
    """
    outc = getattr(model, 'outc', None)
    if outc is None:
        return False
    # OutConv wraps a Conv2d as .c_block
    conv = getattr(outc, 'c_block', outc)
    if not isinstance(conv, nn.Conv2d):
        return False
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return True
