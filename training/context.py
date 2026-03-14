from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PreviewContext:
    """Groups parameters for preview generation."""
    model: torch.nn.Module
    output_dir: str
    device: torch.device
    current_epoch: int
    global_step: int
    preview_save_count: int = 0
    preview_refresh_rate: int = 5
    use_mask_input: bool = False
    use_bce_dice: bool = False
    use_amp: bool = False
    use_auto_mask: bool = False
    auto_mask_gamma: float = 1.0
    diff_amplify: float = 5.0
