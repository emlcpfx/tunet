from dataclasses import dataclass

import torch


@dataclass
class InferenceConfig:
    """Groups parameters for tiled inference."""
    resolution: int
    stride: int
    device: torch.device
    batch_size: int = 1
    use_amp: bool = False
    half_res: bool = False
    use_mask_input: bool = False
    loss_mode: str = 'l1'
