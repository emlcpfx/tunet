import torch
import torchvision.transforms as T


def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def diff_heatmap(a_denorm, b_denorm, amplify=5.0):
    """Compute amplified difference heatmap between two denormalized images.
    Returns (3, H, W) tensor with heat colormap: black -> red -> yellow -> white."""
    diff = (a_denorm - b_denorm).abs().mean(dim=0, keepdim=True)
    diff = (diff * amplify).clamp(0, 1)
    r = (diff * 3.0).clamp(0, 1)
    g = ((diff - 0.33) * 3.0).clamp(0, 1)
    b = ((diff - 0.66) * 3.0).clamp(0, 1)
    return torch.cat([r, g, b], dim=0)


def refine_auto_mask(raw_diff, noise_threshold=0.01, gamma=1.0):
    """Apply blur + steep sigmoid to raw |src-dst| diff tensor.
    Input: (B, 1, H, W) raw diff in [0, 1].
    Returns: (B, 1, H, W) mask in [0, 1] ready for weight_map formula.
    gamma < 1.0 expands white coverage, gamma > 1.0 contracts it."""
    res = raw_diff.shape[-1]
    kernel_size = max(31, int(31 * res / 256) | 1)
    diff = T.functional.gaussian_blur(raw_diff, kernel_size=kernel_size)
    max_vals = diff.amax(dim=(-2, -1), keepdim=True) + 1e-8
    significant = (max_vals > noise_threshold).float()
    diff = diff / max_vals
    diff = torch.sigmoid(20.0 * (diff - 0.15))
    min_vals = diff.amin(dim=(-2, -1), keepdim=True)
    max_vals2 = diff.amax(dim=(-2, -1), keepdim=True)
    diff = (diff - min_vals) / (max_vals2 - min_vals + 1e-8)
    if gamma != 1.0:
        diff = diff.pow(gamma)
    diff = diff * significant
    return diff


def compute_auto_mask(src, dst):
    """Compute auto-mask from |src - dst| tensors (for preview use only).
    Returns (B, 1, H, W) mask in [0, 1]."""
    diff = (src - dst).abs().mean(dim=1, keepdim=True)
    return refine_auto_mask(diff)
