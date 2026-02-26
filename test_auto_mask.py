"""
Test auto-mask approaches on real src/dst pairs.
Run: python test_auto_mask.py
Outputs comparison images to ./auto_mask_tests/
"""
import os, glob, random
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import make_grid

# --- Config ---
SRC_DIR = r"D:/Work/Dropbox/Documents_DB/Work_DB/Bouffant_db/tunet_models/input/Vanity_v001/Paint_INPUT"
DST_DIR = r"D:/Work/Dropbox/Documents_DB/Work_DB/Bouffant_db/tunet_models/input/Vanity_v001/Paint_OUTPUT"
OUT_DIR = "./auto_mask_tests"
NUM_PAIRS = 4
CROP_SIZE = 512

os.makedirs(OUT_DIR, exist_ok=True)


def load_image(path):
    """Load image (supports EXR, PNG, JPG, etc.) -> PIL RGB."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.exr', '.hdr'):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"cv2 failed to read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # EXR is float, clamp to [0,1] for display
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)
    else:
        return Image.open(path).convert("RGB")


def load_pair(src_path, dst_path, crop_size=512):
    """Load and center-crop an image pair to tensors in [-1, 1]."""
    src = load_image(src_path)
    dst = load_image(dst_path)
    src = TF.center_crop(src, crop_size)
    dst = TF.center_crop(dst, crop_size)
    src_t = TF.to_tensor(src) * 2.0 - 1.0  # [0,1] -> [-1,1]
    dst_t = TF.to_tensor(dst) * 2.0 - 1.0
    return src_t, dst_t


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def diff_heatmap(a, b, amplify=5.0):
    """Heat colormap of |a - b|: black -> red -> yellow -> white."""
    diff = (a - b).abs().mean(dim=0, keepdim=True)
    diff = (diff * amplify).clamp(0, 1)
    r = (diff * 3.0).clamp(0, 1)
    g = ((diff - 0.33) * 3.0).clamp(0, 1)
    b = ((diff - 0.66) * 3.0).clamp(0, 1)
    return torch.cat([r, g, b], dim=0)


# ============================================================
# MASK APPROACHES — edit/add here, best one goes into train.py
# ============================================================

def mask_v1_current(src, dst):
    """Current: blur + noise_floor(0.1) + gamma(0.5)"""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=31).squeeze(0)
    diff = diff / (diff.max() + 1e-8)
    diff = (diff - 0.1).clamp(0)
    diff = diff / (diff.max() + 1e-8)
    diff = diff.pow(0.5)
    return diff


def mask_v2_hard_threshold(src, dst):
    """Percentile threshold: keep top 15% of diff values, zero the rest."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=31).squeeze(0)
    threshold = torch.quantile(diff, 0.85)
    diff = (diff - threshold).clamp(0)
    diff = diff / (diff.max() + 1e-8)
    # Gentle expansion
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=21).squeeze(0)
    diff = diff / (diff.max() + 1e-8)
    return diff


def mask_v3_binary_dilate(src, dst):
    """Binary mask: threshold at top 10%, then dilate with large blur."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    # Find strong differences
    threshold = torch.quantile(diff, 0.90)
    binary = (diff > threshold).float()
    # Dilate by blurring the binary mask, then re-threshold
    dilated = TF.gaussian_blur(binary.unsqueeze(0), kernel_size=51).squeeze(0)
    dilated = (dilated > 0.05).float()
    # Soft edges
    soft = TF.gaussian_blur(dilated.unsqueeze(0), kernel_size=21).squeeze(0)
    soft = soft / (soft.max() + 1e-8)
    return soft


def mask_v4_steep_sigmoid(src, dst):
    """Steep sigmoid curve: sharp transition from 0 to 1."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=31).squeeze(0)
    diff = diff / (diff.max() + 1e-8)
    # Steep sigmoid centered at 0.15 — kills noise, snaps signal to ~1
    steepness = 20.0
    center = 0.15
    diff = torch.sigmoid(steepness * (diff - center))
    # Renormalize
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    return diff


def mask_v5_topk_dilate(src, dst):
    """Find hotspot peaks, dilate generously."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    # Small blur first to merge nearby pixels
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=11).squeeze(0)
    # Keep only top 5% as seeds
    threshold = torch.quantile(diff, 0.95)
    seeds = (diff > threshold).float()
    # Large dilation blur to expand seeds into regions
    expanded = TF.gaussian_blur(seeds.unsqueeze(0), kernel_size=71).squeeze(0)
    expanded = expanded / (expanded.max() + 1e-8)
    # Clamp low values to clean up
    expanded = (expanded - 0.02).clamp(0)
    expanded = expanded / (expanded.max() + 1e-8)
    return expanded


def mask_v6_adaptive_floor(src, dst):
    """Adaptive floor: use median as noise estimate, subtract it."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=31).squeeze(0)
    # Use median as adaptive noise floor (robust to outliers)
    noise_floor = diff.median() * 2.0
    diff = (diff - noise_floor).clamp(0)
    diff = diff / (diff.max() + 1e-8)
    # Moderate gamma
    diff = diff.pow(0.6)
    return diff


def mask_v7_sigmoid_soft_edge(src, dst):
    """Sigmoid on raw diff (no pre-blur dilation), then soft edges only."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    diff = diff / (diff.max() + 1e-8)
    # Steep sigmoid: snaps to 0/1 without spreading
    steepness = 25.0
    center = 0.12
    diff = torch.sigmoid(steepness * (diff - center))
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    # Small blur just to soften jagged edges, not to dilate
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=9).squeeze(0)
    diff = diff / (diff.max() + 1e-8)
    return diff


def mask_v8_sigmoid_soft_edge_med(src, dst):
    """Like v7 but slightly more spread with kernel=15."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    diff = diff / (diff.max() + 1e-8)
    steepness = 20.0
    center = 0.10
    diff = torch.sigmoid(steepness * (diff - center))
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=15).squeeze(0)
    diff = diff / (diff.max() + 1e-8)
    return diff


def mask_v9_sigmoid_soft_edge_wide(src, dst):
    """Like v7 but lower center threshold to catch more subtle diffs."""
    diff = (src - dst).abs().mean(dim=0, keepdim=True)
    diff = diff / (diff.max() + 1e-8)
    steepness = 30.0
    center = 0.08
    diff = torch.sigmoid(steepness * (diff - center))
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    diff = TF.gaussian_blur(diff.unsqueeze(0), kernel_size=9).squeeze(0)
    diff = diff / (diff.max() + 1e-8)
    return diff


APPROACHES = [
    ("v4_steep_sigmoid", mask_v4_steep_sigmoid),
    ("v7_sig_soft_k9", mask_v7_sigmoid_soft_edge),
    ("v8_sig_soft_k15", mask_v8_sigmoid_soft_edge_med),
    ("v9_sig_wide_k9", mask_v9_sigmoid_soft_edge_wide),
]


def main():
    # Find matching pairs
    src_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.*")))
    dst_files = sorted(glob.glob(os.path.join(DST_DIR, "*.*")))

    if not src_files or not dst_files:
        print(f"No images found!\n  SRC_DIR: {SRC_DIR}\n  DST_DIR: {DST_DIR}")
        return

    # Match by filename
    dst_map = {os.path.basename(f): f for f in dst_files}
    pairs = [(s, dst_map[os.path.basename(s)]) for s in src_files if os.path.basename(s) in dst_map]

    if not pairs:
        print("No matching pairs found between src and dst!")
        return

    print(f"Found {len(pairs)} pairs. Sampling {NUM_PAIRS}...")
    selected = random.sample(pairs, min(NUM_PAIRS, len(pairs)))

    for pair_idx, (src_path, dst_path) in enumerate(selected):
        fname = os.path.splitext(os.path.basename(src_path))[0]
        print(f"\nPair {pair_idx + 1}: {fname}")
        src_t, dst_t = load_pair(src_path, dst_path, CROP_SIZE)

        # Build comparison grid: each row = one approach
        # Columns: src | dst | diff_heatmap | mask
        rows = []
        for name, fn in APPROACHES:
            mask = fn(src_t, dst_t)  # (1, H, W)
            mask_3ch = mask.repeat(3, 1, 1)  # grayscale -> RGB
            heatmap = diff_heatmap(denorm(src_t), denorm(dst_t))
            rows.extend([denorm(src_t), denorm(dst_t), heatmap, mask_3ch])
            print(f"  {name}: mask range [{mask.min():.3f}, {mask.max():.3f}], "
                  f"nonzero: {(mask > 0.01).float().mean() * 100:.1f}%")

        grid = make_grid(torch.stack(rows), nrow=4, padding=3, normalize=False,
                         pad_value=0.3)
        img = TF.to_pil_image(grid)

        # Add labels
        out_path = os.path.join(OUT_DIR, f"compare_{fname}.jpg")
        img.save(out_path, quality=95)
        print(f"  Saved: {out_path}")

    # Also save a legend
    print(f"\nRows per image (top to bottom):")
    for i, (name, _) in enumerate(APPROACHES):
        print(f"  Row {i + 1}: {name}")
    print(f"\nColumns: src | dst | diff_heatmap | mask")
    print(f"\nAll outputs in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
