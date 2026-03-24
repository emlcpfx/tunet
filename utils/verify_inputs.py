"""Pre-flight dataset verification.

Scans source/destination directories and reports issues (missing files,
dimension mismatches, corrupt images, undersized images) without building
the full training dataset.
"""

import os
import glob
import logging
from collections import defaultdict

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def verify_dataset(src_dir, dst_dir, mask_dir=None, resolution=512, color_space='srgb'):
    """Verify a training dataset and return a structured report.

    Returns dict with keys:
        total_src: int - total source files found
        valid_pairs: int - pairs that pass all checks
        issues: list of (filepath, reason) tuples
        summary: dict mapping reason -> count
    """
    from PIL import Image, UnidentifiedImageError
    from image_io import load_image_linear

    issues = []
    valid_pairs = 0
    is_linear = color_space == 'linear'

    src_files = sorted(glob.glob(os.path.join(os.path.abspath(src_dir), '*.*')))
    # Filter to known image extensions
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif', '.exr'}
    src_files = [f for f in src_files if os.path.splitext(f)[1].lower() in image_exts]

    if not src_files:
        return {
            'total_src': 0, 'valid_pairs': 0,
            'issues': [('(directory)', f'No image files found in {src_dir}')],
            'summary': {'No images found': 1},
        }

    dst_dir_abs = os.path.abspath(dst_dir)
    mask_dir_abs = os.path.abspath(mask_dir) if mask_dir else None

    for src_path in src_files:
        base_name = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir_abs, base_name)

        # Check destination exists
        if not os.path.exists(dst_path):
            issues.append((base_name, 'Destination missing'))
            continue

        # Check mask if required
        if mask_dir_abs:
            stem = os.path.splitext(base_name)[0]
            mask_exts = ['.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff']
            mask_found = False
            for mext in mask_exts:
                if os.path.exists(os.path.join(mask_dir_abs, stem + mext)):
                    mask_found = True
                    break
            if not mask_found:
                issues.append((base_name, 'Mask missing'))
                continue

        # Try opening both files and checking dimensions
        try:
            if is_linear:
                img_s = load_image_linear(src_path)
                img_d = load_image_linear(dst_path)
                h_s, w_s = img_s.shape[:2]
                h_d, w_d = img_d.shape[:2]
            else:
                img_s = Image.open(src_path)
                img_s.load()  # force decode
                w_s, h_s = img_s.size
                img_s.close()
                img_d = Image.open(dst_path)
                img_d.load()
                w_d, h_d = img_d.size
                img_d.close()
        except UnidentifiedImageError:
            issues.append((base_name, 'Corrupt or unreadable'))
            continue
        except Exception as e:
            issues.append((base_name, f'Load error: {type(e).__name__}'))
            continue

        # Dimension mismatch
        if (w_s, h_s) != (w_d, h_d):
            issues.append((base_name, f'Dimension mismatch: src={w_s}x{h_s} dst={w_d}x{h_d}'))
            continue

        # Too small for resolution
        if w_s < resolution or h_s < resolution:
            issues.append((base_name, f'Too small ({w_s}x{h_s} < {resolution}px)'))
            continue

        valid_pairs += 1

    # Build summary
    summary = defaultdict(int)
    for _, reason in issues:
        # Normalize dimension/size reasons for grouping
        if reason.startswith('Dimension mismatch'):
            summary['Dimension mismatch'] += 1
        elif reason.startswith('Too small'):
            summary['Too small for resolution'] += 1
        elif reason.startswith('Load error'):
            summary['Load error'] += 1
        else:
            summary[reason] += 1

    return {
        'total_src': len(src_files),
        'valid_pairs': valid_pairs,
        'issues': issues,
        'summary': dict(summary),
    }
