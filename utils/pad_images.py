"""
Pad small images to meet minimum training resolution.
Useful for images that are slightly smaller than training resolution.
"""

import os
import argparse
from PIL import Image
from pathlib import Path


def pad_image_to_size(img, target_size, pad_mode='black'):
    """
    Pad an image to target size.

    Args:
        img: PIL Image
        target_size: (width, height) or single int for square
        pad_mode: 'black', 'white', 'mirror', or 'edge'
    """
    if isinstance(target_size, int):
        target_w = target_h = target_size
    else:
        target_w, target_h = target_size

    orig_w, orig_h = img.size

    # Already big enough
    if orig_w >= target_w and orig_h >= target_h:
        return img

    # Calculate padding needed
    pad_w = max(0, target_w - orig_w)
    pad_h = max(0, target_h - orig_h)

    # Center the image
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    if pad_mode == 'black':
        # Pad with black (0, 0, 0)
        new_img = Image.new(img.mode, (orig_w + pad_w, orig_h + pad_h), (0, 0, 0))
        new_img.paste(img, (left, top))
    elif pad_mode == 'white':
        # Pad with white (255, 255, 255)
        new_img = Image.new(img.mode, (orig_w + pad_w, orig_h + pad_h), (255, 255, 255))
        new_img.paste(img, (left, top))
    elif pad_mode == 'mirror':
        # Mirror/reflect the edges
        import numpy as np
        img_array = np.array(img)
        padded = np.pad(img_array,
                       ((top, bottom), (left, right), (0, 0)),
                       mode='reflect')
        new_img = Image.fromarray(padded)
    elif pad_mode == 'edge':
        # Extend edge pixels
        import numpy as np
        img_array = np.array(img)
        padded = np.pad(img_array,
                       ((top, bottom), (left, right), (0, 0)),
                       mode='edge')
        new_img = Image.fromarray(padded)
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    return new_img


def process_directory(input_dir, output_dir, target_size, pad_mode='black', in_place=False):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if in_place:
        output_path = input_path
        print("WARNING: Processing in-place. Original files will be overwritten!")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.exr']

    processed = 0
    skipped = 0

    for img_file in input_path.iterdir():
        if img_file.suffix.lower() not in extensions:
            continue

        try:
            img = Image.open(img_file).convert('RGB')
            orig_size = img.size

            # Check if padding needed
            if isinstance(target_size, int):
                needs_padding = orig_size[0] < target_size or orig_size[1] < target_size
            else:
                needs_padding = orig_size[0] < target_size[0] or orig_size[1] < target_size[1]

            if needs_padding:
                padded_img = pad_image_to_size(img, target_size, pad_mode)
                output_file = output_path / img_file.name
                padded_img.save(output_file)
                print(f"✓ Padded: {img_file.name} {orig_size} → {padded_img.size}")
                processed += 1
            else:
                if not in_place:
                    # Copy unchanged
                    output_file = output_path / img_file.name
                    img.save(output_file)
                skipped += 1

        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")

    print(f"\nDone! Processed: {processed}, Skipped (already big enough): {skipped}")


def main():
    parser = argparse.ArgumentParser(description='Pad small images to minimum size')
    parser.add_argument('input_dir', help='Directory containing images to pad')
    parser.add_argument('--output_dir', help='Output directory (default: input_dir + "_padded")')
    parser.add_argument('--size', type=int, default=512, help='Target size (default: 512)')
    parser.add_argument('--mode', choices=['black', 'white', 'mirror', 'edge'],
                       default='black', help='Padding mode (default: black)')
    parser.add_argument('--in_place', action='store_true',
                       help='Modify files in place (WARNING: overwrites originals!)')

    args = parser.parse_args()

    if args.output_dir is None and not args.in_place:
        args.output_dir = args.input_dir + "_padded"

    print(f"Padding images to {args.size}×{args.size} with '{args.mode}' padding")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir if args.output_dir else '(in-place)'}")
    print()

    process_directory(args.input_dir, args.output_dir or args.input_dir,
                     args.size, args.mode, args.in_place)


if __name__ == '__main__':
    main()
