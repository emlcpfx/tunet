import os
import argparse
import math
import re
from glob import glob
from PIL import Image
import logging
import time
from types import SimpleNamespace
import numpy as np

# Enable OpenEXR support in OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torch.amp import autocast

# --- Imports from extracted modules ---
from models import create_model
from config import dict_to_namespace
from image_io import load_image_any_format, load_mask_image, denormalize, NORM_MEAN, NORM_STD

# --- Updated load_model_and_config ---
def load_model_and_config(checkpoint_path, device):
    """Loads TuNet, automatically detecting config from checkpoint.""" # Changed description
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine Config Source
    if 'config' in checkpoint:
        logging.info("Detected 'config' key (YAML-based training).")
        config_source = checkpoint['config']
        if isinstance(config_source, dict): config_source = dict_to_namespace(config_source)
        is_new_format = True
    elif 'args' in checkpoint:
        logging.info("Detected 'args' key (argparse-based training).")
        config_source = checkpoint['args']
        if isinstance(config_source, argparse.Namespace): config_source = SimpleNamespace(**vars(config_source))
        is_new_format = False
    else:
        raise ValueError("Checkpoint missing configuration ('config' or 'args').")

    # Extract Parameters Safely
    default_hidden_size = 64; default_loss = 'l1'; default_resolution = 512; default_bilinear = True

    # <-- CHANGED: Extract model size with fallback -->
    model_size_saved = default_hidden_size # Start with default
    if is_new_format:
        # Try new key first, then old key within model section
        model_config = getattr(config_source, 'model', SimpleNamespace())
        model_size_saved = getattr(model_config, 'model_size_dims', getattr(model_config, 'unet_hidden_size', default_hidden_size))
        loss_mode = getattr(getattr(config_source, 'training', SimpleNamespace()), 'loss', default_loss)
        resolution = getattr(getattr(config_source, 'data', SimpleNamespace()), 'resolution', default_resolution)
        bilinear_mode = default_bilinear # Use class default
    else: # Old format (args)
        # Try new key first (in case args somehow had it), then old key
        model_size_saved = getattr(config_source, 'model_size_dims', getattr(config_source, 'unet_hidden_size', default_hidden_size))
        loss_mode = getattr(config_source, 'loss', default_loss)
        resolution = getattr(config_source, 'resolution', default_resolution)
        bilinear_mode = getattr(config_source, 'bilinear', default_bilinear)

    logging.info(f"Checkpoint parameters: Saved Model Size={model_size_saved}, Loss Mode='{loss_mode}', Resolution={resolution}, Bilinear={bilinear_mode}")

    # Calculate Effective Hidden Size (using extracted model_size_saved)
    hq_default_bump_size = 96 # Match training script
    effective_model_size = model_size_saved # Use the correctly extracted size
    default_size_for_bump = 64 # Match training script condition
    if loss_mode == 'l1+lpips' and model_size_saved == default_size_for_bump:
        effective_model_size = hq_default_bump_size
        logging.info(f"Applying model size bump logic: Effective Size = {effective_model_size}")
    else:
        logging.info(f"Effective Model Size = {effective_model_size}")

    # Extract mask config
    use_mask_input = False
    n_input_ch = checkpoint.get('n_input_channels', 3)
    if n_input_ch == 4:
        use_mask_input = True
    elif is_new_format:
        mask_config = getattr(config_source, 'mask', SimpleNamespace())
        use_mask_input = getattr(mask_config, 'use_mask_input', False)
        if use_mask_input: n_input_ch = 4
    if use_mask_input:
        logging.info("Model uses mask input channel (4-channel input).")

    # Instantiate Model — detect model_type from checkpoint metadata
    model_type_saved = checkpoint.get('model_type', 'unet')
    if model_type_saved == 'unet' and is_new_format:
        model_type_saved = getattr(getattr(config_source, 'model', SimpleNamespace()), 'model_type', 'unet')
    recurrence_t = 2
    if is_new_format:
        recurrence_t = getattr(getattr(config_source, 'model', SimpleNamespace()), 'recurrence_steps', 2)
    model = create_model(model_type=model_type_saved, n_ch=n_input_ch, n_cls=3, hidden_size=effective_model_size, bilinear=bilinear_mode, t=recurrence_t)
    logging.info(f"Model type: {model_type_saved}, hidden_size={effective_model_size}")

    # Load State Dict
    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully and set to evaluation mode.")

    return model, resolution, use_mask_input, loss_mode

# --- create_blend_mask ---
# ... (remains the same) ...
def create_blend_mask(resolution, device):
    hann_1d = torch.hann_window(resolution, periodic=False, device=device)
    hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
    return hann_2d.view(1, 1, resolution, resolution)

# --- Main Inference Function ---
# ... (process_image remains the same) ...
def process_image(model, image_path, output_path, resolution, stride, device, batch_size, transform, denormalize_fn, use_amp, half_res=False, mask_path=None, use_mask_input=False, loss_mode='l1', src_image=None):
    logging.info(f"Processing: {os.path.basename(image_path)}")
    start_time = time.time()
    try:
        img = src_image if src_image is not None else load_image_any_format(image_path)
        orig_width, orig_height = img.size

        # Downscale to half resolution if requested
        if half_res:
            img_width = orig_width // 2
            img_height = orig_height // 2
            img = img.resize((img_width, img_height), Image.LANCZOS)
            logging.info(f"Half-res mode: {orig_width}x{orig_height} -> {img_width}x{img_height}")
        else:
            img_width, img_height = orig_width, orig_height

        if img_width < resolution or img_height < resolution:
            logging.warning(f"Image smaller than resolution {resolution}. Skipping.")
            return
    except Exception as e:
        logging.error(f"Failed to load: {image_path}: {e}")
        return
    # Load mask if needed
    mask_full = None
    if use_mask_input and mask_path:
        try:
            mask_full = load_mask_image(mask_path)
            if half_res:
                mask_full = cv2.resize(mask_full, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logging.error(f"Failed to load mask: {mask_path}: {e}")
            return
    slice_coords = []
    possible_y = list(range(0, img_height - resolution, stride)) + ([img_height - resolution] if img_height > resolution else [0])
    possible_x = list(range(0, img_width - resolution, stride)) + ([img_width - resolution] if img_width > resolution else [0])
    unique_y = sorted(list(set(possible_y)))
    unique_x = sorted(list(set(possible_x)))
    for y in unique_y:
        for x in unique_x:
            slice_coords.append((x, y, x + resolution, y + resolution))
    num_slices = len(slice_coords)
    if num_slices == 0:
        logging.warning(f"No slices generated for {os.path.basename(image_path)}.")
        return
    logging.info(f"Generated {num_slices} slices.")
    output_canvas_cpu = torch.zeros(1, 3, img_height, img_width, dtype=torch.float32)
    weight_map_cpu = torch.zeros(1, 1, img_height, img_width, dtype=torch.float32)
    blend_mask = create_blend_mask(resolution, device)
    blend_mask_cpu = blend_mask.cpu()[0]
    processed_count = 0
    device_type = device.type
    with torch.no_grad():
        for i in range(0, num_slices, batch_size):
            batch_coords = slice_coords[i:min(i + batch_size, num_slices)]
            batch_src_list = [img.crop(coords) for coords in batch_coords]
            batch_src_tensor_list = [transform(src_pil) for src_pil in batch_src_list]
            batch_src_tensor = torch.stack(batch_src_tensor_list).to(device, non_blocking=True)
            # Concatenate mask as 4th channel if needed
            if use_mask_input and mask_full is not None:
                batch_mask_list = []
                for coords in batch_coords:
                    x, y, _, _ = coords
                    mask_slice = mask_full[y:y + resolution, x:x + resolution]
                    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_slice)).unsqueeze(0).float()
                    batch_mask_list.append(mask_tensor)
                batch_mask_tensor = torch.stack(batch_mask_list).to(device, non_blocking=True)
                batch_src_tensor = torch.cat([batch_src_tensor, batch_mask_tensor], dim=1)
            with autocast(device_type=device_type, enabled=use_amp):
                batch_output_tensor = model(batch_src_tensor)
                if loss_mode == 'bce+dice':
                    batch_output_tensor = torch.sigmoid(batch_output_tensor)
            batch_output_tensor_cpu = batch_output_tensor.cpu().float()
            for j, coords in enumerate(batch_coords):
                x, y, _, _ = coords
                if loss_mode == 'bce+dice':
                    output_slice_cpu = torch.clamp(batch_output_tensor_cpu[j], 0, 1)
                else:
                    output_slice_cpu = denormalize_fn(batch_output_tensor_cpu[j].unsqueeze(0))[0]
                output_canvas_cpu[0, :, y:y + resolution, x:x + resolution] += output_slice_cpu * blend_mask_cpu
                weight_map_cpu[0, :, y:y + resolution, x:x + resolution] += blend_mask_cpu
            processed_count += len(batch_coords)
            if processed_count % (batch_size * 10) == 0 or processed_count == num_slices:
                logging.info(f"  Processed {processed_count}/{num_slices} slices...")
    weight_map_cpu = torch.clamp(weight_map_cpu, min=1e-8)
    final_image_tensor = output_canvas_cpu / weight_map_cpu
    try:
        final_pil_image = to_pil_image(torch.clamp(final_image_tensor[0], 0, 1))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_pil_image.save(output_path)
        end_time = time.time()
        logging.info(f"Saved result: {output_path} ({end_time - start_time:.2f} sec)")
    except Exception as e:
        logging.error(f"Failed to save {output_path}: {e}")


def process_image_batch(model, frame_imgs, write_paths, resolution, stride, device,
                        transform, denormalize_fn, use_amp, half_res=False, loss_mode='l1'):
    """Process multiple frames in a single GPU forward pass.

    Most effective when each frame produces few tiles (e.g. 1 tile for 512×512 frames).
    All tiles from all frames are stacked into one batch and run through the model once.

    Args:
        frame_imgs:  list of PIL Images (pre-loaded)
        write_paths: list of output file paths, one per frame
    """
    frame_records = []   # per-frame metadata + accumulation canvas
    all_tile_tensors = []
    tile_frame_idx = []  # flat tile index -> frame index
    tile_slice_idx = []  # flat tile index -> slice index within frame

    for fi, img in enumerate(frame_imgs):
        orig_w, orig_h = img.size
        if half_res:
            img_w, img_h = orig_w // 2, orig_h // 2
            img = img.resize((img_w, img_h), Image.LANCZOS)
        else:
            img_w, img_h = orig_w, orig_h

        if img_w < resolution or img_h < resolution:
            logging.warning(f"Frame {fi} ({img_w}×{img_h}) smaller than resolution {resolution}, skipping.")
            frame_records.append(None)
            continue

        possible_y = list(range(0, img_h - resolution, stride)) + ([img_h - resolution] if img_h > resolution else [0])
        possible_x = list(range(0, img_w - resolution, stride)) + ([img_w - resolution] if img_w > resolution else [0])
        slice_coords = [(x, y, x + resolution, y + resolution)
                        for y in sorted(set(possible_y)) for x in sorted(set(possible_x))]

        canvas = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32)
        weight_map = torch.zeros(1, 1, img_h, img_w, dtype=torch.float32)
        frame_records.append({
            'img_w': img_w, 'img_h': img_h,
            'slice_coords': slice_coords,
            'canvas': canvas, 'weight_map': weight_map,
        })

        for si, coords in enumerate(slice_coords):
            all_tile_tensors.append(transform(img.crop(coords)))
            tile_frame_idx.append(fi)
            tile_slice_idx.append(si)

    if not all_tile_tensors:
        return

    # One GPU call for all tiles across all frames
    blend_mask_cpu = create_blend_mask(resolution, device).cpu()[0]
    batch = torch.stack(all_tile_tensors).to(device, non_blocking=True)
    device_type = device.type

    with torch.no_grad():
        with autocast(device_type=device_type, enabled=use_amp):
            outputs = model(batch)
            if loss_mode == 'bce+dice':
                outputs = torch.sigmoid(outputs)

    outputs_cpu = outputs.cpu().float()

    # Scatter output tiles back to per-frame canvases
    for ti in range(len(all_tile_tensors)):
        fi = tile_frame_idx[ti]
        rec = frame_records[fi]
        if rec is None:
            continue
        x, y, _, _ = rec['slice_coords'][tile_slice_idx[ti]]
        if loss_mode == 'bce+dice':
            out_slice = torch.clamp(outputs_cpu[ti], 0, 1)
        else:
            out_slice = denormalize_fn(outputs_cpu[ti].unsqueeze(0))[0]
        rec['canvas'][0, :, y:y + resolution, x:x + resolution] += out_slice * blend_mask_cpu
        rec['weight_map'][0, :, y:y + resolution, x:x + resolution] += blend_mask_cpu

    # Save each frame
    for fi, (rec, write_path) in enumerate(zip(frame_records, write_paths)):
        if rec is None:
            continue
        try:
            weight_map = torch.clamp(rec['weight_map'], min=1e-8)
            final = rec['canvas'] / weight_map
            pil = to_pil_image(torch.clamp(final[0], 0, 1))
            out_dir = os.path.dirname(write_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            pil.save(write_path)
        except Exception as e:
            logging.error(f"Failed to save frame {fi} to {write_path}: {e}")


# --- Main Execution ---
# ... (remains the same) ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference using trained TuNet with tiled processing.') # Updated description
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images')
    parser.add_argument('--overlap_factor', type=float, default=0.5, help='Overlap factor for slices (0.0 to <1.0). Default: 0.5')
    parser.add_argument('--batch_size', type=int, default=1, help='Slice batch size (adjust based on GPU memory)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for inference')
    parser.add_argument('--use_amp', action='store_true', help='Enable AMP for inference')
    parser.add_argument('--half_res', action='store_true', help='Process at half resolution for ~4x speedup (upscale in comp)')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory containing mask images (required if model uses mask input)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()],
        force=True )

    if not os.path.exists(args.checkpoint): logging.error(f"Checkpoint not found: {args.checkpoint}"); exit(1)
    if not os.path.isdir(args.input_dir): logging.error(f"Input dir not found: {args.input_dir}"); exit(1)
    if not (0.0 <= args.overlap_factor < 1.0): logging.error("overlap_factor must be [0.0, 1.0)"); exit(1)
    if args.batch_size <= 0: logging.error("batch_size must be positive."); exit(1)

    selected_device = args.device
    if selected_device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        selected_device = 'cpu'
    device = torch.device(selected_device)
    logging.info(f"Using device: {device}")
    if args.use_amp and selected_device == 'cpu':
        logging.warning("AMP disabled on CPU.")
        args.use_amp = False

    try:
        model, resolution, use_mask_input, loss_mode = load_model_and_config(args.checkpoint, device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        exit(1)

    if use_mask_input and not args.mask_dir:
        logging.error("This model requires mask input (4-channel) but --mask_dir was not provided.")
        exit(1)
    if args.mask_dir and not os.path.isdir(args.mask_dir):
        logging.error(f"Mask dir not found: {args.mask_dir}")
        exit(1)

    transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD) ])
    overlap_pixels = int(resolution * args.overlap_factor)
    stride = max(1, resolution - overlap_pixels)
    logging.info(f"Inference Parameters: Resolution={resolution}, Overlap Factor={args.overlap_factor}, Stride={stride}")

    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp', '*.exr']
    input_files = sorted([f for ext in img_extensions for f in glob(os.path.join(args.input_dir, ext))])
    if not input_files:
        logging.error(f"No supported image files found in {args.input_dir}")
        exit(1)
    logging.info(f"Found {len(input_files)} images to process.")
    os.makedirs(args.output_dir, exist_ok=True)

    total_start_time = time.time()
    for i, img_path in enumerate(input_files):
        logging.info(f"--- Processing image {i+1}/{len(input_files)} ---")
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # Extract frame number from end of filename (e.g., "shot_name_0001" -> "shot_name", "0001")
        match = re.match(r'^(.+?)_?(\d+)$', basename)
        if match:
            name_part = match.group(1)
            frame_num = match.group(2)
            output_filename = f"{name_part}_tunet_{frame_num}.png"
        else:
            # Fallback if no frame number found
            output_filename = f"{basename}_tunet.png"
        output_path = os.path.join(args.output_dir, output_filename)

        # Find matching mask file if mask_dir provided
        mask_path = None
        if args.mask_dir and use_mask_input:
            mask_exts = ['.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff']
            for mext in mask_exts:
                candidate = os.path.join(args.mask_dir, basename + mext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                logging.warning(f"No mask found for {basename}, using zeros (no mask focus)")

        process_image(model, img_path, output_path, resolution, stride, device,
                      args.batch_size, transform, denormalize, args.use_amp, args.half_res,
                      mask_path=mask_path, use_mask_input=use_mask_input, loss_mode=loss_mode)

    total_end_time = time.time()
    logging.info(f"--- Inference finished ---")
    logging.info(f"Processed {len(input_files)} images.")
    logging.info(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    logging.info(f"Average time per image: {(total_end_time - total_start_time)/len(input_files) if input_files else 0:.2f} seconds")
