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

# Try to import OpenEXR as fallback
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

# --- Helper to convert nested dict to nested SimpleNamespace ---
# (Copied from train.py)
def dict_to_namespace(d):
    if isinstance(d,dict):
        safe_d={};
        for k,v in d.items(): safe_key=k.replace('-','_'); safe_d[safe_key]=dict_to_namespace(v)
        return SimpleNamespace(**safe_d)
    elif isinstance(d,list): return [dict_to_namespace(item) for item in d]
    else: return d

# --- UNet Model Definition ---
# !! Must be IDENTICAL to the training script !!
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels)
        self.d_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.d_block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m_block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.m_block(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            up_out_channels = in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2)
            conv_in_channels = up_out_channels + skip_channels
        conv_in_channels = max(1, conv_in_channels)
        out_channels = max(1, out_channels)
        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            pad_left = diffX // 2; pad_right = diffX - pad_left
            pad_top = diffY // 2; pad_bottom = diffY - pad_top
            x1 = nn.functional.pad(x1, [pad_left, pad_right, pad_top, pad_bottom])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c_block = nn.Conv2d(max(1, in_channels), max(1, out_channels), kernel_size=1)
    def forward(self, x): return self.c_block(x)

class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        if n_ch <= 0 or n_cls <= 0 or hidden_size <= 0:
            raise ValueError("n_ch, n_cls, hidden_size must be positive")
        self.n_ch, self.n_cls, self.hidden_size, self.bilinear = n_ch, n_cls, hidden_size, bilinear
        h = hidden_size
        chs = {'enc1': max(1, h), 'enc2': max(1, h*2), 'enc3': max(1, h*4), 'enc4': max(1, h*8), 'bottle': max(1, h*16)}
        self.inc = DoubleConv(n_ch, chs['enc1'])
        self.down1 = Down(chs['enc1'], chs['enc2'])
        self.down2 = Down(chs['enc2'], chs['enc3'])
        self.down3 = Down(chs['enc3'], chs['enc4'])
        self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear)
        self.up2 = Up(chs['enc4'], chs['enc3'], chs['enc3'], bilinear)
        self.up3 = Up(chs['enc3'], chs['enc2'], chs['enc2'], bilinear)
        self.up4 = Up(chs['enc2'], chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# --- Helper Functions ---
NORM_MEAN = [0.5, 0.5, 0.5]; NORM_STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def _load_exr_full_frame(image_path):
    """Load an EXR file as a float32 RGB numpy array using the displayWindow for full-frame size.

    EXR files have a displayWindow (full frame) and dataWindow (bounding box of actual data).
    OpenCV only reads the dataWindow, which gives wrong dimensions for compositing EXRs.
    This function always returns a full displayWindow-sized image with dataWindow placed correctly.
    """
    if HAS_OPENEXR:
        exr_file = OpenEXR.InputFile(image_path)
        header = exr_file.header()

        disp = header['displayWindow']
        disp_width = disp.max.x - disp.min.x + 1
        disp_height = disp.max.y - disp.min.y + 1

        dw = header['dataWindow']
        dw_width = dw.max.x - dw.min.x + 1
        dw_height = dw.max.y - dw.min.y + 1

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ['R', 'G', 'B']
        available = list(header['channels'].keys())
        if not all(c in available for c in channels):
            ch_name = available[0]
            channel_data = [exr_file.channel(ch_name, FLOAT)] * 3
        else:
            channel_data = [exr_file.channel(c, FLOAT) for c in channels]

        img_channels = []
        for data in channel_data:
            arr = np.frombuffer(data, dtype=np.float32).reshape((dw_height, dw_width))
            img_channels.append(arr)
        data_img = np.stack(img_channels, axis=2)

        if (dw.min.x == disp.min.x and dw.min.y == disp.min.y and
                dw_width == disp_width and dw_height == disp_height):
            return data_img

        full_img = np.zeros((disp_height, disp_width, 3), dtype=np.float32)
        x_off = dw.min.x - disp.min.x
        y_off = dw.min.y - disp.min.y
        full_img[y_off:y_off + dw_height, x_off:x_off + dw_width, :] = data_img
        return full_img

    img_cv = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img_cv is not None:
        if len(img_cv.shape) == 3 and img_cv.shape[2] >= 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)[:, :, :3].astype(np.float32)
        elif len(img_cv.shape) == 2:
            img_cv = np.stack([img_cv, img_cv, img_cv], axis=2).astype(np.float32)
        logging.warning(f"Loaded EXR with OpenCV (no displayWindow support): {image_path}")
        return img_cv

    raise ValueError(f"Failed to load EXR file: {image_path}. Install OpenEXR library: pip install OpenEXR")

def load_image_any_format(image_path):
    """Load an image in any format including EXR. Returns a PIL Image in RGB mode."""
    _, ext = os.path.splitext(image_path.lower())

    if ext == '.exr':
        img_float = _load_exr_full_frame(image_path)

        # Convert to 8-bit for PIL (no transforms - treat like PNG/JPG)
        # EXR values assumed to be in [0, 1] range like normalized PNG
        img_8bit = np.clip(img_float * 255, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        return Image.fromarray(img_8bit, mode='RGB')
    else:
        # Use PIL for standard formats
        return Image.open(image_path).convert('RGB')

def load_mask_image(image_path):
    """Load a mask image as single-channel (H,W) float32 numpy array in [0,1]."""
    _, ext = os.path.splitext(image_path.lower())
    if ext == '.exr':
        if HAS_OPENEXR:
            exr_file = OpenEXR.InputFile(image_path)
            header = exr_file.header()

            disp = header['displayWindow']
            disp_width = disp.max.x - disp.min.x + 1
            disp_height = disp.max.y - disp.min.y + 1

            dw = header['dataWindow']
            dw_width = dw.max.x - dw.min.x + 1
            dw_height = dw.max.y - dw.min.y + 1

            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            available = list(header['channels'].keys())
            ch_name = 'Y' if 'Y' in available else available[0]
            data = exr_file.channel(ch_name, FLOAT)
            mask_data = np.frombuffer(data, dtype=np.float32).reshape((dw_height, dw_width))

            if (dw.min.x == disp.min.x and dw.min.y == disp.min.y and
                    dw_width == disp_width and dw_height == disp_height):
                return np.clip(mask_data, 0.0, 1.0)

            mask = np.zeros((disp_height, disp_width), dtype=np.float32)
            x_off = dw.min.x - disp.min.x
            y_off = dw.min.y - disp.min.y
            mask[y_off:y_off + dw_height, x_off:x_off + dw_width] = mask_data
            return np.clip(mask, 0.0, 1.0)

        img_cv = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img_cv is None:
            raise ValueError(f"Failed to load mask EXR: {image_path}. Install OpenEXR library: pip install OpenEXR")
        logging.warning(f"Loaded mask EXR with OpenCV (no displayWindow support): {image_path}")
        if len(img_cv.shape) == 3:
            mask = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_cv.shape[2] >= 3 else img_cv[:, :, 0]
        else:
            mask = img_cv
        return np.clip(mask.astype(np.float32), 0.0, 1.0)
    else:
        img = Image.open(image_path).convert('L')
        return np.array(img, dtype=np.float32) / 255.0

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

    # Instantiate Model
    model = UNet(n_ch=n_input_ch, n_cls=3, hidden_size=effective_model_size, bilinear=bilinear_mode)

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
