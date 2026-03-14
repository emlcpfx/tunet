# Cross-Platform support

import os
import warnings
import argparse
import math
import glob
from PIL import Image, UnidentifiedImageError
import logging
import time
import random
import yaml
import copy
from types import SimpleNamespace
import itertools # For infinite dataloader cycle
import signal # For graceful shutdown signal handling
import re # For checkpoint pruning regex
import numpy as np
import platform # OS detection

# Enable OpenEXR support in OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import lpips
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Extracted module imports ---
from distributed import setup_ddp, cleanup_ddp, get_rank, get_world_size, is_main_process, CURRENT_OS
from models import create_model, UNet, MSRNet
from image_io import load_image_any_format, load_mask_image, load_exr_full_frame, NORM_MEAN, NORM_STD, denormalize
from config import config_to_dict, dict_to_sns, merge_configs
from training.loss import dice_loss, diff_heatmap, refine_auto_mask, compute_auto_mask
from training.context import PreviewContext
from training.previews import save_previews, capture_preview_batch, save_val_previews, capture_val_preview_batch
from training.validation import run_validation
from training.checkpoint import prune_checkpoints
from training.dataloader_utils import cycle, collate_skip_none, auto_detect_num_workers

# --- Data Loading and Augmentation ---

# --- Augmentation Creation Function ---
def create_augmentations(augmentation_list, has_mask=False, use_auto_mask=False):
    transforms = []
    uses_albumentations = False
    if not isinstance(augmentation_list, list):
        # Keep warning minimal
        # logging.warning(f"Augmentation list is not a list, using defaults.")
        return None

    for i, aug_config in enumerate(augmentation_list):
        if isinstance(aug_config, SimpleNamespace):
            if not hasattr(aug_config, '_target_'): logging.error(f"Aug {i}: Skip SimpleNamespace missing _target_"); continue
            target_str = aug_config._target_; params = {k: v for k, v in vars(aug_config).items() if k != '_target_'}
        elif isinstance(aug_config, dict):
            if '_target_' not in aug_config: logging.error(f"Aug {i}: Skip dict missing _target_"); continue
            target_str = aug_config['_target_']; params = {k: v for k, v in aug_config.items() if k != '_target_'}
        else: logging.error(f"Aug {i}: Skip invalid type {type(aug_config)}"); continue

        try:
            target_cls, is_torchvision, is_albumentation = None, False, False
            if target_str.startswith('torchvision.transforms.'):
                parts = target_str.split('.');
                if len(parts) == 3: cls_name = parts[-1]; target_cls = getattr(T, cls_name, None); is_torchvision = True
                else: logging.error(f"Aug {i}: Skip bad torchvision format {target_str}"); continue
                # Use DEBUG for mixing warning
                if is_torchvision and uses_albumentations: logging.debug(f"Aug {i}: Mixing Torchvision {cls_name} with Albumentations.")
            elif target_str.startswith('albumentations.'):
                parts = target_str.split('.')
                if len(parts) == 3 and parts[1] == 'pytorch': mod, cls_name = A.pytorch, parts[-1]
                elif len(parts) == 2: mod, cls_name = A, parts[-1]
                else: logging.error(f"Aug {i}: Skip bad albumentations format {target_str}"); continue
                target_cls = getattr(mod, cls_name, None); is_albumentation = True; uses_albumentations = True
            else: logging.error(f"Aug {i}: Skip unsupported lib {target_str}"); continue
            if target_cls is None: logging.error(f"Aug {i}: Class not found {target_str}"); continue

            try:
                transform_instance = target_cls(**params)
                transforms.append(transform_instance)
                # Use DEBUG for successful addition
                logging.debug(f"Added augmentation {i}: {target_str}")
            except Exception as e: logging.error(f"Aug {i}: Error instantiating {target_str}: {e}") # Keep error
        except Exception as e: logging.error(f"Aug {i}: Fail process {target_str}: {e}", exc_info=True) # Keep error for outer try

    if not transforms: return None
    # Use DEBUG for pipeline type confirmation
    if uses_albumentations:
        logging.debug(f"Creating Albumentations pipeline ({len(transforms)} steps).")
        additional_targets = {}
        if has_mask: additional_targets['loss_mask'] = 'mask'
        if use_auto_mask: additional_targets['auto_mask'] = 'mask'
        if additional_targets: return A.Compose(transforms, additional_targets=additional_targets)
        return A.Compose(transforms)
    else: logging.debug(f"Creating Torchvision pipeline ({len(transforms)} steps)."); return T.Compose(transforms)

# --- Augmented Dataset Class ---
class AugmentedImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0,
                 src_transforms=None, dst_transforms=None, shared_transforms=None,
                 final_transform=None, mask_dir=None, use_auto_mask=False, skip_empty_patches=False, skip_empty_threshold=1.0):
        self.src_dir = os.path.abspath(src_dir)
        self.dst_dir = os.path.abspath(dst_dir)
        if not os.path.isdir(self.src_dir): raise FileNotFoundError(f"Src dir not found: {self.src_dir}")
        if not os.path.isdir(self.dst_dir): raise FileNotFoundError(f"Dst dir not found: {self.dst_dir}")
        self.mask_dir = os.path.abspath(mask_dir) if mask_dir else None
        if self.mask_dir and not os.path.isdir(self.mask_dir): raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")
        self.resolution, self.overlap_factor = resolution, overlap_factor
        self.src_transforms, self.dst_transforms = src_transforms, dst_transforms
        self.shared_transforms, self.final_transform = shared_transforms, final_transform
        self.use_auto_mask = use_auto_mask
        self.skip_empty_patches = skip_empty_patches
        self.skip_empty_threshold = skip_empty_threshold
        self._patch_diffs = []
        self.slice_info, self.skipped_count, self.processed_files, self.total_slices_generated, self.empty_patches_skipped = [], 0, 0, 0, 0
        self.skipped_file_reasons = []
        overlap_pixels = int(resolution * overlap_factor); self.stride = max(1, resolution - overlap_pixels)
        src_glob_pattern = os.path.join(self.src_dir, '*.*')
        # Use DEBUG for file search pattern
        if is_main_process(): logging.debug(f"Dataset searching: {src_glob_pattern}")
        src_files = sorted(glob.glob(src_glob_pattern))
        if not src_files: logging.error(f"No source files found in '{self.src_dir}'"); # Keep Error
        # Use DEBUG for number of files found
        if is_main_process(): logging.debug(f"Found {len(src_files)} potential source files.")

        for i, src_path in enumerate(src_files):
            base_name = os.path.basename(src_path); dst_path = os.path.join(self.dst_dir, base_name)
            if not os.path.exists(dst_path): self.skipped_count += 1; self.skipped_file_reasons.append((base_name, "Dst missing")); continue
            mask_path = None
            if self.mask_dir:
                stem = os.path.splitext(base_name)[0]
                mask_exts = ['.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff']
                for mext in mask_exts:
                    candidate = os.path.join(self.mask_dir, stem + mext)
                    if os.path.exists(candidate): mask_path = candidate; break
                if mask_path is None: self.skipped_count += 1; self.skipped_file_reasons.append((base_name, "Mask missing")); continue
            try:
                img_s = load_image_any_format(src_path); img_d = load_image_any_format(dst_path)
                w_s, h_s = img_s.size; w_d, h_d = img_d.size
                if (w_s, h_s) != (w_d, h_d): img_s.close(); img_d.close(); self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Dims mismatch: src={w_s}x{h_s} dst={w_d}x{h_d}")); continue
                if w_s < resolution or h_s < resolution: img_s.close(); img_d.close(); self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Too small")); continue
                # Pre-compute per-pixel diff map for empty patch filtering
                diff_map = None
                if self.skip_empty_patches:
                    src_np = np.array(img_s).astype(np.float32)
                    dst_np = np.array(img_d).astype(np.float32)
                    diff_map = np.abs(src_np - dst_np).mean(axis=2)  # (H, W) in [0, 255]
                img_s.close(); img_d.close()
                num_slices_for_file = 0
                y_coords = list(range(0, max(0, h_s - resolution) + 1, self.stride)); x_coords = list(range(0, max(0, w_s - resolution) + 1, self.stride))
                if (h_s > resolution) and ((h_s - resolution) % self.stride != 0): y_coords.append(h_s - resolution)
                if (w_s > resolution) and ((w_s - resolution) % self.stride != 0): x_coords.append(w_s - resolution)
                unique_y = sorted(list(set(y_coords))); unique_x = sorted(list(set(x_coords)))
                for y in unique_y:
                    if not (0 <= y <= h_s - resolution): continue
                    for x in unique_x:
                        if not (0 <= x <= w_s - resolution): continue
                        # Skip patches where src and dst are essentially identical
                        if diff_map is not None:
                            patch_diff = diff_map[y:y+resolution, x:x+resolution]
                            patch_max = patch_diff.max()
                            self._patch_diffs.append(patch_max)
                            if patch_max < self.skip_empty_threshold:
                                self.empty_patches_skipped += 1
                                continue
                        crop_box = (x, y, x + resolution, y + resolution)
                        info_dict = {'src_path': src_path, 'dst_path': dst_path, 'crop_box': crop_box}
                        if mask_path: info_dict['mask_path'] = mask_path
                        self.slice_info.append(info_dict); num_slices_for_file += 1
                if num_slices_for_file > 0: self.processed_files += 1; self.total_slices_generated += num_slices_for_file
            except (FileNotFoundError, UnidentifiedImageError) as file_err:
                 self.skipped_count += 1; self.skipped_file_reasons.append((base_name, type(file_err).__name__));
                 # Keep Warning for bad files
                 if is_main_process(): logging.warning(f"Skipping {base_name}: {file_err}")
            except Exception as e:
                self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Error: {type(e).__name__}"))
                # Keep Warning for other errors
                if is_main_process(): logging.warning(f"Skipping {base_name} due to error: {e}", exc_info=False)

        if is_main_process():
            # Keep INFO for overall summary
            empty_str = f" (filtered {self.empty_patches_skipped} empty patches)" if self.empty_patches_skipped > 0 else ""
            logging.info(f"Dataset Init: Processed {self.processed_files}/{len(src_files)} files -> {self.total_slices_generated} slices{empty_str}. Skipped {self.skipped_count} files.")
            if self._patch_diffs:
                diffs = np.array(self._patch_diffs)
                pcts = np.percentile(diffs, [0, 25, 50, 75, 90, 95, 100])
                logging.info(f"Patch diff stats (max_pixel/255): min={pcts[0]:.2f} p25={pcts[1]:.2f} p50={pcts[2]:.2f} p75={pcts[3]:.2f} p90={pcts[4]:.2f} p95={pcts[5]:.2f} max={pcts[6]:.2f} | threshold={self.skip_empty_threshold:.1f}")
                self._patch_diffs = []
            # Use DEBUG for skip reasons unless count is high? Maybe keep top N as WARNING.
            if self.skipped_count > 0:
                 limit = 5
                 logging.warning(f"--- Top {min(limit, self.skipped_count)} File Skip Reasons ---") # Keep Warning
                 for i, (name, reason) in enumerate(self.skipped_file_reasons):
                      if i >= limit: logging.warning(f"  ... ({self.skipped_count - limit} more)"); break # Keep Warning
                      logging.warning(f"  - {name}: {reason}") # Keep Warning
                 logging.warning("-----------------------------") # Keep Warning
        if len(self.slice_info) == 0:
             err_msg = f"CRITICAL: Dataset has 0 usable slices. "
             if self.processed_files > 0: err_msg += f"Processed {self.processed_files} files but generated no slices. Check resolution/overlap."
             elif self.skipped_count > 0: err_msg += f"All {self.skipped_count} potential files skipped."
             else: err_msg += f"No source files found/processed in '{self.src_dir}'."
             logging.error(err_msg) # Keep Error
             raise ValueError(err_msg)
        elif is_main_process():
             # Keep INFO for final confirmation
             logging.info(f"Dataset ready. Total usable slices: {len(self.slice_info)}")

    def __len__(self): return len(self.slice_info)
    def __getitem__(self, idx):
        try: info = self.slice_info[idx]; src_path, dst_path, crop_box = info['src_path'], info['dst_path'], info['crop_box']
        except IndexError: logging.error(f"Index {idx} out of bounds for slice_info (len {len(self.slice_info)})."); return None
        try: src_img = load_image_any_format(src_path); dst_img = load_image_any_format(dst_path)
        except Exception as load_e: logging.error(f"Item {idx}: Img load error ({os.path.basename(src_path)}): {load_e}"); return None
        # Load mask if configured
        mask_slice = None
        if self.mask_dir and 'mask_path' in info:
            try: mask_full = load_mask_image(info['mask_path'])
            except Exception as e: logging.error(f"Item {idx}: Mask load error: {e}"); return None
            x1, y1, x2, y2 = crop_box
            mask_slice = mask_full[y1:y2, x1:x2]
        try:
            src_slice_pil = src_img.crop(crop_box); dst_slice_pil = dst_img.crop(crop_box)
            src_img.close(); dst_img.close()
            use_numpy = isinstance(self.shared_transforms, A.Compose) or isinstance(self.src_transforms, A.Compose) or isinstance(self.dst_transforms, A.Compose)
            if use_numpy: src_slice_current, dst_slice_current = np.array(src_slice_pil), np.array(dst_slice_pil)
            else: src_slice_current, dst_slice_current = src_slice_pil, dst_slice_pil

            # Compute auto-mask raw diff BEFORE augmentation (no border artifacts)
            auto_mask_slice = None
            if self.use_auto_mask:
                src_np = src_slice_current if use_numpy else np.array(src_slice_current)
                dst_np = dst_slice_current if use_numpy else np.array(dst_slice_current)
                auto_mask_slice = np.abs(src_np.astype(np.float32) - dst_np.astype(np.float32)).mean(axis=2) / 255.0

            if self.shared_transforms:
                if isinstance(self.shared_transforms, A.Compose):
                    aug_kwargs = {'image': src_slice_current, 'mask': dst_slice_current}
                    if mask_slice is not None: aug_kwargs['loss_mask'] = mask_slice
                    if auto_mask_slice is not None: aug_kwargs['auto_mask'] = auto_mask_slice
                    aug = self.shared_transforms(**aug_kwargs)
                    src_slice_current, dst_slice_current = aug['image'], aug['mask']
                    if mask_slice is not None: mask_slice = aug['loss_mask']
                    if auto_mask_slice is not None: auto_mask_slice = aug['auto_mask']
                elif isinstance(self.shared_transforms, T.Compose): seed = random.randint(0, 2**32-1); torch.manual_seed(seed); random.seed(seed); src_slice_current = self.shared_transforms(src_slice_current); torch.manual_seed(seed); random.seed(seed); dst_slice_current = self.shared_transforms(dst_slice_current)
            if self.src_transforms:
                if isinstance(self.src_transforms, A.Compose): src_slice_current = self.src_transforms(image=src_slice_current)['image']
                elif isinstance(self.src_transforms, T.Compose): src_slice_current = self.src_transforms(src_slice_current)
            if self.dst_transforms:
                if isinstance(self.dst_transforms, A.Compose): dst_slice_current = self.dst_transforms(image=dst_slice_current)['image']
                elif isinstance(self.dst_transforms, T.Compose): dst_slice_current = self.dst_transforms(dst_slice_current)

            if self.final_transform:
                if use_numpy: src_final_input, dst_final_input = Image.fromarray(src_slice_current), Image.fromarray(dst_slice_current)
                else: src_final_input, dst_final_input = src_slice_current, dst_slice_current
                src_tensor, dst_tensor = self.final_transform(src_final_input), self.final_transform(dst_final_input)
            else:
                if not use_numpy: src_slice_current, dst_slice_current = np.array(src_slice_current), np.array(dst_slice_current)
                src_tensor = torch.from_numpy(src_slice_current.transpose(2, 0, 1)).float() / 255.0; dst_tensor = torch.from_numpy(dst_slice_current.transpose(2, 0, 1)).float() / 255.0
                norm = T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()); src_tensor, dst_tensor = norm(src_tensor), norm(dst_tensor)
            # Convert mask to tensor (1, H, W) in [0,1] - no normalization
            if mask_slice is not None:
                mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_slice)).unsqueeze(0).float()
                mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
                return src_tensor, dst_tensor, mask_tensor
            # Auto-mask: return raw diff as 3rd element (blur+sigmoid applied later in training)
            if auto_mask_slice is not None:
                auto_mask_tensor = torch.from_numpy(np.ascontiguousarray(auto_mask_slice)).unsqueeze(0).float()
                auto_mask_tensor = torch.clamp(auto_mask_tensor, 0.0, 1.0)
                return src_tensor, dst_tensor, auto_mask_tensor
            return src_tensor, dst_tensor
        except Exception as e: logging.error(f"Item {idx}: Transform error ({os.path.basename(src_path)}): {e}", exc_info=False); return None

# --- Source-Only Dataset (for val preview without dst) ---
class SourceOnlySlicingDataset(Dataset):
    """Loads source images only. For validation preview when no dst is available."""
    def __init__(self, src_dir, resolution, overlap_factor=0.0, final_transform=None):
        self.src_dir = os.path.abspath(src_dir); self.resolution = resolution; self.final_transform = final_transform
        self.slice_info = []
        overlap_pixels = int(resolution * overlap_factor); self.stride = max(1, resolution - overlap_pixels)
        src_files = sorted(glob.glob(os.path.join(self.src_dir, '*.*')))
        for src_path in src_files:
            try:
                img = load_image_any_format(src_path); w, h = img.size; img.close()
                if w < resolution or h < resolution:
                    self.slice_info.append({'src_path': src_path, 'crop_box': None})
                    continue
                y_coords = list(range(0, max(0, h - resolution) + 1, self.stride)); x_coords = list(range(0, max(0, w - resolution) + 1, self.stride))
                if (h > resolution) and ((h - resolution) % self.stride != 0): y_coords.append(h - resolution)
                if (w > resolution) and ((w - resolution) % self.stride != 0): x_coords.append(w - resolution)
                for y in sorted(set(y_coords)):
                    if not (0 <= y <= h - resolution): continue
                    for x in sorted(set(x_coords)):
                        if not (0 <= x <= w - resolution): continue
                        self.slice_info.append({'src_path': src_path, 'crop_box': (x, y, x + resolution, y + resolution)})
            except Exception: continue
        if not self.slice_info: raise ValueError(f"No valid slices from source files in {src_dir}")
        logging.info(f"Val preview dataset (src-only): {len(self.slice_info)} slices from {src_dir}")

    def __len__(self): return len(self.slice_info)
    def __getitem__(self, idx):
        info = self.slice_info[idx]
        try:
            src_img = load_image_any_format(info['src_path'])
            if info['crop_box'] is not None: src_slice = src_img.crop(info['crop_box'])
            else: src_slice = src_img.resize((self.resolution, self.resolution), Image.LANCZOS)
            src_img.close()
            if self.final_transform: src_tensor = self.final_transform(src_slice)
            else: src_np = np.array(src_slice); src_tensor = torch.from_numpy(src_np.transpose(2, 0, 1)).float() / 255.0; norm = T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()); src_tensor = norm(src_tensor)
            return src_tensor
        except Exception as e: logging.error(f"SrcOnly item {idx} error: {e}"); return None

# --- Signal Handler ---
shutdown_requested = False
def handle_signal(signum, frame):
    global shutdown_requested
    try: sig_name = signal.Signals(signum).name
    except: sig_name = f"Signal {signum}"
    current_rank = get_rank()
    if not shutdown_requested:
        print(f"\n[Rank {current_rank}] Received {sig_name}. Requesting graceful shutdown...") # Keep print
        logging.warning(f"Received {sig_name}. Requesting graceful shutdown...") # Keep Warning
        shutdown_requested = True
    else:
        print(f"\n[Rank {current_rank}] Shutdown already requested. Received {sig_name} again. Terminating forcefully.") # Keep print
        logging.warning("Shutdown already requested. Terminating forcefully.") # Keep Warning
        os._exit(1)

# --- Training Helper Functions ---

def _setup_device(config):
    """Determine the training device and type based on platform and availability.

    Returns (device, device_type) where device_type is 'cuda', 'mps', or 'cpu'.
    """
    if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    device_type = device.type  # 'cuda', 'mps', or 'cpu'
    return device, device_type


def _setup_logging(config, rank, device, world_size):
    """Configure console and file logging for training.

    Sets up root logger with console handler (all ranks) and file handler (main process only).
    """
    log_level = logging.INFO if is_main_process() else logging.WARNING
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    file_formatter = logging.Formatter(f'%(asctime)s [R{rank}|{CURRENT_OS}|%(levelname)s] %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    if is_main_process():
        try:
            output_dir_abs = os.path.abspath(config.data.output_dir)
            os.makedirs(output_dir_abs, exist_ok=True)
            log_file = os.path.join(output_dir_abs, 'training.log')
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            logging.info("=" * 60)
            logging.info(f" Starting Training ")
            logging.info(f" Device: {device} | World Size: {world_size}")
            logging.info(f" Output Dir: {output_dir_abs}")
            try:
                logging.debug("--- Effective Configuration ---\n" + yaml.dump(
                    config_to_dict(config), indent=2, default_flow_style=False, sort_keys=False
                ) + "-----------------------------")
            except Exception:
                logging.debug(f" Config Object: {config}")
            logging.info("=" * 60)
            logging.info(">>> Press Ctrl+C for graceful shutdown <<<")
        except Exception as log_setup_e:
            logging.error(f"Error setting up file logging: {log_setup_e}")


def _build_datasets(config, src_transforms, dst_transforms, shared_transforms, standard_transform,
                    use_masks, use_auto_mask, skip_empty, device_type, world_size, rank):
    """Create training and validation datasets and dataloaders.

    Returns (dataset, dataloader, dataloader_iter, val_dataloader, has_val_preview, has_val_loss).
    Raises on fatal errors.
    """
    dataset = AugmentedImagePairSlicingDataset(
        config.data.src_dir, config.data.dst_dir, config.data.resolution,
        config.data.overlap_factor, src_transforms, dst_transforms, shared_transforms,
        standard_transform, mask_dir=config.data.mask_dir if use_masks else None,
        use_auto_mask=use_auto_mask, skip_empty_patches=skip_empty,
        skip_empty_threshold=config.mask.skip_empty_threshold)
    if is_main_process():
        logging.info(f"Dataset Size: {len(dataset)} slices.")
        global_bs = config.training.batch_size * world_size
        est_bpp = math.ceil(len(dataset) / global_bs) if global_bs > 0 else 0
        logging.debug(f"Global Batch: {global_bs} | Est Batches/Pass: {est_bpp} | Iter/Epoch: {config.training.iterations_per_epoch}")
    if len(dataset) < world_size:
        raise ValueError(f"Dataset size ({len(dataset)}) < World size ({world_size}).")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    pin = True if device_type == 'cuda' else False
    persist = True if config.dataloader.num_workers > 0 else False
    prefetch = config.dataloader.prefetch_factor if config.dataloader.num_workers > 0 else None
    if CURRENT_OS == 'Windows' and config.dataloader.num_workers > 0 and is_main_process():
        logging.warning("Using num_workers > 0 on Windows. If issues occur, try num_workers=0.")
    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, sampler=sampler,
        num_workers=config.dataloader.num_workers, pin_memory=pin,
        prefetch_factor=prefetch, persistent_workers=persist,
        collate_fn=collate_skip_none, drop_last=True)
    dataloader_iter = cycle(dataloader)

    # --- Validation Dataset & DataLoader ---
    val_dataloader = None
    has_val_preview = bool(config.data.val_src_dir)
    has_val_loss = bool(config.data.val_src_dir and config.data.val_dst_dir)
    if has_val_loss and is_main_process():
        try:
            val_dataset = AugmentedImagePairSlicingDataset(
                config.data.val_src_dir, config.data.val_dst_dir, config.data.resolution,
                config.data.overlap_factor, None, None, None, standard_transform)
            logging.info(f"Validation Dataset Size: {len(val_dataset)} slices.")
            val_dataloader = DataLoader(
                val_dataset, batch_size=config.training.batch_size, shuffle=False,
                num_workers=0, collate_fn=collate_skip_none, drop_last=False)
        except Exception as e:
            logging.warning(f"Validation dataset init failed: {e}. Validation disabled.")
            val_dataloader = None

    return dataset, dataloader, dataloader_iter, val_dataloader, has_val_preview, has_val_loss


def _build_model(config, device, device_type, world_size, rank, n_input_ch, eff_size,
                 use_lpips, use_l2, use_bce_dice, use_amp_eff, loss_fn_lpips, model_type):
    """Create the model, optimizer, scaler, and load checkpoints if available.

    Returns (model, optimizer, scaler, start_epoch, start_step, ckpt_prefix, use_amp_eff).
    """
    recurrence_t = getattr(config.model, 'recurrence_steps', 2)
    model = create_model(model_type=model_type, n_ch=n_input_ch, n_cls=3,
                         hidden_size=eff_size, t=recurrence_t).to(device)
    if is_main_process():
        logging.debug(f"{model_type.upper()} instantiated: hidden_size={eff_size}, device={device}.")
    if world_size > 1 and device_type == 'cuda':
        if model_type == 'unet':
            if is_main_process(): logging.debug("Converting BatchNorm to SyncBatchNorm.")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            if is_main_process(): logging.debug("MSRNet uses GroupNorm; skipping SyncBatchNorm conversion.")
        dist.barrier()

    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-5)

    # Determine effective AMP usage
    use_amp_requested = config.training.use_amp
    if use_amp_requested:
        if device_type == 'cuda':
            use_amp_eff = True
        elif device_type == 'mps':
            if is_main_process(): logging.warning("AMP requested but device is MPS. Disabling AMP (limited support).")
            use_amp_eff = False
        else:
            if is_main_process(): logging.warning("AMP requested but device is CPU. AMP disabled.")
            use_amp_eff = False
    else:
        use_amp_eff = False

    scaler = GradScaler(enabled=use_amp_eff)
    if is_main_process(): logging.info(f"Optimizer: AdamW | AMP Enabled: {use_amp_eff}")
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss() if use_l2 else None
    criterion_bce = nn.BCEWithLogitsLoss() if use_bce_dice else None

    # Checkpoint naming
    ckpt_prefix = getattr(config, '_config_stem',
                          os.path.basename(os.path.normpath(config.data.output_dir)))
    def_h, bump_h = 64, 96

    # --- Resume from checkpoint ---
    start_epoch, start_step = 0, 0
    latest_ckpt = os.path.join(config.data.output_dir, f'{ckpt_prefix}_tunet_latest.pth')
    # Fallback chain: config_stem name -> folder name -> legacy tunet_
    if is_main_process() and not os.path.exists(latest_ckpt):
        folder_prefix = os.path.basename(os.path.normpath(config.data.output_dir))
        folder_ckpt = os.path.join(config.data.output_dir, f'{folder_prefix}_tunet_latest.pth')
        legacy_ckpt = os.path.join(config.data.output_dir, 'tunet_latest.pth')
        if os.path.exists(folder_ckpt):
            logging.info(f"Found checkpoint with folder name: {folder_prefix}_tunet_latest.pth")
            latest_ckpt = folder_ckpt
        elif os.path.exists(legacy_ckpt):
            logging.info(f"Found legacy checkpoint: tunet_latest.pth")
            latest_ckpt = legacy_ckpt
    exists_list = [False]
    if is_main_process(): exists_list[0] = os.path.exists(latest_ckpt)
    if world_size > 1: dist.broadcast_object_list(exists_list, src=0)
    if exists_list[0]:
        if is_main_process(): logging.info(f"Attempting resume from checkpoint: {latest_ckpt}")
        try:
            ckpt = torch.load(latest_ckpt, map_location='cpu')
            if 'config' not in ckpt: raise ValueError("Checkpoint missing 'config'")
            ckpt_cfg = dict_to_sns(ckpt['config'])
            ckpt_loss = getattr(getattr(ckpt_cfg, 'training', SimpleNamespace()), 'loss', 'l1')
            ckpt_base = getattr(getattr(ckpt_cfg, 'model', SimpleNamespace()), 'model_size_dims', def_h)
            ckpt_needs_bump = ((ckpt_loss == 'l1+lpips' and ckpt_base == def_h)
                               or (ckpt_loss == 'weighted' and ckpt_base == def_h
                                   and getattr(getattr(ckpt_cfg, 'training', SimpleNamespace()), 'lpips_weight', 0) > 0))
            ckpt_eff_size = bump_h if ckpt_needs_bump else ckpt_base
            if ckpt_loss != config.training.loss:
                raise ValueError(f"Loss mismatch: Ckpt='{ckpt_loss}', Current='{config.training.loss}'")
            if ckpt_eff_size != eff_size:
                raise ValueError(f"Effective size mismatch: Ckpt={ckpt_eff_size}, Current={eff_size}")
            ckpt_n_ch = ckpt.get('n_input_channels', 3)
            if ckpt_n_ch != n_input_ch:
                raise ValueError(f"Input channel mismatch: Ckpt={ckpt_n_ch}, Current={n_input_ch}. Cannot resume with different mask input setting.")
            ckpt_model_type = ckpt.get('model_type', getattr(getattr(ckpt_cfg, 'model', SimpleNamespace()), 'model_type', 'unet'))
            if ckpt_model_type != model_type:
                raise ValueError(f"Model type mismatch: Ckpt='{ckpt_model_type}', Current='{model_type}'. Cannot resume across architectures.")
            if is_main_process(): logging.debug("Checkpoint config compatible.")
            state_dict = ckpt['model_state_dict']
            is_ddp_ckpt = any(k.startswith('module.') for k in state_dict)
            if world_size > 1 and not is_ddp_ckpt:
                logging.debug("Adding 'module.' prefix to load non-DDP checkpoint into DDP model.")
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif world_size == 1 and is_ddp_ckpt:
                logging.debug("Removing 'module.' prefix to load DDP checkpoint into non-DDP model.")
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            if is_main_process(): logging.debug("Loaded model state_dict.")

            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                logging.debug("Loaded optimizer state_dict.")
            else:
                if is_main_process(): logging.warning("Optimizer state missing in checkpoint.")

            if use_amp_eff and ckpt.get('scaler_state_dict'):
                try:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                    logging.debug("Loaded GradScaler state_dict.")
                except Exception as se:
                    logging.warning(f"Could not load GradScaler state: {se}.")
            elif use_amp_eff:
                if is_main_process(): logging.warning("AMP enabled, but scaler state missing in ckpt.")

            start_step = ckpt.get('global_step', 0) + 1
            start_epoch = start_step // config.training.iterations_per_epoch
            if is_main_process():
                logging.info(f"Resuming from Global Step {start_step} (Logical Epoch {start_epoch + 1}).")
            del ckpt
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logging.error(f"Checkpoint load failed: {e}. Starting fresh.", exc_info=True)
            start_epoch, start_step = 0, 0
            optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-5)
            scaler = GradScaler(enabled=use_amp_eff)
    else:
        # No checkpoint — check if fine-tuning from an existing model
        finetune_path = getattr(config.training, 'finetune_from', None)
        if finetune_path:
            if is_main_process():
                logging.info("=" * 60)
                logging.info("FINE-TUNING MODE")
                logging.info(f"  Loading model weights from: {finetune_path}")
                logging.info(f"  Training on new data in: {config.data.src_dir}")
                logging.info(f"  Optimizer & step counter start fresh (step 0).")
                logging.info("=" * 60)
            try:
                ft_ckpt = torch.load(finetune_path, map_location='cpu')
                if 'model_state_dict' not in ft_ckpt:
                    raise KeyError("Checkpoint missing 'model_state_dict'. Is this a valid tunet checkpoint?")
                ft_state = ft_ckpt['model_state_dict']
                if any(k.startswith('module.') for k in ft_state):
                    ft_state = {k.replace('module.', '', 1): v for k, v in ft_state.items()}
                if world_size > 1 and not any(k.startswith('module.') for k in ft_state):
                    ft_state = {'module.' + k: v for k, v in ft_state.items()}
                model.load_state_dict(ft_state)
                if is_main_process():
                    logging.info("Fine-tune weights loaded successfully. Starting training from step 0.")
                del ft_ckpt
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                logging.error(f"Failed to load fine-tune checkpoint: {e}", exc_info=True)
                logging.warning("Falling back to training from scratch.")
        else:
            if is_main_process(): logging.info("No checkpoint found. Starting fresh.")

    # --- DDP Wrap ---
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cuda_device_ids = [local_rank] if device_type == 'cuda' else None
        model = DDP(model,
                    device_ids=cuda_device_ids,
                    output_device=(local_rank if device_type == 'cuda' else None),
                    find_unused_parameters=False)
        if is_main_process(): logging.debug("Model wrapped with DDP.")
        dist.barrier()

    return model, optimizer, scaler, start_epoch, start_step, ckpt_prefix, use_amp_eff, criterion_l1, criterion_l2, criterion_bce


def _compute_training_step(model, model_input, dst, optimizer, scaler, criterion_l1, criterion_l2,
                           loss_fn_lpips, criterion_bce, config, device, device_type, use_amp_eff,
                           use_bce_dice, use_lpips, use_weighted, use_l2, use_mask_loss, use_auto_mask,
                           mask_batch, auto_mask_raw):
    """Run forward pass, compute loss, backward pass, and optimizer step.

    Handles both AMP and non-AMP paths. Returns (l1_val, lp_val, l2_val, out_tensor, loss_finite).
    """
    optimizer.zero_grad(set_to_none=True)

    # --- Forward + Loss (shared logic for AMP and non-AMP) ---
    def _forward_and_loss(model_input, dst):
        out = model(model_input)
        if use_bce_dice:
            dst_01 = dst * 0.5 + 0.5
            bce_total = torch.tensor(0.0, device=device)
            dc_total = torch.tensor(0.0, device=device)
            n_ch = out.shape[1]
            for ch in range(n_ch):
                out_ch = out[:, ch:ch + 1, :, :]
                dst_ch = dst_01[:, ch:ch + 1, :, :]
                bce_total = bce_total + criterion_bce(out_ch, dst_ch)
                dc_total = dc_total + dice_loss(torch.sigmoid(out_ch), dst_ch)
            l1 = (bce_total + dc_total) / n_ch
            lp = torch.tensor(0.0, device=device)
            loss = l1
            out = torch.sigmoid(out)
        else:
            if use_mask_loss and mask_batch is not None:
                weight_map = 1.0 + mask_batch * (config.mask.mask_weight - 1.0)
                l1 = (torch.abs(out - dst) * weight_map).mean()
            elif use_auto_mask and auto_mask_raw is not None:
                auto_mask_w = refine_auto_mask(auto_mask_raw, gamma=config.mask.auto_mask_gamma)
                weight_map = 1.0 + auto_mask_w * (config.mask.mask_weight - 1.0)
                l1 = (torch.abs(out - dst) * weight_map).mean()
            else:
                weight_map = None
                l1 = criterion_l1(out, dst)
            l2_val = torch.tensor(0.0, device=device)
            if use_l2 and criterion_l2 is not None:
                if weight_map is not None:
                    l2_val = ((out - dst) ** 2 * weight_map).mean()
                else:
                    l2_val = criterion_l2(out, dst)
            lp = torch.tensor(0.0, device=device)
            if use_lpips and loss_fn_lpips:
                try:
                    lp = loss_fn_lpips(out, dst).mean()
                except Exception:
                    lp = torch.tensor(0.0, device=device)
            if use_weighted:
                loss = (config.training.l1_weight * l1
                        + config.training.l2_weight * l2_val
                        + config.training.lpips_weight * lp)
            else:
                loss = l1 + config.training.lambda_lpips * lp
        l2_out = l2_val if not use_bce_dice else torch.tensor(0.0, device=device)
        return out, l1, lp, l2_out, loss

    # --- AMP vs non-AMP execution ---
    if use_amp_eff:
        with autocast(device_type=device_type, enabled=True):
            out, l1, lp, l2_val, loss = _forward_and_loss(model_input, dst)
    else:
        out, l1, lp, l2_val, loss = _forward_and_loss(model_input, dst)

    # --- Check for NaN/Inf loss ---
    if not torch.isfinite(loss):
        return l1.detach().item(), 0.0, 0.0, out, False

    # --- Backward + Optimizer Step ---
    if use_amp_eff:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        if not torch.isfinite(grad_norm):
            logging.warning(f"NaN/Inf gradient (norm={grad_norm:.2e})! Skipping update.")
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()

    return l1.detach().item(), lp.detach().item() if use_lpips else 0.0, l2_val.detach().item() if use_l2 else 0.0, out, True


# --- Training Function ---
def train(config):
    global shutdown_requested, scaler # Make scaler global for use in save_previews
    training_successful = False; final_global_step = 0

    # --- Signal Registration ---
    signal.signal(signal.SIGINT, handle_signal)
    if CURRENT_OS != 'Windows': signal.signal(signal.SIGTERM, handle_signal)

    # --- DDP & Device Setup ---
    try: setup_ddp()
    except Exception as ddp_e: logging.error(f"FATAL: DDP setup failed: {ddp_e}. Exiting.", exc_info=True); return # Keep Error
    rank = get_rank(); world_size = get_world_size()

    device, device_type = _setup_device(config)

    _setup_logging(config, rank, device, world_size)

    # --- Standard Transform ---
    standard_transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()) ])

    # --- Create Augmentation Pipelines ---
    use_auto_mask = config.mask.use_auto_mask
    src_transforms, dst_transforms, shared_transforms = None, None, None
    try:
        datasets_config = getattr(config, 'dataloader', SimpleNamespace()).datasets
        src_list = getattr(datasets_config, 'src_augs', []); dst_list = getattr(datasets_config, 'dst_augs', []); shared_list = getattr(datasets_config, 'shared_augs', [])
        src_transforms = create_augmentations(src_list if isinstance(src_list, list) else [])
        dst_transforms = create_augmentations(dst_list if isinstance(dst_list, list) else [])
        shared_transforms = create_augmentations(shared_list if isinstance(shared_list, list) else [], has_mask=bool(config.data.mask_dir), use_auto_mask=use_auto_mask)
        if is_main_process(): logging.debug("Augmentation pipelines created (if configured).")
    except Exception as e: logging.error(f"FATAL: Augmentation creation failed: {e}. Exiting.", exc_info=True); cleanup_ddp(); return

    # --- Mask Settings ---
    use_masks = config.data.mask_dir is not None and (config.mask.use_mask_loss or config.mask.use_mask_input)
    use_mask_loss = use_masks and config.mask.use_mask_loss
    use_mask_input = use_masks and config.mask.use_mask_input
    n_input_ch = 4 if use_mask_input else 3
    if is_main_process() and use_masks:
        logging.info(f"Mask enabled: loss_weighting={use_mask_loss} (weight={config.mask.mask_weight}), input_channel={use_mask_input}")
    if is_main_process() and use_auto_mask:
        logging.info(f"Auto-mask enabled: weight={config.mask.mask_weight}, gamma={config.mask.auto_mask_gamma} (auto-generated from |src-dst|, blur+gamma expansion)")
    if is_main_process() and config.mask.skip_empty_patches and use_auto_mask:
        logging.info(f"Skip empty patches: enabled (max pixel diff threshold={config.mask.skip_empty_threshold}/255)")

    # --- Dataset & DataLoader ---
    dataset, dataloader, dataloader_iter = None, None, None
    try:
        skip_empty = config.mask.skip_empty_patches and use_auto_mask
        dataset, dataloader, dataloader_iter, val_dataloader, has_val_preview, has_val_loss = _build_datasets(
            config, src_transforms, dst_transforms, shared_transforms, standard_transform,
            use_masks, use_auto_mask, skip_empty, device_type, world_size, rank)
    except Exception as e: logging.error(f"FATAL: Dataset/DataLoader init failed: {e}. Exiting.", exc_info=True); cleanup_ddp(); return

    # --- Model & Loss Setup ---
    model, loss_fn_lpips = None, None
    eff_size = config.model.model_size_dims; use_lpips = False; use_bce_dice = False; use_weighted = False; use_l2 = False; def_h, bump_h = 64, 96
    if config.training.loss == 'bce+dice':
        use_bce_dice = True
        if is_main_process(): logging.info(f"Using BCE+Dice loss (segmentation mode). Hidden size: {eff_size}.")
    elif config.training.loss == 'weighted':
        use_weighted = True
        needs_lpips = config.training.lpips_weight > 0
        use_l2 = config.training.l2_weight > 0
        if needs_lpips:
            use_lpips = True
            if config.model.model_size_dims == def_h: eff_size = bump_h; msg = f"bumping effective hidden size to {eff_size}"
            else: msg = f"using configured hidden size {eff_size}"
            if is_main_process(): logging.debug(f"Weighted loss with LPIPS, {msg}.")
        if is_main_process():
            logging.info(f"Using weighted loss: L1={config.training.l1_weight}, L2={config.training.l2_weight}, LPIPS={config.training.lpips_weight}. Hidden size: {eff_size}.")
        if needs_lpips:
            status = {'success': False, 'error': None}
            if is_main_process():
                try:
                    if device_type != 'cuda': logging.debug(f"LPIPS running on {device_type}. May be slow.")
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                    [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]; status['success'] = True; logging.info("LPIPS model initialized.")
                except Exception as e: logging.error(f"LPIPS init failed: {e}. Disabling LPIPS component.", exc_info=True); status['error'] = str(e); use_lpips = False
            if world_size > 1: dist.broadcast_object_list([status], src=0); status = status[0]; use_lpips = status['success']
            if rank > 0 and use_lpips:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                    [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]
                except Exception as e_rank_n: logging.error(f"LPIPS init Rank {rank} failed: {e_rank_n}.", exc_info=True)
            if not use_lpips and is_main_process(): logging.warning(f"LPIPS disabled due to init failure. Weighted loss will use L1+L2 only.")
    elif config.training.loss == 'l1+lpips':
        use_lpips = True
        if config.model.model_size_dims == def_h: eff_size = bump_h; msg = f"bumping effective hidden size to {eff_size}"
        else: msg = f"using configured hidden size {eff_size}"
        if is_main_process(): logging.debug(f"LPIPS enabled, {msg}. Lambda={config.training.lambda_lpips}")
        status = {'success': False, 'error': None}
        if is_main_process():
            try:
                if device_type != 'cuda': logging.debug(f"LPIPS running on {device_type}. May be slow.")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]; status['success'] = True; logging.info("LPIPS model initialized.")
            except Exception as e: logging.error(f"LPIPS init failed: {e}. Disabling.", exc_info=True); status['error'] = str(e); use_lpips = False
        if world_size > 1: dist.broadcast_object_list([status], src=0); status = status[0]; use_lpips = status['success']
        if rank > 0 and use_lpips:
             try:
                 with warnings.catch_warnings():
                     warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                     loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                 [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]
             except Exception as e_rank_n: logging.error(f"LPIPS init Rank {rank} failed: {e_rank_n}. Inconsistency likely!", exc_info=True); logging.warning(f"Rank {rank} proceeding with LPIPS potentially uninitialized!")
        if not use_lpips and config.training.loss == 'l1+lpips':
             if is_main_process(): logging.warning(f"LPIPS disabled due to init failure. Using L1 ONLY.")
    else:
        if is_main_process(): logging.debug(f"Using L1 loss only. Hidden size: {eff_size}.")

    # --- Model Instantiation, SyncBN, Resume, DDP Wrap ---
    model, optimizer, scaler = None, None, None
    start_epoch, start_step = 0, 0
    model_type = getattr(config.model, 'model_type', 'unet')
    try:
        (model, optimizer, scaler, start_epoch, start_step, ckpt_prefix,
         use_amp_eff, criterion_l1, criterion_l2, criterion_bce) = _build_model(
            config, device, device_type, world_size, rank, n_input_ch, eff_size,
            use_lpips, use_l2, use_bce_dice, False, loss_fn_lpips, model_type)
    except Exception as model_err: logging.error(f"FATAL: Model setup/resume error: {model_err}. Exiting.", exc_info=True); cleanup_ddp(); return

    # --- Training Loop Variables ---
    max_steps = getattr(config.training, 'max_steps', 0)
    global_step = start_step; iter_epoch = config.training.iterations_per_epoch
    fixed_src, fixed_dst, fixed_mask = None, None, None; preview_count = 0
    val_fixed_src, val_fixed_dst = None, None; val_preview_count = 0
    ep_l1, ep_lpips, ep_steps = 0.0, 0.0, 0
    batch_times = []
    val_interval = config.logging.val_interval

    # --- Early Stopping / Plateau Detection ---
    es_cfg = getattr(config, 'early_stopping', SimpleNamespace(enabled=False))
    es_enabled = getattr(es_cfg, 'enabled', False) and is_main_process()
    es_patience = getattr(es_cfg, 'patience', 30)
    es_min_epochs = getattr(es_cfg, 'min_epochs', 10)
    es_smoothing = getattr(es_cfg, 'smoothing', 0.9)
    es_stop = getattr(es_cfg, 'stop', False)
    es_best_loss = float('inf')
    es_best_epoch = 0
    es_smoothed_loss = None
    es_plateau_saved = False

    # --- Capture Initial Preview Batch ---
    preview_interval = config.logging.preview_batch_interval
    if is_main_process() and preview_interval > 0:
        fixed_src, fixed_dst, fixed_mask, fixed_auto_mask = capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms, standard_transform, use_masks=use_masks, use_auto_mask=use_auto_mask, skip_empty=skip_empty, skip_empty_threshold=config.mask.skip_empty_threshold)
        if fixed_src is None: logging.warning("Initial preview batch capture failed.")
        if has_val_preview:
            val_fixed_src, val_fixed_dst = capture_val_preview_batch(config, standard_transform)
            if val_fixed_src is None: logging.warning("Initial val preview batch capture failed.")

    # --- Progressive Multi-Resolution Setup ---
    progressive_schedule = []  # list of (epoch_index, resolution) for warm-up stages
    current_training_res = config.data.resolution
    if config.training.progressive_resolution:
        full_res = config.data.resolution
        # Compute progressive stages: quarter -> half -> full
        quarter = max(64, (full_res // 4) // 16 * 16)  # Round down to nearest 16
        half = max(64, (full_res // 2) // 16 * 16)
        stages = []
        if quarter < full_res:
            stages.append(quarter)
        if half < full_res and half > quarter:
            stages.append(half)
        # Build schedule: epoch 0 -> stages[0], epoch 1 -> stages[1], epoch 2+ -> full
        for i, res in enumerate(stages):
            progressive_schedule.append((i, res))
        if is_main_process() and progressive_schedule:
            stage_str = ' -> '.join([f'{res}px (epoch {ep+1})' for ep, res in progressive_schedule])
            logging.info(f"Progressive resolution: {stage_str} -> {full_res}px (epoch {len(progressive_schedule)+1}+)")
        # Start at the appropriate progressive resolution (accounting for resume)
        if progressive_schedule:
            resume_epoch = start_step // iter_epoch if start_step > 0 else 0
            current_training_res = config.data.resolution  # default to full
            for ep_idx, res in progressive_schedule:
                if resume_epoch == ep_idx:
                    current_training_res = res
                    break
            if current_training_res != config.data.resolution:
                if is_main_process(): logging.info(f"Progressive: starting at {current_training_res}px resolution")
                try:
                    dataset = AugmentedImagePairSlicingDataset(config.data.src_dir, config.data.dst_dir, current_training_res, config.data.overlap_factor, src_transforms, dst_transforms, shared_transforms, standard_transform, mask_dir=config.data.mask_dir if use_masks else None, use_auto_mask=use_auto_mask, skip_empty_patches=skip_empty, skip_empty_threshold=config.mask.skip_empty_threshold)
                    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
                    pin = True if device_type == 'cuda' else False; persist = True if config.dataloader.num_workers > 0 else False
                    prefetch = config.dataloader.prefetch_factor if config.dataloader.num_workers > 0 else None
                    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=config.dataloader.num_workers, pin_memory=pin, prefetch_factor=prefetch, persistent_workers=persist, collate_fn=collate_skip_none, drop_last=True)
                    dataloader_iter = cycle(dataloader)
                    if is_main_process(): logging.info(f"Progressive: dataset recreated at {current_training_res}px ({len(dataset)} slices)")
                except Exception as e:
                    logging.error(f"Progressive: failed to create dataset at {current_training_res}px: {e}. Using full resolution.", exc_info=True)
                    current_training_res = config.data.resolution
                    progressive_schedule = []

    # --- Main Training Loop ---
    model.train()
    try:
        while True:
            if shutdown_requested:
                if is_main_process(): logging.info(f"Shutdown requested @ step {global_step}.")
                break

            if max_steps > 0 and global_step >= max_steps:
                if is_main_process(): logging.info(f"Reached max_steps ({max_steps}) @ step {global_step}. Stopping.")
                break

            loop_iter_start = time.time()
            current_ep_idx = global_step // iter_epoch; is_new_epoch = (global_step % iter_epoch == 0)
            if is_new_epoch:
                if isinstance(dataloader.sampler, DistributedSampler): dataloader.sampler.set_epoch(current_ep_idx)
                if is_main_process() and (global_step > 0 or start_step > 0): logging.info(f"--- Epoch {current_ep_idx + 1} Start (Step {global_step}) ---")
                ep_l1, ep_lpips, ep_steps = 0.0, 0.0, 0

                # --- Progressive resolution: check if we need to change resolution ---
                if config.training.progressive_resolution:
                    target_res = config.data.resolution  # default to full
                    for ep_idx, res in progressive_schedule:
                        if current_ep_idx == ep_idx:
                            target_res = res
                            break
                    if target_res != current_training_res:
                        current_training_res = target_res
                        if is_main_process(): logging.info(f"Progressive: switching to {current_training_res}px resolution")
                        try:
                            dataset = AugmentedImagePairSlicingDataset(config.data.src_dir, config.data.dst_dir, current_training_res, config.data.overlap_factor, src_transforms, dst_transforms, shared_transforms, standard_transform, mask_dir=config.data.mask_dir if use_masks else None, use_auto_mask=use_auto_mask, skip_empty_patches=skip_empty, skip_empty_threshold=config.mask.skip_empty_threshold)
                            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
                            pin = True if device_type == 'cuda' else False; persist = True if config.dataloader.num_workers > 0 else False
                            prefetch = config.dataloader.prefetch_factor if config.dataloader.num_workers > 0 else None
                            dataloader = DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=config.dataloader.num_workers, pin_memory=pin, prefetch_factor=prefetch, persistent_workers=persist, collate_fn=collate_skip_none, drop_last=True)
                            dataloader_iter = cycle(dataloader)
                            if is_main_process(): logging.info(f"Progressive: dataset recreated at {current_training_res}px ({len(dataset)} slices)")
                        except Exception as e:
                            logging.error(f"Progressive: failed to recreate dataset at {current_training_res}px: {e}", exc_info=True)

            data_load_start = time.time()
            try: batch = next(dataloader_iter)
            except Exception as e: logging.error(f"S{global_step}: Batch load error: {e}, skipping.", exc_info=True); global_step += 1; continue
            if batch is None: logging.warning(f"S{global_step}: Skipped batch (collation error)."); global_step += 1; continue
            # Unpack batch (2-tuple or 3-tuple with mask/auto_mask)
            auto_mask_raw = None
            if use_masks:
                src, dst, mask_batch = batch
                mask_batch = mask_batch.to(device)
            elif use_auto_mask:
                src, dst, auto_mask_raw = batch
                auto_mask_raw = auto_mask_raw.to(device)
                mask_batch = None
            else:
                src, dst = batch
                mask_batch = None
            data_load_time = time.time() - data_load_start

            # ── send inputs to the same device as the model ──
            transfer_start = time.time()
            src = src.to(device)
            dst = dst.to(device)
            transfer_time = time.time() - transfer_start

            # Build model input (concatenate mask as 4th channel if enabled)
            model_input = torch.cat([src, mask_batch], dim=1) if use_mask_input and mask_batch is not None else src

            # --- Forward, Loss, Backward, Optimize ---
            compute_start = time.time()
            batch_l1, batch_lp, batch_l2, out, loss_finite = _compute_training_step(
                model, model_input, dst, optimizer, scaler, criterion_l1, criterion_l2,
                loss_fn_lpips, criterion_bce, config, device, device_type, use_amp_eff,
                use_bce_dice, use_lpips, use_weighted, use_l2, use_mask_loss, use_auto_mask,
                mask_batch, auto_mask_raw)

            if not loss_finite:
                logging.error(f"S{global_step}: NaN/Inf loss ({batch_l1})! Skip update.")
                global_step += 1
                continue

            compute_time = time.time() - compute_start
            _has_weighted_loss = (use_mask_loss and mask_batch is not None) or use_auto_mask
            batch_l1_raw = criterion_l1(out.detach(), dst).item() if _has_weighted_loss else batch_l1
            if world_size > 1:
                l1_t = torch.tensor(batch_l1, device=device); lp_t = torch.tensor(batch_lp, device=device)
                dist.all_reduce(l1_t, op=dist.ReduceOp.AVG); dist.all_reduce(lp_t, op=dist.ReduceOp.AVG)
                batch_l1 = l1_t.item(); batch_lp = lp_t.item()
            ep_l1 += batch_l1; ep_lpips += batch_lp; ep_steps += 1

            iter_time = time.time() - loop_iter_start; batch_times.append(iter_time); batch_times = batch_times[-100:]
            avg_time = sum(batch_times) / len(batch_times) if batch_times else 0.0

            global_step += 1; final_global_step = global_step

            if is_main_process() and global_step % config.logging.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'… (D:{data_load_time:.3f} T:{transfer_time:.3f} C:{compute_time:.3f})')
                avg_ep_l1 = ep_l1 / ep_steps if ep_steps > 0 else 0.0
                avg_ep_lp = ep_lpips / ep_steps if use_lpips and ep_steps > 0 else 0.0
                steps_in_ep = global_step % iter_epoch or iter_epoch
                loss_label = 'BCE+Dice' if use_bce_dice else 'L1'
                log_msg = (f'Epoch[{current_ep_idx + 1}] Step[{global_step}] ({steps_in_ep}/{iter_epoch}), '
                           f'{loss_label}:{batch_l1:.4f}(Avg:{avg_ep_l1:.4f})')
                if _has_weighted_loss:
                    log_msg += f'[raw:{batch_l1_raw:.4f}]'
                if use_l2: log_msg += f', L2:{batch_l2:.4f}'
                if use_lpips: log_msg += (f', LPIPS:{batch_lp:.4f}(Avg:{avg_ep_lp:.4f})')
                log_msg += (f', LR:{current_lr:.1e}, T/Step:{avg_time:.3f}s '
                            f'(D:{data_load_time:.3f} T:{transfer_time:.3f} C:{compute_time:.3f})')
                logging.info(log_msg)

            if is_main_process() and preview_interval > 0 and global_step % preview_interval == 0:
                refresh = (fixed_src is None) or (config.logging.preview_refresh_rate > 0 and preview_count > 0 and (preview_count % config.logging.preview_refresh_rate == 0))
                if refresh:
                     new_src, new_dst, new_mask, new_auto = capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms, standard_transform, use_masks=use_masks, use_auto_mask=use_auto_mask, skip_empty=skip_empty, skip_empty_threshold=config.mask.skip_empty_threshold)
                     if new_src is not None: fixed_src, fixed_dst, fixed_mask, fixed_auto_mask = new_src, new_dst, new_mask, new_auto
                     elif fixed_src is None: logging.warning(f"S{global_step}: Preview refresh failed (no initial).")
                     else: logging.warning(f"S{global_step}: Preview refresh failed, using old.")
                if fixed_src is not None:
                    preview_ctx = PreviewContext(model=model, output_dir=config.data.output_dir, device=device, current_epoch=current_ep_idx, global_step=global_step, preview_save_count=preview_count, preview_refresh_rate=config.logging.preview_refresh_rate, use_mask_input=use_mask_input, use_bce_dice=use_bce_dice, use_amp=use_amp_eff, use_auto_mask=use_auto_mask, auto_mask_gamma=config.mask.auto_mask_gamma, diff_amplify=float(config.logging.diff_amplify))
                    save_previews(preview_ctx, fixed_src, fixed_dst, fixed_mask_batch=fixed_mask, fixed_auto_mask_batch=fixed_auto_mask); preview_count += 1
                elif preview_count == 0: logging.warning(f"S{global_step}: Skipping preview (no batch).")

            save_interval = config.saving.save_iterations_interval; save_now = (save_interval > 0 and global_step % save_interval == 0)
            epoch_end_now = (global_step % iter_epoch == 0) and global_step > 0
            if is_main_process() and (save_now or epoch_end_now):
                 ckpt_ep = global_step // iter_epoch; ep_ckpt_path = os.path.join(config.data.output_dir, f'{ckpt_prefix}_tunet_epoch_{ckpt_ep:09d}.pth') if epoch_end_now else None
                 latest_path = os.path.join(config.data.output_dir, f'{ckpt_prefix}_tunet_latest.pth'); cfg_dict = config_to_dict(config)
                 m_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                 # Save scaler state only if AMP was effectively used
                 scaler_state = scaler.state_dict() if use_amp_eff else None
                 ckpt_data = {'epoch': ckpt_ep, 'global_step': global_step, 'model_state_dict': m_state, 'optimizer_state_dict': optimizer.state_dict(),
                              'scaler_state_dict': scaler_state, 'config': cfg_dict, 'effective_model_size': eff_size, 'n_input_channels': n_input_ch, 'model_type': model_type}
                 try:
                     reason = "interval" if save_now else "epoch end"
                     torch.save(ckpt_data, latest_path)
                     logging.info(f"Saved latest checkpoint ({reason}) @ Step {global_step}")
                     if ep_ckpt_path:
                          torch.save(ckpt_data, ep_ckpt_path)
                          logging.info(f"Saved epoch checkpoint: {os.path.basename(ep_ckpt_path)}")
                          if hasattr(config.saving, 'keep_last_checkpoints') and config.saving.keep_last_checkpoints >= 0:
                              prune_checkpoints(config.data.output_dir, config.saving.keep_last_checkpoints, ckpt_prefix)
                 except Exception as e: logging.error(f"Checkpoint save failed @ Step {global_step}: {e}", exc_info=True)

            # --- Validation ---
            run_val_loss_now = False
            if is_main_process() and val_dataloader is not None:
                if epoch_end_now: run_val_loss_now = True
                elif val_interval > 0 and global_step % val_interval == 0: run_val_loss_now = True
            if run_val_loss_now:
                run_validation(model, val_dataloader, device, device_type, use_amp_eff, use_lpips, loss_fn_lpips, use_bce_dice, criterion_l1, current_ep_idx, global_step, lambda_lpips=config.training.lambda_lpips, use_mask_input=use_mask_input)
            # Val preview runs on its own schedule (epoch end or val_interval), even without dst
            run_val_preview_now = is_main_process() and has_val_preview and val_fixed_src is not None and (epoch_end_now or (val_interval > 0 and global_step % val_interval == 0))
            if run_val_preview_now:
                val_refresh = config.logging.preview_refresh_rate > 0 and val_preview_count > 0 and (val_preview_count % config.logging.preview_refresh_rate == 0)
                if val_refresh:
                    new_val_src, new_val_dst = capture_val_preview_batch(config, standard_transform)
                    if new_val_src is not None: val_fixed_src, val_fixed_dst = new_val_src, new_val_dst
                    else: logging.warning(f"S{global_step}: Val preview refresh failed, using old.")
                val_ctx = PreviewContext(model=model, output_dir=config.data.output_dir, device=device, current_epoch=current_ep_idx, global_step=global_step, use_mask_input=use_mask_input, use_bce_dice=use_bce_dice, use_amp=use_amp_eff, use_auto_mask=use_auto_mask, diff_amplify=float(config.logging.diff_amplify))
                save_val_previews(val_ctx, val_fixed_src, val_fixed_dst); val_preview_count += 1

            # --- Early Stopping / Plateau Detection ---
            if es_enabled and epoch_end_now and ep_steps > 0:
                ep_avg_loss = ep_l1 / ep_steps
                completed_epoch = global_step // iter_epoch
                # Update EMA of epoch loss
                if es_smoothed_loss is None:
                    es_smoothed_loss = ep_avg_loss
                else:
                    es_smoothed_loss = es_smoothing * es_smoothed_loss + (1 - es_smoothing) * ep_avg_loss
                # Track best smoothed loss
                if es_smoothed_loss < es_best_loss:
                    es_best_loss = es_smoothed_loss
                    es_best_epoch = completed_epoch
                    es_plateau_saved = False
                # Check for plateau
                epochs_since_best = completed_epoch - es_best_epoch
                if completed_epoch >= es_min_epochs and epochs_since_best >= es_patience:
                    if not es_plateau_saved:
                        # Save plateau checkpoint
                        plateau_path = os.path.join(config.data.output_dir, f'{ckpt_prefix}_tunet_plateau.pth')
                        try:
                            m_state_p = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                            scaler_state_p = scaler.state_dict() if use_amp_eff else None
                            cfg_dict_p = config_to_dict(config)
                            plateau_data = {'epoch': completed_epoch, 'global_step': global_step, 'model_state_dict': m_state_p, 'optimizer_state_dict': optimizer.state_dict(),
                                            'scaler_state_dict': scaler_state_p, 'config': cfg_dict_p, 'effective_model_size': eff_size, 'n_input_channels': n_input_ch, 'model_type': model_type}
                            torch.save(plateau_data, plateau_path)
                            logging.warning(f"PLATEAU DETECTED: No improvement for {epochs_since_best} epochs (best smoothed loss {es_best_loss:.6f} @ epoch {es_best_epoch}). Saved {os.path.basename(plateau_path)}")
                            es_plateau_saved = True
                        except Exception as e:
                            logging.error(f"Plateau checkpoint save failed: {e}", exc_info=True)
                    if es_stop:
                        logging.warning(f"EARLY STOPPING: Training halted after {completed_epoch} epochs (patience={es_patience}).")
                        shutdown_requested = True
                elif completed_epoch >= es_min_epochs and epochs_since_best >= es_patience // 2:
                    logging.info(f"[Plateau Watch] No improvement for {epochs_since_best}/{es_patience} epochs (best smoothed: {es_best_loss:.6f} @ ep {es_best_epoch})")

        # --- End of Training Loop ---
        training_successful = True
    except KeyboardInterrupt:
        if is_main_process(): logging.warning("KeyboardInterrupt. Shutting down...")
        if not shutdown_requested: handle_signal(signal.SIGINT, None)
    except Exception as loop_err:
        logging.error("FATAL Training loop error:", exc_info=True);
        shutdown_requested = True
    finally:
        max_steps_reached = max_steps > 0 and final_global_step >= max_steps
        if (shutdown_requested or max_steps_reached) and is_main_process():
            logging.info(f"Performing final checkpoint save (State @ Step {final_global_step})...")
            try:
                 final_ep = final_global_step // iter_epoch; latest_path = os.path.join(config.data.output_dir, f'{ckpt_prefix}_tunet_latest.pth'); cfg_dict = config_to_dict(config)
                 if model and optimizer and scaler:
                     m_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                     scaler_state = scaler.state_dict() if use_amp_eff else None # Check effective use
                     final_data = {'epoch': final_ep, 'global_step': final_global_step, 'model_state_dict': m_state, 'optimizer_state_dict': optimizer.state_dict(),
                                   'scaler_state_dict': scaler_state, 'config': cfg_dict, 'effective_model_size': eff_size, 'n_input_channels': n_input_ch, 'model_type': model_type}
                     torch.save(final_data, latest_path); logging.info(f"Final checkpoint saved: {latest_path}")
                 else: logging.error("Cannot save final checkpoint: Model/Optimizer/Scaler missing.")
            except Exception as e: logging.error(f"Final checkpoint save failed: {e}", exc_info=True)

        if is_main_process():
            status = "successfully" if training_successful and not shutdown_requested else "gracefully" if shutdown_requested else "with errors"
            logging.info(f"Training finished {status} at Step {final_global_step}.")
        cleanup_ddp();
        if is_main_process(): logging.info("Script finished.")


# --- Main Execution (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet via YAML config using DDP (Cross-Platform)')
    parser.add_argument('--config', type=str, required=True, help='Path to the USER YAML configuration file')
    parser.add_argument('--training.batch_size', type=int, dest='training_batch_size', default=None, help='Override training batch_size')
    parser.add_argument('--training.lr', type=float, dest='training_lr', default=None, help='Override training learning_rate')
    parser.add_argument('--data.output_dir', type=str, dest='data_output_dir', default=None, help='Override data output_dir')
    cli_args = parser.parse_args()

    # --- Initial Config Loading Logging (Simple Console) ---
    initial_log_level = logging.INFO
    initial_log_format = '%(asctime)s [CONFIG] %(message)s' # Simple format for config phase
    logging.basicConfig(level=initial_log_level, format=initial_log_format, handlers=[logging.StreamHandler()], force=True) # Use force=True to override potential previous basicConfigs
    logging.info("Script starting...")
    logging.info(f"User config: {cli_args.config}")

    # --- Load Base Config ---
    base_config_dict = {}
    try:
        user_cfg_abs = os.path.abspath(cli_args.config); user_cfg_dir = os.path.dirname(user_cfg_abs); script_dir = os.path.dirname(os.path.abspath(__file__))
        base_opts = [os.path.join(user_cfg_dir, 'base', 'base.yaml'), os.path.join(script_dir, 'base', 'base.yaml')]
        base_path = next((p for p in base_opts if os.path.exists(p)), None)
        if not base_path: raise FileNotFoundError(f"Base config ('base/base.yaml') not found in {base_opts}")
        with open(base_path, 'r') as f: base_config_dict = yaml.safe_load(f) or {}
        logging.info(f"Loaded base config: {base_path}") # Keep INFO
    except Exception as e: logging.error(f"CRITICAL: Failed loading base config: {e}. Exiting.", exc_info=True); exit(1) # Keep Error

    # --- Load User Config ---
    user_config_dict = {}
    try:
        user_cfg_abs = os.path.abspath(cli_args.config)
        if not os.path.exists(user_cfg_abs): raise FileNotFoundError(f"User config not found: {user_cfg_abs}")
        with open(user_cfg_abs, 'r') as f: user_config_dict = yaml.safe_load(f) or {}
        logging.info(f"Loaded user config: {user_cfg_abs}") # Keep INFO
    except Exception as e: logging.error(f"CRITICAL: Failed loading user config: {e}. Exiting.", exc_info=True); exit(1) # Keep Error

    # --- Merge & Overrides ---
    try: merged_dict = merge_configs(base_config_dict, user_config_dict)
    except Exception as e: logging.error(f"CRITICAL: Config merge failed: {e}. Exiting.", exc_info=True); exit(1) # Keep Error
    overrides = {}
    if cli_args.training_batch_size is not None: overrides.setdefault('training', {})['batch_size'] = cli_args.training_batch_size
    if cli_args.training_lr is not None: overrides.setdefault('training', {})['lr'] = cli_args.training_lr
    if cli_args.data_output_dir is not None: overrides.setdefault('data', {})['output_dir'] = os.path.abspath(cli_args.data_output_dir)
    if overrides: merged_dict = merge_configs(merged_dict, overrides); logging.info(f"Applied CLI overrides: {overrides}") # Keep INFO

    # --- Convert to Namespace & Validate ---
    try: config = dict_to_sns(merged_dict)
    except Exception as e: logging.error(f"CRITICAL: Config conversion failed: {e}. Exiting.", exc_info=True); exit(1) # Keep Error

    # --- Configuration Validation (Keep concise, errors are critical) ---
    missing, error_msgs = [], []
    req_keys = {'data': ['src_dir', 'dst_dir', 'output_dir', 'resolution', 'overlap_factor'], 'model': ['model_size_dims'], 'training': ['iterations_per_epoch', 'batch_size', 'lr', 'loss', 'lambda_lpips', 'use_amp'], 'saving': ['keep_last_checkpoints']}
    # Check required sections and keys
    for sec, keys in req_keys.items():
        sec_obj = getattr(config, sec, None)
        if sec_obj is None:
            error_msgs.append(f"Missing required section: '{sec}'")
            missing.append(sec)
            continue # Skip keys check if section is missing
        # If section exists, check its keys
        for k in keys:
            if getattr(sec_obj, k, None) is None:
                error_msgs.append(f"Missing/null required value: '{sec}.{k}'")
                missing.append(f"{sec}.{k}")


    # Defaulting (use simple getattr with default, avoid extra logs here)
    config.dataloader = getattr(config, 'dataloader', SimpleNamespace())
    raw_workers = getattr(config.dataloader, 'num_workers', -1)
    if raw_workers == -1:
        resolved_workers, auto_reason = auto_detect_num_workers(config.data.resolution, CURRENT_OS)
        config.dataloader.num_workers = resolved_workers
        logging.info(f"num_workers auto-detected: {resolved_workers} ({auto_reason})")
    else:
        config.dataloader.num_workers = raw_workers
        logging.info(f"num_workers from config: {raw_workers}")
    config.dataloader.prefetch_factor = getattr(config.dataloader, 'prefetch_factor', 2 if config.dataloader.num_workers > 0 else None)
    config.training.max_steps = getattr(config.training, 'max_steps', 0)
    config.training.finetune_from = getattr(config.training, 'finetune_from', None)
    config.saving = getattr(config, 'saving', SimpleNamespace()); config.saving.save_iterations_interval = getattr(config.saving, 'save_iterations_interval', 0)
    config.logging = getattr(config, 'logging', SimpleNamespace()); config.logging.log_interval = getattr(config.logging, 'log_interval', 50)
    config.logging.preview_batch_interval = getattr(config.logging, 'preview_batch_interval', 500); config.logging.preview_refresh_rate = getattr(config.logging, 'preview_refresh_rate', 5); config.logging.diff_amplify = getattr(config.logging, 'diff_amplify', 5.0)
    # Mask defaults
    config.data.mask_dir = getattr(config.data, 'mask_dir', None)
    config.mask = getattr(config, 'mask', SimpleNamespace())
    config.mask.use_mask_loss = getattr(config.mask, 'use_mask_loss', False)
    config.mask.mask_weight = getattr(config.mask, 'mask_weight', 10.0)
    config.mask.use_mask_input = getattr(config.mask, 'use_mask_input', False)
    config.mask.use_auto_mask = getattr(config.mask, 'use_auto_mask', False)
    config.mask.skip_empty_patches = getattr(config.mask, 'skip_empty_patches', False)
    config.mask.skip_empty_threshold = getattr(config.mask, 'skip_empty_threshold', 1.0)
    config.mask.auto_mask_gamma = getattr(config.mask, 'auto_mask_gamma', 1.0)
    # Model defaults
    config.model.model_type = getattr(config.model, 'model_type', 'unet')
    config.model.recurrence_steps = getattr(config.model, 'recurrence_steps', 2)
    # Progressive resolution & weighted loss defaults
    config.training.progressive_resolution = getattr(config.training, 'progressive_resolution', False)
    config.training.l1_weight = getattr(config.training, 'l1_weight', 1.0)
    config.training.l2_weight = getattr(config.training, 'l2_weight', 0.0)
    config.training.lpips_weight = getattr(config.training, 'lpips_weight', 0.1)
    # Validation defaults
    config.data.val_src_dir = getattr(config.data, 'val_src_dir', None)
    config.data.val_dst_dir = getattr(config.data, 'val_dst_dir', None)
    config.logging.val_interval = getattr(config.logging, 'val_interval', 0)
    # Final Value Checks
    if not missing:
        try: # Combine checks for brevity
            if config.training.iterations_per_epoch <= 0: error_msgs.append("train.iter_epoch<=0")
            if config.training.batch_size <= 0: error_msgs.append("train.batch_size<=0")
            if not isinstance(config.training.lr, (int, float)) or config.training.lr <= 0: error_msgs.append("train.lr<=0")
            if config.training.loss not in ['l1', 'l1+lpips', 'weighted', 'bce+dice']: error_msgs.append("train.loss invalid")
            if config.training.lambda_lpips < 0: error_msgs.append("train.lambda_lpips<0")
            if config.training.loss == 'weighted':
                if config.training.l1_weight < 0: error_msgs.append("train.l1_weight<0")
                if config.training.l2_weight < 0: error_msgs.append("train.l2_weight<0")
                if config.training.lpips_weight < 0: error_msgs.append("train.lpips_weight<0")
                if config.training.l1_weight == 0 and config.training.l2_weight == 0 and config.training.lpips_weight == 0:
                    error_msgs.append("weighted loss: at least one weight must be > 0")
            if not (0.0 <= config.data.overlap_factor < 1.0): error_msgs.append("data.overlap invalid")
            ds=16;
            if config.data.resolution<=0 or config.data.resolution % ds !=0: error_msgs.append(f"data.resolution not >0 & div by {ds}")
            if not os.path.isdir(config.data.src_dir): error_msgs.append(f"data.src_dir missing: {config.data.src_dir}") # Added path to msg
            if not os.path.isdir(config.data.dst_dir): error_msgs.append(f"data.dst_dir missing: {config.data.dst_dir}") # Added path to msg
            if config.model.model_size_dims <= 0: error_msgs.append("model.size<=0")
            if config.model.model_type not in ['unet', 'msrn']: error_msgs.append(f"model.model_type invalid: '{config.model.model_type}' (must be 'unet' or 'msrn')")
            if config.model.recurrence_steps < 1: error_msgs.append("model.recurrence_steps must be >= 1")
            if config.dataloader.num_workers < 0: error_msgs.append("loader.workers<0")
            if config.dataloader.num_workers>0 and config.dataloader.prefetch_factor is not None and config.dataloader.prefetch_factor<2: error_msgs.append("loader.prefetch<2 invalid")
            if config.saving.save_iterations_interval < 0: error_msgs.append("save.interval<0")
            if config.saving.keep_last_checkpoints < -1: error_msgs.append("save.keep<-1")
            if config.logging.log_interval <= 0: error_msgs.append("log.interval<=0")
            if config.logging.preview_batch_interval < 0: error_msgs.append("log.preview<0")
            if config.logging.preview_refresh_rate < 0: error_msgs.append("log.refresh<0")
            # Mask validation
            if config.mask.use_mask_loss or config.mask.use_mask_input:
                if not config.data.mask_dir: error_msgs.append("mask_dir required when mask features enabled")
                elif not os.path.isdir(config.data.mask_dir): error_msgs.append(f"data.mask_dir missing: {config.data.mask_dir}")
            if config.mask.mask_weight < 1.0: error_msgs.append("mask.mask_weight must be >= 1.0")
            # Validation dir checks
            if config.data.val_src_dir and not os.path.isdir(config.data.val_src_dir): error_msgs.append(f"data.val_src_dir missing: {config.data.val_src_dir}")
            if config.data.val_dst_dir and not os.path.isdir(config.data.val_dst_dir): error_msgs.append(f"data.val_dst_dir missing: {config.data.val_dst_dir}")
            if config.training.finetune_from and not os.path.isfile(config.training.finetune_from): error_msgs.append(f"training.finetune_from not found: {config.training.finetune_from}")
            # val_src_dir alone enables preview-only validation; both enables preview + loss
        except Exception as e: error_msgs.append(f"Validation error: {e}")
    # Abort on Errors
    if error_msgs or missing:
         print("\n" + "="*30 + " CONFIG ERRORS " + "="*30); all_errs = sorted(list(set(error_msgs + [f"Missing: {m}" for m in missing])))
         for i, msg in enumerate(all_errs): print(f"  {i+1}. {msg}")
         print("="*82 + "\nPlease fix config and restart."); exit(1)
    else: logging.info("[CONFIG] Configuration validated.") # Keep simple INFO confirmation

    # DDP Env Var Check (Keep as Warning)
    if os.environ.get("WORLD_SIZE", "1") != "1" and os.environ.get("LOCAL_RANK", None) is None:
        logging.warning("DDP run detected (WORLD_SIZE>1) but LOCAL_RANK not set. Use 'torchrun' or ensure env vars are set.") # Keep Warning

    # --- Store config file stem for checkpoint naming ---
    config._config_stem = os.path.splitext(os.path.basename(cli_args.config))[0]

    # --- Start Training ---
    try: train(config)
    except Exception as main_err: logging.error("Unhandled error outside train loop:", exc_info=True); cleanup_ddp(); exit(1) # Saiu
