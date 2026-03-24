import os
import logging

import numpy as np
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2

import torch

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
NORM_STD = torch.tensor([0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Log-space encoding for linear HDR data
# ---------------------------------------------------------------------------

def linear_to_log(x):
    """Compress HDR linear values via log1p. Works on numpy arrays or torch tensors."""
    if isinstance(x, np.ndarray):
        return np.log1p(x)
    return torch.log1p(x)


def log_to_linear(x):
    """Recover HDR linear values via expm1. Works on numpy arrays or torch tensors."""
    if isinstance(x, np.ndarray):
        return np.expm1(x)
    return torch.expm1(x)


# ---------------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------------

def denormalize(tensor):
    """Denormalize a tensor from [-1,1] back to [0,1] (sRGB path)."""
    if tensor is None:
        return None
    try:
        mean = NORM_MEAN.to(tensor.device).view(1, 3, 1, 1)
        std = NORM_STD.to(tensor.device).view(1, 3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)
    except Exception as e:
        logging.error(f"Denormalize error: {e}")
        return tensor


def denormalize_linear(tensor):
    """Denormalize from normalized log-space back to scene-linear HDR values.

    Pipeline: [-1,1] → un-normalize to log-space → expm1 → linear.
    No upper clamp — HDR values above 1.0 are preserved.
    """
    if tensor is None:
        return None
    try:
        mean = NORM_MEAN.to(tensor.device).view(1, 3, 1, 1)
        std = NORM_STD.to(tensor.device).view(1, 3, 1, 1)
        log_space = tensor * std + mean
        return torch.expm1(log_space.clamp(min=0.0))
    except Exception as e:
        logging.error(f"Denormalize linear error: {e}")
        return tensor


def load_exr_full_frame(image_path):
    """Load an EXR file as a float32 RGB numpy array using the displayWindow.

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

        if (dw.min.x == disp.min.x and dw.min.y == disp.min.y
                and dw_width == disp_width and dw_height == disp_height):
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
        img_float = load_exr_full_frame(image_path)
        np.nan_to_num(img_float, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        img_8bit = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_8bit, mode='RGB')
    else:
        return Image.open(image_path).convert('RGB')


def load_image_linear(image_path):
    """Load an image preserving float32 linear values (H,W,3).

    EXR files are kept as scene-linear float32.  Other formats are read as
    uint8 and converted to [0,1] float32 (assumed to already be linear or
    close enough for the user's purposes).
    """
    _, ext = os.path.splitext(image_path.lower())
    if ext == '.exr':
        img_float = load_exr_full_frame(image_path)
        np.nan_to_num(img_float, copy=False, nan=0.0, posinf=10.0, neginf=0.0)
        return np.clip(img_float, 0.0, None)  # allow >1.0, clamp negatives
    else:
        img = Image.open(image_path).convert('RGB')
        return np.array(img, dtype=np.float32) / 255.0


def save_exr(image_np, output_path):
    """Save a float32 (H,W,3) RGB numpy array as an EXR file via OpenCV."""
    bgr = cv2.cvtColor(image_np.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])


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

            if (dw.min.x == disp.min.x and dw.min.y == disp.min.y
                    and dw_width == disp_width and dw_height == disp_height):
                return np.clip(mask_data, 0.0, 1.0)

            mask = np.zeros((disp_height, disp_width), dtype=np.float32)
            x_off = dw.min.x - disp.min.x
            y_off = dw.min.y - disp.min.y
            mask[y_off:y_off + dw_height, x_off:x_off + dw_width] = mask_data
            return np.clip(mask, 0.0, 1.0)

        img_cv = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img_cv is None:
            raise ValueError(f"Failed to load mask EXR: {image_path}. Install OpenEXR: pip install OpenEXR")
        logging.warning(f"Loaded mask EXR with OpenCV (no displayWindow support): {image_path}")
        if len(img_cv.shape) == 3:
            mask = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_cv.shape[2] >= 3 else img_cv[:, :, 0]
        else:
            mask = img_cv
        return np.clip(mask.astype(np.float32), 0.0, 1.0)
    else:
        img = Image.open(image_path).convert('L')
        return np.array(img, dtype=np.float32) / 255.0
