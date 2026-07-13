import os
import struct
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


def _exr_samples_along(min_v, max_v, sampling):
    """Number of stored samples along one axis for a (possibly subsampled) EXR channel.

    Mirrors OpenEXR's storage semantics: samples sit at coordinates divisible by
    ``sampling`` within the inclusive ``[min_v, max_v]`` data-window range.
    """
    return (max_v // sampling) - (-(-min_v // sampling)) + 1


def _read_exr_channel_2d(exr_file, header, ch_name, dw):
    """Read one EXR channel as a full-resolution (dw_height, dw_width) float32 array.

    Channels may be *subsampled* (``xSampling``/``ySampling`` > 1), in which case the
    raw buffer holds fewer samples than the data window's pixel count. Reshaping the
    buffer straight to the data-window size then fails with
    ``cannot reshape array of size N into shape (H, W)``. Here we reshape to the actual
    stored sample grid and nearest-neighbour upsample back to full resolution.
    """
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    dw_width = dw.max.x - dw.min.x + 1
    dw_height = dw.max.y - dw.min.y + 1

    data = exr_file.channel(ch_name, FLOAT)
    arr = np.frombuffer(data, dtype=np.float32)

    # Fast path: buffer already matches the full data window.
    if arr.size == dw_width * dw_height:
        return arr.reshape((dw_height, dw_width))

    # Subsampled channel: reshape to the stored grid, then upsample.
    ch = header['channels'].get(ch_name)
    xs = int(getattr(ch, 'xSampling', 1) or 1)
    ys = int(getattr(ch, 'ySampling', 1) or 1)
    sw = _exr_samples_along(dw.min.x, dw.max.x, xs)
    sh = _exr_samples_along(dw.min.y, dw.max.y, ys)

    if (xs, ys) == (1, 1) or arr.size != sw * sh:
        # Header sampling doesn't explain the mismatch (e.g. a data window that
        # doesn't match the stored channel resolution). Infer a uniform integer
        # downscale factor so the file is still usable instead of skipped.
        for f in range(2, 9):
            fw = -(-dw_width // f)
            fh = -(-dw_height // f)
            if arr.size == fw * fh:
                xs = ys = f
                sw, sh = fw, fh
                break
        else:
            raise ValueError(
                f"channel '{ch_name}' has {arr.size} samples, incompatible with data "
                f"window {dw_width}x{dw_height} ({dw_width * dw_height} px)"
            )

    arr = arr.reshape((sh, sw))
    if (xs, ys) != (1, 1):
        arr = np.repeat(np.repeat(arr, ys, axis=0), xs, axis=1)[:dw_height, :dw_width]
    return arr


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

        channels = ['R', 'G', 'B']
        available = list(header['channels'].keys())
        if not all(c in available for c in channels):
            arr = _read_exr_channel_2d(exr_file, header, available[0], dw)
            img_channels = [arr, arr, arr]
        else:
            img_channels = [_read_exr_channel_2d(exr_file, header, c, dw) for c in channels]
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


def load_dpx(image_path):
    """Load a DPX (SMPTE 268M) file as a float32 (H, W, 3) RGB array in [0, 1].

    Self-contained reader — DPX is a fixed-layout format, so no external library is
    required. Handles the common uncompressed flavours produced by film-scan / VFX
    pipelines: 8-, 10-, 12- and 16-bit, RGB / RGBA / single-channel.

    Values are the raw normalized code values (log DPX is *not* linearized here) so
    the file is treated like any other non-EXR image by the rest of the pipeline.
    Raises ValueError on formats we don't decode (e.g. RLE compression), so the
    dataset skips the file with a clear reason instead of crashing.
    """
    with open(image_path, 'rb') as f:
        raw = f.read()
    if len(raw) < 812:
        raise ValueError(f"DPX truncated or too small ({len(raw)} bytes): {image_path}")

    magic = raw[:4]
    if magic == b'SDPX':
        end = '>'          # big-endian
    elif magic == b'XPDS':
        end = '<'          # little-endian
    else:
        raise ValueError(f"Not a DPX file (bad magic {magic!r}): {image_path}")

    u16 = lambda off: struct.unpack_from(end + 'H', raw, off)[0]
    u32 = lambda off: struct.unpack_from(end + 'I', raw, off)[0]

    data_off_file = u32(4)          # generic header: offset to image data
    width = u32(772)
    height = u32(776)
    descriptor = raw[800]           # image element 0 descriptor
    bit_depth = raw[803]
    encoding = u16(806)             # 0 = none, 1 = RLE
    elem_off = u32(808)
    data_off = elem_off if elem_off not in (0, 0xFFFFFFFF) else data_off_file

    if width == 0 or height == 0:
        raise ValueError(f"DPX has zero dimensions ({width}x{height}): {image_path}")
    if encoding == 1:
        raise ValueError(f"RLE-compressed DPX not supported: {image_path}")

    if descriptor == 50:            # RGB
        ch = 3
    elif descriptor in (51, 52):    # RGBA / ABGR
        ch = 4
    elif descriptor in (1, 2, 3, 4, 6):  # single component (red/green/blue/alpha/luma)
        ch = 1
    else:
        raise ValueError(f"Unsupported DPX descriptor {descriptor} (need RGB/RGBA/luma): {image_path}")

    buf = raw[data_off:]

    if bit_depth == 10:
        if ch != 3:
            raise ValueError(f"10-bit DPX only supported for RGB, got descriptor {descriptor}: {image_path}")
        need = width * height
        if len(buf) < need * 4:
            raise ValueError(f"DPX pixel data short: need {need * 4} bytes, have {len(buf)} "
                             f"(line padding or non-standard packing?): {image_path}")
        words = np.frombuffer(buf, dtype=np.dtype(end + 'u4'), count=need).astype(np.uint32)
        r = (words >> 22) & 0x3FF
        g = (words >> 12) & 0x3FF
        b = (words >> 2) & 0x3FF
        arr = np.stack([r, g, b], axis=-1).astype(np.float32).reshape(height, width, 3) / 1023.0
    else:
        if bit_depth == 8:
            dt, maxv = np.uint8, 255.0
        elif bit_depth == 12:
            dt, maxv = np.dtype(end + 'u2'), 4095.0   # Method A: 12 bits right-justified in 16
        elif bit_depth == 16:
            dt, maxv = np.dtype(end + 'u2'), 65535.0
        else:
            raise ValueError(f"Unsupported DPX bit depth {bit_depth}: {image_path}")
        need = width * height * ch
        itemsize = np.dtype(dt).itemsize
        if len(buf) < need * itemsize:
            raise ValueError(f"DPX pixel data short: need {need * itemsize} bytes, have {len(buf)}: {image_path}")
        samples = np.frombuffer(buf, dtype=dt, count=need).astype(np.float32)
        if bit_depth == 12:
            samples = np.mod(samples, 4096.0)         # mask to low 12 bits
        arr = (samples / maxv).reshape(height, width, ch)

    if ch == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif ch == 4:
        if descriptor == 52:        # ABGR -> RGB
            arr = arr[:, :, 3:0:-1]
        else:                       # RGBA -> RGB
            arr = arr[:, :, :3]
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_tiff(image_path):
    """Load a TIFF as float32 (H, W, 3) RGB, preserving bit depth.

    Reads via OpenCV so 16-bit and 32-bit-float TIFFs keep their precision — PIL's
    ``convert('RGB')`` would collapse them to 8-bit. Integer TIFFs are normalized to
    [0, 1] by their bit depth; float TIFFs are returned as-is (values may exceed 1.0,
    like EXR). Falls back to PIL for anything OpenCV can't decode.
    """
    img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        pil = Image.open(image_path).convert('RGB')
        return np.array(pil, dtype=np.float32) / 255.0

    if img.dtype == np.uint8:
        maxv = 255.0
    elif img.dtype == np.uint16:
        maxv = 65535.0
    elif img.dtype in (np.float32, np.float64):
        maxv = 1.0                       # already normalized / scene-linear; preserve range
    else:
        maxv = float(np.iinfo(img.dtype).max)

    arr = img.astype(np.float32)
    if arr.ndim == 2:                    # grayscale -> RGB
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[2] == 4:              # BGRA -> RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    else:                                # BGR (>=3 channels) -> RGB
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)

    if maxv != 1.0:
        arr = arr / maxv
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_image_any_format(image_path):
    """Load an image in any format including EXR. Returns a PIL Image in RGB mode."""
    _, ext = os.path.splitext(image_path.lower())
    if ext == '.exr':
        img_float = load_exr_full_frame(image_path)
        np.nan_to_num(img_float, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        img_8bit = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_8bit, mode='RGB')
    elif ext == '.dpx':
        img_float = load_dpx(image_path)  # (H,W,3) float32 in [0,1]
        img_8bit = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_8bit, mode='RGB')
    elif ext in ('.tif', '.tiff'):
        img_float = load_tiff(image_path)  # (H,W,3) float32, bit-depth preserved
        img_8bit = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_8bit, mode='RGB')
    else:
        return Image.open(image_path).convert('RGB')


def load_image_srgb(image_path):
    """Load an image as float32 (H, W, 3) RGB in [0, 1] at full bit depth.

    The sRGB / display-referred counterpart to load_image_linear: values are the
    normalized code values clamped to [0, 1] (no log encoding). Unlike the legacy
    8-bit PIL path, this preserves 16-bit precision for DPX / TIFF / EXR sources —
    which matters for VFX, where banding from an 8-bit round-trip is unacceptable.
    """
    _, ext = os.path.splitext(image_path.lower())
    if ext == '.exr':
        img = load_exr_full_frame(image_path)
        np.nan_to_num(img, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(img, 0.0, 1.0)
    elif ext == '.dpx':
        img = load_dpx(image_path)
        return np.clip(img, 0.0, 1.0)
    elif ext in ('.tif', '.tiff'):
        img = load_tiff(image_path)
        np.nan_to_num(img, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(img, 0.0, 1.0)
    else:
        img = Image.open(image_path).convert('RGB')
        return np.ascontiguousarray(np.asarray(img, dtype=np.float32) / 255.0)


def load_image_linear(image_path):
    """Load an image preserving float32 linear values (H,W,3).

    EXR, DPX and TIFF are read at full bit depth (16-bit / 32-bit-float preserved).
    Other formats are read as uint8 and converted to [0,1] float32 (assumed to
    already be linear or close enough for the user's purposes).
    """
    _, ext = os.path.splitext(image_path.lower())
    if ext == '.exr':
        img_float = load_exr_full_frame(image_path)
        np.nan_to_num(img_float, copy=False, nan=0.0, posinf=10.0, neginf=0.0)
        return np.clip(img_float, 0.0, None)  # allow >1.0, clamp negatives
    elif ext == '.dpx':
        img_float = load_dpx(image_path)  # (H,W,3) float32 in [0,1]
        np.nan_to_num(img_float, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(img_float, 0.0, None)
    elif ext in ('.tif', '.tiff'):
        img_float = load_tiff(image_path)  # (H,W,3) float32, bit-depth preserved
        np.nan_to_num(img_float, copy=False, nan=0.0, posinf=10.0, neginf=0.0)
        return np.clip(img_float, 0.0, None)  # allow >1.0 for float TIFFs, clamp negatives
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

            available = list(header['channels'].keys())
            ch_name = 'Y' if 'Y' in available else available[0]
            mask_data = _read_exr_channel_2d(exr_file, header, ch_name, dw)

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
    elif ext in ('.tif', '.tiff'):
        rgb = load_tiff(image_path)          # (H,W,3) float32, bit-depth preserved
        return np.clip(rgb.mean(axis=2), 0.0, 1.0)
    else:
        img = Image.open(image_path).convert('L')
        return np.array(img, dtype=np.float32) / 255.0
