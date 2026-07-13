"""Diagnose image files that fail to load or pair (EXR, DPX, TIFF).

When the dataset skips a file — a reshape error, a "Dst missing", a format the
loader can't decode — run this on the offending file(s) to see exactly why. It
dumps the format-specific structure (EXR windows/channels, DPX header, TIFF
dtype), then loads the file through TuNet's real loader to confirm it succeeds.

Project assumption: DPX and TIFF inputs are **16-bit sRGB**. Files that deviate
(wrong bit depth, unexpected channel layout) are flagged so mismatched exports
get caught before they silently degrade training.

Usage:
    python -m scripts.image_debug path/to/file.dpx
    python -m scripts.image_debug path/to/dir             # scans *.exr/*.dpx/*.tif(f)
    python -m scripts.image_debug "src/*.dpx" --quiet      # only report problems
"""

import os
import sys
import glob
import struct
import argparse

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

_SCANNED_EXTS = ('.exr', '.dpx', '.tif', '.tiff')
_ASSUMED_BITS = 16          # project assumption for DPX / TIFF
_PIXEL_TYPE_NAME = {0: 'UINT (32-bit)', 1: 'HALF (16-bit)', 2: 'FLOAT (32-bit)'}
_DPX_DESCRIPTORS = {50: 'RGB', 51: 'RGBA', 52: 'ABGR', 6: 'luma',
                    1: 'red', 2: 'green', 3: 'blue', 4: 'alpha'}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _samples_along(min_v, max_v, sampling):
    """Stored sample count along one axis for a (possibly subsampled) EXR channel."""
    return (max_v // sampling) - (-(-min_v // sampling)) + 1


def _fmt_factor(expected, actual):
    if actual == 0:
        return "actual is empty"
    if expected % actual == 0:
        return f"{expected // actual}x too few"
    if actual % expected == 0:
        return f"{actual // expected}x too many"
    return f"ratio {expected / actual:.4f}"


def _confirm_load(path, lines, problems):
    """Load through the real loader (linear path) and report the result."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from image_io.image_loader import load_image_linear
        arr = load_image_linear(path)
        lines.append(f"  load_image_linear -> shape {arr.shape}, dtype {arr.dtype} "
                     f"(min={float(arr.min()):.4g}, max={float(arr.max()):.4g})  OK")
    except Exception as e:
        problems.append(f"load_image_linear still fails: {type(e).__name__}: {e}")


def _emit(lines, problems, warnings, quiet):
    if quiet and not problems and not warnings:
        return
    print("\n".join(lines))
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    if warnings:
        print("  WARNINGS (deviates from 16-bit sRGB assumption):")
        for w in warnings:
            print(f"    - {w}")
    if not problems and not warnings:
        print("  No problems detected.")
    print()


# --------------------------------------------------------------------------- #
# EXR
# --------------------------------------------------------------------------- #

def diagnose_exr(path, quiet):
    lines = [f"=== {path}  [EXR] ==="]
    problems, warnings = [], []

    if not HAS_OPENEXR:
        lines.append("  OpenEXR not installed (pip install OpenEXR); limited info.")
        if HAS_CV2:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                problems.append("OpenCV could not read the file")
            else:
                lines.append(f"  OpenCV shape: {img.shape}, dtype: {img.dtype}")
        _confirm_load(path, lines, problems)
        _emit(lines, problems, warnings, quiet)
        return bool(problems)

    try:
        exr = OpenEXR.InputFile(path)
    except Exception as e:
        problems.append(f"OpenEXR could not open the file: {type(e).__name__}: {e}")
        _emit(lines, problems, warnings, quiet)
        return True

    header = exr.header()
    disp, dw = header['displayWindow'], header['dataWindow']
    disp_w = disp.max.x - disp.min.x + 1
    disp_h = disp.max.y - disp.min.y + 1
    dw_w = dw.max.x - dw.min.x + 1
    dw_h = dw.max.y - dw.min.y + 1

    lines.append(f"  compression : {header.get('compression')}")
    lines.append(f"  displayWindow: {disp_w} x {disp_h}")
    lines.append(f"  dataWindow   : {dw_w} x {dw_h}")
    if (dw.min.x, dw.min.y, dw_w, dw_h) != (disp.min.x, disp.min.y, disp_w, disp_h):
        lines.append("  note: dataWindow != displayWindow (loader pads to full frame)")

    full_px = dw_w * dw_h
    lines.append(f"  channels ({len(header['channels'])}): "
                 f"expected full-window samples = {dw_w}*{dw_h} = {full_px}")

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    for name, ch in header['channels'].items():
        xs = int(getattr(ch, 'xSampling', 1) or 1)
        ys = int(getattr(ch, 'ySampling', 1) or 1)
        ptype = getattr(getattr(ch, 'type', None), 'v', None)
        stored_w = _samples_along(dw.min.x, dw.max.x, xs)
        stored_h = _samples_along(dw.min.y, dw.max.y, ys)
        stored = stored_w * stored_h
        try:
            actual = np.frombuffer(exr.channel(name, FLOAT), dtype=np.float32).size
        except Exception as e:
            actual = -1
            problems.append(f"channel '{name}': could not read buffer: {e}")
        line = (f"    '{name}': {_PIXEL_TYPE_NAME.get(ptype, ptype)}, sampling=({xs},{ys}) "
                f"-> stored {stored_w}x{stored_h} = {stored}")
        if actual >= 0 and actual != stored:
            line += f"  ⚠ buffer has {actual}"
        lines.append(line)
        if actual >= 0 and actual != full_px:
            if actual == stored and (xs, ys) != (1, 1):
                problems.append(f"channel '{name}' is SUBSAMPLED ({xs}x{ys}); buffer {actual} vs "
                                f"full window {full_px} ({_fmt_factor(full_px, actual)}). "
                                f"Loader upsamples it.")
            else:
                problems.append(f"channel '{name}' buffer {actual} != full window {full_px} "
                                f"({_fmt_factor(full_px, actual)}); data window may not match "
                                f"stored resolution.")

    _confirm_load(path, lines, problems)
    _emit(lines, problems, warnings, quiet)
    return bool(problems)


# --------------------------------------------------------------------------- #
# DPX
# --------------------------------------------------------------------------- #

def diagnose_dpx(path, quiet):
    lines = [f"=== {path}  [DPX] ==="]
    problems, warnings = [], []

    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) < 812:
        problems.append(f"file truncated or too small ({len(raw)} bytes)")
        _emit(lines, problems, warnings, quiet)
        return True

    magic = raw[:4]
    if magic == b'SDPX':
        end, endname = '>', 'big-endian'
    elif magic == b'XPDS':
        end, endname = '<', 'little-endian'
    else:
        problems.append(f"bad magic {magic!r} — not a DPX file")
        _emit(lines, problems, warnings, quiet)
        return True

    u16 = lambda o: struct.unpack_from(end + 'H', raw, o)[0]
    u32 = lambda o: struct.unpack_from(end + 'I', raw, o)[0]

    width, height = u32(772), u32(776)
    descriptor, bit_depth = raw[800], raw[803]
    packing, encoding = u16(804), u16(806)
    desc_name = _DPX_DESCRIPTORS.get(descriptor, f'unknown({descriptor})')
    enc_name = {0: 'none', 1: 'RLE'}.get(encoding, f'unknown({encoding})')
    pack_name = {0: 'packed', 1: 'Method A (filled)', 2: 'Method B'}.get(packing, f'unknown({packing})')

    lines.append(f"  endianness : {endname}")
    lines.append(f"  dimensions : {width} x {height}")
    lines.append(f"  descriptor : {descriptor} ({desc_name})")
    lines.append(f"  bit depth  : {bit_depth}")
    lines.append(f"  packing    : {packing} ({pack_name})   encoding: {encoding} ({enc_name})")

    if encoding == 1:
        problems.append("RLE-compressed DPX is not supported by the loader")
    if descriptor not in (50, 51, 52, 1, 2, 3, 4, 6):
        problems.append(f"descriptor {descriptor} not supported (need RGB/RGBA/luma)")
    if bit_depth != _ASSUMED_BITS:
        warnings.append(f"bit depth is {bit_depth}-bit, project assumes {_ASSUMED_BITS}-bit sRGB "
                        f"(loader still decodes it, but the export may be inconsistent with the rest)")

    _confirm_load(path, lines, problems)
    _emit(lines, problems, warnings, quiet)
    return bool(problems or warnings)


# --------------------------------------------------------------------------- #
# TIFF
# --------------------------------------------------------------------------- #

def diagnose_tiff(path, quiet):
    lines = [f"=== {path}  [TIFF] ==="]
    problems, warnings = [], []

    if not HAS_CV2:
        warnings.append("OpenCV not available; loader will fall back to PIL (8-bit only)")
        _confirm_load(path, lines, problems)
        _emit(lines, problems, warnings, quiet)
        return bool(problems or warnings)

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        problems.append("OpenCV could not decode this TIFF (unusual compression/tiling?); "
                        "loader falls back to PIL, which is 8-bit only")
    else:
        ch = 1 if img.ndim == 2 else img.shape[2]
        bits = {np.uint8: 8, np.uint16: 16, np.float32: 32, np.float64: 64}.get(img.dtype.type, '?')
        lines.append(f"  dimensions : {img.shape[1]} x {img.shape[0]}")
        lines.append(f"  channels   : {ch}")
        lines.append(f"  dtype      : {img.dtype} ({bits}-bit)")
        lines.append(f"  value range: [{float(img.min()):.4g}, {float(img.max()):.4g}]")
        if img.dtype == np.uint8:
            warnings.append("dtype is 8-bit; project assumes 16-bit sRGB (no extra precision to train on)")
        elif img.dtype in (np.float32, np.float64):
            warnings.append(f"dtype is float ({bits}-bit); project assumes 16-bit *integer* sRGB "
                            f"— float TIFFs are treated as scene-linear, not sRGB code values")
        elif img.dtype != np.uint16:
            warnings.append(f"dtype is {img.dtype}; project assumes 16-bit sRGB")

    _confirm_load(path, lines, problems)
    _emit(lines, problems, warnings, quiet)
    return bool(problems or warnings)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def diagnose_file(path, quiet=False):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.exr':
        return diagnose_exr(path, quiet)
    if ext == '.dpx':
        return diagnose_dpx(path, quiet)
    if ext in ('.tif', '.tiff'):
        return diagnose_tiff(path, quiet)
    print(f"=== {path} ===\n  Unsupported extension '{ext}' (handles .exr/.dpx/.tif/.tiff)\n")
    return True


def _collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in _SCANNED_EXTS:
                files.extend(sorted(glob.glob(os.path.join(p, '*' + ext))))
        elif any(c in p for c in '*?[]'):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="Diagnose EXR/DPX/TIFF load failures.")
    ap.add_argument('paths', nargs='+', help="file(s), a directory, or a glob pattern")
    ap.add_argument('--quiet', action='store_true', help="only print files with problems/warnings")
    args = ap.parse_args()

    files = _collect(args.paths)
    if not files:
        print("No EXR/DPX/TIFF files matched.")
        return 1

    print(f"Assumption: DPX and TIFF are 16-bit sRGB. Scanning {len(files)} file(s).\n")
    flagged = sum(diagnose_file(f, quiet=args.quiet) for f in files)
    print(f"Scanned {len(files)} file(s); {flagged} flagged.")
    return 1 if flagged else 0


if __name__ == '__main__':
    sys.exit(main())
