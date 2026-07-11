"""Diagnose EXR files that fail to load (reshape mismatches, subsampling, bad windows).

When the dataset skips a file with an error like
``cannot reshape array of size N into shape (H, W)``, run this on the offending
file(s) to see exactly why. It dumps the display/data windows, every channel's
pixel type and sampling, the expected-vs-actual sample counts, and then tries to
load the file through TuNet's real loader so you can confirm it now succeeds.

Usage:
    python -m scripts.exr_debug path/to/file.exr
    python -m scripts.exr_debug path/to/dir            # scans *.exr in the dir
    python -m scripts.exr_debug "renders/*.exr" --quiet # only report problems
"""

import os
import sys
import glob
import argparse

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

import numpy as np


_PIXEL_TYPE_NAME = {0: 'UINT (32-bit)', 1: 'HALF (16-bit)', 2: 'FLOAT (32-bit)'}


def _samples_along(min_v, max_v, sampling):
    """Stored sample count along one axis for a channel with the given sampling."""
    return (max_v // sampling) - (-(-min_v // sampling)) + 1


def _fmt_factor(expected, actual):
    if actual == 0:
        return "actual is empty"
    if expected % actual == 0:
        return f"{expected // actual}x too few"
    if actual % expected == 0:
        return f"{actual // expected}x too many"
    return f"ratio {expected / actual:.4f}"


def diagnose_file(path, quiet=False):
    """Print a diagnostic report for one EXR. Returns True if a problem was found."""
    header_lines = [f"=== {path} ==="]
    problems = []

    if not HAS_OPENEXR:
        header_lines.append("  OpenEXR not installed — install with: pip install OpenEXR")
        header_lines.append("  (falling back to OpenCV, which only reads the data window)")
        import cv2
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            problems.append("OpenCV could not read the file at all")
        else:
            header_lines.append(f"  OpenCV shape: {img.shape}, dtype: {img.dtype}")
        _emit(header_lines, problems, quiet)
        return bool(problems)

    try:
        exr = OpenEXR.InputFile(path)
    except Exception as e:
        problems.append(f"OpenEXR could not open the file: {type(e).__name__}: {e}")
        _emit(header_lines, problems, quiet)
        return True

    header = exr.header()
    disp = header['displayWindow']
    dw = header['dataWindow']
    disp_w = disp.max.x - disp.min.x + 1
    disp_h = disp.max.y - disp.min.y + 1
    dw_w = dw.max.x - dw.min.x + 1
    dw_h = dw.max.y - dw.min.y + 1

    header_lines.append(f"  compression : {header.get('compression')}")
    header_lines.append(f"  displayWindow: {disp_w} x {disp_h}  "
                        f"[({disp.min.x},{disp.min.y})..({disp.max.x},{disp.max.y})]")
    header_lines.append(f"  dataWindow   : {dw_w} x {dw_h}  "
                        f"[({dw.min.x},{dw.min.y})..({dw.max.x},{dw.max.y})]")
    if (dw.min.x, dw.min.y, dw_w, dw_h) != (disp.min.x, disp.min.y, disp_w, disp_h):
        header_lines.append("  note: dataWindow != displayWindow (loader pads to full frame)")

    full_px = dw_w * dw_h
    header_lines.append(f"  channels ({len(header['channels'])}): "
                        f"expected full-window samples = {dw_w}*{dw_h} = {full_px}")

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    for name, ch in header['channels'].items():
        xs = int(getattr(ch, 'xSampling', 1) or 1)
        ys = int(getattr(ch, 'ySampling', 1) or 1)
        ptype = getattr(getattr(ch, 'type', None), 'v', None)
        ptype_name = _PIXEL_TYPE_NAME.get(ptype, f'type={ptype}')
        stored_w = _samples_along(dw.min.x, dw.max.x, xs)
        stored_h = _samples_along(dw.min.y, dw.max.y, ys)
        stored = stored_w * stored_h

        # actual buffer length when read as FLOAT (what the loader does)
        try:
            actual = np.frombuffer(exr.channel(name, FLOAT), dtype=np.float32).size
        except Exception as e:
            actual = -1
            problems.append(f"channel '{name}': could not read buffer: {e}")

        line = (f"    '{name}': {ptype_name}, sampling=({xs},{ys}) -> "
                f"stored grid {stored_w}x{stored_h} = {stored}")
        if actual >= 0 and actual != stored:
            line += f"  ⚠ buffer has {actual}"
        header_lines.append(line)

        if actual >= 0 and actual != full_px:
            if actual == stored and (xs, ys) != (1, 1):
                problems.append(
                    f"channel '{name}' is SUBSAMPLED ({xs}x{ys}); buffer {actual} vs "
                    f"full window {full_px} ({_fmt_factor(full_px, actual)}). "
                    f"This is the reshape-mismatch cause. Loader now upsamples it.")
            else:
                problems.append(
                    f"channel '{name}' buffer {actual} != full window {full_px} "
                    f"({_fmt_factor(full_px, actual)}) and sampling doesn't explain it "
                    f"— likely a data window that doesn't match stored resolution.")

    # Confirm the real loader now handles it.
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from image_io.image_loader import load_exr_full_frame
        loaded = load_exr_full_frame(path)
        header_lines.append(f"  load_exr_full_frame -> shape {loaded.shape}, dtype {loaded.dtype} "
                            f"(min={float(loaded.min()):.4g}, max={float(loaded.max()):.4g})  OK")
    except Exception as e:
        problems.append(f"load_exr_full_frame still fails: {type(e).__name__}: {e}")

    _emit(header_lines, problems, quiet)
    return bool(problems)


def _emit(header_lines, problems, quiet):
    if quiet and not problems:
        return
    print("\n".join(header_lines))
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  No loader problems detected.")
    print()


def _collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.exr'))))
        elif any(ch in p for ch in '*?[]'):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="Diagnose EXR load/reshape failures.")
    ap.add_argument('paths', nargs='+', help="EXR file(s), a directory, or a glob pattern")
    ap.add_argument('--quiet', action='store_true', help="only print files that have problems")
    args = ap.parse_args()

    files = _collect(args.paths)
    if not files:
        print("No EXR files matched.")
        return 1
    if not HAS_OPENEXR:
        print("WARNING: OpenEXR is not installed; diagnostics are limited. "
              "Install it with: pip install OpenEXR\n")

    problem_count = 0
    for f in files:
        if diagnose_file(f, quiet=args.quiet):
            problem_count += 1

    print(f"Scanned {len(files)} file(s); {problem_count} with problems.")
    return 1 if problem_count else 0


if __name__ == '__main__':
    sys.exit(main())
