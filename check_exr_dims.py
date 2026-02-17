#!/usr/bin/env python3
import os
import sys

# Enable OpenEXR support
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False
    print("WARNING: cv2 not available")

try:
    import OpenEXR
    import Imath
    has_openexr = True
except ImportError:
    has_openexr = False
    print("WARNING: OpenEXR not available")

src_file = '/home/elevy/Downloads/ACTS HANDS INPUT/ACTS_src/A001_A008_0423HF_001.02476662.exr'
dst_file = '/home/elevy/Downloads/ACTS HANDS INPUT/ACTS_dst/A001_A008_0423HF_001.02476662.exr'

print("\n=== Checking SRC file ===")
print(f"File: {src_file}")

if has_cv2:
    img = cv2.imread(src_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is not None:
        print(f"OpenCV shape: {img.shape}")
        print(f"OpenCV dimensions (H, W): ({img.shape[0]}, {img.shape[1]})")
        print(f"OpenCV dimensions (W, H): ({img.shape[1]}, {img.shape[0]})")
    else:
        print("OpenCV failed to load")

if has_openexr:
    try:
        exr_file = OpenEXR.InputFile(src_file)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        print(f"OpenEXR dimensions (W, H): ({width}, {height})")
    except Exception as e:
        print(f"OpenEXR error: {e}")

print("\n=== Checking DST file ===")
print(f"File: {dst_file}")

if has_cv2:
    img = cv2.imread(dst_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is not None:
        print(f"OpenCV shape: {img.shape}")
        print(f"OpenCV dimensions (H, W): ({img.shape[0]}, {img.shape[1]})")
        print(f"OpenCV dimensions (W, H): ({img.shape[1]}, {img.shape[0]})")
    else:
        print("OpenCV failed to load")

if has_openexr:
    try:
        exr_file = OpenEXR.InputFile(dst_file)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        print(f"OpenEXR dimensions (W, H): ({width}, {height})")
    except Exception as e:
        print(f"OpenEXR error: {e}")
