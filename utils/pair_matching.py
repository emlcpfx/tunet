"""Flexible source/destination file pair matching.

Supports:
  1. Exact name match (101.exr -> 101.exr)
  2. Cross-extension match (101.exr -> 101.png)
  3. Suffix-swapped match (101_src.exr -> 101_dst.exr, 101_source.png -> 101_dest.png)
"""

import os
import re

# Suffixes that identify a file as "source" or "destination"
_SRC_SUFFIXES = ('_src', '_source', '_in', '_input')
_DST_SUFFIXES = ('_dst', '_dest', '_destination', '_out', '_output', '_target')

_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif', '.exr', '.dpx')

# Pre-compiled pattern: match any src suffix at end of stem (case-insensitive)
_SRC_SUFFIX_RE = re.compile(
    r'(?i)(' + '|'.join(re.escape(s) for s in _SRC_SUFFIXES) + r')$'
)


def _strip_src_suffix(stem):
    """Strip a source suffix from a filename stem. Returns (core, suffix) or (stem, None)."""
    m = _SRC_SUFFIX_RE.search(stem)
    if m:
        return stem[:m.start()], m.group()
    return stem, None


def find_dst_file(src_path, dst_dir):
    """Find the destination file that pairs with a source file.

    Returns the full destination path, or None if no match found.

    Search order:
      1. Exact basename match
      2. Same stem, different image extension
      3. Swap src-suffix for each dst-suffix (exact ext, then other exts)
    """
    basename = os.path.basename(src_path)
    stem, src_ext = os.path.splitext(basename)

    # Some pipelines emit a stray trailing dot in the frame name (e.g. "shot_01001..dpx"
    # -> stem "shot_01001."). Try both the literal stem and a dot-trimmed variant so a
    # source's dot convention doesn't have to match the destination's exactly.
    stems = [stem]
    trimmed = stem.rstrip('.')
    if trimmed != stem:
        stems.append(trimmed)

    # 1 & 2: Try exact name, then cross-extension (for each stem variant)
    for s in stems:
        for ext in _unique_exts(src_ext):
            candidate = os.path.join(dst_dir, s + ext)
            if os.path.exists(candidate):
                return candidate

    # 3: Suffix swap — only if the stem ends with a recognized src suffix.
    # A stray trailing dot may sit before the src suffix; strip it for matching but
    # also try re-appending it so either dot convention on the destination is found.
    had_trailing_dot = stem != trimmed
    for s in stems:
        core, matched_suffix = _strip_src_suffix(s)
        if matched_suffix is None:
            continue
        for dst_sfx in _DST_SUFFIXES:
            dst_stems = [core + dst_sfx]
            if had_trailing_dot:
                dst_stems.append(core + dst_sfx + '.')
            for dst_stem in dst_stems:
                for ext in _unique_exts(src_ext):
                    candidate = os.path.join(dst_dir, dst_stem + ext)
                    if os.path.exists(candidate):
                        return candidate

    return None


def _unique_exts(primary_ext):
    """Yield the primary extension first, then other image extensions (no duplicates)."""
    yield primary_ext
    for ext in _IMAGE_EXTS:
        if ext != primary_ext:
            yield ext
