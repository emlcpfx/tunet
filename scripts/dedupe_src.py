#!/usr/bin/env python3
"""Find near-duplicate src frames in a training dataset so you can prune
before painting their dst counterparts.

The problem: when you're hand-painting dst frames to fix a smudge / dust /
artifact on a shot, every redundant near-identical src frame doubles your
paint-work for no extra training signal. This tool finds those clusters of
near-identical src frames, picks one to keep per cluster, and lists the
rest so you can delete them before starting the dst paint pass.

How it works:
  - Each src image is fingerprinted: converted to grayscale, downsampled to
    32x32, stored as a flat array.
  - Pairwise L1 distance is computed across all fingerprints. The metric is
    mean per-pixel difference normalized to [0, 1] — so 0.0 = identical,
    1.0 = max possible difference (fully black vs fully white).
  - Frames within --threshold (default 0.05) of each other are grouped via
    union-find (transitive: if A~B and B~C, all three end up in one group).
  - For each group, the "representative" is the frame with the lowest
    average distance to the others — the one most central to the cluster.
    Everything else in the group is suggested for deletion.

Usage:
    python scripts/dedupe_src.py <src_dir>
        — report-only; lists groups + which files to delete

    python scripts/dedupe_src.py <src_dir> --threshold 0.08
        — looser similarity threshold (more aggressive grouping)

    python scripts/dedupe_src.py <src_dir> --delete
        — actually delete the suggested files. PROMPTS BEFORE deleting;
          pass --yes to skip confirmation.

    python scripts/dedupe_src.py <src_dir> --html report.html
        — write a visual side-by-side report you can open in a browser
          to confirm groupings before deleting.
"""

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
FINGERPRINT_SIZE = 32  # 32x32 grayscale per image — good speed/accuracy tradeoff


def list_images(src_dir: str) -> list[str]:
    """List image files in src_dir (non-recursive), sorted by filename."""
    out = []
    for ent in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, ent)
        if not os.path.isfile(p):
            continue
        if Path(ent).suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def fingerprint(path: str) -> np.ndarray:
    """Compute a tiny grayscale fingerprint for fast comparison.

    Returns a (FINGERPRINT_SIZE * FINGERPRINT_SIZE,) uint8 array. EXR/HDR
    isn't supported here (PIL can't decode it); add the format if you need.
    """
    with Image.open(path) as im:
        # convert handles RGBA, palette, etc. transparently
        gs = im.convert('L').resize((FINGERPRINT_SIZE, FINGERPRINT_SIZE), Image.BILINEAR)
        return np.array(gs, dtype=np.uint8).flatten()


def distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Mean per-pixel absolute difference normalized to [0, 1].

    Two identical images → 0.0. Two random-noise images → ~0.33. A black
    image vs white image → 1.0. For 'near-duplicate' typical values are
    below 0.05; visually distinct frames are usually above 0.10.
    """
    return float(np.abs(fp1.astype(np.int16) - fp2.astype(np.int16)).mean()) / 255.0


class UnionFind:
    """Standard union-find for grouping frames transitively."""
    def __init__(self, n: int):
        self.parent = list(range(n))
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_near_duplicate_groups(
    paths: list[str], fingerprints: np.ndarray, threshold: float
) -> dict[int, list[int]]:
    """Return {group_id: [file_index, ...]} for all groups with >1 member."""
    n = len(paths)
    uf = UnionFind(n)
    # O(n^2) pairwise — fine for datasets up to a few thousand frames; the
    # actual comparison is cheap (vectorized numpy on 1024-element arrays).
    fp_int = fingerprints.astype(np.int16)
    for i in range(n):
        # Vectorize the inner loop: distance from i to all j>i in one shot
        diffs = np.abs(fp_int[i+1:] - fp_int[i]).mean(axis=1) / 255.0
        for off, d in enumerate(diffs):
            if d < threshold:
                uf.union(i, i + 1 + off)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    # Only keep groups with multiple members
    return {gid: idxs for gid, idxs in groups.items() if len(idxs) > 1}


def pick_representative(idxs: list[int], fingerprints: np.ndarray) -> tuple[int, dict[int, float]]:
    """Pick the most-central frame in a group (lowest avg distance to others).

    Returns (representative_idx, {member_idx: distance_from_representative}).
    """
    n = len(idxs)
    fp_int = fingerprints[idxs].astype(np.int16)
    # Pairwise distances within the group
    pairwise = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = float(np.abs(fp_int[i] - fp_int[j]).mean()) / 255.0
            pairwise[i, j] = d
            pairwise[j, i] = d
    avg_dist = pairwise.mean(axis=1)
    rep_local = int(np.argmin(avg_dist))
    rep_global = idxs[rep_local]
    return rep_global, {idxs[k]: pairwise[rep_local, k] for k in range(n)}


def write_html_report(
    out_path: str,
    src_dir: str,
    groups: list[tuple[int, list[int], dict[int, float]]],
    paths: list[str],
    threshold: float,
) -> None:
    """Side-by-side visual confirmation HTML. Thumbnails inline as data URIs.

    Big-ish file for many groups (each thumbnail ~5-10KB base64'd), but
    self-contained — no separate image dir, no broken paths when moved.
    """
    import base64, io
    THUMB = 200  # px

    def thumb_data_uri(path: str) -> str:
        with Image.open(path) as im:
            im = im.convert('RGB')
            im.thumbnail((THUMB, THUMB), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/jpeg;base64,{b64}'

    parts = []
    parts.append('<!doctype html><html><head><meta charset="utf-8">')
    parts.append(f'<title>Near-duplicate src frames — {os.path.basename(src_dir)}</title>')
    parts.append('<style>')
    parts.append('body{font-family:system-ui,sans-serif;margin:20px;background:#fafafa;color:#222}')
    parts.append('h1{font-size:18px}h2{font-size:14px;margin-top:24px;border-bottom:1px solid #ccc;padding-bottom:4px}')
    parts.append('.group{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}')
    parts.append('.f{border:2px solid #ccc;padding:4px;background:#fff;text-align:center}')
    parts.append('.f.keep{border-color:#16A34A;background:#f0fdf4}')
    parts.append('.f.del{border-color:#EF4444;opacity:.75}')
    parts.append('.f img{display:block;width:200px;height:auto}')
    parts.append('.f .name{font-size:10px;font-family:monospace;margin-top:4px;word-break:break-all;max-width:200px}')
    parts.append('.f .badge{font-size:9px;text-transform:uppercase;letter-spacing:0.5px;font-weight:bold;margin-top:2px}')
    parts.append('.f.keep .badge{color:#16A34A}.f.del .badge{color:#EF4444}')
    parts.append('.f .dist{font-size:9px;color:#666;margin-top:2px}')
    parts.append('</style></head><body>')
    parts.append(f'<h1>Near-duplicate src frames in {src_dir}</h1>')
    parts.append(f'<p>Threshold: {threshold:.3f}. Groups: {len(groups)}.</p>')
    parts.append('<p style="color:#666;font-size:12px">Green = keep (most central frame in group). Red = candidate for deletion.</p>')
    for gi, (gid, idxs, dists) in enumerate(groups, 1):
        rep = min(dists.items(), key=lambda kv: kv[1])[0]
        parts.append(f'<h2>Group {gi} ({len(idxs)} frames, max dist {max(dists.values()):.3f})</h2>')
        parts.append('<div class="group">')
        for idx in idxs:
            is_rep = (idx == rep)
            klass = 'keep' if is_rep else 'del'
            label = 'KEEP' if is_rep else 'DELETE'
            parts.append(f'<div class="f {klass}">')
            parts.append(f'<img src="{thumb_data_uri(paths[idx])}">')
            parts.append(f'<div class="name">{os.path.basename(paths[idx])}</div>')
            parts.append(f'<div class="badge">{label}</div>')
            parts.append(f'<div class="dist">dist from keep: {dists[idx]:.3f}</div>')
            parts.append('</div>')
        parts.append('</div>')
    parts.append('</body></html>')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src_dir', help='Path to src folder (non-recursive)')
    ap.add_argument('--threshold', type=float, default=0.05,
        help='Max mean-pixel-difference (0-1) to consider near-duplicate. '
             'Default 0.05. Try 0.08 for looser grouping, 0.03 for stricter.')
    ap.add_argument('--delete', action='store_true', help='Actually delete the suggested files')
    ap.add_argument('--yes', action='store_true', help='Skip the confirmation prompt when --delete is on')
    ap.add_argument('--html', metavar='PATH', help='Write a visual HTML report to this path')
    args = ap.parse_args()

    if not os.path.isdir(args.src_dir):
        print(f'ERROR: not a directory: {args.src_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Scanning {args.src_dir} ...', flush=True)
    paths = list_images(args.src_dir)
    if not paths:
        print('No images found.')
        return
    print(f'Found {len(paths)} images.')

    t0 = time.time()
    print('Computing fingerprints ...', end='', flush=True)
    fingerprints = np.stack([fingerprint(p) for p in paths])
    print(f' done ({time.time()-t0:.1f}s)')

    t0 = time.time()
    print(f'Finding groups (threshold={args.threshold}) ...', end='', flush=True)
    groups_raw = find_near_duplicate_groups(paths, fingerprints, args.threshold)
    print(f' done ({time.time()-t0:.1f}s)')

    if not groups_raw:
        print(f'\nNo near-duplicate groups found at threshold {args.threshold}. Nothing to prune.')
        return

    # Compute representative + per-member distances for each group
    groups = []
    for gid, idxs in groups_raw.items():
        rep, dists = pick_representative(idxs, fingerprints)
        # Sort by distance from rep (rep first), then by name
        idxs_sorted = sorted(idxs, key=lambda i: (i != rep, dists[i], paths[i]))
        groups.append((gid, idxs_sorted, dists))
    # Sort groups by size desc, then by first filename
    groups.sort(key=lambda g: (-len(g[1]), paths[g[1][0]]))

    n_dup = sum(len(idxs) for _, idxs, _ in groups)
    n_keep = len(groups)
    n_delete = n_dup - n_keep
    n_unique = len(paths) - n_dup

    print(f'\nResult: {len(groups)} groups, {n_dup} frames in groups, {n_unique} unique frames.')
    print(f'Keep recommendations: {n_keep + n_unique} total ({n_keep} group reps + {n_unique} unique).')
    print(f'Delete recommendations: {n_delete} frames.\n')

    # Text report
    print('=' * 72)
    print('GROUPS (sorted by size)')
    print('=' * 72)
    for gi, (gid, idxs, dists) in enumerate(groups, 1):
        rep = idxs[0]  # already sorted with rep first
        max_d = max(dists.values())
        print(f'\nGroup {gi}: {len(idxs)} frames, max distance {max_d:.3f}')
        for idx in idxs:
            mark = 'KEEP  ' if idx == rep else 'delete'
            d_str = '(rep) ' if idx == rep else f'(d={dists[idx]:.3f})'
            print(f'  [{mark}] {d_str.ljust(10)} {os.path.basename(paths[idx])}')

    # HTML report
    if args.html:
        print(f'\nWriting HTML report → {args.html} ...', end='', flush=True)
        write_html_report(args.html, args.src_dir, groups, paths, args.threshold)
        print(' done')
        print(f'Open in a browser to confirm groupings visually before deleting.')

    # Delete
    to_delete = [paths[idx] for _, idxs, _ in groups for idx in idxs[1:]]  # all but the rep (idxs[0])
    if args.delete:
        if not args.yes:
            print(f'\nAbout to DELETE {len(to_delete)} files. Confirm? [y/N] ', end='', flush=True)
            resp = input().strip().lower()
            if resp not in ('y', 'yes'):
                print('Aborted.')
                return
        for p in to_delete:
            try:
                os.remove(p)
                print(f'  deleted: {os.path.basename(p)}')
            except Exception as e:
                print(f'  FAILED to delete {p}: {e}', file=sys.stderr)
        print(f'\nDeleted {len(to_delete)} files.')
    else:
        print(f'\n(Report only — pass --delete to actually remove the {len(to_delete)} suggested files.)')


if __name__ == '__main__':
    main()
