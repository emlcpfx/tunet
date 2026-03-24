# ==============================================================================
# Package TuNet Distribution for GitHub Release
#
# Takes the full PyInstaller dist and creates:
#   1. A manifest.json with file hashes and version info
#   2. Compressed archive parts (split to stay under GitHub's 2 GB limit)
#
# Usage:  conda run -n tunet python scripts/package_release.py [--version X.Y.Z]
# Output: scripts/release/ directory with manifest.json + .zip parts
# ==============================================================================

import os
import sys
import json
import hashlib
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).parent.parent
DIST_DIR = PROJECT_ROOT / "dist" / "TuNet"
RELEASE_DIR = PROJECT_ROOT / "scripts" / "release"
MAX_PART_SIZE = 1_900_000_000  # ~1.9 GB per part (under GitHub's 2 GB limit)


def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def build_file_manifest(dist_dir):
    """Build a manifest of all files with their sizes and hashes."""
    files = {}
    for root, dirs, filenames in os.walk(dist_dir):
        for fname in filenames:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, dist_dir).replace('\\', '/')
            size = os.path.getsize(fpath)
            files[rel_path] = {
                'size': size,
                'sha256': compute_sha256(fpath),
            }
    return files


def create_archive_parts(dist_dir, release_dir, version):
    """Create split zip archives of the distribution."""
    os.makedirs(release_dir, exist_ok=True)

    # Collect all files sorted by path
    all_files = []
    for root, dirs, filenames in os.walk(dist_dir):
        for fname in filenames:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, dist_dir)
            all_files.append((rel_path, fpath))
    all_files.sort()

    parts = []
    part_num = 1
    current_zip = None
    current_size = 0

    def start_new_part():
        nonlocal current_zip, current_size, part_num
        if current_zip:
            current_zip.close()
        part_name = f"tunet-{version}-part{part_num}.zip"
        part_path = os.path.join(release_dir, part_name)
        current_zip = zipfile.ZipFile(part_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6)
        current_size = 0
        parts.append(part_name)
        part_num += 1
        return part_path

    current_part_path = start_new_part()

    for rel_path, fpath in all_files:
        file_size = os.path.getsize(fpath)
        arc_name = rel_path.replace('\\', '/')

        # Add file to current zip
        current_zip.write(fpath, arc_name)

        # Check ACTUAL zip size on disk (flush first)
        current_zip.fp.flush()
        actual_size = current_zip.fp.tell()

        print(f"  Part {part_num - 1}: {arc_name} ({file_size / 1024 / 1024:.1f} MB) [zip: {actual_size / 1024 / 1024:.0f} MB]")

        # If we've exceeded the limit, close this part and start a new one
        # (next file goes in new part)
        if actual_size > MAX_PART_SIZE:
            current_part_path = start_new_part()

    if current_zip:
        current_zip.close()

    return parts


def main():
    parser = argparse.ArgumentParser(description='Package TuNet for GitHub Release')
    parser.add_argument('--version', type=str, default='0.1.0',
                        help='Version string (e.g. 0.1.0)')
    args = parser.parse_args()

    if not DIST_DIR.is_dir():
        print(f"ERROR: Distribution not found at {DIST_DIR}")
        print("Run 'pyinstaller tunet.spec' first.")
        sys.exit(1)

    print("=" * 60)
    print(f"  TuNet Release Packager v{args.version}")
    print("=" * 60)

    # Build file manifest
    print("\nBuilding file manifest (computing hashes)...")
    files = build_file_manifest(DIST_DIR)
    total_size = sum(f['size'] for f in files.values())
    print(f"  {len(files)} files, {total_size / 1024 / 1024 / 1024:.2f} GB total")

    # Create archive parts
    print(f"\nCreating archive parts (max {MAX_PART_SIZE / 1024 / 1024 / 1024:.1f} GB each)...")
    parts = create_archive_parts(DIST_DIR, RELEASE_DIR, args.version)

    # Compute part hashes
    print("\nComputing archive hashes...")
    part_info = []
    for part_name in parts:
        part_path = os.path.join(RELEASE_DIR, part_name)
        size = os.path.getsize(part_path)
        sha = compute_sha256(part_path)
        part_info.append({
            'filename': part_name,
            'size': size,
            'sha256': sha,
        })
        print(f"  {part_name}: {size / 1024 / 1024:.0f} MB (SHA-256: {sha[:16]}...)")

    # Write manifest
    manifest = {
        'version': args.version,
        'created': datetime.now(timezone.utc).isoformat(),
        'platform': 'win-x64',
        'python': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
        'parts': part_info,
        'files': files,
        'total_size': total_size,
    }

    manifest_path = os.path.join(RELEASE_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Release package created!")
    print(f"{'=' * 60}")
    print(f"  Output:   {RELEASE_DIR}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Parts:    {len(parts)}")
    download_size = sum(p['size'] for p in part_info)
    print(f"  Download: {download_size / 1024 / 1024 / 1024:.2f} GB compressed")
    print(f"  Install:  {total_size / 1024 / 1024 / 1024:.2f} GB uncompressed")
    print(f"\n  Upload instructions:")
    print(f"  1. Create GitHub Release with tag 'v{args.version}'")
    print(f"  2. Upload ALL files from {RELEASE_DIR}/:")
    print(f"     - manifest.json")
    for p in parts:
        print(f"     - {p}")
    print(f"  3. The installer will auto-detect and download from the release.")
    print(f"\n  GitHub Release URL pattern:")
    print(f"  https://github.com/YOUR_USER/tunet/releases/download/v{args.version}/")


if __name__ == '__main__':
    main()
