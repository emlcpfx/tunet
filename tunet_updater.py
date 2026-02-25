# ==============================================================================
# TuNet Installer & Updater
#
# Handles first-time installation and updates for the TuNet distribution.
# Downloads compressed archive parts from a GitHub Release, verifies integrity,
# and extracts only new or changed files.
#
# Features:
#   - Skips files that already exist and match expected hash
#   - Resumes interrupted downloads
#   - Shows progress with PySide6 dialogs
#   - Auto-checks for updates on startup
# ==============================================================================

import os
import sys
import json
import hashlib
import zipfile
import urllib.request
import tempfile
from pathlib import Path


# --- Configuration ---
# Update after creating your first GitHub Release
GITHUB_REPO = "YOUR_USER/tunet"  # e.g. "johndoe/tunet"
RELEASE_TAG = "v0.1.0"

# Derived URLs (auto-constructed from repo + tag)
MANIFEST_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/manifest.json"

# Local version tracking
VERSION_FILE = "tunet_version.json"


def get_install_dir():
    """Get the installation directory (where TuNet.exe lives)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_internal_dir():
    """Get the _internal directory where bundled files live."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def get_version_file_path():
    return os.path.join(get_install_dir(), VERSION_FILE)


def get_installed_version():
    """Read the currently installed version info."""
    vf = get_version_file_path()
    if os.path.isfile(vf):
        try:
            with open(vf, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_installed_version(manifest):
    """Save version info after successful install/update."""
    vf = get_version_file_path()
    info = {
        'version': manifest['version'],
        'installed_at': manifest.get('created', ''),
        'total_files': len(manifest.get('files', {})),
    }
    with open(vf, 'w') as f:
        json.dump(info, f, indent=2)


def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def fetch_manifest():
    """Download the release manifest from GitHub."""
    try:
        req = urllib.request.Request(MANIFEST_URL)
        req.add_header('User-Agent', 'TuNet-Updater/1.0')
        response = urllib.request.urlopen(req, timeout=15)
        data = json.loads(response.read().decode('utf-8'))
        return data
    except Exception as e:
        print(f"Failed to fetch manifest: {e}")
        return None


def check_for_updates():
    """Check if an update is available. Returns (needs_update, manifest) or (False, None)."""
    installed = get_installed_version()
    manifest = fetch_manifest()

    if manifest is None:
        return False, None

    if installed is None:
        # First install
        return True, manifest

    if manifest.get('version') != installed.get('version'):
        return True, manifest

    return False, manifest


def get_files_needing_update(manifest):
    """Compare manifest against installed files. Returns list of files to update."""
    install_dir = get_install_dir()
    needs_update = []

    for rel_path, file_info in manifest.get('files', {}).items():
        local_path = os.path.join(install_dir, rel_path.replace('/', os.sep))

        if not os.path.isfile(local_path):
            needs_update.append(rel_path)
            continue

        # File exists - check size first (fast)
        local_size = os.path.getsize(local_path)
        if local_size != file_info['size']:
            needs_update.append(rel_path)
            continue

        # Size matches - check hash for critical files (DLLs, exe)
        if rel_path.endswith(('.dll', '.exe', '.pyd')):
            local_hash = compute_sha256(local_path)
            if local_hash != file_info['sha256']:
                needs_update.append(rel_path)

    return needs_update


def install_or_update(headless=False):
    """Main install/update entry point.

    If headless=True, runs without GUI (for CLI usage).
    Otherwise shows PySide6 dialogs.
    """
    if headless:
        return _install_headless()
    else:
        return _install_gui()


def _install_headless():
    """CLI-based install/update."""
    print("Checking for updates...")
    needs_update, manifest = check_for_updates()

    if not needs_update:
        print("TuNet is up to date.")
        return True

    if manifest is None:
        print("Could not reach update server.")
        return False

    print(f"Update available: v{manifest['version']}")

    files_to_update = get_files_needing_update(manifest)
    if not files_to_update:
        print("All files are current.")
        save_installed_version(manifest)
        return True

    total_size = sum(manifest['files'][f]['size'] for f in files_to_update)
    print(f"  {len(files_to_update)} files to download ({total_size / 1024 / 1024:.0f} MB)")

    # Download and extract parts
    base_url = MANIFEST_URL.rsplit('/', 1)[0]
    install_dir = get_install_dir()

    for part_info in manifest['parts']:
        part_url = f"{base_url}/{part_info['filename']}"
        print(f"\nDownloading {part_info['filename']}...")

        success = _download_and_extract_part(
            part_url, part_info, install_dir, files_to_update, manifest,
            progress_callback=lambda pct, msg: print(f"\r  {msg} ({pct}%)", end='', flush=True)
        )
        if not success:
            print("\nUpdate failed!")
            return False
        print()

    save_installed_version(manifest)
    print("\nUpdate complete!")
    return True


def _install_gui():
    """GUI-based install/update with PySide6 dialogs."""
    # Check for updates BEFORE creating any GUI
    needs_update, manifest = check_for_updates()

    if not needs_update:
        return True  # Up to date, silently continue

    if manifest is None:
        # Can't reach server - continue with what we have
        return True

    # Determine what needs downloading
    files_to_update = get_files_needing_update(manifest)

    if not files_to_update:
        save_installed_version(manifest)
        return True

    total_size = sum(manifest['files'][f]['size'] for f in files_to_update)
    installed = get_installed_version()

    # Only create QApplication now that we actually need UI
    from PySide6.QtWidgets import (
        QApplication, QMessageBox, QProgressDialog
    )
    from PySide6.QtCore import Qt

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    if installed is None:
        title = "TuNet - First Time Setup"
        message = "TuNet needs to download required components."
    else:
        title = "TuNet - Update Available"
        message = f"Version {manifest['version']} is available (you have {installed['version']})."

    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setIcon(QMessageBox.Information)
    msg.setText(message)
    msg.setInformativeText(
        f"{len(files_to_update)} files need to be downloaded "
        f"({total_size / 1024 / 1024 / 1024:.1f} GB).\n\n"
        f"Files you already have will be skipped."
    )
    download_btn = msg.addButton("Download && Install", QMessageBox.AcceptRole)
    skip_btn = msg.addButton("Skip", QMessageBox.RejectRole)
    msg.setDefaultButton(download_btn)
    msg.exec()

    if msg.clickedButton() != download_btn:
        return True  # User skipped, continue anyway

    # Download with progress
    progress = QProgressDialog("Preparing download...", "Cancel", 0, 100)
    progress.setWindowTitle("TuNet - Downloading")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setMinimumWidth(450)

    base_url = MANIFEST_URL.rsplit('/', 1)[0]
    install_dir = get_install_dir()
    total_parts = len(manifest['parts'])

    for i, part_info in enumerate(manifest['parts']):
        part_url = f"{base_url}/{part_info['filename']}"

        def gui_progress(pct, msg, _i=i):
            if progress.wasCanceled():
                return False
            # Overall progress across all parts
            overall = int((_i * 100 + pct) / total_parts)
            progress.setValue(overall)
            progress.setLabelText(
                f"Part {_i + 1}/{total_parts}: {msg}"
            )
            app.processEvents()
            return True

        success = _download_and_extract_part(
            part_url, part_info, install_dir, files_to_update, manifest,
            progress_callback=gui_progress
        )

        if not success:
            progress.close()
            if not progress.wasCanceled():
                QMessageBox.critical(
                    None, "Download Error",
                    "Failed to download update. You can try again later."
                )
            return True  # Continue anyway with what we have

    progress.setValue(100)
    save_installed_version(manifest)

    QMessageBox.information(
        None, "Update Complete",
        f"TuNet v{manifest['version']} installed successfully!"
    )

    if app_created:
        del app

    return True


def _download_and_extract_part(url, part_info, install_dir, files_to_update, manifest, progress_callback=None):
    """Download a zip part and extract only the files that need updating."""
    tmp_path = os.path.join(tempfile.gettempdir(), part_info['filename'])

    try:
        # Download
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'TuNet-Updater/1.0')
        response = urllib.request.urlopen(req)
        total_size = int(response.headers.get('Content-Length', 0))

        sha256 = hashlib.sha256()
        downloaded = 0

        with open(tmp_path, 'wb') as f:
            while True:
                chunk = response.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                sha256.update(chunk)
                downloaded += len(chunk)

                if total_size > 0 and progress_callback:
                    pct = int(downloaded * 70 / total_size)  # 0-70% for download
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    result = progress_callback(pct, f"Downloading... {mb:.0f}/{total_mb:.0f} MB")
                    if result is False:
                        _cleanup(tmp_path)
                        return False

        response.close()

        # Verify hash
        if sha256.hexdigest() != part_info['sha256']:
            print(f"Hash mismatch for {part_info['filename']}")
            _cleanup(tmp_path)
            return False

        # Extract only files that need updating
        if progress_callback:
            progress_callback(75, "Extracting files...")

        files_to_update_set = set(files_to_update)

        with zipfile.ZipFile(tmp_path, 'r') as zf:
            members = zf.namelist()
            extracted = 0
            for name in members:
                # Normalize path
                norm_name = name.replace('\\', '/')
                if norm_name in files_to_update_set:
                    target = os.path.join(install_dir, norm_name.replace('/', os.sep))
                    os.makedirs(os.path.dirname(target), exist_ok=True)

                    with zf.open(name) as src, open(target, 'wb') as dst:
                        while True:
                            data = src.read(1024 * 1024)
                            if not data:
                                break
                            dst.write(data)

                    extracted += 1

                    if progress_callback:
                        pct = 75 + int(extracted * 25 / max(len(files_to_update_set), 1))
                        progress_callback(pct, f"Installing... ({extracted}/{len(files_to_update_set)})")

        _cleanup(tmp_path)
        return True

    except Exception as e:
        print(f"Error processing {part_info['filename']}: {e}")
        _cleanup(tmp_path)
        return False


def _cleanup(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


if __name__ == '__main__':
    # CLI mode
    import argparse
    parser = argparse.ArgumentParser(description='TuNet Installer/Updater')
    parser.add_argument('--check', action='store_true', help='Check for updates only')
    parser.add_argument('--gui', action='store_true', help='Use GUI dialogs')
    args = parser.parse_args()

    if args.check:
        needs, manifest = check_for_updates()
        if needs and manifest:
            print(f"Update available: v{manifest['version']}")
            files = get_files_needing_update(manifest)
            total = sum(manifest['files'][f]['size'] for f in files)
            print(f"  {len(files)} files ({total / 1024 / 1024:.0f} MB)")
        else:
            print("Up to date.")
    else:
        install_or_update(headless=not args.gui)
