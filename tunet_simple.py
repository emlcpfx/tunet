# ==============================================================================
# TuNet Simple UI — Preset-based workflow for AE users
#
# A streamlined PySide6 interface that replaces the full UI with a
# preset-driven workflow. Pick a task, point to a folder, and go.
#
# Folder structure expected:
#   project_folder/
#     src/    — source images (before)
#     dst/    — destination images (after)
#     model/  — created automatically for checkpoints
#
# require: pip install PySide6
# ==============================================================================

import sys
import os
import json
import math
import subprocess
import threading
import signal
import yaml
import logging
import time
import platform
from pathlib import Path
from glob import glob as globglob

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog,
    QFormLayout, QMessageBox, QSizePolicy, QScrollArea,
    QSplitter, QProgressBar, QGroupBox, QFrame, QSpinBox,
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QFileSystemWatcher, QTimer
from PySide6.QtGui import QPixmap, QTextCursor, QFont


# =============================================================================
# Helper Classes (shared with main UI)
# =============================================================================

class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


class ProcessStreamReader(QObject):
    """Reads a subprocess stdout/stderr stream in a thread and emits lines."""
    new_text = Signal(str)

    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self):
        for line in iter(self.stream.readline, ''):
            self.new_text.emit(line)


# =============================================================================
# Presets
# =============================================================================

# Each preset defines training settings optimized for a specific task.
# The model_size_dims and resolution get overridden by auto-detection,
# but the preset provides defaults and task-specific settings.

PRESETS = {
    "Beauty": {
        "description": (
            "Skin retouching, blemish removal, frequency separation cleanup.\n"
            "Uses auto-mask to focus on areas that changed between src and dst."
        ),
        "loss": "l1+lpips",
        "lambda_lpips": 0.1,
        "lr": 5e-4,
        "mask": {
            "use_mask_loss": True,
            "mask_weight": 10.0,
            "use_mask_input": False,
            "use_auto_mask": True,
        },
        "augmentations": [
            {"_target_": "albumentations.HorizontalFlip", "p": 0.5},
        ],
        "batch_size": 4,
        "default_model_size": 128,
        "default_resolution": 512,
    },
    "Roto / Matte": {
        "description": (
            "Segmentation and matte extraction.\n"
            "Source = plate, Destination = black/white matte.\n"
            "Uses BCE+Dice loss optimized for binary output."
        ),
        "loss": "bce+dice",
        "lambda_lpips": 1.0,
        "lr": 1e-4,
        "mask": {},
        "augmentations": [
            {"_target_": "albumentations.HorizontalFlip", "p": 0.5},
            {
                "_target_": "albumentations.Affine",
                "scale": [0.85, 1.15],
                "translate_percent": [-0.1, 0.1],
                "rotate": [-5, 5],
                "shear": [-2, 2],
                "interpolation": 2,
                "keep_ratio": True,
                "p": 0.4,
            },
        ],
        "batch_size": 4,
        "default_model_size": 128,
        "default_resolution": 512,
    },
    "Cleanup / Paint": {
        "description": (
            "Wire removal, rig removal, paint fixes.\n"
            "Source = plate with wires/rigs, Destination = clean plate.\n"
            "Uses auto-mask to focus learning on the removed areas."
        ),
        "loss": "l1+lpips",
        "lambda_lpips": 0.15,
        "lr": 1e-4,
        "mask": {
            "use_mask_loss": True,
            "mask_weight": 15.0,
            "use_mask_input": False,
            "use_auto_mask": True,
        },
        "augmentations": [
            {"_target_": "albumentations.HorizontalFlip", "p": 0.3},
            {
                "_target_": "albumentations.Affine",
                "scale": [0.9, 1.1],
                "translate_percent": [-0.05, 0.05],
                "rotate": [-2, 2],
                "shear": [-1, 1],
                "interpolation": 2,
                "keep_ratio": True,
                "p": 0.3,
            },
        ],
        "batch_size": 4,
        "default_model_size": 128,
        "default_resolution": 512,
    },
    "Color / Grade": {
        "description": (
            "Color correction, look transfer, grade matching.\n"
            "Source = original grade, Destination = target grade.\n"
            "Pure pixel loss for accurate color reproduction."
        ),
        "loss": "l1",
        "lambda_lpips": 0.0,
        "lr": 3e-4,
        "mask": {},
        "augmentations": [
            {"_target_": "albumentations.HorizontalFlip", "p": 0.5},
            {
                "_target_": "albumentations.RandomGamma",
                "gamma_limit": [60, 140],
                "p": 0.3,
            },
        ],
        "batch_size": 4,
        "default_model_size": 64,
        "default_resolution": 512,
    },
    "Denoise": {
        "description": (
            "Noise reduction and grain management.\n"
            "Source = noisy plate, Destination = clean/denoised plate.\n"
            "Perceptual loss helps preserve texture while removing noise."
        ),
        "loss": "l1+lpips",
        "lambda_lpips": 0.05,
        "lr": 1e-4,
        "mask": {},
        "augmentations": [
            {"_target_": "albumentations.HorizontalFlip", "p": 0.5},
        ],
        "batch_size": 4,
        "default_model_size": 128,
        "default_resolution": 512,
    },
    "General": {
        "description": (
            "General purpose image-to-image translation.\n"
            "Good starting point when none of the other presets fit.\n"
            "Balanced settings with moderate augmentation."
        ),
        "loss": "l1+lpips",
        "lambda_lpips": 0.1,
        "lr": 1e-4,
        "mask": {},
        "augmentations": [
            {"_target_": "albumentations.HorizontalFlip", "p": 0.5},
            {
                "_target_": "albumentations.Affine",
                "scale": [0.9, 1.1],
                "translate_percent": [-0.1, 0.1],
                "rotate": [-3, 3],
                "shear": [-1, 1],
                "interpolation": 2,
                "keep_ratio": True,
                "p": 0.4,
            },
        ],
        "batch_size": 4,
        "default_model_size": 128,
        "default_resolution": 512,
    },
}

# =============================================================================
# Auto-detection logic
# =============================================================================

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr', '.bmp', '.webp'}


def detect_image_info(folder_path):
    """Scan a folder and return (width, height, count) of the first valid image."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return None, None, 0

    count = 0
    first_w, first_h = None, None

    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            count += 1
            if first_w is None:
                try:
                    from PIL import Image
                    with Image.open(f) as img:
                        first_w, first_h = img.size
                except Exception:
                    pass

    return first_w, first_h, count


def recommend_settings(width, height, preset_name):
    """Given image dimensions and a preset, recommend resolution and model size."""
    preset = PRESETS.get(preset_name, PRESETS["General"])
    min_dim = min(width, height) if width and height else 1920

    # Resolution: pick largest that fits the image well
    # For very large images (4K+), cap at 1024 to keep VRAM manageable
    if min_dim >= 3000:
        resolution = 1024
    elif min_dim >= 1500:
        resolution = 768
    elif min_dim >= 1000:
        resolution = 512
    elif min_dim >= 600:
        resolution = 384
    else:
        resolution = 256

    # Model size: scale with resolution
    # Bigger patches need more capacity to learn the mapping
    if resolution >= 1024:
        model_size = 64   # keep smaller at high res to save VRAM
    elif resolution >= 768:
        model_size = 128
    elif resolution >= 512:
        model_size = 128
    else:
        model_size = 64

    # For roto/matte, slightly smaller model is fine since it's binary output
    if preset_name == "Roto / Matte":
        model_size = min(model_size, 128)

    # For color grading, smaller model is sufficient
    if preset_name == "Color / Grade":
        model_size = min(model_size, 64)

    # Batch size: reduce for high-res / large model combos
    batch_size = preset.get("batch_size", 4)
    if resolution >= 1024 and model_size >= 128:
        batch_size = max(1, batch_size // 2)
    elif resolution >= 768 and model_size >= 128:
        batch_size = max(2, batch_size // 2)

    return resolution, model_size, batch_size


# =============================================================================
# Session paths
# =============================================================================
_APP_DIR = Path(__file__).parent
_SESSION_FILE = _APP_DIR / 'tunet_simple_session.yaml'


# =============================================================================
# SimpleWindow
# =============================================================================

class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TuNet Simple")
        self.setGeometry(100, 100, 780, 650)

        # State
        self.process = None
        self.config_file_path = Path("config_from_simple_ui.yaml")
        self._detected_width = None
        self._detected_height = None
        self._detected_count = 0

        # Preview state
        self.preview_image_path = None
        self.original_pixmap = None

        # Inference state
        self.inference_running = False
        self.inference_stop_requested = False
        self._cached_model = None
        self._cached_checkpoint = None
        self._cached_resolution = None
        self._cached_loss_mode = None

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Vertical)

        # Top: settings
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(8, 8, 8, 4)

        self._create_preset_section(top_layout)
        self._create_folder_section(top_layout)
        self._create_info_section(top_layout)
        self._create_buttons(top_layout)

        top_scroll = QScrollArea()
        top_scroll.setWidgetResizable(True)
        top_scroll.setWidget(top_widget)
        splitter.addWidget(top_scroll)

        # Bottom: console + preview
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFontFamily("monospace")
        self.console_output.setMinimumHeight(120)
        bottom_layout.addWidget(self.console_output, stretch=2)

        # Preview area
        preview_frame = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_frame)
        self.preview_label = QLabel("No preview yet")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(250)
        preview_layout.addWidget(self.preview_label)
        bottom_layout.addWidget(preview_frame, stretch=1)

        splitter.addWidget(bottom_widget)
        splitter.setSizes([320, 300])

        main_layout.addWidget(splitter)

        # Timers
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(3000)
        self.preview_timer.timeout.connect(self._update_preview)

        self.process_monitor = QTimer(self)
        self.process_monitor.setInterval(500)
        self.process_monitor.timeout.connect(self._check_process_status)

        # Logging handler
        self._log_handler = _QTextEditLogHandler(self.console_output)
        self._log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(self._log_handler)
        logging.getLogger().setLevel(logging.INFO)

        # Load session
        self._load_session()

    # =========================================================================
    # UI creation
    # =========================================================================

    def _create_preset_section(self, parent_layout):
        grp = QGroupBox("Task Preset")
        layout = QVBoxLayout(grp)

        self.preset_combo = QComboBox()
        for name in PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentText("Beauty")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        # Preset description
        self.preset_desc = QLabel(PRESETS["Beauty"]["description"])
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet("color: #888; font-style: italic; padding: 4px;")

        layout.addWidget(self.preset_combo)
        layout.addWidget(self.preset_desc)
        parent_layout.addWidget(grp)

    def _create_folder_section(self, parent_layout):
        grp = QGroupBox("Project Folder")
        layout = QVBoxLayout(grp)

        hint = QLabel(
            "Select a folder containing src/ and dst/ subfolders.\n"
            "A model/ subfolder will be created automatically."
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        folder_row = QHBoxLayout()
        self.folder_input = QPushButton("Select Project Folder...")
        self.folder_input.setStyleSheet("text-align: left; padding: 6px 12px;")
        self.folder_input.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_input)
        layout.addLayout(folder_row)

        self.folder_status = QLabel("")
        self.folder_status.setWordWrap(True)
        layout.addWidget(self.folder_status)

        parent_layout.addWidget(grp)

    def _create_info_section(self, parent_layout):
        grp = QGroupBox("Auto-Detected Settings")
        form = QFormLayout(grp)

        self.info_resolution = QLabel("—")
        self.info_model_size = QLabel("—")
        self.info_batch_size = QLabel("—")
        self.info_image_count = QLabel("—")
        self.info_image_dims = QLabel("—")

        form.addRow("Image Dimensions:", self.info_image_dims)
        form.addRow("Image Count:", self.info_image_count)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        form.addRow(sep)

        form.addRow("Training Resolution:", self.info_resolution)
        form.addRow("Model Capacity:", self.info_model_size)
        form.addRow("Batch Size:", self.info_batch_size)

        parent_layout.addWidget(grp)

    def _create_buttons(self, parent_layout):
        layout = QHBoxLayout()

        self.train_btn = QPushButton("Train")
        self.train_btn.setStyleSheet(
            "background-color: #a8e6cf; color: #1a1a1a; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px;")
        self.train_btn.clicked.connect(self._start_training)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "background-color: #ff8a80; color: #1a1a1a; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)

        self.inference_btn = QPushButton("Run Inference")
        self.inference_btn.setStyleSheet(
            "background-color: #b3e5fc; color: #1a1a1a; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px;")
        self.inference_btn.clicked.connect(self._run_inference)

        layout.addWidget(self.train_btn)
        layout.addWidget(self.stop_btn)
        layout.addStretch()
        layout.addWidget(self.inference_btn)

        parent_layout.addLayout(layout)

    # =========================================================================
    # Folder browsing & auto-detection
    # =========================================================================

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self._set_project_folder(folder)

    def _set_project_folder(self, folder):
        self.folder_input.setText(folder)
        self._validate_folder(folder)

    def _validate_folder(self, folder):
        folder_path = Path(folder)
        src = folder_path / "src"
        dst = folder_path / "dst"
        model = folder_path / "model"

        issues = []
        if not folder_path.is_dir():
            issues.append("Folder does not exist.")
        else:
            if not src.is_dir():
                issues.append("Missing src/ subfolder.")
            if not dst.is_dir():
                issues.append("Missing dst/ subfolder.")

        if issues:
            self.folder_status.setText("  ".join(issues))
            self.folder_status.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self._clear_detection()
            return

        # Create model folder if needed
        if not model.is_dir():
            try:
                model.mkdir(parents=True, exist_ok=True)
                self.folder_status.setText(
                    f"OK — Created model/ folder.\n"
                    f"  src: {src}\n  dst: {dst}\n  model: {model}")
            except Exception as e:
                self.folder_status.setText(f"OK but failed to create model/: {e}")
                self.folder_status.setStyleSheet("color: #f57c00;")
                return
        else:
            self.folder_status.setText(
                f"OK\n  src: {src}\n  dst: {dst}\n  model: {model}")

        self.folder_status.setStyleSheet("color: #388e3c;")

        # Auto-detect image sizes
        w, h, count = detect_image_info(str(src))
        self._detected_width = w
        self._detected_height = h
        self._detected_count = count

        if w and h:
            self.info_image_dims.setText(f"{w} x {h}")
            self.info_image_count.setText(str(count))
            self._update_recommended_settings()
        else:
            self.info_image_dims.setText("No images found in src/")
            self.info_image_count.setText("0")
            self._clear_detection()

    def _clear_detection(self):
        self.info_resolution.setText("—")
        self.info_model_size.setText("—")
        self.info_batch_size.setText("—")
        self._detected_width = None
        self._detected_height = None
        self._detected_count = 0

    def _update_recommended_settings(self):
        preset_name = self.preset_combo.currentText()
        if self._detected_width and self._detected_height:
            res, model_sz, batch = recommend_settings(
                self._detected_width, self._detected_height, preset_name)
            self.info_resolution.setText(f"{res}")
            self.info_model_size.setText(f"{model_sz}")
            self.info_batch_size.setText(f"{batch}")

    # =========================================================================
    # Preset handling
    # =========================================================================

    def _on_preset_changed(self, name):
        preset = PRESETS.get(name, PRESETS["General"])
        self.preset_desc.setText(preset["description"])
        self._update_recommended_settings()

    # =========================================================================
    # Config generation
    # =========================================================================

    def _get_project_folder(self):
        text = self.folder_input.text()
        if text.startswith("Select"):
            return None
        return text

    def _build_config(self):
        """Build a full training config from preset + auto-detected settings."""
        folder = self._get_project_folder()
        if not folder:
            return None

        folder_path = Path(folder)
        src_dir = str(folder_path / "src")
        dst_dir = str(folder_path / "dst")
        model_dir = str(folder_path / "model")

        preset_name = self.preset_combo.currentText()
        preset = PRESETS.get(preset_name, PRESETS["General"])

        # Get recommended settings
        if self._detected_width and self._detected_height:
            resolution, model_size, batch_size = recommend_settings(
                self._detected_width, self._detected_height, preset_name)
        else:
            resolution = preset["default_resolution"]
            model_size = preset["default_model_size"]
            batch_size = preset["batch_size"]

        config = {
            'data': {
                'src_dir': src_dir,
                'dst_dir': dst_dir,
                'output_dir': model_dir,
                'resolution': resolution,
                'overlap_factor': 0.25,
            },
            'model': {
                'model_size_dims': model_size,
                'model_type': 'unet',
            },
            'training': {
                'iterations_per_epoch': 500,
                'batch_size': batch_size,
                'max_steps': 0,
                'use_amp': True,
                'loss': preset['loss'],
                'lambda_lpips': preset['lambda_lpips'],
                'lr': preset['lr'],
                'progressive_resolution': resolution >= 768,
            },
            'logging': {
                'log_interval': 5,
                'preview_batch_interval': 35,
                'preview_refresh_rate': 5,
            },
            'saving': {
                'keep_last_checkpoints': 4,
            },
            'dataloader': {
                'num_workers': -1,
                'datasets': {
                    'shared_augs': preset['augmentations'],
                },
            },
        }

        # Mask settings
        if preset.get('mask'):
            config['mask'] = preset['mask'].copy()

        return config

    # =========================================================================
    # Training
    # =========================================================================

    def _find_train_script(self):
        """Locate train.py relative to this script."""
        candidates = [
            _APP_DIR / 'train.py',
            Path(sys.executable).parent / 'train.py',
        ]
        if getattr(sys, 'frozen', False):
            candidates.insert(0, Path(sys._MEIPASS) / 'train.py')

        for c in candidates:
            if c.is_file():
                return str(c)
        return None

    def _start_training(self):
        if self.process:
            QMessageBox.warning(self, "Busy", "Training is already running.")
            return

        config = self._build_config()
        if not config:
            QMessageBox.warning(self, "No Folder", "Please select a project folder first.")
            return

        # Validate folders exist
        src_dir = Path(config['data']['src_dir'])
        dst_dir = Path(config['data']['dst_dir'])
        model_dir = Path(config['data']['output_dir'])

        if not src_dir.is_dir():
            QMessageBox.warning(self, "Missing Data", f"Source folder not found:\n{src_dir}")
            return
        if not dst_dir.is_dir():
            QMessageBox.warning(self, "Missing Data", f"Destination folder not found:\n{dst_dir}")
            return

        # Ensure model dir exists
        model_dir.mkdir(parents=True, exist_ok=True)

        train_script = self._find_train_script()
        if not train_script:
            QMessageBox.warning(self, "Missing Script", "Cannot find train.py.")
            return

        # Write config
        self.config_file_path = Path(f"{model_dir.name}_simple.yaml")
        with open(self.config_file_path, 'w') as f:
            yaml.dump(config, f, Dumper=IndentDumper, sort_keys=False,
                      default_flow_style=False, indent=2)

        # Build command
        current_os = platform.system()
        if current_os == 'Linux':
            command = ['torchrun', '--standalone', '--nnodes=1', '--nproc_per_node=1',
                       train_script, '--config', str(self.config_file_path)]
        elif current_os in ('Windows', 'Darwin'):
            command = [sys.executable, train_script, '--config', str(self.config_file_path)]
        else:
            QMessageBox.critical(self, "Unsupported OS", f"'{current_os}' is not supported.")
            return

        self.console_output.clear()
        preset_name = self.preset_combo.currentText()
        self.console_output.append(f"Preset: {preset_name}")
        self.console_output.append(f"Resolution: {config['data']['resolution']}  |  "
                                   f"Model: {config['model']['model_size_dims']}  |  "
                                   f"Batch: {config['training']['batch_size']}")
        self.console_output.append(f"Loss: {config['training']['loss']}  |  "
                                   f"LR: {config['training']['lr']}")
        if config.get('mask', {}).get('use_auto_mask'):
            self.console_output.append("Auto-mask: enabled")
        self.console_output.append(f"\nCommand: {' '.join(command)}\n")

        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if current_os == "Windows" else 0

        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=creation_flags,
                encoding='utf-8', errors='replace')
        except Exception as e:
            QMessageBox.critical(self, "Process Error", f"Failed to start training:\n{e}")
            return

        self.stream_reader = ProcessStreamReader(self.process.stdout)
        self.stream_reader.new_text.connect(self._append_text)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.inference_btn.setEnabled(False)

        # Watch for preview images
        model_dir_str = config['data']['output_dir']
        self.preview_image_path = str(Path(model_dir_str) / 'training_preview.jpg')
        self.preview_timer.start()
        self.process_monitor.start()

    def _stop_training(self):
        if self.process and self.process.poll() is None:
            self.console_output.append("\n--- Sending stop signal ---\n")
            if platform.system() == "Windows":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)

    @Slot()
    def _check_process_status(self):
        if not self.process:
            return
        if self.process.poll() is not None:
            self.process_monitor.stop()
            self._on_training_finished()

    def _on_training_finished(self):
        self.preview_timer.stop()
        self.console_output.append("\n--- Training Finished ---\n")
        self.process = None
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.inference_btn.setEnabled(True)
        self._update_preview()

    @Slot()
    def _append_text(self, text):
        self.console_output.moveCursor(QTextCursor.End)
        self.console_output.insertPlainText(text)

    # =========================================================================
    # Inference
    # =========================================================================

    def _run_inference(self):
        """Run inference using the trained model on the src folder."""
        if self.process:
            QMessageBox.warning(self, "Busy", "Training is running. Stop it first.")
            return

        folder = self._get_project_folder()
        if not folder:
            QMessageBox.warning(self, "No Folder", "Please select a project folder first.")
            return

        folder_path = Path(folder)
        model_dir = folder_path / "model"
        src_dir = folder_path / "src"

        # Find latest checkpoint
        checkpoints = sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            QMessageBox.warning(self, "No Model",
                                f"No .pth checkpoint found in:\n{model_dir}\n\n"
                                "Train a model first.")
            return

        latest_ckpt = checkpoints[-1]
        input_dir = str(src_dir)
        output_dir = str(folder_path / "output")

        self.console_output.clear()
        self.console_output.append(f"Running inference...")
        self.console_output.append(f"  Checkpoint: {latest_ckpt.name}")
        self.console_output.append(f"  Input: {input_dir}")
        self.console_output.append(f"  Output: {output_dir}\n")

        # Run inference in a thread
        self.inference_running = True
        self.inference_stop_requested = False
        self.inference_btn.setEnabled(False)
        self.train_btn.setEnabled(False)

        thread = threading.Thread(
            target=self._inference_worker,
            args=(str(latest_ckpt), input_dir, output_dir),
            daemon=True
        )
        thread.start()

    def _inference_worker(self, checkpoint_path, input_dir, output_dir):
        """Run inference in background thread."""
        try:
            # Import inference module
            sys.path.insert(0, str(_APP_DIR))
            from inference import load_checkpoint_for_inference, infer_folder

            model, resolution, loss_mode = load_checkpoint_for_inference(checkpoint_path)
            if model is None:
                QTimer.singleShot(0, lambda: self._inference_done("Failed to load model."))
                return

            os.makedirs(output_dir, exist_ok=True)

            infer_folder(
                model=model,
                input_dir=input_dir,
                output_dir=output_dir,
                resolution=resolution,
                stride=None,  # auto
                half_res=False,
                skip_existing=True,
                use_bce_dice=(loss_mode == 'bce+dice'),
            )

            QTimer.singleShot(0, lambda: self._inference_done("Inference complete."))

        except ImportError:
            # Fallback: run inference.py as subprocess
            QTimer.singleShot(0, lambda: self._inference_subprocess(
                checkpoint_path, input_dir, output_dir))
        except Exception as e:
            QTimer.singleShot(0, lambda: self._inference_done(f"Inference failed: {e}"))

    def _inference_subprocess(self, checkpoint_path, input_dir, output_dir):
        """Fallback: run inference as a subprocess."""
        inference_script = None
        for candidate in [_APP_DIR / 'inference.py',
                          Path(sys.executable).parent / 'inference.py']:
            if candidate.is_file():
                inference_script = str(candidate)
                break

        if not inference_script:
            self._inference_done("Cannot find inference.py")
            return

        command = [
            sys.executable, inference_script,
            '--checkpoint', checkpoint_path,
            '--input', input_dir,
            '--output', output_dir,
            '--skip-existing',
        ]

        self.console_output.append(f"Command: {' '.join(command)}\n")

        try:
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=creation_flags,
                encoding='utf-8', errors='replace')

            self.stream_reader = ProcessStreamReader(self.process.stdout)
            self.stream_reader.new_text.connect(self._append_text)
            self.process_monitor.start()

        except Exception as e:
            self._inference_done(f"Failed to start inference: {e}")

    def _inference_done(self, message):
        self.console_output.append(f"\n{message}\n")
        self.inference_running = False
        self.inference_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.process = None

    # =========================================================================
    # Preview
    # =========================================================================

    @Slot()
    def _update_preview(self):
        if not self.preview_image_path:
            return
        path = Path(self.preview_image_path)
        if path.is_file():
            try:
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    self.original_pixmap = pixmap
                    # Scale to fit preview area
                    scaled = pixmap.scaled(
                        self.preview_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation)
                    self.preview_label.setPixmap(scaled)
            except Exception:
                pass

    # =========================================================================
    # Session save/load
    # =========================================================================

    def _save_session(self):
        try:
            session = {
                'preset': self.preset_combo.currentText(),
                'project_folder': self._get_project_folder() or '',
            }
            with open(_SESSION_FILE, 'w') as f:
                yaml.dump(session, f, default_flow_style=False)
        except Exception:
            pass

    def _load_session(self):
        if _SESSION_FILE.exists():
            try:
                with open(_SESSION_FILE, 'r') as f:
                    session = yaml.safe_load(f)
                if session:
                    preset = session.get('preset', 'Beauty')
                    if preset in PRESETS:
                        self.preset_combo.setCurrentText(preset)
                    folder = session.get('project_folder', '')
                    if folder and Path(folder).is_dir():
                        self._set_project_folder(folder)
            except Exception:
                pass

    def closeEvent(self, event):
        self._save_session()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        event.accept()


# =============================================================================
# Log handler
# =============================================================================

class _QTextEditLogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + '\n'
        QTimer.singleShot(0, lambda: self._append(msg))

    def _append(self, msg):
        self.text_widget.moveCursor(QTextCursor.End)
        self.text_widget.insertPlainText(msg)


# =============================================================================
# Entry point
# =============================================================================

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = SimpleWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
