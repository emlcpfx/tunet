# ==============================================================================
# TuNet Combined UI — Training + Inference
#
# A unified PySide6 desktop application combining training control and
# batch inference into a single window, organized by workflow stage.
#
# Tabs: Data | Training | Previews | Export | Inference | About
#
# require: pip install PySide6
# ==============================================================================

import sys
import os
import json
import math
import queue
import shutil
import subprocess
import tempfile
import threading
import signal
import yaml
import logging
import time
import re
from pathlib import Path
from glob import glob as globglob
import platform

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QFileDialog, QFormLayout, QMessageBox, QSizePolicy, QScrollArea,
    QSplitter, QProgressDialog, QDoubleSpinBox, QListWidget, QGroupBox,
    QProgressBar, QStackedWidget, QFrame,
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QFileSystemWatcher, QTimer, QThread
from PySide6.QtGui import QPixmap, QTextCursor


# =============================================================================
# Helper Classes
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


class FileCopyWorker(QObject):
    """Copies a file in a background thread with progress reporting."""
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)
    _is_cancelled = False

    def run(self, source_path_str, dest_path_str):
        self._is_cancelled = False
        source_path = Path(source_path_str)
        dest_path = Path(dest_path_str)
        try:
            total_size = source_path.stat().st_size
            bytes_copied = 0
            chunk_size = 4096 * 4
            with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
                while True:
                    if self._is_cancelled:
                        self.error.emit("Copy operation cancelled.")
                        if dest_path.exists():
                            dest_path.unlink()
                        return
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    bytes_copied += len(chunk)
                    percent = int((bytes_copied / total_size) * 100)
                    self.progress.emit(percent)
            self.finished.emit(str(dest_path))
        except Exception as e:
            self.error.emit(f"File copy failed: {e}")

    def cancel(self):
        self._is_cancelled = True


class ZoomPanScrollArea(QScrollArea):
    """Scroll area with mouse-wheel zoom and middle-click pan."""
    zoom_changed = Signal(float)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.zoom_changed.emit(factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, '_panning', False) and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class QTextEditLogHandler(logging.Handler):
    """Logging handler that writes to a QTextEdit from any thread."""
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
# Session file paths
# =============================================================================
_APP_DIR = Path(__file__).parent
_SESSION_FILE = _APP_DIR / 'tunet_session.yaml'
_LEGACY_INFERENCE_SETTINGS = _APP_DIR / 'inference_gui_settings.json'


# =============================================================================
# MainWindow
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"TuNet — {platform.system()}")
        self.setGeometry(100, 100, 1000, 750)

        # --- Training state ---
        self.process = None
        self.utility_process = None
        self.config_file_path = Path("config_from_ui.yaml")
        self.training_queue = []
        self.queue_running = False
        self.queue_stop_requested = False
        self._last_config_dir = ""
        self.copy_thread = None
        self.conversion_target = None

        # --- Preview state ---
        self.preview_image_path = None
        self.val_preview_image_path = None
        self.original_pixmap = None
        self.val_original_pixmap = None
        self._preview_zoom_factor = None
        self._val_preview_zoom_factor = None

        # --- Inference state ---
        self.inference_running = False
        self.inference_stop_requested = False
        self.inference_queue = []
        self._cached_model = None
        self._cached_checkpoint = None
        self._cached_resolution = None
        self._cached_loss_mode = None

        # --- Build UI ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left side: tabs + console + bottom controls
        left_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFontFamily("monospace")

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.console_output)
        splitter.setSizes([550, 150])

        left_layout.addWidget(splitter)
        left_layout.addLayout(self._create_control_panel())
        self.main_layout.addLayout(left_layout, stretch=1)

        # Right side: context-sensitive sidebar
        self.main_layout.addLayout(self._create_sidebar())

        # --- Create tabs ---
        self._create_data_tab()          # Tab 0
        self._create_training_tab()      # Tab 1
        self._create_previews_tab()      # Tab 2
        self._create_export_tab()        # Tab 3
        self._create_inference_tab()     # Tab 4
        self._create_about_tab()         # Tab 5

        # Remember the inference tab index for sidebar switching
        self._inference_tab_index = 4

        # --- Signals ---
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # --- File watcher for previews ---
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.fileChanged.connect(self._on_watched_file_changed)

        # --- Timers ---
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(2500)
        self.preview_timer.timeout.connect(self._update_all_previews)

        self.process_monitor = QTimer(self)
        self.process_monitor.setInterval(500)
        self.process_monitor.timeout.connect(self._check_process_status)

        self.utility_monitor = QTimer(self)
        self.utility_monitor.setInterval(500)
        self.utility_monitor.timeout.connect(self._check_utility_status)

        # --- Logging handler for inference ---
        self._log_handler = QTextEditLogHandler(self.console_output)
        self._log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(self._log_handler)
        logging.getLogger().setLevel(logging.INFO)

        # --- Populate defaults ---
        self._populate_default_script_path()
        self._load_session()

    # =========================================================================
    # Tab creation
    # =========================================================================

    def _create_data_tab(self):
        """Tab 1: Data — paths, patch extraction, masks, augmentation."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Training Script ---
        grp_script = QGroupBox("Training Script")
        form_script = QFormLayout(grp_script)
        self.train_script_input = self._create_path_selector("Training Script", is_file=True)
        self.train_script_input.setToolTip(
            "Path to train.py that runs the training loop. Usually auto-detected.")
        form_script.addRow("Script:", self.train_script_input)
        if platform.system() == 'Linux':
            self.nproc_input = QSpinBox(minimum=1, maximum=16, value=1)
            self.nproc_input.setToolTip("Number of GPUs for distributed training via torchrun.")
            form_script.addRow("GPUs (nproc):", self.nproc_input)
        else:
            self.nproc_input = None
        layout.addWidget(grp_script)

        # --- Source & Target Data ---
        grp_data = QGroupBox("Source & Target Data")
        form_data = QFormLayout(grp_data)
        self.src_dir_input = self._create_path_selector("Source Directory")
        self.src_dir_input.setToolTip(
            "Folder with source/input images the model learns to transform FROM (the 'before' images).")
        self.dst_dir_input = self._create_path_selector("Destination Directory")
        self.dst_dir_input.setToolTip(
            "Folder with target/output images the model learns to transform TO (the 'after' images). "
            "Must have matching filenames.")
        self.mask_dir_input = self._create_path_selector("Mask Directory")
        self.mask_dir_input.setToolTip(
            "Optional. Grayscale mask images (white = important regions). "
            "Not needed if Auto Mask is enabled.")
        self.auto_mask_hint = QLabel("")
        self.auto_mask_hint.setStyleSheet("color: gray; font-style: italic;")
        form_data.addRow("Source Directory:", self.src_dir_input)
        form_data.addRow("Target Directory:", self.dst_dir_input)
        form_data.addRow("Mask Directory:", self.mask_dir_input)
        form_data.addRow("", self.auto_mask_hint)
        layout.addWidget(grp_data)

        # --- Validation Data ---
        grp_val = QGroupBox("Validation Data (optional)")
        form_val = QFormLayout(grp_val)
        self.val_src_dir_input = self._create_path_selector("Val Source Directory")
        self.val_src_dir_input.setToolTip(
            "Separate source images used only for validation (never trained on). "
            "Helps monitor generalization.")
        self.val_dst_dir_input = self._create_path_selector("Val Destination Directory")
        self.val_dst_dir_input.setToolTip(
            "Matching target images for validation.")
        form_val.addRow("Val Source:", self.val_src_dir_input)
        form_val.addRow("Val Target:", self.val_dst_dir_input)
        layout.addWidget(grp_val)

        # --- Patch Extraction ---
        grp_patch = QGroupBox("Patch Extraction")
        form_patch = QFormLayout(grp_patch)
        self.resolution_input = QComboBox()
        self.resolution_input.setEditable(True)
        self.resolution_input.addItems(["256", "384", "512", "640", "768", "896", "928", "960", "1024"])
        self.resolution_input.setCurrentText("512")
        self.resolution_input.setToolTip(
            "Size of square patches extracted for training. Larger = more context but more VRAM. "
            "512 is a good default; 768-1024 for large-scale structure.")
        self.overlap_factor_input = QComboBox()
        self.overlap_factor_input.addItems(["0.0", "0.25", "0.5", "0.75"])
        self.overlap_factor_input.setCurrentText("0.25")
        self.overlap_factor_input.setToolTip(
            "How much neighboring patches overlap. Higher = more training patches but slower epochs. "
            "0.25 is a good balance.")
        form_patch.addRow("Resolution:", self.resolution_input)
        form_patch.addRow("Overlap Factor:", self.overlap_factor_input)
        layout.addWidget(grp_patch)

        # --- Mask Behavior ---
        grp_mask = QGroupBox("Mask Behavior")
        form_mask = QFormLayout(grp_mask)

        self.use_mask_loss_input = QCheckBox("Weight loss by mask (white = important)")
        self.use_mask_loss_input.setToolTip(
            "Training loss is weighted by mask: white pixels count more, "
            "making the model focus on those areas.")
        self.mask_weight_input = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=100.0, value=10.0, singleStep=1.0)
        self.mask_weight_input.setToolTip(
            "How much more important masked (white) regions are. "
            "10 = white pixels contribute 10x more to loss. Only active when mask loss is checked.")
        self.mask_weight_input.setEnabled(False)
        self.use_mask_loss_input.toggled.connect(self.mask_weight_input.setEnabled)

        self.use_mask_input_input = QCheckBox("Feed mask as 4th input channel to model")
        self.use_mask_input_input.setToolTip(
            "Feed the mask to the model as a 4th channel alongside RGB. "
            "Changes architecture (cannot toggle mid-training).")
        self.use_auto_mask_input = QCheckBox("Auto-generate masks from |src - dst| difference")
        self.use_auto_mask_input.setToolTip(
            "Automatically create masks from the difference between source and target. "
            "No mask files needed on disk.")
        self.use_auto_mask_input.toggled.connect(
            lambda checked: self.auto_mask_hint.setText(
                "(Mask directory not needed with Auto Mask)" if checked else ""))

        self.skip_empty_patches_input = QCheckBox("Skip empty patches")
        self.skip_empty_patches_input.setToolTip(
            "Filter out training patches where source and destination are identical. "
            "Speeds up training when only parts of the image have changes. Requires Auto Mask.")
        self.skip_empty_patches_input.setEnabled(False)
        self.use_auto_mask_input.toggled.connect(self.skip_empty_patches_input.setEnabled)

        self.skip_empty_threshold_input = QDoubleSpinBox()
        self.skip_empty_threshold_input.setRange(0.1, 20.0)
        self.skip_empty_threshold_input.setSingleStep(0.5)
        self.skip_empty_threshold_input.setDecimals(1)
        self.skip_empty_threshold_input.setValue(1.0)
        self.skip_empty_threshold_input.setToolTip(
            "Max pixel difference threshold (0-255 scale) below which a patch is skipped. "
            "If no pixel in the crop differs by more than this, the crop is considered empty.")
        self.skip_empty_threshold_input.setEnabled(False)
        self.skip_empty_patches_input.toggled.connect(self.skip_empty_threshold_input.setEnabled)

        mask_weight_row = QHBoxLayout()
        mask_weight_row.addWidget(self.use_mask_loss_input)
        mask_weight_row.addWidget(QLabel("Weight:"))
        mask_weight_row.addWidget(self.mask_weight_input)
        form_mask.addRow(mask_weight_row)
        form_mask.addRow(self.use_mask_input_input)
        form_mask.addRow(self.use_auto_mask_input)
        skip_empty_row = QHBoxLayout()
        skip_empty_row.addWidget(self.skip_empty_patches_input)
        skip_empty_row.addWidget(QLabel("Threshold:"))
        skip_empty_row.addWidget(self.skip_empty_threshold_input)
        form_mask.addRow(skip_empty_row)
        layout.addWidget(grp_mask)

        # --- Data Augmentation ---
        grp_aug = QGroupBox("Data Augmentation")
        form_aug = QFormLayout(grp_aug)

        # Horizontal Flip
        self.hflip_check = QCheckBox("Horizontal Flip")
        self.hflip_check.setToolTip("Randomly flip images horizontally. Disable for text or directional content.")
        self.hflip_p = QSpinBox(minimum=0, maximum=100, value=50)
        self.hflip_p.setSuffix("%")
        self.hflip_p.setToolTip("Chance each sample gets flipped.")
        self.hflip_p.setEnabled(False)
        self.hflip_check.toggled.connect(self.hflip_p.setEnabled)
        hflip_row = QHBoxLayout()
        hflip_row.addWidget(self.hflip_check)
        hflip_row.addWidget(self.hflip_p)
        form_aug.addRow(hflip_row)

        # Affine
        self.affine_check = QCheckBox("Random Affine")
        self.affine_check.setToolTip(
            "Random scale/translate/rotate/shear. Helps robustness to alignment differences.")
        self.affine_p = QSpinBox(minimum=0, maximum=100, value=40)
        self.affine_p.setSuffix("%")
        self.affine_p.setToolTip("Chance each sample gets an affine transform.")
        affine_header = QHBoxLayout()
        affine_header.addWidget(self.affine_check)
        affine_header.addWidget(self.affine_p)
        form_aug.addRow(affine_header)

        self.affine_scale_min = QDoubleSpinBox(decimals=2, minimum=0.1, maximum=5.0, value=0.9, singleStep=0.05)
        self.affine_scale_max = QDoubleSpinBox(decimals=2, minimum=0.1, maximum=5.0, value=1.1, singleStep=0.05)
        self.affine_scale_min.setToolTip("Min zoom. 0.9 = up to 10% zoom out.")
        self.affine_scale_max.setToolTip("Max zoom. 1.1 = up to 10% zoom in.")
        form_aug.addRow("  Scale:", self._create_range_layout(self.affine_scale_min, self.affine_scale_max))

        self.affine_translate_min = QDoubleSpinBox(decimals=2, minimum=-0.5, maximum=0.5, value=-0.1, singleStep=0.01)
        self.affine_translate_max = QDoubleSpinBox(decimals=2, minimum=-0.5, maximum=0.5, value=0.1, singleStep=0.01)
        form_aug.addRow("  Translate %:", self._create_range_layout(self.affine_translate_min, self.affine_translate_max))

        self.affine_rotate_min = QSpinBox(minimum=-180, maximum=180, value=-3)
        self.affine_rotate_max = QSpinBox(minimum=-180, maximum=180, value=3)
        self.affine_rotate_min.setToolTip("Rotation range in degrees.")
        form_aug.addRow("  Rotate:", self._create_range_layout(self.affine_rotate_min, self.affine_rotate_max))

        self.affine_shear_min = QSpinBox(minimum=-45, maximum=45, value=-1)
        self.affine_shear_max = QSpinBox(minimum=-45, maximum=45, value=1)
        self.affine_shear_min.setToolTip("Shear range in degrees. Skews the image diagonally.")
        form_aug.addRow("  Shear:", self._create_range_layout(self.affine_shear_min, self.affine_shear_max))

        self.affine_interpolation = QSpinBox(minimum=0, maximum=5, value=2)
        self.affine_interpolation.setToolTip("0=Nearest, 1=Bilinear, 2=Cubic (recommended).")
        self.affine_keep_ratio = QCheckBox()
        self.affine_keep_ratio.setChecked(True)
        self.affine_keep_ratio.setToolTip("Keep aspect ratio when scaling. Important for paired images.")
        form_aug.addRow("  Interpolation:", self.affine_interpolation)
        form_aug.addRow("  Keep Ratio:", self.affine_keep_ratio)

        # Collect affine sub-widgets for conditional enable
        self._affine_sub_widgets = [
            self.affine_p, self.affine_scale_min, self.affine_scale_max,
            self.affine_translate_min, self.affine_translate_max,
            self.affine_rotate_min, self.affine_rotate_max,
            self.affine_shear_min, self.affine_shear_max,
            self.affine_interpolation, self.affine_keep_ratio,
        ]
        for w in self._affine_sub_widgets:
            w.setEnabled(False)
        self.affine_check.toggled.connect(
            lambda checked: [w.setEnabled(checked) for w in self._affine_sub_widgets])

        # Random Gamma
        self.gamma_check = QCheckBox("Random Gamma")
        self.gamma_check.setToolTip("Randomly adjust brightness curve. Helps with varying exposure.")
        self.gamma_p = QSpinBox(minimum=0, maximum=100, value=20)
        self.gamma_p.setSuffix("%")
        self.gamma_p.setToolTip("Chance each sample gets a gamma adjustment.")
        gamma_header = QHBoxLayout()
        gamma_header.addWidget(self.gamma_check)
        gamma_header.addWidget(self.gamma_p)
        form_aug.addRow(gamma_header)

        self.gamma_limit_min = QSpinBox(minimum=0, maximum=255, value=40)
        self.gamma_limit_max = QSpinBox(minimum=0, maximum=255, value=160)
        self.gamma_limit_min.setToolTip("100 = no change. Below 100 = darken, above 100 = brighten.")
        form_aug.addRow("  Gamma Limit:", self._create_range_layout(self.gamma_limit_min, self.gamma_limit_max))

        self._gamma_sub_widgets = [self.gamma_p, self.gamma_limit_min, self.gamma_limit_max]
        for w in self._gamma_sub_widgets:
            w.setEnabled(False)
        self.gamma_check.toggled.connect(
            lambda checked: [w.setEnabled(checked) for w in self._gamma_sub_widgets])

        # Color Augmentation (brightness, contrast, saturation)
        self.color_check = QCheckBox("Color Augmentation")
        self.color_check.setToolTip(
            "Randomly adjust brightness, contrast, and saturation.\n"
            "Applied identically to source and target pairs.\n"
            "Helps the model generalize across different lighting conditions and color grades.\n\n"
            "Use when: your dataset has varied lighting, or you want robustness to color shifts.\n"
            "Avoid when: precise color matching is critical (e.g. exact color correction tasks).")
        self.color_p = QSpinBox(minimum=0, maximum=100, value=30)
        self.color_p.setSuffix("%")
        self.color_p.setToolTip("Chance each sample gets a color adjustment.")
        color_header = QHBoxLayout()
        color_header.addWidget(self.color_check)
        color_header.addWidget(self.color_p)
        form_aug.addRow(color_header)

        self.color_brightness_min = QDoubleSpinBox(decimals=2, minimum=-1.0, maximum=1.0, value=-0.2, singleStep=0.05)
        self.color_brightness_max = QDoubleSpinBox(decimals=2, minimum=-1.0, maximum=1.0, value=0.2, singleStep=0.05)
        self.color_brightness_min.setToolTip(
            "Brightness adjustment range. 0 = no change.\n"
            "Negative = darker, positive = brighter. ±0.2 is a safe default.")
        form_aug.addRow("  Brightness:", self._create_range_layout(self.color_brightness_min, self.color_brightness_max))

        self.color_contrast_min = QDoubleSpinBox(decimals=2, minimum=-1.0, maximum=1.0, value=-0.2, singleStep=0.05)
        self.color_contrast_max = QDoubleSpinBox(decimals=2, minimum=-1.0, maximum=1.0, value=0.2, singleStep=0.05)
        self.color_contrast_min.setToolTip(
            "Contrast adjustment range. 0 = no change.\n"
            "Negative = lower contrast, positive = higher contrast.")
        form_aug.addRow("  Contrast:", self._create_range_layout(self.color_contrast_min, self.color_contrast_max))

        self.color_saturation_min = QSpinBox(minimum=-100, maximum=100, value=-30)
        self.color_saturation_max = QSpinBox(minimum=-100, maximum=100, value=30)
        self.color_saturation_min.setToolTip(
            "Saturation shift range. 0 = no change.\n"
            "Negative = desaturate toward grayscale, positive = boost color intensity.")
        form_aug.addRow("  Saturation:", self._create_range_layout(self.color_saturation_min, self.color_saturation_max))

        self._color_sub_widgets = [
            self.color_p, self.color_brightness_min, self.color_brightness_max,
            self.color_contrast_min, self.color_contrast_max,
            self.color_saturation_min, self.color_saturation_max,
        ]
        for w in self._color_sub_widgets:
            w.setEnabled(False)
        self.color_check.toggled.connect(
            lambda checked: [w.setEnabled(checked) for w in self._color_sub_widgets])

        layout.addWidget(grp_aug)
        layout.addStretch()

        # Wrap in scroll area for small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        self.tabs.addTab(scroll, "Data")

    def _apply_preset(self, preset_name):
        """Apply a training preset, adjusting relevant settings."""
        if preset_name == "Custom":
            return
        presets = {
            "Beauty / Paint Fix": {
                "model_type": "msrn",
                "model_size_dims": "128",
                "resolution": "512",
                "overlap_factor": "0.5",
                "loss": "l1+lpips",
                "lambda_lpips": 0.2,
                "lr": 1e-4,
                "use_auto_mask": True,
                "skip_empty_patches": True,
                "progressive_resolution": False,
            },
            "Roto / Matte": {
                "model_type": "unet",
                "model_size_dims": "128",
                "resolution": "512",
                "overlap_factor": "0.25",
                "loss": "bce+dice",
                "lambda_lpips": 0.0,
                "lr": 3e-4,
                "use_auto_mask": False,
                "skip_empty_patches": False,
                "progressive_resolution": True,
            },
        }
        p = presets.get(preset_name)
        if not p:
            return
        self.model_type_input.setCurrentText(p["model_type"])
        self.model_size_dims_input.setCurrentText(p["model_size_dims"])
        self.resolution_input.setCurrentText(p["resolution"])
        self.overlap_factor_input.setCurrentText(p["overlap_factor"])
        self.loss_input.setCurrentText(p["loss"])
        # Set LR by finding matching preset value
        for i, (_, val) in enumerate(self.lr_presets):
            if abs(val - p["lr"]) < 1e-8:
                self.lr_input.setCurrentIndex(i)
                break
        # Set LPIPS lambda by finding matching preset value
        for i, (_, val) in enumerate(self.lambda_presets):
            if abs(val - p["lambda_lpips"]) < 1e-8:
                self.lambda_lpips_input.setCurrentIndex(i)
                break
        self.use_auto_mask_input.setChecked(p["use_auto_mask"])
        self.skip_empty_patches_input.setChecked(p["skip_empty_patches"])
        self.progressive_res_check.setChecked(p.get("progressive_resolution", False))

    def _create_training_tab(self):
        """Tab 2: Training — model, optimization, schedule, logging."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Preset ---
        grp_preset = QGroupBox("Preset")
        form_preset = QFormLayout(grp_preset)
        self.preset_input = QComboBox()
        self.preset_input.addItems(["Custom", "Beauty / Paint Fix", "Roto / Matte"])
        self.preset_input.setToolTip(
            "Quick-start presets that configure model, loss, and patch settings.\n"
            "Select a preset then adjust individual settings as needed.\n"
            "Changing any setting afterwards keeps your changes (won't revert).")
        self.preset_input.currentTextChanged.connect(self._apply_preset)
        form_preset.addRow("Training Preset:", self.preset_input)
        layout.addWidget(grp_preset)

        # --- Model ---
        grp_model = QGroupBox("Model")
        form_model = QFormLayout(grp_model)
        self.model_folder_input = self._create_path_selector("Model Folder", is_output=True)
        self.model_folder_input.setToolTip(
            "Folder where checkpoints, logs, and preview images are saved. "
            "Each training run should have its own folder.")
        self.model_type_input = QComboBox()
        self.model_type_input.addItems(["unet", "msrn"])
        self.model_type_input.setCurrentText("unet")
        self.model_type_input.setToolTip(
            "Architecture type. 'unet' is fast and general-purpose. "
            "'msrn' uses attention and recurrence for better fine detail but trains slower.")
        self.model_size_dims_input = QComboBox()
        self.model_size_dims_input.addItems(["32", "64", "128", "256", "512"])
        self.model_size_dims_input.setCurrentText("128")
        self.model_size_dims_input.setToolTip(
            "Hidden layer width. Larger = more capacity but more VRAM. "
            "64 = lightweight, 128 = balanced, 256+ = complex transformations.")
        form_model.addRow("Output Folder:", self.model_folder_input)
        form_model.addRow("Model Type:", self.model_type_input)
        form_model.addRow("Model Capacity:", self.model_size_dims_input)
        layout.addWidget(grp_model)

        # --- Optimization ---
        grp_opt = QGroupBox("Optimization")
        form_opt = QFormLayout(grp_opt)

        self.lr_input = QComboBox()
        self.lr_presets = [
            ("Slow (5e-5) — Fine texture, lots of data", 5e-5),
            ("Default (1e-4) — General purpose", 1e-4),
            ("Medium (3e-4) — Moderate changes", 3e-4),
            ("Fast (5e-4) — Beauty work, subtle fixes", 5e-4),
            ("Aggressive (1e-3) — Large changes, quick experiments", 1e-3),
        ]
        for label, _ in self.lr_presets:
            self.lr_input.addItem(label)
        self.lr_input.setCurrentIndex(1)
        self.lr_input.setToolTip(
            "How fast the model updates weights. Start with Default (1e-4). "
            "Slower = safer, faster = quicker convergence but riskier.")
        form_opt.addRow("Learning Rate:", self.lr_input)

        self.loss_input = QComboBox()
        self.loss_input.addItems(["l1", "l1+lpips", "weighted", "bce+dice"])
        self.loss_input.setToolTip(
            "'l1' = pixel-level absolute difference (sharp, stable).\n"
            "'l1+lpips' = pixel + perceptual similarity (better textures).\n"
            "'weighted' = custom mix of L1 + L2 + LPIPS with individual weight sliders.\n"
            "'bce+dice' = for binary mask/segmentation outputs.")
        form_opt.addRow("Loss Function:", self.loss_input)

        self.lambda_lpips_input = QComboBox()
        self.lambda_presets = [
            ("Off (0.0) — Pure L1 pixel loss", 0.0),
            ("Light (0.05) — Mostly pixel accuracy", 0.05),
            ("Default (0.1) — Recommended balance", 0.1),
            ("Medium (0.2) — More perceptual texture", 0.2),
            ("High (0.3) — Strong perceptual push", 0.3),
            ("Risky (0.5) — LPIPS dominates, watch for artifacts", 0.5),
        ]
        for label, _ in self.lambda_presets:
            self.lambda_lpips_input.addItem(label)
        self.lambda_lpips_input.setCurrentIndex(2)
        self.lambda_lpips_input.setToolTip(
            "Balance between pixel accuracy (L1) and perceptual quality (LPIPS). "
            "Only active when loss is 'l1+lpips'.")
        self.lambda_lpips_input.setEnabled(False)
        form_opt.addRow("LPIPS Lambda:", self.lambda_lpips_input)

        # --- Weighted loss controls (visible only when loss = "weighted") ---
        self.l1_weight_input = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=10.0, value=1.0, singleStep=0.05)
        self.l1_weight_input.setToolTip(
            "Weight for L1 (Mean Absolute Error) loss.\n"
            "Treats all pixel errors equally. Good baseline for sharpness.\n"
            "Default 1.0. Set to 0 to disable.")
        self.l2_weight_input = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=10.0, value=0.0, singleStep=0.05)
        self.l2_weight_input.setToolTip(
            "Weight for L2 (Mean Squared Error) loss.\n"
            "Penalizes large errors more than small ones — smoother results.\n"
            "Try 0.1–0.5 alongside L1 for a blend of sharp + smooth.")
        self.lpips_weight_input = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=10.0, value=0.1, singleStep=0.05)
        self.lpips_weight_input.setToolTip(
            "Weight for LPIPS perceptual loss.\n"
            "Matches structures and textures rather than raw pixels.\n"
            "Makes models less brittle to small misalignments.\n"
            "Keep below 0.5 to avoid artifacts. 0.1 is a safe start.")
        weighted_row = QHBoxLayout()
        weighted_row.addWidget(QLabel("L1:"))
        weighted_row.addWidget(self.l1_weight_input)
        weighted_row.addWidget(QLabel("L2:"))
        weighted_row.addWidget(self.l2_weight_input)
        weighted_row.addWidget(QLabel("LPIPS:"))
        weighted_row.addWidget(self.lpips_weight_input)
        self.weighted_loss_widget = QWidget()
        self.weighted_loss_widget.setLayout(weighted_row)
        self.weighted_loss_widget.setVisible(False)
        form_opt.addRow("Loss Weights:", self.weighted_loss_widget)

        def _on_loss_changed(t):
            self.lambda_lpips_input.setEnabled(t == "l1+lpips")
            self.weighted_loss_widget.setVisible(t == "weighted")
        self.loss_input.currentTextChanged.connect(_on_loss_changed)

        self.use_amp_input = QCheckBox("Enable fp16 Mixed Precision")
        self.use_amp_input.setChecked(True)
        self.use_amp_input.setToolTip(
            "Nearly 2x faster, less VRAM, negligible quality impact. "
            "Disable only if you see NaN losses.")
        form_opt.addRow("Mixed Precision:", self.use_amp_input)
        layout.addWidget(grp_opt)

        # --- Fine-tune ---
        grp_finetune = QGroupBox("Fine-tune (Optional)")
        form_finetune = QFormLayout(grp_finetune)
        self.finetune_from_input = self._create_path_selector(
            "Fine-tune From", is_file=True, file_filter="PyTorch Checkpoints (*.pth)")
        self.finetune_from_input.setToolTip(
            "Pick an existing .pth model to fine-tune on new data.\n"
            "Only model weights are loaded — optimizer and step counter start fresh.\n"
            "Just point Source/Destination to your new dataset and hit Train.\n\n"
            "Leave empty to train from scratch (or auto-resume if output folder has a checkpoint).")
        form_finetune.addRow("Starting Checkpoint:", self.finetune_from_input)
        layout.addWidget(grp_finetune)

        # --- Schedule ---
        grp_sched = QGroupBox("Schedule")
        form_sched = QFormLayout(grp_sched)

        self.batch_size_input = QSpinBox(minimum=1, maximum=256, value=4)
        self.batch_size_input.setToolTip(
            "Patches per training step. Larger = more stable gradients but more VRAM. "
            "Reduce to 2 if you get out-of-memory errors.")
        form_sched.addRow("Batch Size (per GPU):", self.batch_size_input)

        self.iter_per_epoch_input = QSpinBox(minimum=1, maximum=10000, value=500)
        self.iter_per_epoch_input.setToolTip(
            "Steps before saving a checkpoint and running validation. "
            "Lower = more frequent saves.")
        form_sched.addRow("Iterations per Epoch:", self.iter_per_epoch_input)

        self.max_steps_input = QSpinBox(minimum=0, maximum=10000000, value=0)
        self.max_steps_input.setSpecialValueText("Unlimited")
        self.max_steps_input.setToolTip(
            "Total steps before auto-stopping. 0 = train until manually stopped. "
            "Required for queue items.")
        form_sched.addRow("Max Steps:", self.max_steps_input)

        self.progressive_res_check = QCheckBox("Progressive Multi-Resolution")
        self.progressive_res_check.setToolTip(
            "Start training at lower resolutions and progressively increase to full.\n"
            "Speeds up early training ~2x by learning coarse structure first.\n\n"
            "How it works:\n"
            "  Epoch 1 → trains at 1/4 resolution (fast, learns shapes & layout)\n"
            "  Epoch 2 → trains at 1/2 resolution (medium detail)\n"
            "  Epoch 3+ → trains at full resolution (fine detail)\n\n"
            "Best for: large datasets, high resolutions (512+), long training runs.\n"
            "Skip when: resolution is already small (256), very short runs, or fine-tuning.")
        form_sched.addRow(self.progressive_res_check)

        self.num_workers_input = QComboBox()
        self.num_workers_presets = [
            ("Auto (Recommended)", -1),
            ("0 — Disabled (debug)", 0),
            ("2 — Light", 2),
            ("4 — Moderate", 4),
            ("8 — Heavy (many CPU cores)", 8),
        ]
        for label, _ in self.num_workers_presets:
            self.num_workers_input.addItem(label)
        self.num_workers_input.setCurrentIndex(0)
        self.num_workers_input.setToolTip(
            "CPU threads loading data in parallel. 'Auto' picks based on hardware. "
            "Set to 0 for debugging.")
        form_sched.addRow("DataLoader Workers:", self.num_workers_input)
        layout.addWidget(grp_sched)

        # --- Logging & Checkpoints ---
        grp_log = QGroupBox("Logging & Checkpoints")
        form_log = QFormLayout(grp_log)

        self.log_interval_input = QSpinBox(minimum=1, maximum=1000, value=5)
        self.log_interval_input.setToolTip("Print loss to console every N steps.")
        form_log.addRow("Log Interval:", self.log_interval_input)

        self.preview_interval_input = QSpinBox(minimum=0, maximum=1000, value=35)
        self.preview_interval_input.setToolTip(
            "Save preview image (src / dst / model output) every N steps. 0 = disable.")
        form_log.addRow("Preview Interval:", self.preview_interval_input)

        self.preview_refresh_input = QSpinBox(minimum=0, maximum=1000, value=5)
        self.preview_refresh_input.setToolTip(
            "How many preview saves before the viewer refreshes.")
        form_log.addRow("Preview Refresh Rate:", self.preview_refresh_input)

        self.keep_checkpoints_input = QSpinBox(minimum=1, maximum=50, value=4)
        self.keep_checkpoints_input.setToolTip(
            "Number of old checkpoints to keep (plus the latest).")
        form_log.addRow("Keep Checkpoints:", self.keep_checkpoints_input)
        layout.addWidget(grp_log)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        self.tabs.addTab(scroll, "Training")

    def _create_inference_tab(self):
        """Tab 5: Inference — apply a trained model to new images."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Model ---
        grp_model = QGroupBox("Model")
        form_model = QFormLayout(grp_model)

        self.inf_checkpoint_input = self._create_path_selector("Checkpoint", is_file=True, file_filter="PyTorch Checkpoints (*.pth)")
        self.inf_checkpoint_input.setToolTip(
            "Path to a trained .pth checkpoint. Architecture and settings are auto-detected.")
        form_model.addRow("Checkpoint:", self.inf_checkpoint_input)

        self.inf_use_latest_btn = QPushButton("Use Latest from Training Folder")
        self.inf_use_latest_btn.setToolTip(
            "Find the most recent .pth checkpoint from the Model Output Folder in the Training tab.")
        self.inf_use_latest_btn.clicked.connect(self._inf_use_latest_checkpoint)
        form_model.addRow("", self.inf_use_latest_btn)
        layout.addWidget(grp_model)

        # --- Input / Output ---
        grp_io = QGroupBox("Input / Output")
        form_io = QFormLayout(grp_io)

        self.inf_input_dir = self._create_path_selector("Input Directory")
        self.inf_input_dir.setToolTip(
            "Folder of images to process. Supports PNG, JPG, TIFF, EXR, BMP, WebP.")
        form_io.addRow("Input Directory:", self.inf_input_dir)

        self.inf_output_root = self._create_path_selector("Output Root Directory")
        self.inf_output_root.setToolTip(
            "Root folder for output. Each run creates a versioned subfolder: "
            "inputname_modelname_v001 (auto-increments).")
        form_io.addRow("Output Root:", self.inf_output_root)
        layout.addWidget(grp_io)

        # --- Processing Options ---
        grp_opts = QGroupBox("Processing Options")
        form_opts = QFormLayout(grp_opts)

        stride_row = QHBoxLayout()
        self.inf_stride = QSpinBox(minimum=64, maximum=1024, value=256, singleStep=64)
        self.inf_stride.setToolTip(
            "Step size in pixels between overlapping tiles. Smaller = better blending but slower.")
        self.inf_auto_stride = QCheckBox("Auto")
        self.inf_auto_stride.setChecked(True)
        self.inf_auto_stride.setToolTip(
            "Compute optimal stride from the first image's dimensions for complete pixel coverage.")
        self.inf_stride.setEnabled(False)
        self.inf_auto_stride.toggled.connect(lambda c: self.inf_stride.setEnabled(not c))
        stride_row.addWidget(self.inf_stride)
        stride_row.addWidget(self.inf_auto_stride)
        form_opts.addRow("Stride:", stride_row)

        self.inf_half_res = QCheckBox("Half Resolution (~4x faster)")
        self.inf_half_res.setToolTip("Process at half resolution. Useful for quick previews.")
        form_opts.addRow(self.inf_half_res)

        self.inf_skip_existing = QCheckBox("Skip Existing Files")
        self.inf_skip_existing.setChecked(True)
        self.inf_skip_existing.setToolTip(
            "Skip files that already exist. Resume partial runs without re-processing.")
        form_opts.addRow(self.inf_skip_existing)
        layout.addWidget(grp_opts)

        # --- Progress ---
        grp_progress = QGroupBox("Progress")
        progress_layout = QVBoxLayout(grp_progress)
        self.inf_progress_bar = QProgressBar()
        self.inf_progress_bar.setValue(0)
        self.inf_progress_label = QLabel("")
        progress_layout.addWidget(self.inf_progress_bar)
        progress_layout.addWidget(self.inf_progress_label)
        layout.addWidget(grp_progress)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.inf_run_btn = QPushButton("Run Inference")
        self.inf_run_btn.setStyleSheet("background-color: #a8e6cf; color: #1a1a1a;")
        self.inf_run_btn.clicked.connect(self._inf_run_single)
        self.inf_run_queue_btn = QPushButton("Run Queue")
        self.inf_run_queue_btn.setStyleSheet("background-color: #a8e6cf; color: #1a1a1a;")
        self.inf_run_queue_btn.clicked.connect(self._inf_run_queue)
        self.inf_stop_btn = QPushButton("Stop")
        self.inf_stop_btn.setStyleSheet("background-color: #ff8a80; color: #1a1a1a;")
        self.inf_stop_btn.setEnabled(False)
        self.inf_stop_btn.clicked.connect(self._inf_request_stop)
        btn_layout.addWidget(self.inf_run_btn)
        btn_layout.addWidget(self.inf_run_queue_btn)
        btn_layout.addWidget(self.inf_stop_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        self.tabs.addTab(scroll, "Inference")

    def _create_previews_tab(self):
        """Tab 4: Previews — training + validation preview with toggle."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # Toggle buttons
        toggle_layout = QHBoxLayout()
        self.preview_train_btn = QPushButton("Training Preview")
        self.preview_train_btn.setCheckable(True)
        self.preview_train_btn.setChecked(True)
        self.preview_train_btn.setToolTip("Show latest training preview (src / dst / model output).")
        self.preview_val_btn = QPushButton("Validation Preview")
        self.preview_val_btn.setCheckable(True)
        self.preview_val_btn.setToolTip("Show latest validation preview (unseen data).")

        self.preview_train_btn.clicked.connect(lambda: self._switch_preview('train'))
        self.preview_val_btn.clicked.connect(lambda: self._switch_preview('val'))
        toggle_layout.addWidget(self.preview_train_btn)
        toggle_layout.addWidget(self.preview_val_btn)
        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)

        # Scroll area (shared for both previews)
        self.scroll_area = ZoomPanScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.preview_label = QLabel("Waiting for preview image...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.zoom_changed.connect(self._on_preview_wheel_zoom)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # Zoom controls
        controls_layout = QHBoxLayout()
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["Fit", "50%", "100%", "200%"])
        self.zoom_combo.setToolTip("Zoom level. Mouse wheel to zoom, middle-click to pan.")
        self.zoom_combo.activated.connect(lambda: self._on_zoom_combo_changed(self.zoom_combo.currentText()))
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Zoom:"))
        controls_layout.addWidget(self.zoom_combo)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Column labels
        self.labels_container = QWidget()
        labels_layout = QHBoxLayout(self.labels_container)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        lbl_src = QLabel("src data")
        lbl_dst = QLabel("dst data")
        lbl_model = QLabel("Model result")
        lbl_src.setAlignment(Qt.AlignLeft)
        lbl_dst.setAlignment(Qt.AlignCenter)
        lbl_model.setAlignment(Qt.AlignRight)
        labels_layout.addWidget(lbl_src)
        labels_layout.addWidget(lbl_dst)
        labels_layout.addWidget(lbl_model)
        main_layout.addWidget(self.labels_container)

        # Track which preview is active
        self._active_preview = 'train'

        self.tabs.addTab(tab, "Previews")

    def _create_export_tab(self):
        """Tab 5: Export — convert checkpoint for VFX apps."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addStretch()

        info = QLabel("Export your trained model checkpoint for use in\nthird-party compositing applications.")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        layout.addSpacing(15)

        self.convert_flame_btn = QPushButton("Export for Flame / After Effects")
        self.convert_flame_btn.setToolTip(
            "Convert checkpoint to ONNX for Autodesk Flame or Adobe After Effects.")
        self.convert_flame_btn.clicked.connect(lambda: self._start_conversion('flame'))
        layout.addWidget(self.convert_flame_btn)

        self.convert_nuke_btn = QPushButton("Export for Nuke")
        self.convert_nuke_btn.setToolTip(
            "Convert checkpoint for Foundry Nuke. A .nk script file is also generated.")
        self.convert_nuke_btn.clicked.connect(lambda: self._start_conversion('nuke'))
        layout.addWidget(self.convert_nuke_btn)

        self.copy_before_convert_check = QCheckBox("Copy checkpoint to export subfolder first")
        self.copy_before_convert_check.setChecked(True)
        self.copy_before_convert_check.setToolTip(
            "Copy checkpoint into a subfolder before converting. Keeps exports separate from training.")
        self.copy_before_convert_check.setStyleSheet("margin-top: 10px;")
        layout.addWidget(self.copy_before_convert_check, alignment=Qt.AlignCenter)

        layout.addStretch()
        self.tabs.addTab(tab, "Export")

    def _create_about_tab(self):
        """Tab 6: About / Info."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignCenter)

        # --- Original project credit ---
        orig_title = QLabel("TuNet")
        orig_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_author = QLabel("Created by tpo.comp")
        orig_author.setStyleSheet("font-size: 14px;")
        orig_author.setAlignment(Qt.AlignCenter)
        orig_desc = QLabel(
            "A direct, pixel-level mapping from source to destination images\n"
            "via an encoder-decoder network.")
        orig_desc.setAlignment(Qt.AlignCenter)
        orig_desc.setWordWrap(True)
        orig_link = QLabel('<a href="https://github.com/tpc2233/tunet">github.com/tpc2233/tunet</a>')
        orig_link.setOpenExternalLinks(True)
        orig_link.setAlignment(Qt.AlignCenter)

        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        # --- Fork credit ---
        fork_title = QLabel("VFX Tools Fork")
        fork_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        fork_title.setAlignment(Qt.AlignCenter)
        fork_author = QLabel("Maintained by emlcpfx")
        fork_author.setStyleSheet("font-size: 13px;")
        fork_author.setAlignment(Qt.AlignCenter)
        fork_changes = QLabel(
            "Changes since fork:\n"
            "\u2022 Unified PyQt GUI for training, inference, and export\n"
            "\u2022 Render queue with progress monitoring\n"
            "\u2022 Live preview with pan/zoom\n"
            "\u2022 Auto-matte / auto-mask generation\n"
            "\u2022 MSRN architecture option and BigCat features\n"
            "\u2022 Checkpoint resume, validation, and improved naming\n"
            "\u2022 Export to Flame (.gizmo) and Nuke (.nk) formats\n"
            "\u2022 Multi-OS support")
        fork_changes.setAlignment(Qt.AlignCenter)
        fork_changes.setWordWrap(True)
        fork_link = QLabel('<a href="https://github.com/emlcpfx/tunet">github.com/emlcpfx/tunet</a>')
        fork_link.setOpenExternalLinks(True)
        fork_link.setAlignment(Qt.AlignCenter)

        # --- Layout ---
        layout.addStretch()
        layout.addWidget(orig_title)
        layout.addWidget(orig_author)
        layout.addSpacing(8)
        layout.addWidget(orig_desc)
        layout.addSpacing(4)
        layout.addWidget(orig_link)
        layout.addSpacing(20)
        layout.addWidget(separator)
        layout.addSpacing(20)
        layout.addWidget(fork_title)
        layout.addWidget(fork_author)
        layout.addSpacing(8)
        layout.addWidget(fork_changes)
        layout.addSpacing(4)
        layout.addWidget(fork_link)
        layout.addStretch()
        self.tabs.addTab(tab, "About")

    # =========================================================================
    # UI helpers
    # =========================================================================

    def _create_path_selector(self, label_text, is_output=False, is_file=False, file_filter=None):
        """Create a line edit + browse button widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit()
        button = QPushButton("Browse...")
        if is_file:
            if file_filter:
                button.clicked.connect(lambda: self._select_file(line_edit, file_filter))
            else:
                button.clicked.connect(lambda: self._select_file(line_edit, "Python Files (*.py)"))
        else:
            button.clicked.connect(lambda: self._select_directory(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(button)
        if is_output:
            line_edit.textChanged.connect(self._setup_preview_watcher)
        return widget

    def _select_directory(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            line_edit.setText(dir_path)

    def _select_file(self, line_edit, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def _create_range_layout(self, min_widget, max_widget):
        layout = QHBoxLayout()
        layout.addWidget(min_widget)
        layout.addWidget(QLabel("to"))
        layout.addWidget(max_widget)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    @staticmethod
    def _get_path(widget):
        """Extract path string from a path selector widget."""
        return widget.findChild(QLineEdit).text()

    @staticmethod
    def _set_path(widget, text):
        """Set path string on a path selector widget."""
        le = widget.findChild(QLineEdit)
        if le:
            le.setText(text)

    # =========================================================================
    # Sidebar — context-sensitive queue
    # =========================================================================

    def _create_sidebar(self):
        sidebar_layout = QVBoxLayout()
        self.sidebar_label = QLabel("Training Queue")
        sidebar_layout.addWidget(self.sidebar_label)

        self.sidebar_stack = QStackedWidget()
        self.sidebar_stack.setMinimumWidth(200)
        self.sidebar_stack.setMaximumWidth(280)

        # --- Page 0: Training Queue ---
        train_queue_page = QWidget()
        tq_layout = QVBoxLayout(train_queue_page)
        tq_layout.setContentsMargins(0, 0, 0, 0)
        self.queue_listbox = QListWidget()
        self.queue_listbox.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.queue_listbox.setSelectionMode(QListWidget.ExtendedSelection)
        tq_layout.addWidget(self.queue_listbox, stretch=1)

        self.queue_add_btn = QPushButton("Add to Queue")
        self.queue_add_btn.clicked.connect(self._add_to_training_queue)
        self.queue_remove_btn = QPushButton("Remove")
        self.queue_remove_btn.clicked.connect(self._remove_from_training_queue)
        self.queue_clear_btn = QPushButton("Clear")
        self.queue_clear_btn.clicked.connect(self._clear_training_queue)
        self.queue_run_btn = QPushButton("Run Queue")
        self.queue_run_btn.clicked.connect(self._run_training_queue)
        self.queue_run_btn.setStyleSheet("background-color: #a8e6cf; color: #1a1a1a;")
        tq_layout.addWidget(self.queue_add_btn)
        tq_layout.addWidget(self.queue_remove_btn)
        tq_layout.addWidget(self.queue_clear_btn)
        tq_layout.addWidget(self.queue_run_btn)
        self.sidebar_stack.addWidget(train_queue_page)

        # --- Page 1: Inference Queue ---
        inf_queue_page = QWidget()
        iq_layout = QVBoxLayout(inf_queue_page)
        iq_layout.setContentsMargins(0, 0, 0, 0)
        self.inf_queue_listbox = QListWidget()
        self.inf_queue_listbox.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.inf_queue_listbox.setSelectionMode(QListWidget.ExtendedSelection)
        iq_layout.addWidget(self.inf_queue_listbox, stretch=1)

        self.inf_queue_add_btn = QPushButton("Add to Queue")
        self.inf_queue_add_btn.clicked.connect(self._inf_add_to_queue)
        self.inf_queue_remove_btn = QPushButton("Remove")
        self.inf_queue_remove_btn.clicked.connect(self._inf_remove_from_queue)
        self.inf_queue_clear_btn = QPushButton("Clear")
        self.inf_queue_clear_btn.clicked.connect(self._inf_clear_queue)
        iq_layout.addWidget(self.inf_queue_add_btn)
        iq_layout.addWidget(self.inf_queue_remove_btn)
        iq_layout.addWidget(self.inf_queue_clear_btn)
        self.sidebar_stack.addWidget(inf_queue_page)

        sidebar_layout.addWidget(self.sidebar_stack, stretch=1)
        return sidebar_layout

    # =========================================================================
    # Bottom control panel
    # =========================================================================

    def _create_control_panel(self):
        layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Config")
        self.load_btn.clicked.connect(self._load_config_from_file)
        self.save_btn = QPushButton("Save Config")
        self.save_btn.clicked.connect(self._save_config_to_file)
        self.monitor_btn = QPushButton("Training Monitor")
        self.monitor_btn.setStyleSheet("background-color: #b3e5fc; color: #1a1a1a;")
        self.monitor_btn.clicked.connect(self._launch_training_monitor)
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setStyleSheet("background-color: #a8e6cf; color: #1a1a1a;")
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setStyleSheet("background-color: #ff8a80; color: #1a1a1a;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.monitor_btn)
        layout.addStretch()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        return layout

    # =========================================================================
    # Tab switching / sidebar context
    # =========================================================================

    @Slot(int)
    def _on_tab_changed(self, index):
        if index == self._inference_tab_index:
            self.sidebar_stack.setCurrentIndex(1)
            self.sidebar_label.setText("Inference Queue")
        else:
            self.sidebar_stack.setCurrentIndex(0)
            self.sidebar_label.setText("Training Queue")

        # Refresh previews when switching to previews tab
        tab_name = self.tabs.tabText(index)
        if tab_name == "Previews":
            self._apply_preview_zoom()

    # =========================================================================
    # Preview management
    # =========================================================================

    def _switch_preview(self, mode):
        """Toggle between training and validation preview."""
        self._active_preview = mode
        self.preview_train_btn.setChecked(mode == 'train')
        self.preview_val_btn.setChecked(mode == 'val')
        self._apply_preview_zoom()

    def _apply_preview_zoom(self):
        """Apply current zoom to the active preview pixmap."""
        if self._active_preview == 'train':
            pixmap = self.original_pixmap
            zoom = self._preview_zoom_factor
        else:
            pixmap = self.val_original_pixmap
            zoom = self._val_preview_zoom_factor

        if not pixmap:
            label_text = "Waiting for training preview..." if self._active_preview == 'train' else "Waiting for validation preview..."
            self.preview_label.setText(label_text)
            self.labels_container.setVisible(False)
            return

        self.labels_container.setVisible(True)
        if zoom is None:
            scaled = pixmap.scaled(self.scroll_area.viewport().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            new_w = int(pixmap.width() * zoom)
            new_h = int(pixmap.height() * zoom)
            scaled = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.preview_label.setPixmap(scaled)
        self.preview_label.adjustSize()
        image_width = scaled.width()
        container_width = self.scroll_area.viewport().width()
        margin = max(0, (container_width - image_width) // 2)
        self.labels_container.setContentsMargins(margin, 0, margin, 0)

    def _on_zoom_combo_changed(self, text):
        if self._active_preview == 'train':
            if text == "Fit":
                self._preview_zoom_factor = None
            else:
                self._preview_zoom_factor = float(text.replace('%', '')) / 100.0
        else:
            if text == "Fit":
                self._val_preview_zoom_factor = None
            else:
                self._val_preview_zoom_factor = float(text.replace('%', '')) / 100.0
        self._apply_preview_zoom()

    def _on_preview_wheel_zoom(self, factor):
        if self._active_preview == 'train':
            pixmap = self.original_pixmap
            current_zoom = self._preview_zoom_factor
        else:
            pixmap = self.val_original_pixmap
            current_zoom = self._val_preview_zoom_factor

        if current_zoom is None:
            if pixmap and not pixmap.isNull():
                vp = self.scroll_area.viewport().size()
                current_zoom = min(vp.width() / pixmap.width(), vp.height() / pixmap.height())
            else:
                return

        new_zoom = max(0.05, min(10.0, current_zoom * factor))
        if self._active_preview == 'train':
            self._preview_zoom_factor = new_zoom
        else:
            self._val_preview_zoom_factor = new_zoom
        self._apply_preview_zoom()

    @Slot(str)
    def _setup_preview_watcher(self, model_folder):
        if self.preview_image_path and self.file_watcher.files():
            self.file_watcher.removePath(self.preview_image_path)
        if self.val_preview_image_path and self.val_preview_image_path in self.file_watcher.files():
            self.file_watcher.removePath(self.val_preview_image_path)
        if model_folder and Path(model_folder).is_dir():
            self.preview_image_path = str(Path(model_folder) / "training_preview.jpg")
            self.val_preview_image_path = str(Path(model_folder) / "val_preview.jpg")
            self.file_watcher.addPath(self.preview_image_path)
            self.file_watcher.addPath(self.val_preview_image_path)
            self._update_preview_image()
            self._update_val_preview_image()

    @Slot(str)
    def _on_watched_file_changed(self, path):
        if self.preview_image_path and path == self.preview_image_path:
            self._update_preview_image()
        elif self.val_preview_image_path and path == self.val_preview_image_path:
            self._update_val_preview_image()

    def _update_all_previews(self):
        self._update_preview_image()
        self._update_val_preview_image()

    def _update_preview_image(self):
        if self.preview_image_path and Path(self.preview_image_path).exists():
            try:
                with open(self.preview_image_path, 'rb') as f:
                    image_data = f.read()
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                if not pixmap.isNull():
                    self.original_pixmap = pixmap
                    if self._active_preview == 'train':
                        self._apply_preview_zoom()
            except Exception:
                pass
        else:
            self.original_pixmap = None
            if self._active_preview == 'train':
                self._apply_preview_zoom()

    def _update_val_preview_image(self):
        if self.val_preview_image_path and Path(self.val_preview_image_path).exists():
            try:
                with open(self.val_preview_image_path, 'rb') as f:
                    image_data = f.read()
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                if not pixmap.isNull():
                    self.val_original_pixmap = pixmap
                    if self._active_preview == 'val':
                        self._apply_preview_zoom()
            except Exception:
                pass
        else:
            self.val_original_pixmap = None
            if self._active_preview == 'val':
                self._apply_preview_zoom()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_preview_zoom()

    # =========================================================================
    # Config gather / populate
    # =========================================================================

    def gather_config_from_ui(self):
        """Collect all UI state into a single config dict."""
        gp = self._get_path
        augs = []
        if self.hflip_check.isChecked():
            augs.append({'_target_': 'albumentations.HorizontalFlip', 'p': self.hflip_p.value() / 100.0})
        if self.affine_check.isChecked():
            augs.append({
                '_target_': 'albumentations.Affine',
                'scale': [self.affine_scale_min.value(), self.affine_scale_max.value()],
                'translate_percent': [self.affine_translate_min.value(), self.affine_translate_max.value()],
                'rotate': [self.affine_rotate_min.value(), self.affine_rotate_max.value()],
                'shear': [self.affine_shear_min.value(), self.affine_shear_max.value()],
                'interpolation': self.affine_interpolation.value(),
                'keep_ratio': self.affine_keep_ratio.isChecked(),
                'p': self.affine_p.value() / 100.0,
            })
        if self.gamma_check.isChecked():
            augs.append({
                '_target_': 'albumentations.RandomGamma',
                'gamma_limit': [self.gamma_limit_min.value(), self.gamma_limit_max.value()],
                'p': self.gamma_p.value() / 100.0,
            })
        if self.color_check.isChecked():
            augs.append({
                '_target_': 'albumentations.RandomBrightnessContrast',
                'brightness_limit': [self.color_brightness_min.value(), self.color_brightness_max.value()],
                'contrast_limit': [self.color_contrast_min.value(), self.color_contrast_max.value()],
                'p': self.color_p.value() / 100.0,
            })
            augs.append({
                '_target_': 'albumentations.HueSaturationValue',
                'hue_shift_limit': 0,
                'sat_shift_limit': [self.color_saturation_min.value(), self.color_saturation_max.value()],
                'val_shift_limit': 0,
                'p': self.color_p.value() / 100.0,
            })

        mask_dir = gp(self.mask_dir_input)
        data_config = {
            'src_dir': gp(self.src_dir_input),
            'dst_dir': gp(self.dst_dir_input),
            'output_dir': gp(self.model_folder_input),
            'resolution': int(self.resolution_input.currentText()),
            'overlap_factor': float(self.overlap_factor_input.currentText()),
        }
        if mask_dir:
            data_config['mask_dir'] = mask_dir
        val_src = gp(self.val_src_dir_input)
        val_dst = gp(self.val_dst_dir_input)
        if val_src:
            data_config['val_src_dir'] = val_src
        if val_dst:
            data_config['val_dst_dir'] = val_dst

        lr_value = self.lr_presets[self.lr_input.currentIndex()][1]
        lambda_value = self.lambda_presets[self.lambda_lpips_input.currentIndex()][1]

        config = {
            'data': data_config,
            'model': {
                'model_size_dims': int(self.model_size_dims_input.currentText()),
                'model_type': self.model_type_input.currentText(),
            },
            'training': {
                'finetune_from': gp(self.finetune_from_input) or None,
                'iterations_per_epoch': self.iter_per_epoch_input.value(),
                'batch_size': self.batch_size_input.value(),
                'max_steps': self.max_steps_input.value(),
                'use_amp': self.use_amp_input.isChecked(),
                'loss': self.loss_input.currentText(),
                'lambda_lpips': lambda_value,
                'lr': lr_value,
                'progressive_resolution': self.progressive_res_check.isChecked(),
                'l1_weight': self.l1_weight_input.value(),
                'l2_weight': self.l2_weight_input.value(),
                'lpips_weight': self.lpips_weight_input.value(),
            },
            'logging': {
                'log_interval': self.log_interval_input.value(),
                'preview_batch_interval': self.preview_interval_input.value(),
                'preview_refresh_rate': self.preview_refresh_input.value(),
            },
            'saving': {
                'keep_last_checkpoints': self.keep_checkpoints_input.value(),
            },
            'dataloader': {
                'num_workers': self.num_workers_presets[self.num_workers_input.currentIndex()][1],
                'datasets': {'shared_augs': augs},
            },
        }

        if self.use_mask_loss_input.isChecked() or self.use_mask_input_input.isChecked() or self.use_auto_mask_input.isChecked():
            config['mask'] = {
                'use_mask_loss': self.use_mask_loss_input.isChecked(),
                'mask_weight': self.mask_weight_input.value(),
                'use_mask_input': self.use_mask_input_input.isChecked(),
                'use_auto_mask': self.use_auto_mask_input.isChecked(),
                'skip_empty_patches': self.skip_empty_patches_input.isChecked(),
                'skip_empty_threshold': self.skip_empty_threshold_input.value(),
            }

        # Inference section
        config['inference'] = {
            'checkpoint_path': gp(self.inf_checkpoint_input),
            'input_dir': gp(self.inf_input_dir),
            'output_root': gp(self.inf_output_root),
            'stride': self.inf_stride.value(),
            'auto_stride': self.inf_auto_stride.isChecked(),
            'half_res': self.inf_half_res.isChecked(),
            'skip_existing': self.inf_skip_existing.isChecked(),
            'queue': [{'input_dir': q['input_dir'], 'output_root': q['output_root']}
                      for q in self.inference_queue if q['status'] == 'pending'],
        }

        # UI-only settings
        config['_ui_settings'] = {
            'train_script_path': gp(self.train_script_input),
            'nproc_per_node': self.nproc_input.value() if self.nproc_input else 1,
        }
        return config

    def populate_ui_from_config(self, config):
        """Populate all UI fields from a config dict."""
        sp = self._set_path
        ui_settings = config.get('_ui_settings', {})
        sp(self.train_script_input, ui_settings.get('train_script_path', ''))
        if self.nproc_input:
            self.nproc_input.setValue(ui_settings.get('nproc_per_node', 1))

        # Data
        data = config.get('data', {})
        sp(self.src_dir_input, data.get('src_dir', ''))
        sp(self.dst_dir_input, data.get('dst_dir', ''))
        sp(self.mask_dir_input, data.get('mask_dir', ''))
        sp(self.model_folder_input, data.get('output_dir', ''))
        self.resolution_input.setCurrentText(str(data.get('resolution', 512)))
        self.overlap_factor_input.setCurrentText(str(data.get('overlap_factor', 0.25)))
        sp(self.val_src_dir_input, data.get('val_src_dir', ''))
        sp(self.val_dst_dir_input, data.get('val_dst_dir', ''))

        # Model
        model = config.get('model', {})
        self.model_size_dims_input.setCurrentText(str(model.get('model_size_dims', 128)))
        self.model_type_input.setCurrentText(model.get('model_type', 'unet'))

        # Training
        training = config.get('training', {})
        sp(self.finetune_from_input, training.get('finetune_from', '') or '')
        self.iter_per_epoch_input.setValue(training.get('iterations_per_epoch', 500))
        self.batch_size_input.setValue(training.get('batch_size', 4))
        self.max_steps_input.setValue(training.get('max_steps', 0))
        self.use_amp_input.setChecked(training.get('use_amp', True))
        self.loss_input.setCurrentText(training.get('loss', 'l1'))

        saved_lambda = training.get('lambda_lpips', 0.1)
        best_lambda_idx = 2
        for i, (_, val) in enumerate(self.lambda_presets):
            if abs(val - saved_lambda) < 1e-6:
                best_lambda_idx = i
                break
        self.lambda_lpips_input.setCurrentIndex(best_lambda_idx)

        saved_lr = training.get('lr', 1e-4)
        best_lr_idx = 1
        for i, (_, val) in enumerate(self.lr_presets):
            if abs(val - saved_lr) < 1e-6:
                best_lr_idx = i
                break
        self.lr_input.setCurrentIndex(best_lr_idx)

        # Progressive resolution & weighted loss
        self.progressive_res_check.setChecked(training.get('progressive_resolution', False))
        self.l1_weight_input.setValue(training.get('l1_weight', 1.0))
        self.l2_weight_input.setValue(training.get('l2_weight', 0.0))
        self.lpips_weight_input.setValue(training.get('lpips_weight', 0.1))

        # Logging
        log_cfg = config.get('logging', {})
        self.log_interval_input.setValue(log_cfg.get('log_interval', 5))
        self.preview_interval_input.setValue(log_cfg.get('preview_batch_interval', 35))
        self.preview_refresh_input.setValue(log_cfg.get('preview_refresh_rate', 5))
        self.keep_checkpoints_input.setValue(config.get('saving', {}).get('keep_last_checkpoints', 4))

        # Dataloader workers
        saved_workers = config.get('dataloader', {}).get('num_workers', -1)
        best_workers_idx = 0
        for i, (_, val) in enumerate(self.num_workers_presets):
            if val == saved_workers:
                best_workers_idx = i
                break
        self.num_workers_input.setCurrentIndex(best_workers_idx)

        # Mask
        mask_cfg = config.get('mask', {})
        self.use_mask_loss_input.setChecked(mask_cfg.get('use_mask_loss', False))
        self.mask_weight_input.setValue(mask_cfg.get('mask_weight', 10.0))
        self.use_mask_input_input.setChecked(mask_cfg.get('use_mask_input', False))
        self.use_auto_mask_input.setChecked(mask_cfg.get('use_auto_mask', False))
        self.skip_empty_patches_input.setChecked(mask_cfg.get('skip_empty_patches', False))
        self.skip_empty_threshold_input.setValue(mask_cfg.get('skip_empty_threshold', 1.0))

        # Augmentations
        self.hflip_check.setChecked(False)
        self.affine_check.setChecked(False)
        self.gamma_check.setChecked(False)
        self.color_check.setChecked(False)
        augs = config.get('dataloader', {}).get('datasets', {}).get('shared_augs', [])
        for aug in augs:
            target = aug.get('_target_', '')
            if 'HorizontalFlip' in target:
                self.hflip_check.setChecked(True)
                self.hflip_p.setValue(int(aug.get('p', 0.5) * 100))
            elif 'Affine' in target:
                self.affine_check.setChecked(True)
                self.affine_p.setValue(int(aug.get('p', 0.4) * 100))
                scale = aug.get('scale', [0.9, 1.1])
                self.affine_scale_min.setValue(scale[0])
                self.affine_scale_max.setValue(scale[1])
                translate = aug.get('translate_percent', [-0.1, 0.1])
                self.affine_translate_min.setValue(translate[0])
                self.affine_translate_max.setValue(translate[1])
                rotate = aug.get('rotate', [-3, 3])
                self.affine_rotate_min.setValue(rotate[0])
                self.affine_rotate_max.setValue(rotate[1])
                shear = aug.get('shear', [-1, 1])
                self.affine_shear_min.setValue(shear[0])
                self.affine_shear_max.setValue(shear[1])
                self.affine_interpolation.setValue(aug.get('interpolation', 2))
                self.affine_keep_ratio.setChecked(aug.get('keep_ratio', True))
            elif 'RandomGamma' in target:
                self.gamma_check.setChecked(True)
                self.gamma_p.setValue(int(aug.get('p', 0.2) * 100))
                limit = aug.get('gamma_limit', [40, 160])
                self.gamma_limit_min.setValue(limit[0])
                self.gamma_limit_max.setValue(limit[1])
            elif 'RandomBrightnessContrast' in target:
                self.color_check.setChecked(True)
                self.color_p.setValue(int(aug.get('p', 0.3) * 100))
                bl = aug.get('brightness_limit', [-0.2, 0.2])
                self.color_brightness_min.setValue(bl[0])
                self.color_brightness_max.setValue(bl[1])
                cl = aug.get('contrast_limit', [-0.2, 0.2])
                self.color_contrast_min.setValue(cl[0])
                self.color_contrast_max.setValue(cl[1])
            elif 'HueSaturationValue' in target:
                # Saturation part of color augmentation (paired with RandomBrightnessContrast)
                sl = aug.get('sat_shift_limit', [-30, 30])
                if isinstance(sl, list):
                    self.color_saturation_min.setValue(sl[0])
                    self.color_saturation_max.setValue(sl[1])

        # Inference
        inf = config.get('inference', {})
        if inf:
            sp(self.inf_checkpoint_input, inf.get('checkpoint_path', ''))
            sp(self.inf_input_dir, inf.get('input_dir', ''))
            sp(self.inf_output_root, inf.get('output_root', ''))
            self.inf_stride.setValue(inf.get('stride', 256))
            self.inf_auto_stride.setChecked(inf.get('auto_stride', True))
            self.inf_half_res.setChecked(inf.get('half_res', False))
            self.inf_skip_existing.setChecked(inf.get('skip_existing', True))
            # Restore inference queue
            self.inference_queue = []
            for item in inf.get('queue', []):
                input_d = item.get('input_dir', '')
                output_r = item.get('output_root', '')
                if input_d:
                    self.inference_queue.append({
                        'input_dir': input_d, 'output_root': output_r, 'status': 'pending'
                    })
            self._inf_refresh_queue_display()

    # =========================================================================
    # Config file save/load
    # =========================================================================

    def _save_config_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config As", self._last_config_dir, "YAML Files (*.yaml *.yml)")
        if path:
            self._last_config_dir = str(Path(path).parent)
            config = self.gather_config_from_ui()
            with open(path, 'w') as f:
                yaml.dump(config, f, Dumper=IndentDumper, sort_keys=False, default_flow_style=False, indent=2)
            self.console_output.append(f"Config saved to: {path}\n")

    def _load_config_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", self._last_config_dir, "YAML Files (*.yaml *.yml)")
        if path:
            self._last_config_dir = str(Path(path).parent)
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                self.populate_ui_from_config(config)
                self.console_output.append(f"Config loaded from: {path}\n")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config:\n{e}")

    # =========================================================================
    # Session auto-save / auto-load
    # =========================================================================

    def _save_session(self):
        """Auto-save all UI state on close."""
        try:
            config = self.gather_config_from_ui()
            with open(_SESSION_FILE, 'w') as f:
                yaml.dump(config, f, Dumper=IndentDumper, sort_keys=False, default_flow_style=False, indent=2)
        except Exception:
            pass

    def _load_session(self):
        """Auto-load session state on startup."""
        # Try session file first
        if _SESSION_FILE.exists():
            try:
                with open(_SESSION_FILE, 'r') as f:
                    config = yaml.safe_load(f)
                if config:
                    self.populate_ui_from_config(config)
                    return
            except Exception:
                pass

        # Migrate legacy inference_gui_settings.json if it exists
        if _LEGACY_INFERENCE_SETTINGS.exists():
            try:
                with open(_LEGACY_INFERENCE_SETTINGS, 'r') as f:
                    s = json.load(f)
                self._set_path(self.inf_checkpoint_input, s.get('checkpoint_path', ''))
                self._set_path(self.inf_input_dir, s.get('input_dir', ''))
                self._set_path(self.inf_output_root, s.get('output_root', s.get('output_dir', '')))
                self.inf_stride.setValue(s.get('stride', 256))
                self.inf_auto_stride.setChecked(s.get('auto_stride', True))
                self.inf_half_res.setChecked(s.get('half_res', False))
                self.inf_skip_existing.setChecked(s.get('skip_existing', True))
                # Restore queue
                for item in s.get('queue', []):
                    input_d = item.get('input_dir', '')
                    output_r = item.get('output_root', item.get('output_dir', ''))
                    if input_d and os.path.isdir(input_d):
                        self.inference_queue.append({
                            'input_dir': input_d, 'output_root': output_r, 'status': 'pending'
                        })
                self._inf_refresh_queue_display()
            except Exception:
                pass

    def _populate_default_script_path(self):
        try:
            default_script = _APP_DIR / "train.py"
            if default_script.is_file():
                le = self.train_script_input.findChild(QLineEdit)
                if le:
                    le.setText(str(default_script.resolve()))
        except Exception:
            pass

    # =========================================================================
    # Training process management
    # =========================================================================

    def _launch_training(self, full_config):
        """Launch a training subprocess. Returns True if started."""
        train_script = full_config['_ui_settings']['train_script_path']
        if not train_script or not Path(train_script).is_file():
            QMessageBox.warning(self, "Error", "Training script path is invalid.")
            return False
        model_folder = Path(full_config['data']['output_dir'])
        if not model_folder.is_dir():
            QMessageBox.warning(self, "Error", f"Model Folder does not exist:\n{model_folder}")
            return False

        self._setup_preview_watcher(str(model_folder))

        # Write config for train.py (strip UI-only and inference keys)
        script_config = {k: v for k, v in full_config.items() if k not in ('_ui_settings', 'inference')}
        self.config_file_path = Path(f"{model_folder.name}.yaml")
        with open(self.config_file_path, 'w') as f:
            yaml.dump(script_config, f, Dumper=IndentDumper, sort_keys=False, default_flow_style=False, indent=2)

        current_os = platform.system()
        if current_os == 'Linux':
            nproc = full_config['_ui_settings']['nproc_per_node']
            command = ['torchrun', '--standalone', '--nnodes=1', f'--nproc_per_node={nproc}',
                       train_script, '--config', str(self.config_file_path)]
        elif current_os in ('Windows', 'Darwin'):
            command = [sys.executable, train_script, '--config', str(self.config_file_path)]
        else:
            QMessageBox.critical(self, "Unsupported OS", f"'{current_os}' is not supported.")
            return False

        self.console_output.append(f"OS: {current_os}\nCommand: {' '.join(command)}\n\n")
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0

        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=creation_flags,
                encoding='utf-8', errors='replace')
        except Exception as e:
            QMessageBox.critical(self, "Process Error", f"Failed to start: {e}")
            return False

        self.stream_reader = ProcessStreamReader(self.process.stdout)
        self.stream_reader.new_text.connect(self._append_text)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.queue_run_btn.setEnabled(False)
        self.queue_add_btn.setEnabled(False)
        self.preview_timer.start()
        self.process_monitor.start()
        return True

    def _start_training(self):
        full_config = self.gather_config_from_ui()
        self.console_output.clear()
        self._launch_training(full_config)

    @Slot()
    def _check_process_status(self):
        if not self.process:
            return
        if self.process.poll() is not None:
            self.process_monitor.stop()
            self._on_training_finished()

    def _on_training_finished(self):
        self.preview_timer.stop()
        self.console_output.append("\n--- Training Process Terminated ---\n")
        self.process = None
        QTimer.singleShot(100, self._update_all_previews)

        if self.queue_running:
            for item in self.training_queue:
                if item['status'] == 'processing':
                    item['status'] = 'done'
                    break
            self._refresh_training_queue_display()
            if self.queue_stop_requested:
                self._finish_training_queue()
                return
            self._run_next_training_queue_item()
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.queue_run_btn.setEnabled(True)
            self.queue_add_btn.setEnabled(True)

    def _stop_training(self):
        if self.process and self.process.poll() is None:
            self.console_output.append("\n--- Sending stop signal ---\n")
            if platform.system() == "Windows":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
            if self.queue_running:
                self.queue_stop_requested = True
                self.console_output.append("Queue will stop after current item finishes.\n")

    @Slot()
    def _append_text(self, text):
        self.console_output.moveCursor(QTextCursor.End)
        self.console_output.insertPlainText(text)

    # =========================================================================
    # Training queue
    # =========================================================================

    def _add_to_training_queue(self):
        full_config = self.gather_config_from_ui()
        label = os.path.basename(full_config['data'].get('output_dir', 'untitled'))
        max_steps = full_config.get('training', {}).get('max_steps', 0)
        if max_steps == 0:
            QMessageBox.warning(self, "Max Steps Required",
                                "Set Max Steps > 0 for queue items so training knows when to stop.")
            return
        self.training_queue.append({'config': full_config, 'status': 'pending', 'label': label})
        self._refresh_training_queue_display()
        self.console_output.append(f"Added to queue: {label} (max_steps={max_steps})\n")

    def _remove_from_training_queue(self):
        selected = sorted(self.queue_listbox.selectedIndexes(), key=lambda x: x.row(), reverse=True)
        for idx in selected:
            row = idx.row()
            if row < len(self.training_queue) and self.training_queue[row]['status'] != 'processing':
                self.training_queue.pop(row)
        self._refresh_training_queue_display()

    def _clear_training_queue(self):
        self.training_queue = [q for q in self.training_queue if q['status'] == 'processing']
        self._refresh_training_queue_display()

    def _refresh_training_queue_display(self):
        self.queue_listbox.clear()
        prefix_map = {'pending': '   ', 'processing': '>> ', 'done': 'OK ', 'error': '!! '}
        for i, item in enumerate(self.training_queue):
            prefix = prefix_map.get(item['status'], '   ')
            max_s = item['config'].get('training', {}).get('max_steps', 0)
            text = f"{prefix}{item['label']}  (max_steps={max_s})"
            self.queue_listbox.addItem(text)
            list_item = self.queue_listbox.item(i)
            if item['status'] == 'processing':
                list_item.setForeground(Qt.blue)
            elif item['status'] == 'done':
                list_item.setForeground(Qt.darkGreen)
            elif item['status'] == 'error':
                list_item.setForeground(Qt.red)

    def _run_training_queue(self):
        if self.process:
            QMessageBox.warning(self, "Busy", "Training is already running.")
            return
        pending = [q for q in self.training_queue if q['status'] == 'pending']
        if not pending:
            QMessageBox.information(self, "Queue Empty", "No pending items in the queue.")
            return
        self.queue_running = True
        self.queue_stop_requested = False
        self.console_output.clear()
        self.console_output.append(f"=== Starting Queue ({len(pending)} items) ===\n\n")
        self._run_next_training_queue_item()

    def _run_next_training_queue_item(self):
        pending = [q for q in self.training_queue if q['status'] == 'pending']
        if not pending:
            self._finish_training_queue()
            return
        item = pending[0]
        item['status'] = 'processing'
        self._refresh_training_queue_display()
        done_count = sum(1 for q in self.training_queue if q['status'] == 'done')
        total_count = sum(1 for q in self.training_queue if q['status'] in ('pending', 'processing', 'done'))
        self.console_output.append(f"\n=== Queue item {done_count + 1}/{total_count}: {item['label']} ===\n\n")
        if not self._launch_training(item['config']):
            item['status'] = 'error'
            self._refresh_training_queue_display()
            self._run_next_training_queue_item()

    def _finish_training_queue(self):
        self.queue_running = False
        self.queue_stop_requested = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.queue_run_btn.setEnabled(True)
        self.queue_add_btn.setEnabled(True)
        done_count = sum(1 for q in self.training_queue if q['status'] == 'done')
        total_count = len(self.training_queue)
        self.console_output.append(f"\n=== Queue finished ({done_count}/{total_count} completed) ===\n")
        self._refresh_training_queue_display()

    # =========================================================================
    # Inference — processing logic (ported from inference_gui.py)
    # =========================================================================

    def _inf_use_latest_checkpoint(self):
        """Find the newest .pth in the model output folder."""
        model_folder = self._get_path(self.model_folder_input)
        if not model_folder or not Path(model_folder).is_dir():
            QMessageBox.warning(self, "No Folder", "Set a valid Model Output Folder in the Training tab first.")
            return
        pth_files = sorted(Path(model_folder).glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pth_files:
            QMessageBox.information(self, "No Checkpoints", f"No .pth files found in:\n{model_folder}")
            return
        self._set_path(self.inf_checkpoint_input, str(pth_files[0]))
        self.console_output.append(f"Using checkpoint: {pth_files[0].name}\n")

    def _inf_validate_checkpoint(self):
        path = self._get_path(self.inf_checkpoint_input)
        if not path:
            QMessageBox.warning(self, "Error", "Please select a model checkpoint.")
            return False
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Checkpoint file not found.")
            return False
        return True

    def _inf_validate_inputs(self):
        if not self._inf_validate_checkpoint():
            return False
        input_dir = self._get_path(self.inf_input_dir)
        if not input_dir:
            QMessageBox.warning(self, "Error", "Please select an input directory.")
            return False
        if not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Error", "Input directory not found.")
            return False
        output_root = self._get_path(self.inf_output_root)
        if not output_root:
            QMessageBox.warning(self, "Error", "Please select an Output Root directory.")
            return False
        return True

    def _inf_start_processing(self):
        self.inference_running = True
        self.inference_stop_requested = False
        self.inf_run_btn.setEnabled(False)
        self.inf_run_queue_btn.setEnabled(False)
        self.inf_stop_btn.setEnabled(True)
        self.inf_progress_bar.setValue(0)
        self.inf_progress_label.setText("")

    def _inf_finish_processing(self):
        self.inference_running = False
        self.inference_stop_requested = False
        self.inf_run_btn.setEnabled(True)
        self.inf_run_queue_btn.setEnabled(True)
        self.inf_stop_btn.setEnabled(False)
        self.inf_progress_bar.setValue(0)
        self.inf_progress_label.setText("")
        self._cached_model = None

    def _inf_request_stop(self):
        self.inference_stop_requested = True
        logging.info("Stop requested, finishing current image...")

    def _inf_run_single(self):
        if self.inference_running:
            return
        if not self._inf_validate_inputs():
            return
        self._inf_start_processing()
        thread = threading.Thread(target=self._inf_run_single_thread, daemon=True)
        thread.start()

    def _inf_run_single_thread(self):
        try:
            import torch
            import torchvision.transforms as T
            from inference import load_model_and_config, process_image, denormalize, NORM_MEAN, NORM_STD, load_image_any_format

            model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch = self._inf_load_model()
            input_dir = self._get_path(self.inf_input_dir)
            output_root = self._get_path(self.inf_output_root)
            checkpoint = self._get_path(self.inf_checkpoint_input)
            skip = self.inf_skip_existing.isChecked()
            output_dir = self._inf_resolve_output_dir(input_dir, output_root, checkpoint, skip)
            logging.info(f"Output folder: {output_dir}")
            self._inf_process_folder(model, input_dir, output_dir, resolution, stride,
                                     device, use_amp, transform, loss_mode, tile_batch)
            if not self.inference_stop_requested:
                logging.info("Inference complete!")
                QTimer.singleShot(0, lambda: QMessageBox.information(self, "Done", "Inference completed successfully!"))
        except Exception as e:
            logging.error(f"Error: {e}")
            err_msg = str(e)
            QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", err_msg))
        finally:
            QTimer.singleShot(0, self._inf_finish_processing)

    def _inf_run_queue(self):
        if self.inference_running:
            return
        pending = [q for q in self.inference_queue if q['status'] == 'pending']
        if not pending:
            QMessageBox.information(self, "Queue Empty", "No pending items. Add folders first.")
            return
        if not self._inf_validate_checkpoint():
            return
        self._inf_start_processing()
        thread = threading.Thread(target=self._inf_run_queue_thread, daemon=True)
        thread.start()

    def _inf_run_queue_thread(self):
        try:
            model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch = self._inf_load_model()
            pending = [q for q in self.inference_queue if q['status'] == 'pending']
            total_items = len(pending)

            for qi, item in enumerate(pending):
                if self.inference_stop_requested:
                    logging.info("Queue stopped by user.")
                    break

                logging.info(f"--- Queue item {qi+1}/{total_items}: {os.path.basename(item['input_dir'])} ---")
                item['status'] = 'processing'
                QTimer.singleShot(0, self._inf_refresh_queue_display)

                try:
                    checkpoint = self._get_path(self.inf_checkpoint_input)
                    skip = self.inf_skip_existing.isChecked()
                    output_dir = self._inf_resolve_output_dir(
                        item['input_dir'], item['output_root'], checkpoint, skip)
                    item['resolved_output'] = output_dir
                    QTimer.singleShot(0, self._inf_refresh_queue_display)
                    logging.info(f"Output folder: {output_dir}")
                    self._inf_process_folder(model, item['input_dir'], output_dir,
                                             resolution, stride, device, use_amp, transform, loss_mode, tile_batch)
                    if not self.inference_stop_requested:
                        item['status'] = 'done'
                    else:
                        item['status'] = 'pending'
                except Exception as e:
                    logging.error(f"Error processing {item['input_dir']}: {e}")
                    item['status'] = 'error'

                QTimer.singleShot(0, self._inf_refresh_queue_display)

            if not self.inference_stop_requested:
                done_count = sum(1 for q in self.inference_queue if q['status'] == 'done')
                logging.info(f"Queue complete! {done_count}/{total_items} items processed.")
                QTimer.singleShot(0, lambda: QMessageBox.information(
                    self, "Done", f"Queue complete! {done_count}/{total_items} items processed."))
        except Exception as e:
            logging.error(f"Queue error: {e}")
            err_msg = str(e)
            QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", err_msg))
        finally:
            QTimer.singleShot(0, self._inf_finish_processing)

    def _inf_load_model(self):
        """Load or reuse cached model for inference."""
        import torch
        import torchvision.transforms as T
        from inference import load_model_and_config, NORM_MEAN, NORM_STD

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)
        use_amp = device_str == 'cuda'
        checkpoint = self._get_path(self.inf_checkpoint_input)

        if self._cached_model is not None and self._cached_checkpoint == checkpoint:
            logging.info("Reusing loaded model.")
            model = self._cached_model
            resolution = self._cached_resolution
            loss_mode = self._cached_loss_mode
        else:
            logging.info(f"Using device: {device}")
            logging.info("Loading model...")
            model, resolution, use_mask_input, loss_mode = load_model_and_config(checkpoint, device)
            if os.name != 'nt':
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    logging.info("torch.compile enabled (reduce-overhead).")
                except Exception as e:
                    logging.warning(f"torch.compile unavailable: {e}")
            else:
                logging.info("torch.compile skipped (Windows).")
            self._cached_model = model
            self._cached_checkpoint = checkpoint
            self._cached_resolution = resolution
            self._cached_loss_mode = loss_mode

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD)])
        stride = max(1, self.inf_stride.value())
        logging.info(f"Resolution: {resolution}, Stride: {stride} ({'auto' if self.inf_auto_stride.isChecked() else 'manual'})")

        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            vram_gb = props.total_memory / (1024 ** 3)
            if vram_gb >= 20:
                tile_batch = 16
            elif vram_gb >= 10:
                tile_batch = 8
            elif vram_gb >= 6:
                tile_batch = 4
            else:
                tile_batch = 2
            logging.info(f"GPU: {props.name} ({vram_gb:.0f}GB) -> tile batch = {tile_batch}")
        else:
            tile_batch = 1

        return model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch

    @staticmethod
    def _inf_resolve_output_dir(input_dir, output_root, checkpoint_path, skip_existing=False):
        """Build a versioned output subfolder."""
        input_name = os.path.basename(os.path.normpath(input_dir))
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        base = f"{input_name}_{model_name}"

        if skip_existing:
            latest = None
            v = 1
            while True:
                candidate = os.path.join(output_root, f"{base}_v{v:03d}")
                if os.path.exists(candidate):
                    latest = candidate
                    v += 1
                else:
                    break
            return latest if latest else os.path.join(output_root, f"{base}_v001")
        else:
            v = 1
            while True:
                candidate = os.path.join(output_root, f"{base}_v{v:03d}")
                if not os.path.exists(candidate):
                    return candidate
                v += 1

    @staticmethod
    def _inf_compute_optimal_stride(img_w, img_h, resolution):
        """Minimum stride for full pixel coverage."""
        if img_w <= resolution and img_h <= resolution:
            return resolution

        def min_stride_for_dim(dim):
            excess = dim - resolution
            k = math.ceil(excess / resolution)
            return excess / k

        strides = []
        if img_w > resolution:
            strides.append(min_stride_for_dim(img_w))
        if img_h > resolution:
            strides.append(min_stride_for_dim(img_h))
        return max(1, int(math.floor(min(strides))))

    @staticmethod
    def _inf_is_network_path(path):
        """Return True if path is a UNC share or mapped network drive."""
        norm = os.path.normpath(path)
        if norm.startswith('\\\\'):
            return True
        if os.name == 'nt' and len(norm) >= 2 and norm[1] == ':':
            import ctypes
            drive = norm[:3]
            dtype = ctypes.windll.kernel32.GetDriveTypeW(drive)
            DRIVE_REMOTE = 4
            return dtype == DRIVE_REMOTE
        return False

    def _inf_copy_worker(self, copy_q, tmp_dir):
        """Background thread: move files from tmp to final network destination."""
        while True:
            item = copy_q.get()
            if item is None:
                break
            tmp_path, final_path = item
            try:
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.move(tmp_path, final_path)
            except Exception as e:
                logging.error(f"Async copy failed {os.path.basename(final_path)}: {e}")
            finally:
                copy_q.task_done()

    def _inf_process_folder(self, model, input_dir, output_dir, resolution, stride,
                            device, use_amp, transform, loss_mode, tile_batch=4):
        """Process all images in a folder (ported from inference_gui.py)."""
        from inference import process_image, denormalize, load_image_any_format

        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp', '*.exr']
        input_files = sorted([f for ext in img_extensions
                              for f in globglob(os.path.join(input_dir, ext))])
        if not input_files:
            logging.error(f"No images found in {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        skip_existing = self.inf_skip_existing.isChecked()

        work_items = []
        for img_path in input_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            match = re.match(r'^(.+?)_?(\d+)$', basename)
            if match:
                output_filename = f"{match.group(1)}_tunet_{match.group(2)}.png"
            else:
                output_filename = f"{basename}_tunet.png"
            output_path = os.path.join(output_dir, output_filename)
            work_items.append((img_path, output_path))

        skipped = 0
        if skip_existing:
            filtered = []
            for img_path, output_path in work_items:
                if os.path.exists(output_path):
                    skipped += 1
                else:
                    filtered.append((img_path, output_path))
            work_items = filtered

        logging.info(f"Found {len(input_files)} images in {os.path.basename(input_dir)}"
                     + (f" ({skipped} already done, {len(work_items)} remaining)" if skipped else ""))

        if not work_items:
            logging.info("All files already processed, nothing to do.")
            return

        # Auto-stride
        if self.inf_auto_stride.isChecked():
            try:
                from PIL import Image as _PIL
                with _PIL.open(work_items[0][0]) as _img:
                    first_w, first_h = _img.size
                stride = self._inf_compute_optimal_stride(first_w, first_h, resolution)

                def _tile_pos(dim):
                    return len(set(list(range(0, dim - resolution, stride)) +
                                   ([dim - resolution] if dim > resolution else [0])))

                tiles = _tile_pos(first_w) * _tile_pos(first_h)
                logging.info(f"Auto-stride: {first_w}x{first_h} -> stride={stride}, {tiles} tile(s)/frame")
                QTimer.singleShot(0, lambda s=stride: self.inf_stride.setValue(s))
            except Exception as e:
                logging.warning(f"Auto-stride failed ({e}), using manual stride={stride}")

        # Prefetch
        prefetch_q = queue.Queue(maxsize=2)

        def prefetch_worker():
            for item_pair in work_items:
                if self.inference_stop_requested:
                    prefetch_q.put(None)
                    return
                img_path, _ = item_pair
                try:
                    pil = load_image_any_format(img_path)
                except Exception as e:
                    logging.error(f"Prefetch failed for {os.path.basename(img_path)}: {e}")
                    pil = None
                prefetch_q.put((item_pair, pil))
            prefetch_q.put(None)

        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        # Async copy for network paths
        use_async_copy = self._inf_is_network_path(output_dir)
        tmp_dir = None
        copy_queue_obj = None
        copy_thread = None

        if use_async_copy:
            tmp_dir = tempfile.mkdtemp(prefix='tunet_')
            copy_queue_obj = queue.Queue()
            copy_thread = threading.Thread(target=self._inf_copy_worker,
                                           args=(copy_queue_obj, tmp_dir), daemon=True)
            copy_thread.start()
            logging.info(f"Network output detected — writing to local temp, copying async.")

        total = len(work_items)
        frame_times = []
        processed = 0

        try:
            stop_early = False
            while True:
                if self.inference_stop_requested:
                    stop_early = True
                    break
                prefetched = prefetch_q.get()
                if prefetched is None:
                    break
                (img_path, output_path), pil_img = prefetched
                if pil_img is None:
                    logging.error(f"Skipping {os.path.basename(img_path)} (load failed)")
                    continue
                write_path = (os.path.join(tmp_dir, os.path.basename(output_path))
                              if use_async_copy else output_path)

                t0 = time.perf_counter()
                process_image(model, img_path, write_path, resolution, stride, device,
                              tile_batch, transform, denormalize, use_amp,
                              self.inf_half_res.isChecked(),
                              loss_mode=loss_mode, src_image=pil_img)
                elapsed = time.perf_counter() - t0

                processed += 1
                frame_times.append(elapsed)
                avg = sum(frame_times) / len(frame_times)
                eta_str = self._inf_format_eta(int(avg * (total - processed)))

                if use_async_copy:
                    copy_queue_obj.put((write_path, output_path))
                    qd = copy_queue_obj.qsize()
                    qi = f"  [copy queue: {qd}]" if qd > 1 else ""
                else:
                    qi = ""

                logging.info(f"[{processed}/{total}] {os.path.basename(img_path)}  "
                             f"{elapsed:.2f}s  avg {avg:.2f}s  ETA {eta_str}{qi}")
                pct = int(processed / total * 100)
                QTimer.singleShot(0, lambda p=pct, c=processed, t=total, e=eta_str, fr=elapsed, a=avg:
                                  self._inf_update_progress(p, c, t, e, fr, a))

            if stop_early:
                logging.info(f"Stopped. {processed}/{total} processed, {total - processed} remaining.")
        finally:
            try:
                while True:
                    prefetch_q.get_nowait()
            except queue.Empty:
                pass
            prefetch_thread.join(timeout=5)

            if use_async_copy and copy_queue_obj is not None:
                pending_copies = copy_queue_obj.qsize()
                if pending_copies > 0:
                    logging.info(f"Waiting for {pending_copies} files to finish copying...")
                copy_queue_obj.put(None)
                copy_thread.join()
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

    @staticmethod
    def _inf_format_eta(secs):
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs // 60}m {secs % 60:02d}s"
        else:
            h = secs // 3600
            m = (secs % 3600) // 60
            return f"{h}h {m:02d}m"

    def _inf_update_progress(self, pct, current, total, eta="", frame_time=None, avg_time=None):
        self.inf_progress_bar.setValue(pct)
        parts = [f"{current}/{total}"]
        if frame_time is not None:
            parts.append(f"{frame_time:.2f}s/frame")
        if avg_time is not None:
            parts.append(f"avg {avg_time:.2f}s")
        if eta:
            parts.append(f"ETA {eta}")
        self.inf_progress_label.setText("  ".join(parts))

    # =========================================================================
    # Inference queue management
    # =========================================================================

    def _inf_add_to_queue(self):
        input_d = self._get_path(self.inf_input_dir)
        output_r = self._get_path(self.inf_output_root)
        if not input_d or not os.path.isdir(input_d):
            QMessageBox.warning(self, "Error", "Input directory is invalid.")
            return
        if not output_r:
            QMessageBox.warning(self, "Error", "Output Root is empty.")
            return
        self.inference_queue.append({'input_dir': input_d, 'output_root': output_r, 'status': 'pending'})
        self._inf_refresh_queue_display()
        logging.info(f"Added to inference queue: {os.path.basename(input_d)}")

    def _inf_remove_from_queue(self):
        selected = sorted(self.inf_queue_listbox.selectedIndexes(), key=lambda x: x.row(), reverse=True)
        for idx in selected:
            row = idx.row()
            if row < len(self.inference_queue) and self.inference_queue[row]['status'] != 'processing':
                self.inference_queue.pop(row)
        self._inf_refresh_queue_display()

    def _inf_clear_queue(self):
        self.inference_queue = [q for q in self.inference_queue if q['status'] == 'processing']
        self._inf_refresh_queue_display()

    def _inf_refresh_queue_display(self):
        self.inf_queue_listbox.clear()
        for i, item in enumerate(self.inference_queue):
            status = item['status']
            prefix = {'pending': '   ', 'processing': '>> ', 'done': 'OK ', 'error': '!! '}.get(status, '   ')
            input_name = os.path.basename(item['input_dir'])
            resolved = item.get('resolved_output', '')
            if resolved:
                dest = os.path.basename(resolved)
            else:
                dest = f"{os.path.basename(item['output_root'])}/auto"
            display = f"{prefix}{input_name} -> {dest}"
            self.inf_queue_listbox.addItem(display)
            list_item = self.inf_queue_listbox.item(i)
            if status == 'processing':
                list_item.setForeground(Qt.blue)
            elif status == 'done':
                list_item.setForeground(Qt.darkGreen)
            elif status == 'error':
                list_item.setForeground(Qt.red)

    # =========================================================================
    # Export / conversion
    # =========================================================================

    def _start_conversion(self, target_type):
        model_folder_str = self._get_path(self.model_folder_input)
        if not model_folder_str or not Path(model_folder_str).is_dir():
            QMessageBox.warning(self, "Error", "Model Folder is not set or does not exist.")
            return
        source_path_str, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", model_folder_str, "PyTorch Checkpoints (*.pth)")
        if not source_path_str:
            return

        # Remember the selected file's directory for output
        self._conversion_output_dir = str(Path(source_path_str).parent)

        if self.copy_before_convert_check.isChecked():
            self.conversion_target = target_type
            source_path = Path(source_path_str)
            dest_folder = Path(model_folder_str) / self.conversion_target
            dest_folder.mkdir(parents=True, exist_ok=True)
            new_checkpoint_path = dest_folder / source_path.name

            self.progress_dialog = QProgressDialog(
                f"Copying to {self.conversion_target} folder...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowTitle("Copying Checkpoint")
            self.progress_dialog.setWindowModality(Qt.WindowModal)

            self.copy_thread = QThread()
            self.copy_worker = FileCopyWorker()
            self.copy_worker.moveToThread(self.copy_thread)
            self.progress_dialog.canceled.connect(self.copy_worker.cancel)
            self.copy_worker.progress.connect(self.progress_dialog.setValue)
            self.copy_worker.finished.connect(self._on_copy_finished)
            self.copy_worker.error.connect(self._on_copy_error)
            self.copy_thread.started.connect(
                lambda: self.copy_worker.run(source_path_str, str(new_checkpoint_path)))
            self.copy_thread.start()
            self.progress_dialog.exec()
        else:
            self.console_output.append(f"Converting checkpoint directly: {source_path_str}\n")
            command = self._build_utility_command(target_type, source_path_str)
            if command:
                self._run_utility_script(command)

    @Slot(str)
    def _on_copy_finished(self, new_checkpoint_path):
        self.progress_dialog.close()
        self.console_output.append(f"Copied to: {new_checkpoint_path}\n")
        command = self._build_utility_command(self.conversion_target, new_checkpoint_path)
        if command:
            self._run_utility_script(command)
        self.copy_thread.quit()
        self.copy_thread.wait()

    @Slot(str)
    def _on_copy_error(self, error_message):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Copy Error", error_message)
        self.copy_thread.quit()
        self.copy_thread.wait()

    def _build_utility_command(self, target_type, checkpoint_path):
        if target_type == 'flame':
            script_path = _APP_DIR / "utils" / "convert_flame.py"
            cmd = [sys.executable, str(script_path), '--checkpoint', checkpoint_path, '--use_gpu']
            if hasattr(self, '_conversion_output_dir') and self._conversion_output_dir:
                cmd.extend(['--output_dir', self._conversion_output_dir])
            return cmd
        elif target_type == 'nuke':
            script_path = _APP_DIR / "utils" / "convert_nuke.py"
            return [sys.executable, str(script_path), '--generate_nk',
                    '--checkpoint_pth', checkpoint_path, '--method', 'script']
        return None

    def _run_utility_script(self, command):
        if self.utility_process:
            QMessageBox.warning(self, "Busy", "Another utility script is already running.")
            return
        self.console_output.clear()
        self.console_output.append(f"Running:\n{' '.join(command)}\n\n")
        try:
            self.utility_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='replace')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start: {e}")
            return
        self.utility_stream_reader = ProcessStreamReader(self.utility_process.stdout)
        self.utility_stream_reader.new_text.connect(self._append_text)
        self.convert_flame_btn.setEnabled(False)
        self.convert_nuke_btn.setEnabled(False)
        self.utility_monitor.start()

    @Slot()
    def _check_utility_status(self):
        if not self.utility_process:
            return
        if self.utility_process.poll() is not None:
            self.utility_monitor.stop()
            self.utility_process = None
            self.console_output.append("\n--- Utility script finished. ---\n")
            self.convert_flame_btn.setEnabled(True)
            self.convert_nuke_btn.setEnabled(True)

    # =========================================================================
    # Training Monitor
    # =========================================================================

    def _launch_training_monitor(self):
        monitor_script = _APP_DIR / "training_monitor.py"
        if not monitor_script.is_file():
            QMessageBox.warning(self, "Error", f"Training monitor script not found:\n{monitor_script}")
            return
        model_folder = self._get_path(self.model_folder_input)
        command = [sys.executable, str(monitor_script)]
        if model_folder and Path(model_folder).is_dir():
            log_file = Path(model_folder) / "training.log"
            if log_file.exists():
                command.extend(['--log_file', str(log_file)])
            else:
                command.extend(['--output_dir', model_folder])
        try:
            subprocess.Popen(command)
            self.console_output.append("Launched Training Monitor\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch training monitor:\n{e}")

    # =========================================================================
    # Window events
    # =========================================================================

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.process_monitor.stop()
        self._save_session()

        if self.process and self.process.poll() is None:
            reply = QMessageBox.question(
                self, 'Exit Confirmation',
                "A training process is still running. Exit and terminate it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._stop_training()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
