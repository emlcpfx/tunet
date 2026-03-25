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
    QProgressBar, QStackedWidget, QFrame, QSlider,
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QFileSystemWatcher, QTimer, QThread
from PySide6.QtGui import QPixmap, QTextCursor

from gui import (
    IndentDumper, ProcessStreamReader, FileCopyWorker, ZoomPanScrollArea, QTextEditLogHandler,
    DataTabMixin, TrainingTabMixin, PreviewsTabMixin, ExportTabMixin, InferenceTabMixin, AboutTabMixin,
)


# =============================================================================
# Session file paths
# =============================================================================
_APP_DIR = Path(__file__).parent
_SESSION_FILE = _APP_DIR / 'tunet_session.yaml'
_LEGACY_INFERENCE_SETTINGS = _APP_DIR / 'inference_gui_settings.json'


# =============================================================================
# MainWindow
# =============================================================================

class MainWindow(DataTabMixin, TrainingTabMixin, PreviewsTabMixin, ExportTabMixin, InferenceTabMixin, AboutTabMixin, QMainWindow):
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
        self._cached_color_space = 'srgb'

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
        self.console_output.setPlaceholderText("Console — training output will appear here")

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
        self._create_training_tab()      # Tab 0 (includes project folder)
        self._create_previews_tab()      # Tab 1
        self._create_export_tab()        # Tab 2
        self._create_inference_tab()     # Tab 3
        self._create_about_tab()         # Tab 4

        # Remember the inference tab index for sidebar switching
        self._inference_tab_index = 3

        # --- Signals ---
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.verify_btn.clicked.connect(self._verify_inputs)

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
        self._load_session()

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
    def _combo_value(combo):
        """Extract the numeric/value prefix from a combo item like '512 — Good default'."""
        return combo.currentText().split("\u2014")[0].strip()

    @staticmethod
    def _set_combo_by_prefix(combo, value):
        """Select the combo item whose text starts with the given value."""
        value_str = str(value)
        for i in range(combo.count()):
            if combo.itemText(i).split("\u2014")[0].strip() == value_str:
                combo.setCurrentIndex(i)
                return
        # Fallback: set as raw text (works for editable combos)
        combo.setCurrentText(value_str)

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
        sidebar_layout = QHBoxLayout()
        sidebar_layout.setSpacing(0)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        # --- Toggle button (thin vertical strip, always visible) ---
        self.sidebar_toggle = QPushButton("◀")
        self.sidebar_toggle.setCheckable(True)
        self.sidebar_toggle.setChecked(False)
        self.sidebar_toggle.setProperty("cssClass", "sidebar-toggle")
        self.sidebar_toggle.setFixedWidth(22)
        self.sidebar_toggle.setToolTip("Show / hide the queue panel")
        self.sidebar_toggle.toggled.connect(self._on_sidebar_toggled)
        sidebar_layout.addWidget(self.sidebar_toggle)

        # --- Panel (the part that expands/collapses) ---
        self.sidebar_panel = QWidget()
        self.sidebar_panel.setMinimumWidth(200)
        self.sidebar_panel.setMaximumWidth(280)
        panel_layout = QVBoxLayout(self.sidebar_panel)
        panel_layout.setContentsMargins(4, 0, 0, 0)

        # Queue title label
        self.sidebar_queue_label = QLabel("Training Queue")
        self.sidebar_queue_label.setStyleSheet("font-weight: 600; padding: 4px 0;")
        panel_layout.addWidget(self.sidebar_queue_label)

        # Stacked widget for switching between training/inference queues
        self.sidebar_stack = QStackedWidget()

        # --- Page 0: Training Queue ---
        train_queue_page = QWidget()
        tq_layout = QVBoxLayout(train_queue_page)
        tq_layout.setContentsMargins(0, 0, 0, 0)
        self.queue_listbox = QListWidget()
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
        self.queue_run_btn.setProperty("cssClass", "start")
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

        panel_layout.addWidget(self.sidebar_stack, stretch=1)
        sidebar_layout.addWidget(self.sidebar_panel)

        return sidebar_layout

    def _on_sidebar_toggled(self, expanded):
        self.sidebar_panel.setVisible(expanded)
        self.sidebar_toggle.setText("◀" if expanded else "▶")

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
        self.monitor_btn.setProperty("cssClass", "accent")
        self.monitor_btn.clicked.connect(self._launch_training_monitor)
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setProperty("cssClass", "start")
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setProperty("cssClass", "stop")
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
            self.sidebar_queue_label.setText("Inference Queue")
        else:
            self.sidebar_stack.setCurrentIndex(0)
            self.sidebar_queue_label.setText("Training Queue")

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
            'resolution': int(self._combo_value(self.resolution_input)),
            'overlap_factor': float(self._combo_value(self.overlap_factor_input)),
            'color_space': self.color_space_input.currentText(),
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
                'batch_size': 0 if self.auto_batch_check.isChecked() else self.batch_size_input.value(),
                'max_steps': self.max_steps_input.value(),
                'use_amp': self.use_amp_input.isChecked(),
                'loss': self.loss_input.currentText(),
                'lambda_lpips': lambda_value,
                'lr': lr_value,
                'progressive_resolution': self.progressive_res_check.isChecked(),
                'lr_scheduler': self.lr_scheduler_input.currentText(),
                'l1_weight': self.l1_weight_input.value(),
                'l2_weight': self.l2_weight_input.value(),
                'lpips_weight': self.lpips_weight_input.value(),
            },
            'logging': {
                'log_interval': self.log_interval_input.value(),
                'preview_batch_interval': self.preview_interval_input.value(),
                'preview_refresh_rate': self.preview_refresh_input.value(),
                'diff_amplify': self.diff_amplify_slider.value(),
            },
            'saving': {
                'keep_last_checkpoints': self.keep_checkpoints_input.value(),
            },
            'early_stopping': {
                'enabled': self.es_enabled_input.isChecked(),
                'patience': self.es_patience_input.value(),
                'stop': self.es_stop_input.isChecked(),
            },
            'auto_export': {
                'interval': self.auto_export_interval_input.value(),
                'flame': self.auto_export_flame_check.isChecked(),
                'nuke': self.auto_export_nuke_check.isChecked(),
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
                'auto_mask_gamma': self.auto_mask_gamma_input.value(),
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
            'nproc_per_node': self.nproc_input.value() if self.nproc_input else 1,
        }
        return config

    def populate_ui_from_config(self, config):
        """Populate all UI fields from a config dict."""
        sp = self._set_path
        ui_settings = config.get('_ui_settings', {})
        if self.nproc_input:
            self.nproc_input.setValue(ui_settings.get('nproc_per_node', 1))

        # Data
        data = config.get('data', {})
        sp(self.src_dir_input, data.get('src_dir', ''))
        sp(self.dst_dir_input, data.get('dst_dir', ''))
        sp(self.mask_dir_input, data.get('mask_dir', ''))
        sp(self.model_folder_input, data.get('output_dir', ''))
        self._set_combo_by_prefix(self.resolution_input, data.get('resolution', 512))
        self._set_combo_by_prefix(self.overlap_factor_input, data.get('overlap_factor', 0.25))
        self.color_space_input.setCurrentText(data.get('color_space', 'srgb'))
        sp(self.val_src_dir_input, data.get('val_src_dir', ''))
        sp(self.val_dst_dir_input, data.get('val_dst_dir', ''))
        # Infer project folder from src_dir parent
        src_dir = data.get('src_dir', '')
        if src_dir and os.path.isdir(src_dir):
            sp(self.project_folder_input, str(Path(src_dir).parent))

        # Model
        model = config.get('model', {})
        self.model_size_dims_input.setCurrentText(str(model.get('model_size_dims', 128)))
        self.model_type_input.setCurrentText(model.get('model_type', 'unet'))

        # Training
        training = config.get('training', {})
        sp(self.finetune_from_input, training.get('finetune_from', '') or '')
        self.iter_per_epoch_input.setValue(training.get('iterations_per_epoch', 500))
        saved_batch = training.get('batch_size', 4)
        self.auto_batch_check.setChecked(saved_batch == 0)
        self.batch_size_input.setValue(saved_batch if saved_batch > 0 else 4)
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

        # Progressive resolution, LR scheduler & weighted loss
        self.progressive_res_check.setChecked(training.get('progressive_resolution', False))
        self.lr_scheduler_input.setCurrentText(training.get('lr_scheduler', 'none'))
        self.l1_weight_input.setValue(training.get('l1_weight', 1.0))
        self.l2_weight_input.setValue(training.get('l2_weight', 0.0))
        self.lpips_weight_input.setValue(training.get('lpips_weight', 0.1))

        # Logging
        log_cfg = config.get('logging', {})
        self.log_interval_input.setValue(log_cfg.get('log_interval', 5))
        self.preview_interval_input.setValue(log_cfg.get('preview_batch_interval', 35))
        self.preview_refresh_input.setValue(log_cfg.get('preview_refresh_rate', 5))
        self.diff_amplify_slider.setValue(log_cfg.get('diff_amplify', 5))
        self.keep_checkpoints_input.setValue(config.get('saving', {}).get('keep_last_checkpoints', 4))

        # Early stopping
        es_cfg = config.get('early_stopping', {})
        self.es_enabled_input.setChecked(es_cfg.get('enabled', True))
        self.es_patience_input.setValue(es_cfg.get('patience', 30))
        self.es_stop_input.setChecked(es_cfg.get('stop', False))

        # Auto export
        ae_cfg = config.get('auto_export', {})
        self.auto_export_interval_input.setValue(ae_cfg.get('interval', 0))
        self.auto_export_flame_check.setChecked(ae_cfg.get('flame', False))
        self.auto_export_nuke_check.setChecked(ae_cfg.get('nuke', False))

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
        self.auto_mask_gamma_input.setValue(mask_cfg.get('auto_mask_gamma', 1.0))

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

    # =========================================================================
    # Training process management
    # =========================================================================

    def _launch_training(self, full_config):
        """Launch a training subprocess. Returns True if started."""
        train_script = str(_APP_DIR / "train.py")
        if not Path(train_script).is_file():
            QMessageBox.warning(self, "Error", f"Training script not found:\n{train_script}")
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

        # Stop file: a sentinel file that train.py polls for graceful shutdown
        self._stop_file_path = str(model_folder / '.stop_training')
        try: os.remove(self._stop_file_path)
        except OSError: pass
        command += ['--stop-file', self._stop_file_path]

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

    def _verify_inputs(self):
        """Run pre-flight dataset verification and show results."""
        src_dir = self._get_path(self.src_dir_input)
        dst_dir = self._get_path(self.dst_dir_input)
        if not src_dir or not os.path.isdir(src_dir):
            QMessageBox.warning(self, "Verify Inputs", "Source directory is not set or does not exist.")
            return
        if not dst_dir or not os.path.isdir(dst_dir):
            QMessageBox.warning(self, "Verify Inputs", "Target directory is not set or does not exist.")
            return

        mask_dir = self._get_path(self.mask_dir_input) or None
        if mask_dir and not os.path.isdir(mask_dir):
            mask_dir = None
        resolution = int(self._combo_value(self.resolution_input))
        color_space = self.color_space_input.currentText()

        # Show a progress dialog (verification can take a while on large datasets)
        progress = QProgressDialog("Scanning dataset...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Verify Inputs")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        try:
            from utils.verify_inputs import verify_dataset
            result = verify_dataset(src_dir, dst_dir, mask_dir, resolution, color_space)
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Verify Inputs", f"Verification failed:\n{e}")
            return
        progress.close()

        # Build message
        total = result['total_src']
        valid = result['valid_pairs']
        issues = result['issues']
        summary = result['summary']

        if not issues:
            QMessageBox.information(
                self, "Verify Inputs",
                f"All {total} image pairs passed verification.\n"
                f"Resolution: {resolution}px | Color space: {color_space}")
            return

        lines = [f"Scanned {total} source images: {valid} valid, {len(issues)} with issues.\n"]
        for reason, count in sorted(summary.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason}: {count}")
        lines.append("")

        # Show first 20 individual issues
        lines.append("Details (first 20):")
        for filename, reason in issues[:20]:
            lines.append(f"  {filename}: {reason}")
        if len(issues) > 20:
            lines.append(f"  ... and {len(issues) - 20} more")

        QMessageBox.warning(self, "Verify Inputs", "\n".join(lines))

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
        # Clean up stop file
        if hasattr(self, '_stop_file_path') and self._stop_file_path:
            try: os.remove(self._stop_file_path)
            except OSError: pass
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
            if hasattr(self, '_stop_file_path') and self._stop_file_path:
                # File-based stop: create sentinel file that train.py polls for
                try:
                    with open(self._stop_file_path, 'w') as f:
                        f.write('stop')
                except OSError:
                    pass
            if platform.system() != "Windows":
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
        self._cached_color_space = 'srgb'

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

            model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch, color_space = self._inf_load_model()
            input_dir = self._get_path(self.inf_input_dir)
            output_root = self._get_path(self.inf_output_root)
            checkpoint = self._get_path(self.inf_checkpoint_input)
            skip = self.inf_skip_existing.isChecked()
            output_dir = self._inf_resolve_output_dir(input_dir, output_root, checkpoint, skip)
            logging.info(f"Output folder: {output_dir}")
            self._inf_process_folder(model, input_dir, output_dir, resolution, stride,
                                     device, use_amp, transform, loss_mode, tile_batch,
                                     color_space=color_space)
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
            model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch, color_space = self._inf_load_model()
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
                                             resolution, stride, device, use_amp, transform, loss_mode, tile_batch,
                                             color_space=color_space)
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
            color_space = self._cached_color_space
        else:
            logging.info(f"Using device: {device}")
            logging.info("Loading model...")
            model, resolution, use_mask_input, loss_mode, color_space = load_model_and_config(checkpoint, device)
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
            self._cached_color_space = color_space

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

        return model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch, color_space

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
                            device, use_amp, transform, loss_mode, tile_batch=4,
                            color_space='srgb'):
        """Process all images in a folder (ported from inference_gui.py)."""
        from inference import process_image, denormalize, denormalize_linear, load_image_any_format, load_image_linear
        from inference_config import InferenceConfig
        is_linear = color_space == 'linear'
        denorm_fn = denormalize_linear if is_linear else denormalize

        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp', '*.exr']
        input_files = sorted([f for ext in img_extensions
                              for f in globglob(os.path.join(input_dir, ext))])
        if not input_files:
            logging.error(f"No images found in {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        skip_existing = self.inf_skip_existing.isChecked()

        out_ext = '.exr' if is_linear else '.png'
        work_items = []
        for img_path in input_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            match = re.match(r'^(.+?)_?(\d+)$', basename)
            if match:
                output_filename = f"{match.group(1)}_tunet_{match.group(2)}{out_ext}"
            else:
                output_filename = f"{basename}_tunet{out_ext}"
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
                    img_data = load_image_linear(img_path) if is_linear else load_image_any_format(img_path)
                except Exception as e:
                    logging.error(f"Prefetch failed for {os.path.basename(img_path)}: {e}")
                    img_data = None
                prefetch_q.put((item_pair, img_data))
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
                (img_path, output_path), img_data = prefetched
                if img_data is None:
                    logging.error(f"Skipping {os.path.basename(img_path)} (load failed)")
                    continue
                write_path = (os.path.join(tmp_dir, os.path.basename(output_path))
                              if use_async_copy else output_path)

                t0 = time.perf_counter()
                inf_cfg = InferenceConfig(
                    resolution=resolution, stride=stride, device=device,
                    batch_size=tile_batch, use_amp=use_amp,
                    half_res=self.inf_half_res.isChecked(),
                    loss_mode=loss_mode, color_space=color_space)
                process_image(model, img_path, write_path, inf_cfg, transform, denorm_fn,
                              src_image=img_data)
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
            script_path = _APP_DIR / "exporters" / "flame_exporter.py"
            cmd = [sys.executable, str(script_path), '--checkpoint', checkpoint_path, '--use_gpu']
            if hasattr(self, '_conversion_output_dir') and self._conversion_output_dir:
                cmd.extend(['--output_dir', self._conversion_output_dir])
            return cmd
        elif target_type == 'nuke':
            script_path = _APP_DIR / "exporters" / "nuke_exporter.py"
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
            # Also pass parent dir so the monitor can discover sibling runs
            parent_dir = Path(model_folder).parent
            if parent_dir.is_dir():
                command.extend(['--data_dir', str(parent_dir)])
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
                    self.process.wait(timeout=5)
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

def apply_spark_theme(app):
    """Apply a Spark Cloud Studio-inspired light theme."""
    # -- Palette (directly from spark-tunet.css) --
    BG         = "#F9FAFB"   # gray-100: page background
    WHITE      = "#ffffff"   # cards, inputs
    BORDER     = "#e5e7eb"   # gray-200: card/input borders
    BORDER_DK  = "#D1D5DB"   # gray-300: dividers
    TEXT       = "#111827"   # gray-900: primary text
    TEXT_SEC   = "#374151"   # gray-700: secondary text
    TEXT_DIM   = "#6b7280"   # gray-500: muted/placeholder
    TEXT_FAINT = "#9ca3af"   # gray-400: disabled
    ACCENT     = "#ae69f4"   # spark-purple
    ACCENT_HVR = "#7E3AF2"   # spark-purple-dark (hover)
    ACCENT_ACT = "#6C2BD9"   # spark-purple-deep (pressed)
    ACCENT_LT  = "#F7F4FC"   # spark-purple-light
    ACCENT_HI  = "#c084fc"   # gradient end
    GREEN      = "#16A34A"
    GREEN_BG   = "#f0fdf4"
    RED        = "#EF4444"
    RED_BG     = "#fef2f2"

    app.setStyleSheet(f"""
        /* -- Global -- */
        QWidget {{
            background-color: {BG};
            color: {TEXT};
            font-family: "Plus Jakarta Sans", "Segoe UI", "SF Pro Text", sans-serif;
            font-size: 10pt;
            selection-background-color: {ACCENT_LT};
            selection-color: {ACCENT_HVR};
        }}

        QMainWindow {{
            background-color: {BG};
        }}

        /* -- Tab Widget (Spark flat tabs with bottom-border accent) -- */
        QTabWidget::pane {{
            border: none;
            background-color: {BG};
            top: -1px;
        }}
        QTabBar {{
            background-color: {WHITE};
            border-bottom: 1px solid {BORDER};
        }}
        QTabBar::tab {{
            background-color: transparent;
            color: {TEXT_DIM};
            border: none;
            border-bottom: 2px solid transparent;
            padding: 12px 24px;
            margin-right: 0;
            font-weight: 500;
            font-size: 10pt;
        }}
        QTabBar::tab:selected {{
            color: {ACCENT};
            border-bottom: 2px solid {ACCENT};
            font-weight: 600;
        }}
        QTabBar::tab:hover:!selected {{
            color: {TEXT_SEC};
        }}

        /* -- Group Box (Spark card: 20px 24px padding, 12px radius) -- */
        QGroupBox {{
            background-color: {WHITE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            margin-top: 20px;
            padding: 24px 24px 20px 24px;
            font-weight: 600;
            font-size: 10pt;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 16px;
            padding: 3px 12px;
            background-color: {WHITE};
            border: 1px solid {BORDER};
            border-radius: 6px;
            color: {ACCENT};
        }}

        /* -- Collapsible Section Header -- */
        QPushButton[cssClass="collapse-header"] {{
            background-color: {WHITE};
            color: {ACCENT};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 12px 20px;
            font-weight: 600;
            font-size: 10pt;
            text-align: left;
        }}
        QPushButton[cssClass="collapse-header"]:hover {{
            background-color: {ACCENT_LT};
            border-color: {ACCENT};
        }}

        /* -- Collapsible Section Body -- */
        QWidget[cssClass="collapse-body"] {{
            background-color: {WHITE};
            border: 1px solid {BORDER};
            border-top: none;
            border-radius: 0 0 12px 12px;
        }}

        /* Section description labels */
        QLabel[cssClass="section-desc"] {{
            color: {TEXT_DIM};
            font-size: 9pt;
            font-style: italic;
            padding: 4px 6px 10px 6px;
            background-color: transparent;
        }}

        /* -- Buttons (Spark-style: 10px 20px) -- */
        QPushButton {{
            background-color: {WHITE};
            color: {TEXT_SEC};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            min-height: 22px;
        }}
        QPushButton:hover {{
            background-color: {BG};
            border-color: {ACCENT};
            color: {TEXT};
        }}
        QPushButton:pressed {{
            background-color: {ACCENT_LT};
        }}
        QPushButton:disabled {{
            background-color: {BG};
            color: {TEXT_FAINT};
            border-color: {BORDER};
        }}
        QPushButton:checked {{
            background-color: {ACCENT_LT};
            color: {ACCENT};
            border-color: {ACCENT};
        }}

        /* Primary (purple filled) */
        QPushButton[cssClass="start"] {{
            background-color: {GREEN};
            color: {WHITE};
            border: none;
            font-weight: 600;
        }}
        QPushButton[cssClass="start"]:hover {{
            background-color: #15803D;
        }}
        QPushButton[cssClass="stop"] {{
            background-color: {RED};
            color: {WHITE};
            border: none;
            font-weight: 600;
        }}
        QPushButton[cssClass="stop"]:hover {{
            background-color: #dc2626;
        }}
        QPushButton[cssClass="sidebar-toggle"] {{
            background-color: {ACCENT_LT};
            color: {ACCENT};
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 0;
            font-size: 10pt;
            min-height: 40px;
        }}
        QPushButton[cssClass="sidebar-toggle"]:hover {{
            background-color: {ACCENT};
            color: {WHITE};
            border-color: {ACCENT};
        }}
        QPushButton[cssClass="sidebar-toggle"]:checked {{
            background-color: {ACCENT_LT};
            color: {ACCENT};
            border-color: {BORDER};
        }}
        QPushButton[cssClass="accent"] {{
            background-color: {ACCENT};
            color: {WHITE};
            border: none;
        }}
        QPushButton[cssClass="accent"]:hover {{
            background-color: {ACCENT_HVR};
        }}

        /* -- Inputs (Spark form inputs: 10px 14px) -- */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {WHITE};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 10px 14px;
            min-height: 24px;
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
        QComboBox:on {{
            border-color: {ACCENT};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {TEXT_FAINT};
            margin-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {WHITE};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 4px;
            selection-background-color: {ACCENT_LT};
            selection-color: {ACCENT_HVR};
            outline: none;
        }}

        /* -- Checkbox -- */
        QCheckBox {{
            spacing: 8px;
            color: {TEXT};
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {BORDER_DK};
            border-radius: 4px;
            background-color: {WHITE};
        }}
        QCheckBox::indicator:checked {{
            background-color: {ACCENT};
            border-color: {ACCENT};
        }}
        QCheckBox::indicator:hover {{
            border-color: {ACCENT};
        }}
        QCheckBox:disabled {{
            color: {TEXT_FAINT};
        }}

        /* -- Scroll Area -- */
        QScrollArea {{
            border: none;
            background-color: {BG};
        }}
        QScrollBar:vertical {{
            background-color: transparent;
            width: 8px;
            margin: 0;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background-color: {BORDER_DK};
            min-height: 30px;
            border-radius: 4px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {TEXT_FAINT};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background-color: transparent;
            height: 8px;
            margin: 0;
            border: none;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {BORDER_DK};
            min-width: 30px;
            border-radius: 4px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {TEXT_FAINT};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}

        /* -- Text Edit (console — dark inset like Spark) -- */
        QTextEdit {{
            background-color: #1F2937;
            color: #d1d5db;
            border: 1px solid {BORDER};
            border-radius: 12px;
            font-family: "Fira Mono", "Cascadia Code", "JetBrains Mono", "Consolas", monospace;
            font-size: 9pt;
            padding: 10px;
        }}

        /* -- Splitter -- */
        QSplitter::handle {{
            background-color: {BORDER};
            height: 3px;
        }}
        QSplitter::handle:hover {{
            background-color: {ACCENT};
        }}

        /* -- Progress Bar (Spark purple gradient) -- */
        QProgressBar {{
            background-color: {BORDER};
            border: none;
            border-radius: 6px;
            text-align: center;
            color: {TEXT};
            height: 8px;
            font-size: 9pt;
        }}
        QProgressBar::chunk {{
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {ACCENT}, stop:1 {ACCENT_HI});
            border-radius: 6px;
        }}

        /* -- Labels -- */
        QLabel {{
            background-color: transparent;
            color: {TEXT};
        }}

        /* -- Slider -- */
        QSlider::groove:horizontal {{
            background-color: {BORDER};
            height: 4px;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background-color: {ACCENT};
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        QSlider::handle:horizontal:hover {{
            background-color: {ACCENT_HVR};
        }}

        /* -- List Widget (queues) -- */
        QListWidget {{
            background-color: {WHITE};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 8px;
            font-family: "Fira Mono", "Cascadia Code", "Consolas", monospace;
            font-size: 9pt;
            outline: none;
            padding: 4px;
        }}
        QListWidget::item {{
            padding: 4px 8px;
            border-bottom: 1px solid {BG};
            border-radius: 4px;
        }}
        QListWidget::item:selected {{
            background-color: {ACCENT_LT};
            color: {ACCENT_HVR};
        }}
        QListWidget::item:hover {{
            background-color: {BG};
        }}

        /* -- Frame separator -- */
        QFrame[frameShape="4"] {{
            color: {BORDER};
        }}

        /* -- Tool Tips -- */
        QToolTip {{
            background-color: {WHITE};
            color: {TEXT};
            border: 1px solid {BORDER};
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 9pt;
        }}

        /* -- Form Layout -- */
        QFormLayout {{
            margin: 6px;
        }}
    """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_spark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
