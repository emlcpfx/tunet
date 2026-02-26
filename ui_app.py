# ==============================================================================
# Tunet Training Control UI 
#
# A cross-platform desktop application built with PySide6.
#
# require: pip install PySide6
#
# ==============================================================================

import sys
import os
import subprocess
import threading
import signal
import yaml
from pathlib import Path
import platform
import shutil

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QFileDialog, QFormLayout, QMessageBox, QSizePolicy, QScrollArea,
    QSplitter, QProgressDialog, QDoubleSpinBox, QListWidget
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QFileSystemWatcher, QTimer, QThread
from PySide6.QtGui import QPixmap, QTextCursor


class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


class ProcessStreamReader(QObject):
    new_text = Signal(str)
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self._thread = threading.Thread(target=self.run)
        self._thread.daemon = True
        self._thread.start()
    def run(self):
        for line in iter(self.stream.readline, ''):
            self.new_text.emit(line)

class FileCopyWorker(QObject):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)
    _is_cancelled = False
    def run(self, source_path_str, dest_path_str):
        self._is_cancelled = False; source_path = Path(source_path_str); dest_path = Path(dest_path_str)
        try:
            total_size = source_path.stat().st_size; bytes_copied = 0; chunk_size = 4096 * 4
            with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
                while True:
                    if self._is_cancelled:
                        self.error.emit("Copy operation cancelled.")
                        if dest_path.exists(): dest_path.unlink()
                        return
                    chunk = src.read(chunk_size)
                    if not chunk: break
                    dst.write(chunk); bytes_copied += len(chunk)
                    percent = int((bytes_copied / total_size) * 100); self.progress.emit(percent)
            self.finished.emit(str(dest_path))
        except Exception as e: self.error.emit(f"File copy failed: {e}")
    def cancel(self): self._is_cancelled = True

class ZoomPanScrollArea(QScrollArea):
    zoom_changed = Signal(float)
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.zoom_changed.emit(factor)
        event.accept()
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True; self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor); event.accept()
        else: super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if getattr(self, '_panning', False) and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start; self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else: super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False; self.setCursor(Qt.ArrowCursor); event.accept()
        else: super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Tunet Training UI - Running on {platform.system()}")
        self.setGeometry(100, 100, 900, 700)
        self.process = None; self.utility_process = None; self.config_file_path = Path("config_from_ui.yaml")
        self.preview_image_path = None; self.original_pixmap = None; self.copy_thread = None; self.conversion_target = None
        self.val_preview_image_path = None; self.val_original_pixmap = None; self._preview_zoom_factor = None; self._val_preview_zoom_factor = None
        self.training_queue = []; self.queue_running = False; self.queue_stop_requested = False; self._last_config_dir = ""
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        # Left side: tabs + console + controls
        left_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.console_output = QTextEdit(); self.console_output.setReadOnly(True); self.console_output.setFontFamily("monospace")
        splitter = QSplitter(Qt.Vertical); splitter.addWidget(self.tabs); splitter.addWidget(self.console_output); splitter.setSizes([550, 150])
        left_layout.addWidget(splitter); left_layout.addLayout(self.create_control_panel())
        self.main_layout.addLayout(left_layout, stretch=1)
        # Right side: queue sidebar
        self.main_layout.addLayout(self.create_queue_section())
        self.create_main_tab(); self.create_advanced_tab(); self.create_dataloader_tab()
        self.create_preview_tab(); self.create_val_preview_tab(); self.create_convert_tab(); self.create_about_tab()
        self.file_watcher = QFileSystemWatcher(); self.file_watcher.fileChanged.connect(self.on_watched_file_changed)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.preview_timer = QTimer(self); self.preview_timer.setInterval(2500); self.preview_timer.timeout.connect(self.update_all_previews)
        self.process_monitor = QTimer(self); self.process_monitor.setInterval(500); self.process_monitor.timeout.connect(self.check_process_status)
        self.utility_monitor = QTimer(self); self.utility_monitor.setInterval(500); self.utility_monitor.timeout.connect(self.check_utility_status)
        self.populate_default_script_path()

    def populate_default_script_path(self):
        try:
            app_dir = Path(__file__).parent; default_script_path = app_dir / "train.py"
            if default_script_path.is_file():
                line_edit = self.train_script_input.findChild(QLineEdit)
                if line_edit: line_edit.setText(str(default_script_path.resolve()))
        except (NameError, Exception): pass
    
    def create_main_tab(self):
        tab = QWidget(); layout = QFormLayout(tab); layout.addRow(QLabel("--- Launcher Settings ---")); self.train_script_input = self.create_path_selector("Training Script (.py)", is_file=True); layout.addRow("Training Script:", self.train_script_input)
        if platform.system() == 'Linux': self.nproc_input = QSpinBox(minimum=1, maximum=16, value=1); layout.addRow("GPUs (nproc_per_node):", self.nproc_input)
        else: self.nproc_input = None
        layout.addRow(QLabel(" ")); self.src_dir_input = self.create_path_selector("Source Directory"); self.dst_dir_input = self.create_path_selector("Destination Directory"); self.mask_dir_input = self.create_path_selector("Mask Directory"); self.model_folder_input = self.create_path_selector("Model Folder", is_output=True); self.resolution_input = QComboBox(); self.resolution_input.setEditable(True); self.resolution_input.addItems(["256", "384", "512", "640", "768", "896", "928", "960", "1024"]); self.resolution_input.setCurrentText("512"); self.model_size_dims_input = QComboBox(); self.model_size_dims_input.addItems(["32", "64", "128", "256", "512"]); self.model_size_dims_input.setCurrentText("128"); layout.addRow(QLabel("--- Data Settings ---")); layout.addRow("Source Directory:", self.src_dir_input); layout.addRow("Destination Directory:", self.dst_dir_input); layout.addRow("Mask Directory (optional):", self.mask_dir_input); layout.addRow("Model Folder:", self.model_folder_input); layout.addRow("Resolution:", self.resolution_input)
        self.val_src_dir_input = self.create_path_selector("Val Source Directory"); self.val_dst_dir_input = self.create_path_selector("Val Destination Directory")
        layout.addRow(QLabel("--- Validation Data (optional) ---")); layout.addRow("Val Source Directory:", self.val_src_dir_input); layout.addRow("Val Destination Directory:", self.val_dst_dir_input)
        layout.addRow(QLabel("--- Model Settings ---")); layout.addRow("Model Size Dims:", self.model_size_dims_input); self.tabs.addTab(tab, "Main")
    def create_path_selector(self, label_text, is_output=False, is_file=False):
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0); line_edit = QLineEdit(); button = QPushButton("Browse...")
        if is_file: button.clicked.connect(lambda: self.select_file(line_edit))
        else: button.clicked.connect(lambda: self.select_directory(line_edit))
        layout.addWidget(line_edit); layout.addWidget(button)
        if is_output: line_edit.textChanged.connect(self.setup_preview_watcher)
        return widget
    def select_directory(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory");
        if dir_path: line_edit.setText(dir_path)
    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Python Script", "", "Python Files (*.py)");
        if file_path: line_edit.setText(file_path)
    def create_advanced_tab(self):
        tab = QWidget(); layout = QFormLayout(tab); self.iter_per_epoch_input = QSpinBox(minimum=1, maximum=10000, value=500); self.batch_size_input = QSpinBox(minimum=1, maximum=256, value=4); self.max_steps_input = QSpinBox(minimum=0, maximum=10000000, value=0); self.max_steps_input.setSpecialValueText("Unlimited"); self.use_amp_input = QCheckBox("Enable fp16 Mixed-precision"); self.use_amp_input.setChecked(True); self.loss_input = QComboBox(); self.loss_input.addItems(["l1", "l1+lpips", "bce+dice"]); self.lambda_lpips_input = QComboBox()
        self.lambda_presets = [
            ("Off (0.0) - Pure L1 pixel loss, no perceptual", 0.0),
            ("Light (0.05) - Mostly pixel accuracy, hint of perceptual", 0.05),
            ("Default (0.1) - Recommended balance for subtle work", 0.1),
            ("Medium (0.2) - More perceptual texture guidance", 0.2),
            ("High (0.3) - Strong perceptual push, still safe", 0.3),
            ("Risky (0.5) - LPIPS starts to dominate, watch for artifacts", 0.5),
        ]
        for label, _ in self.lambda_presets: self.lambda_lpips_input.addItem(label)
        self.lambda_lpips_input.setCurrentIndex(2)
        self.lambda_lpips_input.setEnabled(False); self.loss_input.currentTextChanged.connect(lambda t: self.lambda_lpips_input.setEnabled(t == "l1+lpips"))
        self.lr_input = QComboBox()
        self.lr_presets = [
            ("Slow (5e-5) - Preserving very fine texture, lots of data", 5e-5),
            ("Default (1e-4) - General purpose training", 1e-4),
            ("Medium (3e-4) - Moderate changes, medium datasets", 3e-4),
            ("Fast (5e-4) - Beauty work, subtle fixes, small datasets", 5e-4),
            ("Aggressive (1e-3) - Large obvious changes, quick experiments", 1e-3),
        ]
        for label, _ in self.lr_presets: self.lr_input.addItem(label)
        self.lr_input.setCurrentIndex(1)
        self.log_interval_input = QSpinBox(minimum=1, maximum=1000, value=5); self.preview_interval_input = QSpinBox(minimum=0, maximum=1000, value=35); self.preview_refresh_input = QSpinBox(minimum=0, maximum=1000, value=5); self.keep_checkpoints_input = QSpinBox(minimum=1, maximum=50, value=4); layout.addRow(QLabel("--- Training Settings ---")); layout.addRow("Iterations per Epoch:", self.iter_per_epoch_input); layout.addRow("Batch Size (per GPU):", self.batch_size_input); layout.addRow("Max Steps (0=unlimited):", self.max_steps_input); layout.addRow("Use AMP:", self.use_amp_input); layout.addRow("Loss:", self.loss_input); layout.addRow("LPIPS Lambda:", self.lambda_lpips_input); layout.addRow("Learning Rate:", self.lr_input); layout.addRow(QLabel("--- Logging & Saving Settings ---")); layout.addRow("Log Interval:", self.log_interval_input); layout.addRow("Preview Batch Interval:", self.preview_interval_input); layout.addRow("Preview Refresh Rate:", self.preview_refresh_input); layout.addRow("Keep Last Checkpoints:", self.keep_checkpoints_input)
        layout.addRow(QLabel("--- DataLoader Settings ---"))
        self.num_workers_input = QComboBox()
        self.num_workers_presets = [
            ("Auto (Recommended)", -1),
            ("0 - Disabled (single thread)", 0),
            ("2 - Light", 2),
            ("4 - Moderate", 4),
            ("8 - Heavy (high-end GPU + many CPU cores)", 8),
        ]
        for label, _ in self.num_workers_presets: self.num_workers_input.addItem(label)
        self.num_workers_input.setCurrentIndex(0)
        layout.addRow("DataLoader Workers:", self.num_workers_input)
        layout.addRow(QLabel("--- Mask Settings ---"))
        self.use_mask_loss_input = QCheckBox("Weight loss by mask (white=important)")
        layout.addRow("Mask Loss:", self.use_mask_loss_input)
        self.mask_weight_input = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=100.0, value=10.0, singleStep=1.0)
        layout.addRow("Mask Weight:", self.mask_weight_input)
        self.use_mask_input_input = QCheckBox("Feed mask as 4th input channel to model")
        layout.addRow("Mask Input:", self.use_mask_input_input)
        self.use_auto_mask_input = QCheckBox("Auto-generate masks from |src-dst| difference (no mask files needed)")
        layout.addRow("Auto Mask:", self.use_auto_mask_input)
        self.tabs.addTab(tab, "Advanced")

    def create_dataloader_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.addRow(QLabel("--- Shared Augmentations ---"))

        # --- Horizontal Flip ---
        self.hflip_check = QCheckBox("HorizontalFlip")
        self.hflip_p = QSpinBox(minimum=0, maximum=100, value=50); self.hflip_p.setSuffix("%")
        hflip_layout = QHBoxLayout(); hflip_layout.addWidget(self.hflip_check); hflip_layout.addWidget(self.hflip_p); layout.addRow(hflip_layout)

        layout.addRow(QLabel(" ")); # Spacer

        # --- Affine ---
        self.affine_check = QCheckBox("Affine")
        self.affine_p = QSpinBox(minimum=0, maximum=100, value=40); self.affine_p.setSuffix("%")
        affine_p_layout = QHBoxLayout(); affine_p_layout.addWidget(self.affine_check); affine_p_layout.addWidget(self.affine_p); layout.addRow(affine_p_layout)
        
        # Scale
        self.affine_scale_min = QDoubleSpinBox(decimals=2, minimum=0.1, maximum=5.0, value=0.9, singleStep=0.05)
        self.affine_scale_max = QDoubleSpinBox(decimals=2, minimum=0.1, maximum=5.0, value=1.1, singleStep=0.05)
        layout.addRow("  Scale:", self.create_range_layout(self.affine_scale_min, self.affine_scale_max))
        # Translate
        self.affine_translate_min = QDoubleSpinBox(decimals=2, minimum=-0.5, maximum=0.5, value=-0.1, singleStep=0.01)
        self.affine_translate_max = QDoubleSpinBox(decimals=2, minimum=-0.5, maximum=0.5, value=0.1, singleStep=0.01)
        layout.addRow("  Translate %:", self.create_range_layout(self.affine_translate_min, self.affine_translate_max))
        # Rotate
        self.affine_rotate_min = QSpinBox(minimum=-180, maximum=180, value=-3)
        self.affine_rotate_max = QSpinBox(minimum=-180, maximum=180, value=3)
        layout.addRow("  Rotate:", self.create_range_layout(self.affine_rotate_min, self.affine_rotate_max))
        # Shear
        self.affine_shear_min = QSpinBox(minimum=-45, maximum=45, value=-1)
        self.affine_shear_max = QSpinBox(minimum=-45, maximum=45, value=1)
        layout.addRow("  Shear:", self.create_range_layout(self.affine_shear_min, self.affine_shear_max))
        # Interpolation and Keep Ratio
        self.affine_interpolation = QSpinBox(minimum=0, maximum=5, value=2)
        self.affine_keep_ratio = QCheckBox(); self.affine_keep_ratio.setChecked(True)
        layout.addRow("  Interpolation:", self.affine_interpolation)
        layout.addRow("  Keep Ratio:", self.affine_keep_ratio)

        layout.addRow(QLabel(" ")); # Spacer

        # --- Random Gamma ---
        self.gamma_check = QCheckBox("RandomGamma")
        self.gamma_p = QSpinBox(minimum=0, maximum=100, value=20); self.gamma_p.setSuffix("%")
        gamma_p_layout = QHBoxLayout(); gamma_p_layout.addWidget(self.gamma_check); gamma_p_layout.addWidget(self.gamma_p); layout.addRow(gamma_p_layout)
        # Gamma Limit
        self.gamma_limit_min = QSpinBox(minimum=0, maximum=255, value=40)
        self.gamma_limit_max = QSpinBox(minimum=0, maximum=255, value=160)
        layout.addRow("  Gamma Limit:", self.create_range_layout(self.gamma_limit_min, self.gamma_limit_max))
        
        self.tabs.addTab(tab, "Dataloader")

    def create_range_layout(self, min_widget, max_widget):
        layout = QHBoxLayout()
        layout.addWidget(min_widget)
        layout.addWidget(QLabel("to"))
        layout.addWidget(max_widget)
        widget = QWidget(); widget.setLayout(layout)
        return widget
        
    def create_preview_tab(self):
        self.preview_tab = QWidget(); main_layout = QVBoxLayout(self.preview_tab); self.scroll_area = ZoomPanScrollArea(); self.scroll_area.setWidgetResizable(True); self.preview_label = QLabel("Waiting for preview image..."); self.preview_label.setAlignment(Qt.AlignCenter); self.scroll_area.setWidget(self.preview_label); controls_layout = QHBoxLayout(); self.zoom_combo = QComboBox(); self.zoom_combo.addItems(["Fit", "50%", "100%", "200%"]); self.zoom_combo.activated.connect(lambda: self._on_zoom_combo_changed(self.zoom_combo.currentText())); self.scroll_area.zoom_changed.connect(self._on_preview_wheel_zoom); controls_layout.addStretch(); controls_layout.addWidget(QLabel("Zoom:")); controls_layout.addWidget(self.zoom_combo); controls_layout.addStretch(); self.labels_container = QWidget(); labels_layout = QHBoxLayout(self.labels_container); labels_layout.setContentsMargins(0,0,0,0); src_label = QLabel("src data"); dst_label = QLabel("dst data"); model_label = QLabel("Model result"); src_label.setAlignment(Qt.AlignLeft); dst_label.setAlignment(Qt.AlignCenter); model_label.setAlignment(Qt.AlignRight); labels_layout.addWidget(src_label); labels_layout.addWidget(dst_label); labels_layout.addWidget(model_label); main_layout.addWidget(self.scroll_area, stretch=1); main_layout.addLayout(controls_layout); main_layout.addWidget(self.labels_container); self.tabs.addTab(self.preview_tab, "Preview")
    def create_val_preview_tab(self):
        self.val_preview_tab = QWidget(); main_layout = QVBoxLayout(self.val_preview_tab)
        self.val_scroll_area = ZoomPanScrollArea(); self.val_scroll_area.setWidgetResizable(True)
        self.val_preview_label = QLabel("Waiting for validation preview image..."); self.val_preview_label.setAlignment(Qt.AlignCenter)
        self.val_scroll_area.setWidget(self.val_preview_label)
        controls_layout = QHBoxLayout()
        self.val_zoom_combo = QComboBox(); self.val_zoom_combo.addItems(["Fit", "50%", "100%", "200%"])
        self.val_zoom_combo.activated.connect(lambda: self._on_val_zoom_combo_changed(self.val_zoom_combo.currentText())); self.val_scroll_area.zoom_changed.connect(self._on_val_preview_wheel_zoom)
        controls_layout.addStretch(); controls_layout.addWidget(QLabel("Zoom:")); controls_layout.addWidget(self.val_zoom_combo); controls_layout.addStretch()
        self.val_labels_container = QWidget(); labels_layout = QHBoxLayout(self.val_labels_container); labels_layout.setContentsMargins(0,0,0,0)
        src_label = QLabel("val src"); dst_label = QLabel("val dst"); model_label = QLabel("Model result")
        src_label.setAlignment(Qt.AlignLeft); dst_label.setAlignment(Qt.AlignCenter); model_label.setAlignment(Qt.AlignRight)
        labels_layout.addWidget(src_label); labels_layout.addWidget(dst_label); labels_layout.addWidget(model_label)
        main_layout.addWidget(self.val_scroll_area, stretch=1); main_layout.addLayout(controls_layout); main_layout.addWidget(self.val_labels_container)
        self.tabs.addTab(self.val_preview_tab, "Val Preview")
    def create_convert_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); layout.addStretch(); self.convert_flame_btn = QPushButton("Convert to Autodesk Flame / After Effects"); self.convert_flame_btn.clicked.connect(self.run_flame_conversion); self.convert_nuke_btn = QPushButton("Convert to Foundry Nuke"); self.convert_nuke_btn.clicked.connect(self.run_nuke_conversion); self.copy_before_convert_check = QCheckBox("Copy checkpoint to subfolder before converting"); self.copy_before_convert_check.setChecked(True); self.copy_before_convert_check.setStyleSheet("margin-top: 10px;"); layout.addWidget(self.convert_flame_btn); layout.addWidget(self.convert_nuke_btn); layout.addWidget(self.copy_before_convert_check, alignment=Qt.AlignCenter); layout.addStretch(); self.tabs.addTab(tab, "Convert")
    def create_about_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); layout.setAlignment(Qt.AlignCenter); title = QLabel("Tunet by tpo.comp"); title.setStyleSheet("font-size: 18px; font-weight: bold;"); desc_text = ("A direct, pixel-level mapping from src to dst images via an encoder-decoder network.\n" "Supports training, inference, and export to VFX tools."); description = QLabel(desc_text); description.setAlignment(Qt.AlignCenter); description.setWordWrap(True); support_text = "Native inference support in Autodesk Flame or Foundry Nuke."; support = QLabel(support_text); support.setAlignment(Qt.AlignCenter); source_title = QLabel("Source:"); source_link = QLabel('<a href="https://github.com/tpc2233/tunet">https://github.com/tpc2233/tunet</a>'); source_link.setOpenExternalLinks(True); source_link.setAlignment(Qt.AlignCenter); layout.addStretch(); layout.addWidget(title); layout.addSpacing(15); layout.addWidget(description); layout.addSpacing(10); layout.addWidget(support); layout.addSpacing(25); layout.addWidget(source_title); layout.addWidget(source_link); layout.addStretch(); self.tabs.addTab(tab, "About")
    def create_queue_section(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Training Queue"))
        self.queue_listbox = QListWidget(); self.queue_listbox.setMinimumWidth(200); self.queue_listbox.setMaximumWidth(280); self.queue_listbox.setStyleSheet("font-family: monospace; font-size: 10px;"); self.queue_listbox.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.queue_listbox, stretch=1)
        self.queue_add_btn = QPushButton("Add to Queue"); self.queue_add_btn.clicked.connect(self.add_to_queue)
        self.queue_remove_btn = QPushButton("Remove"); self.queue_remove_btn.clicked.connect(self.remove_from_queue)
        self.queue_clear_btn = QPushButton("Clear"); self.queue_clear_btn.clicked.connect(self.clear_queue)
        self.queue_run_btn = QPushButton("Run Queue"); self.queue_run_btn.clicked.connect(self.run_queue); self.queue_run_btn.setStyleSheet("background-color: #a8e6cf; color: #1a1a1a;")
        layout.addWidget(self.queue_add_btn); layout.addWidget(self.queue_remove_btn); layout.addWidget(self.queue_clear_btn); layout.addWidget(self.queue_run_btn)
        return layout

    def create_control_panel(self):
        control_layout = QHBoxLayout(); self.load_btn = QPushButton("Load Config"); self.save_btn = QPushButton("Save Config"); self.monitor_btn = QPushButton("Training Monitor"); self.start_btn = QPushButton("Start Training"); self.stop_btn = QPushButton("Stop Training"); self.load_btn.clicked.connect(self.load_config_from_file); self.save_btn.clicked.connect(self.save_config_to_file); self.monitor_btn.clicked.connect(self.launch_training_monitor); self.start_btn.clicked.connect(self.start_training); self.stop_btn.clicked.connect(self.stop_training); self.stop_btn.setEnabled(False); self.start_btn.setStyleSheet("background-color: #a8e6cf; color: #1a1a1a;"); self.stop_btn.setStyleSheet("background-color: #ff8a80; color: #1a1a1a;"); self.monitor_btn.setStyleSheet("background-color: #b3e5fc; color: #1a1a1a;"); control_layout.addWidget(self.load_btn); control_layout.addWidget(self.save_btn); control_layout.addWidget(self.monitor_btn); control_layout.addStretch(); control_layout.addWidget(self.start_btn); control_layout.addWidget(self.stop_btn); return control_layout

    def launch_training_monitor(self):
        """Launch the training monitor window as a separate process."""
        app_dir = Path(__file__).parent
        monitor_script = app_dir / "training_monitor.py"
        if not monitor_script.is_file():
            QMessageBox.warning(self, "Error", f"Training monitor script not found:\n{monitor_script}")
            return
        # Get the model folder to find training.log
        model_folder = self.model_folder_input.findChild(QLineEdit).text()
        command = [sys.executable, str(monitor_script)]
        if model_folder and Path(model_folder).is_dir():
            log_file = Path(model_folder) / "training.log"
            if log_file.exists():
                command.extend(['--log_file', str(log_file)])
            else:
                command.extend(['--output_dir', model_folder])
        try:
            subprocess.Popen(command)
            self.console_output.append(f"Launched Training Monitor\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch training monitor:\n{e}")

    def gather_config_from_ui(self):
        def get_path(widget): return widget.findChild(QLineEdit).text()
        augs = []
        if self.hflip_check.isChecked():
            augs.append({'_target_': 'albumentations.HorizontalFlip', 'p': self.hflip_p.value() / 100.0})
        
        if self.affine_check.isChecked():
            affine_dict = {
                '_target_': 'albumentations.Affine',
                'scale': [self.affine_scale_min.value(), self.affine_scale_max.value()],
                'translate_percent': [self.affine_translate_min.value(), self.affine_translate_max.value()],
                'rotate': [self.affine_rotate_min.value(), self.affine_rotate_max.value()],
                'shear': [self.affine_shear_min.value(), self.affine_shear_max.value()],
                'interpolation': self.affine_interpolation.value(),
                'keep_ratio': self.affine_keep_ratio.isChecked(),
                'p': self.affine_p.value() / 100.0
            }
            augs.append(affine_dict)

        if self.gamma_check.isChecked():
            gamma_dict = {
                '_target_': 'albumentations.RandomGamma',
                'gamma_limit': [self.gamma_limit_min.value(), self.gamma_limit_max.value()],
                'p': self.gamma_p.value() / 100.0
            }
            augs.append(gamma_dict)

        mask_dir = get_path(self.mask_dir_input)
        data_config = {'src_dir': get_path(self.src_dir_input), 'dst_dir': get_path(self.dst_dir_input), 'output_dir': get_path(self.model_folder_input), 'resolution': int(self.resolution_input.currentText())}
        if mask_dir:
            data_config['mask_dir'] = mask_dir
        val_src_dir = get_path(self.val_src_dir_input); val_dst_dir = get_path(self.val_dst_dir_input)
        if val_src_dir: data_config['val_src_dir'] = val_src_dir
        if val_dst_dir: data_config['val_dst_dir'] = val_dst_dir
        lr_value = self.lr_presets[self.lr_input.currentIndex()][1]
        lambda_value = self.lambda_presets[self.lambda_lpips_input.currentIndex()][1]
        config = {'data': data_config, 'model': {'model_size_dims': int(self.model_size_dims_input.currentText())}, 'training': {'iterations_per_epoch': self.iter_per_epoch_input.value(), 'batch_size': self.batch_size_input.value(), 'max_steps': self.max_steps_input.value(), 'use_amp': self.use_amp_input.isChecked(), 'loss': self.loss_input.currentText(), 'lambda_lpips': lambda_value, 'lr': lr_value}, 'logging': {'log_interval': self.log_interval_input.value(), 'preview_batch_interval': self.preview_interval_input.value(), 'preview_refresh_rate': self.preview_refresh_input.value()}, 'saving': {'keep_last_checkpoints': self.keep_checkpoints_input.value()}, 'dataloader': {'num_workers': self.num_workers_presets[self.num_workers_input.currentIndex()][1], 'datasets': {'shared_augs': augs}}}
        if self.use_mask_loss_input.isChecked() or self.use_mask_input_input.isChecked() or self.use_auto_mask_input.isChecked():
            config['mask'] = {'use_mask_loss': self.use_mask_loss_input.isChecked(), 'mask_weight': self.mask_weight_input.value(), 'use_mask_input': self.use_mask_input_input.isChecked(), 'use_auto_mask': self.use_auto_mask_input.isChecked()}
        config['_ui_settings'] = {'train_script_path': get_path(self.train_script_input), 'nproc_per_node': self.nproc_input.value() if self.nproc_input else 1}
        return config

    def populate_ui_from_config(self, config):
        def get_path_widget(widget): return widget.findChild(QLineEdit)
        ui_settings = config.get('_ui_settings', {}); get_path_widget(self.train_script_input).setText(ui_settings.get('train_script_path', ''))
        if self.nproc_input: self.nproc_input.setValue(ui_settings.get('nproc_per_node', 1))
        get_path_widget(self.src_dir_input).setText(config.get('data', {}).get('src_dir', '')); get_path_widget(self.dst_dir_input).setText(config.get('data', {}).get('dst_dir', '')); get_path_widget(self.mask_dir_input).setText(config.get('data', {}).get('mask_dir', '')); get_path_widget(self.model_folder_input).setText(config.get('data', {}).get('output_dir', '')); self.resolution_input.setCurrentText(str(config.get('data', {}).get('resolution', 512))); self.model_size_dims_input.setCurrentText(str(config.get('model', {}).get('model_size_dims', 128))); self.iter_per_epoch_input.setValue(config.get('training', {}).get('iterations_per_epoch', 500)); self.batch_size_input.setValue(config.get('training', {}).get('batch_size', 4)); self.max_steps_input.setValue(config.get('training', {}).get('max_steps', 0)); self.use_amp_input.setChecked(config.get('training', {}).get('use_amp', True)); self.loss_input.setCurrentText(config.get('training', {}).get('loss', 'l1'))
        # Restore lambda preset from config
        saved_lambda = config.get('training', {}).get('lambda_lpips', 1.0)
        best_lambda_idx = 3  # Default (1.0)
        for i, (_, val) in enumerate(self.lambda_presets):
            if abs(val - saved_lambda) < 1e-6: best_lambda_idx = i; break
        self.lambda_lpips_input.setCurrentIndex(best_lambda_idx)
        # Restore learning rate preset from config
        saved_lr = config.get('training', {}).get('lr', 1e-4)
        best_idx = 1  # Default
        for i, (_, val) in enumerate(self.lr_presets):
            if abs(val - saved_lr) < 1e-6: best_idx = i; break
        self.lr_input.setCurrentIndex(best_idx)
        self.log_interval_input.setValue(config.get('logging', {}).get('log_interval', 5)); self.preview_interval_input.setValue(config.get('logging', {}).get('preview_batch_interval', 35)); self.preview_refresh_input.setValue(config.get('logging', {}).get('preview_refresh_rate', 5)); self.keep_checkpoints_input.setValue(config.get('saving', {}).get('keep_last_checkpoints', 4))
        mask_config = config.get('mask', {}); self.use_mask_loss_input.setChecked(mask_config.get('use_mask_loss', False)); self.mask_weight_input.setValue(mask_config.get('mask_weight', 10.0)); self.use_mask_input_input.setChecked(mask_config.get('use_mask_input', False)); self.use_auto_mask_input.setChecked(mask_config.get('use_auto_mask', False))
        get_path_widget(self.val_src_dir_input).setText(config.get('data', {}).get('val_src_dir', '')); get_path_widget(self.val_dst_dir_input).setText(config.get('data', {}).get('val_dst_dir', ''))

        # Restore num_workers preset from config
        saved_workers = config.get('dataloader', {}).get('num_workers', -1)
        best_workers_idx = 0  # Default to Auto
        for i, (_, val) in enumerate(self.num_workers_presets):
            if val == saved_workers: best_workers_idx = i; break
        self.num_workers_input.setCurrentIndex(best_workers_idx)
        # Reset augs before populating
        self.hflip_check.setChecked(False); self.affine_check.setChecked(False); self.gamma_check.setChecked(False)
        augs = config.get('dataloader', {}).get('datasets', {}).get('shared_augs', [])
        for aug in augs:
            target = aug.get('_target_', '')
            if 'HorizontalFlip' in target:
                self.hflip_check.setChecked(True)
                self.hflip_p.setValue(int(aug.get('p', 0.5) * 100))
            elif 'Affine' in target:
                self.affine_check.setChecked(True)
                self.affine_p.setValue(int(aug.get('p', 0.4) * 100))
                scale = aug.get('scale', [0.9, 1.1]); self.affine_scale_min.setValue(scale[0]); self.affine_scale_max.setValue(scale[1])
                translate = aug.get('translate_percent', [-0.1, 0.1]); self.affine_translate_min.setValue(translate[0]); self.affine_translate_max.setValue(translate[1])
                rotate = aug.get('rotate', [-3, 3]); self.affine_rotate_min.setValue(rotate[0]); self.affine_rotate_max.setValue(rotate[1])
                shear = aug.get('shear', [-1, 1]); self.affine_shear_min.setValue(shear[0]); self.affine_shear_max.setValue(shear[1])
                self.affine_interpolation.setValue(aug.get('interpolation', 2))
                self.affine_keep_ratio.setChecked(aug.get('keep_ratio', True))
            elif 'RandomGamma' in target:
                self.gamma_check.setChecked(True)
                self.gamma_p.setValue(int(aug.get('p', 0.2) * 100))
                limit = aug.get('gamma_limit', [40, 160]); self.gamma_limit_min.setValue(limit[0]); self.gamma_limit_max.setValue(limit[1])

    def _launch_training(self, full_config):
        """Launch a training subprocess from a full config dict. Returns True if started."""
        train_script = full_config['_ui_settings']['train_script_path']
        if not train_script or not Path(train_script).is_file(): QMessageBox.warning(self, "Error", "Training script path is invalid."); return False
        model_folder = Path(full_config['data']['output_dir'])
        if not model_folder.is_dir(): QMessageBox.warning(self, "Error", f"Model Folder does not exist:\n{model_folder}"); return False
        self.setup_preview_watcher(str(model_folder))
        script_config = {k: v for k, v in full_config.items() if k != '_ui_settings'}
        self.config_file_path = Path(f"{model_folder.name}.yaml")
        with open(self.config_file_path, 'w') as f:
            yaml.dump(script_config, f, Dumper=IndentDumper, sort_keys=False, default_flow_style=False, indent=2)
        current_os = platform.system(); command = []
        if current_os == 'Linux':
            nproc = full_config['_ui_settings']['nproc_per_node']
            command = ['torchrun', '--standalone', '--nnodes=1', f'--nproc_per_node={nproc}', train_script, '--config', str(self.config_file_path)]
        elif current_os in ['Windows', 'Darwin']:
            command = [sys.executable, train_script, '--config', str(self.config_file_path)]
        else: QMessageBox.critical(self, "Unsupported OS", f"'{current_os}' is not supported."); return False
        self.console_output.append(f"OS Detected: {current_os}\nRunning command:\n{' '.join(command)}\n\n")
        creation_flags = 0
        if platform.system() == "Windows": creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=creation_flags, encoding='utf-8', errors='replace'
            )
        except Exception as e: QMessageBox.critical(self, "Process Error", f"Failed to start process: {e}"); return False
        self.stream_reader = ProcessStreamReader(self.process.stdout); self.stream_reader.new_text.connect(self.append_text)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.queue_run_btn.setEnabled(False); self.queue_add_btn.setEnabled(False)
        self.preview_timer.start(); self.process_monitor.start()
        return True

    def start_training(self):
        full_config = self.gather_config_from_ui()
        self.console_output.clear()
        self._launch_training(full_config)
    @Slot()
    def check_process_status(self):
        if not self.process: return
        if self.process.poll() is not None:
            self.process_monitor.stop(); self.on_training_finished()
    def on_training_finished(self):
        self.preview_timer.stop(); self.console_output.append("\n--- Training Process Terminated ---")
        self.process = None
        QTimer.singleShot(100, self.update_all_previews)
        if self.queue_running:
            # Mark current item as done
            for item in self.training_queue:
                if item['status'] == 'processing':
                    item['status'] = 'done'; break
            self.refresh_queue_display()
            if self.queue_stop_requested:
                self._finish_queue(); return
            # Run next pending item
            self._run_next_queue_item()
        else:
            self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.queue_run_btn.setEnabled(True); self.queue_add_btn.setEnabled(True)
    def stop_training(self):
        if self.process and self.process.poll() is None:
            self.console_output.append("\n--- Sending stop signal ---")
            if platform.system() == "Windows": self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else: self.process.send_signal(signal.SIGINT)
            if self.queue_running:
                self.queue_stop_requested = True
                self.console_output.append("Queue will stop after current item finishes.")
        else: self.console_output.append("Stop clicked, but no active process found.\n")

    # --- Training Queue ---
    def add_to_queue(self):
        full_config = self.gather_config_from_ui()
        label = os.path.basename(full_config['data'].get('output_dir', 'untitled'))
        max_steps = full_config.get('training', {}).get('max_steps', 0)
        if max_steps == 0:
            QMessageBox.warning(self, "Max Steps Required", "Set Max Steps > 0 for queue items so training knows when to stop.")
            return
        self.training_queue.append({'config': full_config, 'status': 'pending', 'label': label})
        self.refresh_queue_display()
        self.console_output.append(f"Added to queue: {label} (max_steps={max_steps})")

    def remove_from_queue(self):
        selected = sorted(self.queue_listbox.selectedIndexes(), key=lambda x: x.row(), reverse=True)
        for idx in selected:
            row = idx.row()
            if row < len(self.training_queue) and self.training_queue[row]['status'] != 'processing':
                self.training_queue.pop(row)
        self.refresh_queue_display()

    def clear_queue(self):
        self.training_queue = [q for q in self.training_queue if q['status'] == 'processing']
        self.refresh_queue_display()

    def refresh_queue_display(self):
        self.queue_listbox.clear()
        prefix_map = {'pending': '   ', 'processing': '>> ', 'done': 'OK ', 'error': '!! '}
        for item in self.training_queue:
            prefix = prefix_map.get(item['status'], '   ')
            max_s = item['config'].get('training', {}).get('max_steps', 0)
            text = f"{prefix}{item['label']}  (max_steps={max_s})"
            self.queue_listbox.addItem(text)
        # Color items
        for i, item in enumerate(self.training_queue):
            list_item = self.queue_listbox.item(i)
            if item['status'] == 'processing': list_item.setForeground(Qt.blue)
            elif item['status'] == 'done': list_item.setForeground(Qt.darkGreen)
            elif item['status'] == 'error': list_item.setForeground(Qt.red)

    def run_queue(self):
        if self.process: QMessageBox.warning(self, "Busy", "Training is already running."); return
        pending = [q for q in self.training_queue if q['status'] == 'pending']
        if not pending: QMessageBox.information(self, "Queue Empty", "No pending items in the queue."); return
        self.queue_running = True; self.queue_stop_requested = False
        self.console_output.clear()
        self.console_output.append(f"=== Starting Queue ({len(pending)} items) ===\n")
        self._run_next_queue_item()

    def _run_next_queue_item(self):
        pending = [q for q in self.training_queue if q['status'] == 'pending']
        if not pending:
            self._finish_queue(); return
        item = pending[0]; item['status'] = 'processing'
        self.refresh_queue_display()
        done_count = sum(1 for q in self.training_queue if q['status'] == 'done')
        total_count = sum(1 for q in self.training_queue if q['status'] in ('pending', 'processing', 'done'))
        self.console_output.append(f"\n=== Queue item {done_count + 1}/{total_count}: {item['label']} ===\n")
        if not self._launch_training(item['config']):
            item['status'] = 'error'; self.refresh_queue_display()
            self._run_next_queue_item()

    def _finish_queue(self):
        self.queue_running = False; self.queue_stop_requested = False
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.queue_run_btn.setEnabled(True); self.queue_add_btn.setEnabled(True)
        done_count = sum(1 for q in self.training_queue if q['status'] == 'done')
        total_count = len(self.training_queue)
        self.console_output.append(f"\n=== Queue finished ({done_count}/{total_count} completed) ===\n")
        self.refresh_queue_display()
    def run_flame_conversion(self): self.start_conversion_process('flame')
    def run_nuke_conversion(self): self.start_conversion_process('nuke')
    def start_conversion_process(self, target_type):
        model_folder_str = self.model_folder_input.findChild(QLineEdit).text()
        if not model_folder_str or not Path(model_folder_str).is_dir():
            QMessageBox.warning(self, "Error", "Model Folder is not set or does not exist."); return
        source_path_str, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint File", model_folder_str, "PyTorch Checkpoints (*.pth)")
        if not source_path_str: return
        if self.copy_before_convert_check.isChecked():
            self.conversion_target = target_type
            source_path = Path(source_path_str); dest_folder = Path(model_folder_str) / self.conversion_target
            dest_folder.mkdir(parents=True, exist_ok=True); new_checkpoint_path = dest_folder / source_path.name
            self.progress_dialog = QProgressDialog(f"Copying to {self.conversion_target} folder...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowTitle("Copying Checkpoint"); self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.copy_thread = QThread(); self.copy_worker = FileCopyWorker(); self.copy_worker.moveToThread(self.copy_thread)
            self.progress_dialog.canceled.connect(self.copy_worker.cancel); self.copy_worker.progress.connect(self.progress_dialog.setValue)
            self.copy_worker.finished.connect(self.on_copy_finished); self.copy_worker.error.connect(self.on_copy_error)
            self.copy_thread.started.connect(lambda: self.copy_worker.run(source_path_str, str(new_checkpoint_path)))
            self.copy_thread.start(); self.progress_dialog.exec()
        else:
            self.console_output.append(f"Converting checkpoint directly: {source_path_str}\n")
            command = self.build_utility_command(target_type, source_path_str)
            if command: self.run_utility_script(command)
    @Slot(str)
    def on_copy_finished(self, new_checkpoint_path):
        self.progress_dialog.close(); self.console_output.append(f"Checkpoint copied successfully to: {new_checkpoint_path}\n")
        command = self.build_utility_command(self.conversion_target, new_checkpoint_path)
        if command: self.run_utility_script(command)
        self.copy_thread.quit(); self.copy_thread.wait()
    def build_utility_command(self, target_type, checkpoint_path):
        app_dir = Path(__file__).parent
        if target_type == 'flame':
            script_path = app_dir / "utils" / "convert_flame.py"
            return [sys.executable, str(script_path), '--checkpoint', checkpoint_path, '--use_gpu']
        elif target_type == 'nuke':
            script_path = app_dir / "utils" / "convert_nuke.py"
            return [sys.executable, str(script_path), '--generate_nk', '--checkpoint_pth', checkpoint_path, '--method', 'script']
        return None
    @Slot(str)
    def on_copy_error(self, error_message):
        self.progress_dialog.close(); QMessageBox.critical(self, "File Copy Error", error_message)
        self.copy_thread.quit(); self.copy_thread.wait()
    def run_utility_script(self, command):
        if self.utility_process: QMessageBox.warning(self, "Busy", "Another utility script is already running."); return
        self.console_output.clear(); self.console_output.append(f"Running command:\n{' '.join(command)}\n\n")
        try:
            self.utility_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace'
            )
        except Exception as e: QMessageBox.critical(self, "Process Error", f"Failed to start utility script: {e}"); return
        self.utility_stream_reader = ProcessStreamReader(self.utility_process.stdout)
        self.utility_stream_reader.new_text.connect(self.append_text)
        self.convert_flame_btn.setEnabled(False); self.convert_nuke_btn.setEnabled(False)
        self.utility_monitor.start()
    @Slot()
    def check_utility_status(self):
        if not self.utility_process: return
        if self.utility_process.poll() is not None:
            self.utility_monitor.stop(); self.utility_process = None
            self.console_output.append("\n--- Utility script finished. ---")
            self.convert_flame_btn.setEnabled(True); self.convert_nuke_btn.setEnabled(True)
    @Slot(str)
    def setup_preview_watcher(self, model_folder):
        if self.preview_image_path and self.file_watcher.files(): self.file_watcher.removePath(self.preview_image_path)
        if self.val_preview_image_path and self.val_preview_image_path in self.file_watcher.files(): self.file_watcher.removePath(self.val_preview_image_path)
        if model_folder and Path(model_folder).is_dir():
            self.preview_image_path = str(Path(model_folder) / "training_preview.jpg")
            self.val_preview_image_path = str(Path(model_folder) / "val_preview.jpg")
            self.file_watcher.addPath(self.preview_image_path)
            self.file_watcher.addPath(self.val_preview_image_path)
            self.update_preview_image(); self.update_val_preview_image()
    def update_all_previews(self):
        self.update_preview_image(); self.update_val_preview_image()
    @Slot(str)
    def on_watched_file_changed(self, path):
        if self.preview_image_path and path == self.preview_image_path: self.update_preview_image()
        elif self.val_preview_image_path and path == self.val_preview_image_path: self.update_val_preview_image()
    @Slot()
    def update_preview_image(self):
        if self.preview_image_path and Path(self.preview_image_path).exists():
            try:
                with open(self.preview_image_path, 'rb') as f: image_data = f.read()
                pixmap = QPixmap(); pixmap.loadFromData(image_data)
                if not pixmap.isNull(): self.original_pixmap = pixmap; self.apply_zoom()
                else: self.preview_label.setText("Loading preview...")
            except Exception: pass
        else: self.original_pixmap = None; self.apply_zoom()
    @Slot()
    def apply_zoom(self):
        if not self.original_pixmap:
            self.preview_label.setText("Waiting for preview image..."); self.labels_container.setVisible(False)
            return
        self.labels_container.setVisible(True)
        if self._preview_zoom_factor is None:
            scaled_pixmap = self.original_pixmap.scaled(self.scroll_area.viewport().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            new_width = int(self.original_pixmap.width() * self._preview_zoom_factor); new_height = int(self.original_pixmap.height() * self._preview_zoom_factor)
            scaled_pixmap = self.original_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pixmap); self.preview_label.adjustSize()
        image_width = scaled_pixmap.width(); container_width = self.scroll_area.viewport().width()
        margin = max(0, (container_width - image_width) // 2)
        self.labels_container.setContentsMargins(margin, 0, margin, 0)
    @Slot()
    def update_val_preview_image(self):
        if self.val_preview_image_path and Path(self.val_preview_image_path).exists():
            try:
                with open(self.val_preview_image_path, 'rb') as f: image_data = f.read()
                pixmap = QPixmap(); pixmap.loadFromData(image_data)
                if not pixmap.isNull(): self.val_original_pixmap = pixmap; self.apply_val_zoom()
                else: self.val_preview_label.setText("Loading val preview...")
            except Exception: pass
        else: self.val_original_pixmap = None; self.apply_val_zoom()
    @Slot()
    def apply_val_zoom(self):
        if not self.val_original_pixmap:
            self.val_preview_label.setText("Waiting for validation preview image..."); self.val_labels_container.setVisible(False)
            return
        self.val_labels_container.setVisible(True)
        if self._val_preview_zoom_factor is None:
            scaled_pixmap = self.val_original_pixmap.scaled(self.val_scroll_area.viewport().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            new_width = int(self.val_original_pixmap.width() * self._val_preview_zoom_factor); new_height = int(self.val_original_pixmap.height() * self._val_preview_zoom_factor)
            scaled_pixmap = self.val_original_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.val_preview_label.setPixmap(scaled_pixmap); self.val_preview_label.adjustSize()
        image_width = scaled_pixmap.width(); container_width = self.val_scroll_area.viewport().width()
        margin = max(0, (container_width - image_width) // 2)
        self.val_labels_container.setContentsMargins(margin, 0, margin, 0)
    def _on_zoom_combo_changed(self, text):
        if text == "Fit": self._preview_zoom_factor = None
        else: self._preview_zoom_factor = float(text.replace('%', '')) / 100.0
        self.apply_zoom()
    def _on_val_zoom_combo_changed(self, text):
        if text == "Fit": self._val_preview_zoom_factor = None
        else: self._val_preview_zoom_factor = float(text.replace('%', '')) / 100.0
        self.apply_val_zoom()
    def _on_preview_wheel_zoom(self, factor):
        if self._preview_zoom_factor is None:
            if self.original_pixmap and not self.original_pixmap.isNull():
                vp = self.scroll_area.viewport().size()
                self._preview_zoom_factor = min(vp.width() / self.original_pixmap.width(), vp.height() / self.original_pixmap.height())
            else: return
        self._preview_zoom_factor = max(0.05, min(10.0, self._preview_zoom_factor * factor))
        self.apply_zoom()
    def _on_val_preview_wheel_zoom(self, factor):
        if self._val_preview_zoom_factor is None:
            if self.val_original_pixmap and not self.val_original_pixmap.isNull():
                vp = self.val_scroll_area.viewport().size()
                self._val_preview_zoom_factor = min(vp.width() / self.val_original_pixmap.width(), vp.height() / self.val_original_pixmap.height())
            else: return
        self._val_preview_zoom_factor = max(0.05, min(10.0, self._val_preview_zoom_factor * factor))
        self.apply_val_zoom()
    @Slot(int)
    def on_tab_changed(self, index):
        tab_name = self.tabs.tabText(index)
        if tab_name == "Preview": self.update_preview_image()
        elif tab_name == "Val Preview": self.update_val_preview_image()
    def resizeEvent(self, event):
        super().resizeEvent(event); self.apply_zoom(); self.apply_val_zoom()
    def save_config_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config As", self._last_config_dir, "YAML Files (*.yaml *.yml)")
        if path:
            self._last_config_dir = str(Path(path).parent)
            config = self.gather_config_from_ui()
            with open(path, 'w') as f:
                yaml.dump(config, f, Dumper=IndentDumper, sort_keys=False, default_flow_style=False, indent=2)
            self.console_output.append(f"Config saved to: {path}")
    def load_config_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", self._last_config_dir, "YAML Files (*.yaml *.yml)")
        if path:
            self._last_config_dir = str(Path(path).parent)
            try:
                with open(path, 'r') as f: config = yaml.safe_load(f)
                self.populate_ui_from_config(config); self.console_output.append(f"Config loaded from: {path}")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load or parse config file:\n{e}")
    @Slot()
    def append_text(self, text):
        self.console_output.moveCursor(QTextCursor.End)
        self.console_output.insertPlainText(text)
    def closeEvent(self, event):
        self.preview_timer.stop(); self.process_monitor.stop()
        if self.process and self.process.poll() is None:
            reply = QMessageBox.question(self, 'Exit Confirmation',
                                           "A training process is still running. Are you sure you want to exit? The process will be terminated.",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_training()
                try: self.process.wait(timeout=2)
                except subprocess.TimeoutExpired: self.process.terminate()
                event.accept()
            else: event.ignore()
        else: event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
