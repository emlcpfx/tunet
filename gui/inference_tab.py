from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QLabel, QSpinBox, QCheckBox, QPushButton, QProgressBar,
)


class InferenceTabMixin:
    """Mixin that creates the Inference tab (Tab 4)."""

    def _create_inference_tab(self):
        """Tab 5: Inference -- apply a trained model to new images."""
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
        self.inf_run_btn.setProperty("cssClass", "start")
        self.inf_run_btn.clicked.connect(self._inf_run_single)
        self.inf_run_queue_btn = QPushButton("Run Queue")
        self.inf_run_queue_btn.setProperty("cssClass", "start")
        self.inf_run_queue_btn.clicked.connect(self._inf_run_queue)
        self.inf_stop_btn = QPushButton("Stop")
        self.inf_stop_btn.setProperty("cssClass", "stop")
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
