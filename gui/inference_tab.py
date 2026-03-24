from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QLabel, QSpinBox, QCheckBox, QPushButton, QProgressBar,
)

from .widgets import CollapsibleGroupBox


class InferenceTabMixin:
    """Mixin that creates the Inference tab (Tab 4)."""

    def _create_inference_tab(self):
        """Tab 5: Inference -- apply a trained model to new images."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # =================================================================
        # MODEL — always visible
        # =================================================================
        grp_model = QGroupBox("Model")
        form_model = QFormLayout(grp_model)
        desc_model = QLabel("Select a trained checkpoint. Architecture and settings are auto-detected from the file.")
        desc_model.setWordWrap(True)
        desc_model.setProperty("cssClass", "section-desc")
        form_model.addRow(desc_model)

        self.inf_checkpoint_input = self._create_path_selector("Checkpoint", is_file=True, file_filter="PyTorch Checkpoints (*.pth)")
        self.inf_checkpoint_input.setToolTip(
            "Path to a trained .pth checkpoint file.\n"
            "The model type, capacity, and resolution are read from the checkpoint metadata.")
        form_model.addRow("Checkpoint:", self.inf_checkpoint_input)

        self.inf_use_latest_btn = QPushButton("Use Latest from Training Folder")
        self.inf_use_latest_btn.setToolTip(
            "Automatically find the most recent .pth checkpoint\n"
            "from the Output Folder set in the Training tab.")
        self.inf_use_latest_btn.clicked.connect(self._inf_use_latest_checkpoint)
        form_model.addRow("", self.inf_use_latest_btn)
        layout.addWidget(grp_model)

        # =================================================================
        # INPUT / OUTPUT — always visible
        # =================================================================
        grp_io = QGroupBox("Input / Output")
        form_io = QFormLayout(grp_io)

        self.inf_input_dir = self._create_path_selector("Input Directory")
        self.inf_input_dir.setToolTip(
            "Folder of images to process.\n"
            "Supports: PNG, JPG, TIFF, EXR, BMP, WebP")
        form_io.addRow("Input Directory:", self.inf_input_dir)

        self.inf_output_root = self._create_path_selector("Output Root Directory")
        self.inf_output_root.setToolTip(
            "Root folder for output. Each run creates a versioned subfolder:\n"
            "  inputname_modelname_v001 (auto-increments)")
        form_io.addRow("Output Root:", self.inf_output_root)
        layout.addWidget(grp_io)

        # =================================================================
        # PROCESSING OPTIONS — collapsible, good defaults
        # =================================================================
        grp_opts = CollapsibleGroupBox(
            "Processing Options",
            description="Tile stride and quality settings. Defaults work well for most images.",
            collapsed=True)
        form_opts = QFormLayout()

        stride_row = QHBoxLayout()
        self.inf_stride = QSpinBox(minimum=64, maximum=1024, value=256, singleStep=64)
        self.inf_stride.setToolTip(
            "Step size in pixels between overlapping tiles.\n"
            "Smaller = better blending at tile boundaries but slower.\n"
            "256 is a good default for most images.")
        self.inf_auto_stride = QCheckBox("Auto")
        self.inf_auto_stride.setChecked(True)
        self.inf_auto_stride.setToolTip(
            "Compute optimal stride from the first image's dimensions\n"
            "to ensure complete pixel coverage with no gaps.")
        self.inf_stride.setEnabled(False)
        self.inf_auto_stride.toggled.connect(lambda c: self.inf_stride.setEnabled(not c))
        stride_row.addWidget(self.inf_stride)
        stride_row.addWidget(self.inf_auto_stride)
        form_opts.addRow("Stride:", stride_row)

        self.inf_half_res = QCheckBox("Half Resolution (~4× faster)")
        self.inf_half_res.setToolTip(
            "Process at half resolution for quick preview checks.\n"
            "Useful for verifying a model before committing to a full-res run.")
        form_opts.addRow(self.inf_half_res)

        self.inf_skip_existing = QCheckBox("Skip Existing Files")
        self.inf_skip_existing.setChecked(True)
        self.inf_skip_existing.setToolTip(
            "Skip files that already exist in the output folder.\n"
            "Lets you resume a partial run without re-processing.")
        form_opts.addRow(self.inf_skip_existing)
        grp_opts.setContentLayout(form_opts)
        layout.addWidget(grp_opts)

        # =================================================================
        # PROGRESS — always visible
        # =================================================================
        grp_progress = QGroupBox("Progress")
        progress_layout = QVBoxLayout(grp_progress)
        self.inf_progress_bar = QProgressBar()
        self.inf_progress_bar.setValue(0)
        self.inf_progress_label = QLabel("")
        progress_layout.addWidget(self.inf_progress_bar)
        progress_layout.addWidget(self.inf_progress_label)
        layout.addWidget(grp_progress)

        # =================================================================
        # ACTION BUTTONS
        # =================================================================
        btn_layout = QHBoxLayout()
        self.inf_run_btn = QPushButton("Run Inference")
        self.inf_run_btn.setProperty("cssClass", "start")
        self.inf_run_btn.setToolTip("Process all images in the Input Directory with the selected model.")
        self.inf_run_btn.clicked.connect(self._inf_run_single)
        self.inf_run_queue_btn = QPushButton("Run Queue")
        self.inf_run_queue_btn.setProperty("cssClass", "start")
        self.inf_run_queue_btn.setToolTip("Run all jobs in the Inference Queue (see sidebar).")
        self.inf_run_queue_btn.clicked.connect(self._inf_run_queue)
        self.inf_stop_btn = QPushButton("Stop")
        self.inf_stop_btn.setProperty("cssClass", "stop")
        self.inf_stop_btn.setEnabled(False)
        self.inf_stop_btn.setToolTip("Stop the current inference run after the current image finishes.")
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
