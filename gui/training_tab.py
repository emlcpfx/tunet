from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
)


class TrainingTabMixin:
    """Mixin that creates the Training tab (Tab 1) and the _apply_preset method."""

    def _apply_preset(self, preset_name):
        """Apply a training preset, adjusting relevant settings."""
        if preset_name == "Custom":
            return
        presets = {
            "Beauty / Paint Fix": {
                "model_type": "msrn",
                "model_size_dims": "64",
                "resolution": "512",
                "overlap_factor": "0.5",
                "loss": "l1+lpips",
                "lambda_lpips": 0.2,
                "lr": 1e-4,
                "use_auto_mask": True,
                "auto_mask_gamma": 0.5,
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
        self.auto_mask_gamma_input.setValue(p.get("auto_mask_gamma", 1.0))
        self.skip_empty_patches_input.setChecked(p["skip_empty_patches"])
        self.progressive_res_check.setChecked(p.get("progressive_resolution", False))

    def _create_training_tab(self):
        """Tab 2: Training -- model, optimization, schedule, logging."""
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
            ("Slow (5e-5) \u2014 Fine texture, lots of data", 5e-5),
            ("Default (1e-4) \u2014 General purpose", 1e-4),
            ("Medium (3e-4) \u2014 Moderate changes", 3e-4),
            ("Fast (5e-4) \u2014 Beauty work, subtle fixes", 5e-4),
            ("Aggressive (1e-3) \u2014 Large changes, quick experiments", 1e-3),
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
            ("Off (0.0) \u2014 Pure L1 pixel loss", 0.0),
            ("Light (0.05) \u2014 Mostly pixel accuracy", 0.05),
            ("Default (0.1) \u2014 Recommended balance", 0.1),
            ("Medium (0.2) \u2014 More perceptual texture", 0.2),
            ("High (0.3) \u2014 Strong perceptual push", 0.3),
            ("Risky (0.5) \u2014 LPIPS dominates, watch for artifacts", 0.5),
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
            "Penalizes large errors more than small ones \u2014 smoother results.\n"
            "Try 0.1\u20130.5 alongside L1 for a blend of sharp + smooth.")
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
            "Only model weights are loaded \u2014 optimizer and step counter start fresh.\n"
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
            "  Epoch 1 \u2192 trains at 1/4 resolution (fast, learns shapes & layout)\n"
            "  Epoch 2 \u2192 trains at 1/2 resolution (medium detail)\n"
            "  Epoch 3+ \u2192 trains at full resolution (fine detail)\n\n"
            "Best for: large datasets, high resolutions (512+), long training runs.\n"
            "Skip when: resolution is already small (256), very short runs, or fine-tuning.")
        form_sched.addRow(self.progressive_res_check)

        self.num_workers_input = QComboBox()
        self.num_workers_presets = [
            ("Auto (Recommended)", -1),
            ("0 \u2014 Disabled (debug)", 0),
            ("2 \u2014 Light", 2),
            ("4 \u2014 Moderate", 4),
            ("8 \u2014 Heavy (many CPU cores)", 8),
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

        # --- Early Stopping / Plateau Detection ---
        grp_es = QGroupBox("Plateau Detection")
        form_es = QFormLayout(grp_es)

        self.es_enabled_input = QCheckBox("Enable Plateau Detection")
        self.es_enabled_input.setChecked(True)
        self.es_enabled_input.setToolTip(
            "Saves a _plateau.pth checkpoint when loss stops improving.\n"
            "Useful as a safety net for overnight runs.")
        form_es.addRow(self.es_enabled_input)

        self.es_patience_input = QSpinBox(minimum=5, maximum=200, value=30)
        self.es_patience_input.setToolTip(
            "Epochs with no improvement before saving a plateau checkpoint.")
        form_es.addRow("Patience (epochs):", self.es_patience_input)

        self.es_stop_input = QCheckBox("Stop training on plateau")
        self.es_stop_input.setChecked(False)
        self.es_stop_input.setToolTip(
            "Actually stop training when plateau is detected.\n"
            "Off = just save checkpoint and keep going (recommended).")
        form_es.addRow(self.es_stop_input)

        layout.addWidget(grp_es)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        self.tabs.addTab(scroll, "Training")
