from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
)

from .widgets import CollapsibleGroupBox


class TrainingTabMixin:
    """Mixin that creates the Training tab (Tab 1) and the _apply_preset method."""

    def _apply_preset(self, preset_name):
        """Apply a training preset, adjusting relevant settings."""
        if preset_name == "Custom":
            return
        presets = {
            "General (Image-to-Image)": {
                "model_type": "msrn",
                "model_size_dims": "64",
                "resolution": "512",
                "overlap_factor": "0.5",
                "loss": "l1",
                "lambda_lpips": 0.0,
                "lr": 1e-4,
                "use_auto_mask": False,
                "skip_empty_patches": False,
                "progressive_resolution": False,
            },
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
            "Paintout / Cleanup": {
                "model_type": "msrn",
                "model_size_dims": "64",
                "resolution": "512",
                "overlap_factor": "0.75",
                "loss": "weighted",
                "lambda_lpips": 0.0,
                "l1_weight": 0.5,
                "l2_weight": 0.5,
                "lpips_weight": 0.0,
                "lr": 3e-4,
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
        self._set_combo_by_prefix(self.resolution_input, p["resolution"])
        self._set_combo_by_prefix(self.overlap_factor_input, p["overlap_factor"])
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
        # Set weighted loss sliders if preset specifies them
        if "l1_weight" in p:
            self.l1_weight_input.setValue(p["l1_weight"])
        if "l2_weight" in p:
            self.l2_weight_input.setValue(p["l2_weight"])
        if "lpips_weight" in p:
            self.lpips_weight_input.setValue(p["lpips_weight"])
        self.use_auto_mask_input.setChecked(p["use_auto_mask"])
        self.auto_mask_gamma_input.setValue(p.get("auto_mask_gamma", 1.0))
        self.skip_empty_patches_input.setChecked(p["skip_empty_patches"])
        self.progressive_res_check.setChecked(p.get("progressive_resolution", False))

    def _create_training_tab(self):
        """Training tab — project folder, model, optimization, schedule, logging."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # =================================================================
        # PROJECT FOLDER + DATA — at the very top
        # =================================================================
        self._create_data_widgets(layout)

        # =================================================================
        # PRESET — always visible
        # =================================================================
        grp_preset = QGroupBox("Preset")
        form_preset = QFormLayout(grp_preset)
        desc_preset = QLabel("Choose a starting point for your task, then tweak individual settings below.")
        desc_preset.setWordWrap(True)
        desc_preset.setProperty("cssClass", "section-desc")
        form_preset.addRow(desc_preset)
        self.preset_input = QComboBox()
        self.preset_input.addItems(["Custom", "General (Image-to-Image)", "Beauty / Paint Fix", "Paintout / Cleanup", "Roto / Matte"])
        self.preset_input.setToolTip(
            "Quick-start presets that configure model, loss, and patch settings.\n\n"
            "  General — Good all-around starting point for image-to-image tasks\n"
            "  Beauty / Paint Fix — Perceptual loss for faces, skin, hair\n"
            "  Paintout / Cleanup — Weighted loss for removing objects\n"
            "  Roto / Matte — Binary mask output for rotoscoping\n\n"
            "Select a preset then adjust individual settings as needed.\n"
            "Changing any setting afterwards keeps your changes (won't revert).")
        self.preset_input.currentTextChanged.connect(self._apply_preset)
        form_preset.addRow("Training Preset:", self.preset_input)
        layout.addWidget(grp_preset)

        # Hidden: model folder is auto-set by project folder on Data tab
        self.model_folder_input = self._create_path_selector("Model Folder", is_output=True)
        self.model_folder_input.setVisible(False)
        layout.addWidget(self.model_folder_input)

        # =================================================================
        # MODEL — always visible, essential
        # =================================================================
        grp_model = QGroupBox("Model")
        form_model = QFormLayout(grp_model)
        self.model_type_input = QComboBox()
        self.model_type_input.addItems(["unet", "msrn"])
        self.model_type_input.setCurrentText("unet")
        self.model_type_input.setToolTip(
            "Architecture type:\n"
            "  'unet' — Fast, general-purpose encoder-decoder\n"
            "  'msrn' — Attention + recurrence for finer detail, trains slower\n\n"
            "Start with 'unet' unless you need maximum quality on fine textures.")
        self.model_size_dims_input = QComboBox()
        self.model_size_dims_input.addItems(["32", "64", "128", "256", "512"])
        self.model_size_dims_input.setCurrentText("128")
        self.model_size_dims_input.setToolTip(
            "Hidden layer width (model capacity):\n"
            "  64 — Lightweight, fast training\n"
            "  128 — Balanced (recommended)\n"
            "  256+ — High capacity for complex transformations\n\n"
            "Larger = more GPU memory. Double the size ≈ 4× the VRAM.")
        self.finetune_from_input = self._create_path_selector(
            "Fine-tune From", is_file=True, file_filter="PyTorch Checkpoints (*.pth)")
        self.finetune_from_input.setToolTip(
            "Optional: pick an existing .pth to start from instead of training from scratch.\n"
            "Only model weights are loaded — optimizer resets.\n"
            "Leave empty to train from scratch.")

        form_model.addRow("Model Type:", self.model_type_input)
        form_model.addRow("Model Capacity:", self.model_size_dims_input)
        layout.addWidget(grp_model)

        # =================================================================
        # FINE-TUNE — prominent standalone section
        # =================================================================
        grp_finetune = QGroupBox("Fine-tune")
        form_finetune = QFormLayout(grp_finetune)
        form_finetune.addRow("Resume From:", self.finetune_from_input)
        layout.addWidget(grp_finetune)

        # =================================================================
        # OPTIMIZATION — always visible, core settings
        # =================================================================
        grp_opt = QGroupBox("Optimization")
        form_opt = QFormLayout(grp_opt)

        self.lr_input = QComboBox()
        self.lr_presets = [
            ("Slow (5e-5) \u2014 Careful, stable, needs more steps", 5e-5),
            ("Default (1e-4) \u2014 Good starting point", 1e-4),
            ("Medium (3e-4) \u2014 Faster convergence", 3e-4),
            ("Fast (5e-4) \u2014 Quick results, may overshoot", 5e-4),
            ("Aggressive (1e-3) \u2014 Rapid experiments, watch for artifacts", 1e-3),
        ]
        for label, _ in self.lr_presets:
            self.lr_input.addItem(label)
        self.lr_input.setCurrentIndex(1)
        self.lr_input.setToolTip(
            "How fast the model updates its weights each step.\n\n"
            "Start with Default (1e-4) — it works for most tasks including beauty.\n"
            "Go slower (5e-5) if you have lots of data and want stability.\n"
            "Go faster (3e-4+) to iterate quickly, but watch for artifacts.\n\n"
            "If training loss jumps around or diverges, try a slower rate.")
        form_opt.addRow("Learning Rate:", self.lr_input)

        self.loss_input = QComboBox()
        self.loss_input.addItems(["l1", "l2", "l1+lpips", "weighted", "bce+dice"])
        self.loss_input.setToolTip(
            "How the model measures its errors:\n\n"
            "  'l1' — Pixel difference. Sharp, stable. Best all-around choice.\n"
            "  'l2' — Squared difference. Smoother results, can blur detail.\n"
            "  'l1+lpips' — Pixel + perceptual. Best for faces, skin, hair.\n"
            "  'weighted' — Custom L1+L2+LPIPS mix (advanced).\n"
            "  'bce+dice' — For mask/matte output only (black & white targets).")
        form_opt.addRow("Loss Function:", self.loss_input)

        self.lambda_lpips_input = QComboBox()
        self.lambda_presets = [
            ("Off (0.0) \u2014 Pixel-exact only", 0.0),
            ("Light (0.05) \u2014 Slight detail preservation", 0.05),
            ("Default (0.1) \u2014 Good balance", 0.1),
            ("Medium (0.2) \u2014 Prioritizes natural-looking results", 0.2),
            ("High (0.3) \u2014 Looks good but less pixel-accurate", 0.3),
            ("Risky (0.5) \u2014 May hallucinate detail, watch closely", 0.5),
        ]
        for label, _ in self.lambda_presets:
            self.lambda_lpips_input.addItem(label)
        self.lambda_lpips_input.setCurrentIndex(2)
        self.lambda_lpips_input.setToolTip(
            "How much the model cares about 'looking right' vs exact pixel values.\n"
            "Only active when loss is 'l1+lpips'.\n\n"
            "Low values = output matches target pixels closely but may look flat.\n"
            "High values = output looks more natural but may invent detail.\n\n"
            "0.1 is a safe starting point. Go higher for skin/hair, lower for cleanup.")
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
            "Penalizes large errors more — smoother results.\n"
            "Try 0.1–0.5 alongside L1 for a blend of sharp + smooth.")
        self.lpips_weight_input = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=10.0, value=0.1, singleStep=0.05)
        self.lpips_weight_input.setToolTip(
            "Weight for LPIPS perceptual loss.\n"
            "Matches structures/textures rather than raw pixels.\n"
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
            "Trains ~2× faster with less GPU memory. Almost no quality impact.\n"
            "Disable only if you see NaN losses (very rare).")
        form_opt.addRow("Mixed Precision:", self.use_amp_input)

        # --- LR Scheduler ---
        self.lr_scheduler_input = QComboBox()
        self.lr_scheduler_input.addItems(["none", "cosine", "plateau"])
        self.lr_scheduler_input.setToolTip(
            "Controls how the learning rate changes over time:\n\n"
            "  'none' — Constant learning rate (simplest, good default)\n"
            "  'cosine' — Decays then resets each cycle (good for long runs)\n"
            "  'plateau' — Halves LR when loss stops improving (adaptive)")
        form_opt.addRow("LR Scheduler:", self.lr_scheduler_input)

        layout.addWidget(grp_opt)

        # =================================================================
        # PATCH EXTRACTION — collapsible, preset-driven defaults
        # =================================================================
        grp_patch = CollapsibleGroupBox(
            "Patch Extraction",
            description="Images are cut into overlapping tiles for training. "
                        "Presets set these automatically — adjust only if needed.",
            collapsed=False)
        form_patch = QFormLayout()

        self.resolution_input = QComboBox()
        self.resolution_input.setEditable(True)
        self.resolution_input.addItems([
            "256 \u2014 Fast, local fixes",
            "384",
            "512 \u2014 Good default",
            "640",
            "768 \u2014 More context, more VRAM",
            "896",
            "928",
            "960",
            "1024 \u2014 Maximum context",
        ])
        self.resolution_input.setCurrentText("512 \u2014 Good default")
        self.resolution_input.setToolTip(
            "How big of a crop the model sees during each training step.\n\n"
            "Think of it as a magnifying glass sliding over your image:\n"
            "  256 — Small window. Fast, low VRAM. Fine for simple local fixes.\n"
            "  512 — Good default. Sees enough context for most VFX tasks.\n"
            "  768+ — Large window. Better for tasks that need spatial awareness\n"
            "          (relighting, large paintouts) but uses much more VRAM.\n\n"
            "Rule of thumb: if the change between source and target is local\n"
            "(small blemish, fine texture), 512 is plenty. If the change spans\n"
            "large areas, go higher.")
        self.overlap_factor_input = QComboBox()
        self.overlap_factor_input.addItems([
            "0.0 \u2014 No overlap, fastest",
            "0.25 \u2014 Good balance (recommended)",
            "0.5 \u2014 Smoother, for beauty/cleanup",
            "0.75 \u2014 Maximum coverage, slowest",
        ])
        self.overlap_factor_input.setCurrentText("0.25 \u2014 Good balance (recommended)")
        self.overlap_factor_input.setToolTip(
            "How much neighboring crops overlap as they slide across the image.\n\n"
            "  0.0  — No overlap. Fastest, fewest patches. Risk of visible\n"
            "          tile seams at inference. Only for quick tests.\n"
            "  0.25 — Slight overlap. Good balance of speed and coverage.\n"
            "          Works well for most tasks.\n"
            "  0.5  — Half overlap. More patches, smoother blending.\n"
            "          Use for beauty/cleanup where seams would be visible.\n"
            "  0.75 — Heavy overlap. Slowest but maximum data coverage.\n"
            "          Best for paintout/cleanup where every pixel matters.\n\n"
            "Higher overlap = more training data from the same images,\n"
            "but each epoch takes longer.")
        self.color_space_input = QComboBox()
        self.color_space_input.addItems(["srgb", "linear"])
        self.color_space_input.setCurrentText("srgb")
        self.color_space_input.setToolTip(
            "Must match your source image format:\n"
            "  'srgb' — Standard 8-bit images (PNG, JPEG, TIFF)\n"
            "  'linear' — Scene-linear 32-bit float (EXR)\n\n"
            "If unsure, use 'srgb' for most image formats.")
        form_patch.addRow("Resolution:", self.resolution_input)
        form_patch.addRow("Overlap Factor:", self.overlap_factor_input)
        form_patch.addRow("Color Space:", self.color_space_input)
        grp_patch.setContentLayout(form_patch)
        layout.addWidget(grp_patch)

        # =================================================================
        # SCHEDULE — expanded, right after patch extraction
        # =================================================================
        grp_sched = CollapsibleGroupBox(
            "Schedule",
            description="Control batch size, epoch length, and training duration. "
                        "Defaults work well for most tasks.",
            collapsed=False)
        form_sched = QFormLayout()

        self.batch_size_input = QSpinBox(minimum=1, maximum=256, value=4)
        self.batch_size_input.setToolTip(
            "Patches per training step.\n"
            "Larger = more stable gradients but more VRAM.\n"
            "Reduce to 2 if you get out-of-memory errors.")
        self.auto_batch_check = QCheckBox("Auto")
        self.auto_batch_check.setToolTip(
            "Automatically find the largest batch size that fits in GPU memory.\n"
            "Probes VRAM at training start.")
        self.auto_batch_check.toggled.connect(self.batch_size_input.setDisabled)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_size_input)
        batch_row.addWidget(self.auto_batch_check)
        form_sched.addRow("Batch Size (per GPU):", batch_row)

        self.iter_per_epoch_input = QSpinBox(minimum=1, maximum=10000, value=500)
        self.iter_per_epoch_input.setToolTip(
            "Steps before saving a checkpoint and running validation.\n"
            "Lower = more frequent saves, useful for monitoring progress.")
        form_sched.addRow("Iterations per Epoch:", self.iter_per_epoch_input)

        self.max_steps_input = QSpinBox(minimum=0, maximum=10000000, value=0)
        self.max_steps_input.setSpecialValueText("Unlimited")
        self.max_steps_input.setToolTip(
            "Total steps before auto-stopping.\n"
            "0 = train until manually stopped.\n"
            "Set a limit for queue items or overnight runs.")
        form_sched.addRow("Max Steps:", self.max_steps_input)

        self.progressive_res_check = QCheckBox("Progressive Multi-Resolution")
        self.progressive_res_check.setToolTip(
            "Start at low resolution, then scale up to full:\n"
            "  Epoch 1 \u2192 1/4 resolution (learns shapes fast)\n"
            "  Epoch 2 \u2192 1/2 resolution (medium detail)\n"
            "  Epoch 3+ \u2192 Full resolution (fine detail)\n\n"
            "Speeds up early training ~2\u00d7.\n"
            "Best for: large datasets, high resolutions (512+).\n"
            "Skip when: small resolution (256), short runs, fine-tuning.")
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
            "CPU threads loading data in parallel.\n"
            "'Auto' picks based on your hardware \u2014 recommended for most users.")
        form_sched.addRow("DataLoader Workers:", self.num_workers_input)
        grp_sched.setContentLayout(form_sched)
        layout.addWidget(grp_sched)

        # =================================================================
        # MASK BEHAVIOR — expanded, auto-generate on by default
        # =================================================================
        grp_mask = CollapsibleGroupBox(
            "Mask Behavior",
            description="Focus training on important image regions using masks. "
                        "Auto Mask generates them from source/target differences.",
            collapsed=False)
        form_mask = QFormLayout()

        self.use_mask_loss_input = QCheckBox("Weight loss by mask (white = important)")
        self.use_mask_loss_input.setToolTip(
            "Training loss is weighted by mask: white pixels count more,\n"
            "making the model focus on those areas.\n\n"
            "Example: if you have a face mask, the model will prioritize\n"
            "getting the face right over the background.")
        self.mask_weight_input = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=100.0, value=10.0, singleStep=1.0)
        self.mask_weight_input.setToolTip(
            "How much more important masked (white) regions are.\n"
            "10 = white pixels contribute 10× more to loss.")
        self.mask_weight_input.setEnabled(False)
        self.use_mask_loss_input.toggled.connect(self.mask_weight_input.setEnabled)

        self.use_mask_input_input = QCheckBox("Feed mask as 4th input channel to model")
        self.use_mask_input_input.setToolTip(
            "Feed the mask to the model as a 4th channel alongside RGB.\n"
            "The model can then learn to treat masked/unmasked areas differently.\n\n"
            "Warning: Changes architecture — cannot toggle mid-training.")
        self.use_auto_mask_input = QCheckBox("Auto-generate masks from |src − dst| difference")
        self.use_auto_mask_input.setChecked(True)
        self.use_auto_mask_input.setToolTip(
            "Automatically create masks by comparing source and target.\n"
            "Areas that differ become white (important), identical areas become black.\n\n"
            "Great for: beauty work, paint fixes, cleanup — no manual mask files needed.")
        self.use_auto_mask_input.toggled.connect(
            lambda checked: self.auto_mask_hint.setText(
                "(Mask directory not needed with Auto Mask)" if checked else ""))

        self.skip_empty_patches_input = QCheckBox("Skip empty patches")
        self.skip_empty_patches_input.setChecked(True)
        self.skip_empty_patches_input.setToolTip(
            "Skip training patches where source and target are identical.\n"
            "Speeds up training when only parts of the image have changes.\n\n"
            "Requires Auto Mask to be enabled.")
        self.use_auto_mask_input.toggled.connect(self.skip_empty_patches_input.setEnabled)

        self.skip_empty_threshold_input = QDoubleSpinBox()
        self.skip_empty_threshold_input.setRange(0.1, 20.0)
        self.skip_empty_threshold_input.setSingleStep(0.5)
        self.skip_empty_threshold_input.setDecimals(1)
        self.skip_empty_threshold_input.setValue(1.0)
        self.skip_empty_threshold_input.setToolTip(
            "Max pixel difference threshold (0–255 scale) below which a patch is skipped.\n"
            "If no pixel in the crop differs by more than this, the crop is considered empty.\n\n"
            "Increase if too many patches are being skipped.")
        self.skip_empty_threshold_input.setEnabled(False)
        self.skip_empty_patches_input.toggled.connect(self.skip_empty_threshold_input.setEnabled)

        self.auto_mask_gamma_input = QDoubleSpinBox()
        self.auto_mask_gamma_input.setRange(0.1, 3.0)
        self.auto_mask_gamma_input.setSingleStep(0.1)
        self.auto_mask_gamma_input.setDecimals(2)
        self.auto_mask_gamma_input.setValue(1.0)
        self.auto_mask_gamma_input.setToolTip(
            "Gamma curve applied to auto-mask.\n"
            "  < 1.0 — expands white coverage (e.g. 0.5 for subtle beauty work)\n"
            "  > 1.0 — contracts it (tighter focus)\n"
            "  1.0 — neutral, no adjustment")
        self.auto_mask_gamma_input.setEnabled(False)
        self.use_auto_mask_input.toggled.connect(self.auto_mask_gamma_input.setEnabled)

        mask_weight_row = QHBoxLayout()
        mask_weight_row.addWidget(self.use_mask_loss_input)
        mask_weight_row.addWidget(QLabel("Weight:"))
        mask_weight_row.addWidget(self.mask_weight_input)
        form_mask.addRow(mask_weight_row)
        form_mask.addRow(self.use_mask_input_input)
        form_mask.addRow(self.use_auto_mask_input)
        auto_mask_gamma_row = QHBoxLayout()
        auto_mask_gamma_row.addWidget(QLabel("Auto Mask Gamma:"))
        auto_mask_gamma_row.addWidget(self.auto_mask_gamma_input)
        form_mask.addRow(auto_mask_gamma_row)
        skip_empty_row = QHBoxLayout()
        skip_empty_row.addWidget(self.skip_empty_patches_input)
        skip_empty_row.addWidget(QLabel("Threshold:"))
        skip_empty_row.addWidget(self.skip_empty_threshold_input)
        form_mask.addRow(skip_empty_row)
        grp_mask.setContentLayout(form_mask)
        layout.addWidget(grp_mask)

        # =================================================================
        # DATA AUGMENTATION — collapsible, advanced
        # =================================================================
        grp_aug = CollapsibleGroupBox(
            "Data Augmentation",
            description="Randomly transform training samples to improve generalization. "
                        "Disabled by default — enable individual transforms as needed.",
            collapsed=True)
        form_aug = QFormLayout()

        # Horizontal Flip
        self.hflip_check = QCheckBox("Horizontal Flip")
        self.hflip_check.setToolTip(
            "Randomly flip images left-to-right.\n"
            "Good for most tasks. Disable for text or directional content.")
        self.hflip_p = QSpinBox(minimum=0, maximum=100, value=50)
        self.hflip_p.setSuffix("%")
        self.hflip_p.setToolTip("Probability each sample gets flipped.")
        self.hflip_p.setEnabled(False)
        self.hflip_check.toggled.connect(self.hflip_p.setEnabled)
        hflip_row = QHBoxLayout()
        hflip_row.addWidget(self.hflip_check)
        hflip_row.addWidget(self.hflip_p)
        form_aug.addRow(hflip_row)

        # Affine
        self.affine_check = QCheckBox("Random Affine")
        self.affine_check.setToolTip(
            "Random scale / translate / rotate / shear.\n"
            "Helps the model handle slight alignment differences between source and target.")
        self.affine_p = QSpinBox(minimum=0, maximum=100, value=40)
        self.affine_p.setSuffix("%")
        self.affine_p.setToolTip("Probability each sample gets an affine transform.")
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
        self.affine_interpolation.setToolTip("0 = Nearest, 1 = Bilinear, 2 = Cubic (recommended).")
        self.affine_keep_ratio = QCheckBox()
        self.affine_keep_ratio.setChecked(True)
        self.affine_keep_ratio.setToolTip("Keep aspect ratio when scaling.")
        form_aug.addRow("  Interpolation:", self.affine_interpolation)
        form_aug.addRow("  Keep Ratio:", self.affine_keep_ratio)

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
        self.gamma_check.setToolTip(
            "Randomly adjust brightness curve.\n"
            "Helps the model handle images with varying exposure levels.")
        self.gamma_p = QSpinBox(minimum=0, maximum=100, value=20)
        self.gamma_p.setSuffix("%")
        self.gamma_p.setToolTip("Probability each sample gets a gamma adjustment.")
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

        # Color Augmentation
        self.color_check = QCheckBox("Color Augmentation")
        self.color_check.setToolTip(
            "Randomly adjust brightness, contrast, and saturation.\n"
            "Applied identically to source and target pairs.\n\n"
            "Use when: your dataset has varied lighting or color grades.\n"
            "Avoid when: precise color matching is critical.")
        self.color_p = QSpinBox(minimum=0, maximum=100, value=30)
        self.color_p.setSuffix("%")
        self.color_p.setToolTip("Probability each sample gets a color adjustment.")
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
            "Negative = desaturate, positive = boost color intensity.")
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

        grp_aug.setContentLayout(form_aug)
        layout.addWidget(grp_aug)

        # =================================================================
        # LOGGING & CHECKPOINTS — collapsible
        # =================================================================
        grp_log = CollapsibleGroupBox(
            "Logging & Checkpoints",
            description="How often to save progress and generate preview images.",
            collapsed=True)
        form_log = QFormLayout()

        self.log_interval_input = QSpinBox(minimum=1, maximum=1000, value=5)
        self.log_interval_input.setToolTip("Print loss to the console every N steps.")
        form_log.addRow("Log Interval:", self.log_interval_input)

        self.preview_interval_input = QSpinBox(minimum=0, maximum=1000, value=35)
        self.preview_interval_input.setToolTip(
            "Save a preview image (source / target / model output) every N steps.\n"
            "0 = disable previews. View them on the Previews tab.")
        form_log.addRow("Preview Interval:", self.preview_interval_input)

        self.preview_refresh_input = QSpinBox(minimum=0, maximum=1000, value=5)
        self.preview_refresh_input.setToolTip(
            "How many preview saves before the Previews tab refreshes.\n"
            "Keeps the UI responsive during heavy training.")
        form_log.addRow("Preview Refresh Rate:", self.preview_refresh_input)

        self.keep_checkpoints_input = QSpinBox(minimum=1, maximum=50, value=4)
        self.keep_checkpoints_input.setToolTip(
            "Number of old checkpoints to keep on disk (plus the latest).\n"
            "Older checkpoints are auto-deleted to save space.")
        form_log.addRow("Keep Checkpoints:", self.keep_checkpoints_input)
        grp_log.setContentLayout(form_log)
        layout.addWidget(grp_log)

        # =================================================================
        # AUTO EXPORT — collapsible
        # =================================================================
        grp_export = CollapsibleGroupBox(
            "Auto Export",
            description="Automatically export the model at regular intervals during training "
                        "for testing in your compositing app.",
            collapsed=True)
        form_export = QFormLayout()

        self.auto_export_interval_input = QSpinBox(minimum=0, maximum=1000, value=0)
        self.auto_export_interval_input.setSpecialValueText("Disabled")
        self.auto_export_interval_input.setToolTip(
            "Export the model every N epochs during training.\n"
            "0 = disabled. Exports go to an 'exports/' subfolder.\n\n"
            "Useful for testing intermediate models without stopping training.")
        form_export.addRow("Export Every N Epochs:", self.auto_export_interval_input)

        self.auto_export_flame_check = QCheckBox("Flame / After Effects (ONNX + JSON)")
        self.auto_export_flame_check.setToolTip(
            "Export to ONNX format with a Flame JSON sidecar.\n"
            "Works with Autodesk Flame ML nodes and Adobe After Effects ONNX plugins.")
        form_export.addRow(self.auto_export_flame_check)

        self.auto_export_nuke_check = QCheckBox("Nuke (TorchScript .pt + .nk helper)")
        self.auto_export_nuke_check.setToolTip(
            "Export to TorchScript (.pt) with a Nuke helper script (.nk).\n"
            "Open the .nk in Nuke to produce the final .cat file for Inference.")
        form_export.addRow(self.auto_export_nuke_check)

        grp_export.setContentLayout(form_export)
        layout.addWidget(grp_export)

        # =================================================================
        # PLATEAU DETECTION — collapsible
        # =================================================================
        grp_es = CollapsibleGroupBox(
            "Plateau Detection",
            description="Detect when the model stops improving and optionally stop training. "
                        "Acts as a safety net for long or overnight runs.",
            collapsed=True)
        form_es = QFormLayout()

        self.es_enabled_input = QCheckBox("Enable Plateau Detection")
        self.es_enabled_input.setChecked(True)
        self.es_enabled_input.setToolTip(
            "Saves a _plateau.pth checkpoint when loss stops improving.\n"
            "A safety net — you can always roll back to the best model.")
        form_es.addRow(self.es_enabled_input)

        self.es_patience_input = QSpinBox(minimum=5, maximum=200, value=30)
        self.es_patience_input.setToolTip(
            "Number of epochs with no improvement before triggering.\n"
            "30 is a safe default — lower for short runs, higher for noisy data.")
        form_es.addRow("Patience (epochs):", self.es_patience_input)

        self.es_stop_input = QCheckBox("Stop training on plateau")
        self.es_stop_input.setChecked(False)
        self.es_stop_input.setToolTip(
            "Actually stop training when plateau is detected.\n"
            "Off = just save checkpoint and keep going (recommended for most cases).")
        form_es.addRow(self.es_stop_input)

        grp_es.setContentLayout(form_es)
        layout.addWidget(grp_es)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        self.tabs.addTab(scroll, "Training")
