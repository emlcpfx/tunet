import platform

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
)


class DataTabMixin:
    """Mixin that creates the Data tab (Tab 0)."""

    def _create_data_tab(self):
        """Tab 1: Data -- paths, patch extraction, masks, augmentation."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Distributed Training (Linux only) ---
        if platform.system() == 'Linux':
            grp_dist = QGroupBox("Distributed Training")
            form_dist = QFormLayout(grp_dist)
            self.nproc_input = QSpinBox(minimum=1, maximum=16, value=1)
            self.nproc_input.setToolTip("Number of GPUs for distributed training via torchrun.")
            form_dist.addRow("GPUs (nproc):", self.nproc_input)
            layout.addWidget(grp_dist)
        else:
            self.nproc_input = None

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
        self.auto_mask_hint.setStyleSheet("color: #8b919d; font-style: italic;")
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

        self.auto_mask_gamma_input = QDoubleSpinBox()
        self.auto_mask_gamma_input.setRange(0.1, 3.0)
        self.auto_mask_gamma_input.setSingleStep(0.1)
        self.auto_mask_gamma_input.setDecimals(2)
        self.auto_mask_gamma_input.setValue(1.0)
        self.auto_mask_gamma_input.setToolTip(
            "Gamma curve applied to auto-mask. Values < 1.0 expand white coverage "
            "(e.g. 0.5 for subtle beauty work), > 1.0 contracts it, 1.0 = neutral.")
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
            "Negative = darker, positive = brighter. \u00b10.2 is a safe default.")
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
