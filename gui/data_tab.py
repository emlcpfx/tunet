import platform

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QLabel, QSpinBox, QPushButton,
)


class DataTabMixin:
    """Mixin that creates the Data tab (Tab 0) — folder paths only."""

    def _create_data_tab(self):
        """Tab 1: Data -- source, target, validation, and mask directories."""
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

        # =================================================================
        # SOURCE & TARGET DATA
        # =================================================================
        grp_data = QGroupBox("Source & Target Data")
        form_data = QFormLayout(grp_data)
        desc = QLabel("Point these to your training image folders. Source = input (before), Target = output (after).")
        desc.setWordWrap(True)
        desc.setProperty("cssClass", "section-desc")
        form_data.addRow(desc)

        self.src_dir_input = self._create_path_selector("Source Directory")
        self.src_dir_input.setToolTip(
            "Folder with source/input images the model learns to transform FROM.\n"
            "These are the 'before' images — the model sees these at inference time.")
        self.dst_dir_input = self._create_path_selector("Destination Directory")
        self.dst_dir_input.setToolTip(
            "Folder with target/output images the model learns to produce.\n"
            "These are the 'after' images. Must have matching filenames to Source.")
        self.mask_dir_input = self._create_path_selector("Mask Directory")
        self.mask_dir_input.setToolTip(
            "Optional. Grayscale mask images (white = important regions).\n"
            "Not needed if Auto Mask is enabled in the Training tab.")
        self.auto_mask_hint = QLabel("")
        self.auto_mask_hint.setStyleSheet("color: #8b919d; font-style: italic;")
        form_data.addRow("Source Directory:", self.src_dir_input)
        form_data.addRow("Target Directory:", self.dst_dir_input)
        form_data.addRow("Mask Directory:", self.mask_dir_input)
        form_data.addRow("", self.auto_mask_hint)
        layout.addWidget(grp_data)

        # =================================================================
        # VALIDATION DATA
        # =================================================================
        grp_val = QGroupBox("Validation Data (optional)")
        form_val = QFormLayout(grp_val)
        desc_val = QLabel("Separate held-out images to monitor how well the model generalizes.")
        desc_val.setWordWrap(True)
        desc_val.setProperty("cssClass", "section-desc")
        form_val.addRow(desc_val)

        self.val_src_dir_input = self._create_path_selector("Val Source Directory")
        self.val_src_dir_input.setToolTip(
            "Separate source images used only for validation (never trained on).\n"
            "Helps you spot overfitting — if training loss drops but validation loss doesn't.")
        self.val_dst_dir_input = self._create_path_selector("Val Destination Directory")
        self.val_dst_dir_input.setToolTip(
            "Matching target images for the validation source folder.")
        form_val.addRow("Val Source:", self.val_src_dir_input)
        form_val.addRow("Val Target:", self.val_dst_dir_input)
        layout.addWidget(grp_val)

        # --- Verify Inputs ---
        self.verify_btn = QPushButton("Verify Inputs")
        self.verify_btn.setToolTip(
            "Scan source and target directories for problems before training.\n"
            "Checks for: missing targets, dimension mismatches, corrupt files,\n"
            "and images too small for the configured patch resolution.")
        layout.addWidget(self.verify_btn)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        self.tabs.addTab(scroll, "Data")
