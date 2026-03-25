import os
import platform

from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QLabel, QSpinBox, QPushButton, QLineEdit,
)


# Subfolder names to search for (in priority order)
_SRC_NAMES = ("src", "source", "in", "input")
_DST_NAMES = ("dst", "dest", "destination", "out", "output", "target")
_VAL_SRC_NAMES = ("val_src", "val_source", "val_input", "val/src", "validation/src")
_VAL_DST_NAMES = ("val_dst", "val_dest", "val_target", "val/dst", "validation/dst")
_MASK_NAMES = ("mask", "masks", "matte", "mattes")


def _find_subfolder(root, candidates):
    """Return the first existing subfolder matching a candidate name, or ''."""
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return ""


class DataTabMixin:
    """Mixin that builds the project-folder widgets (added to Training tab, not its own tab)."""

    def _create_data_widgets(self, layout):
        """Add project folder picker and hidden path widgets to the given layout."""

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
        # PROJECT FOLDER
        # =================================================================
        grp_project = QGroupBox("Project Folder")
        form_project = QFormLayout(grp_project)
        desc = QLabel(
            "Pick a folder that contains source and destination subfolders "
            "(src/, source/, in/, input/ and dst/, dest/, out/, output/, etc). "
            "Optional: val_src/, val_dst/, mask/. Output saves to model/.")
        desc.setWordWrap(True)
        desc.setProperty("cssClass", "section-desc")
        form_project.addRow(desc)

        self.project_folder_input = self._create_path_selector("Project Folder")
        self.project_folder_input.setToolTip(
            "Root folder for this training project.\n\n"
            "Expected structure:\n"
            "  my_project/\n"
            "    src/          \u2190 source images (required)\n"
            "    dst/          \u2190 target images (required)\n"
            "    val_src/      \u2190 validation source (optional)\n"
            "    val_dst/      \u2190 validation target (optional)\n"
            "    mask/         \u2190 mask images (optional)\n"
            "    model/        \u2190 checkpoints saved here (auto-created)\n\n"
            "Also accepts: source/, target/, input/, output/, dest/, etc.")
        form_project.addRow("Project Folder:", self.project_folder_input)

        self._data_status_label = QLabel("")
        self._data_status_label.setWordWrap(True)
        form_project.addRow("", self._data_status_label)

        self.verify_btn = QPushButton("Verify Inputs")
        self.verify_btn.setToolTip(
            "Scan source and target directories for problems before training.\n"
            "Checks for: missing targets, dimension mismatches, corrupt files,\n"
            "and images too small for the configured patch resolution.")
        form_project.addRow("", self.verify_btn)

        layout.addWidget(grp_project)

        # Hook up auto-detection
        project_line_edit = self.project_folder_input.findChild(QLineEdit)
        if project_line_edit:
            project_line_edit.textChanged.connect(self._on_project_folder_changed)

        # =================================================================
        # Hidden path widgets — still used by gather_config / YAML
        # =================================================================
        self.src_dir_input = self._create_path_selector("Source Directory")
        self.src_dir_input.setVisible(False)
        self.dst_dir_input = self._create_path_selector("Destination Directory")
        self.dst_dir_input.setVisible(False)
        self.mask_dir_input = self._create_path_selector("Mask Directory")
        self.mask_dir_input.setVisible(False)
        self.val_src_dir_input = self._create_path_selector("Val Source Directory")
        self.val_src_dir_input.setVisible(False)
        self.val_dst_dir_input = self._create_path_selector("Val Destination Directory")
        self.val_dst_dir_input.setVisible(False)
        self.auto_mask_hint = QLabel("")
        self.auto_mask_hint.setVisible(False)
        for w in (self.src_dir_input, self.dst_dir_input, self.mask_dir_input,
                  self.val_src_dir_input, self.val_dst_dir_input, self.auto_mask_hint):
            layout.addWidget(w)

    def _on_project_folder_changed(self, root):
        """Auto-detect subfolders and populate hidden path widgets + status."""
        if not root or not os.path.isdir(root):
            self._data_status_label.setText("")
            return

        src = _find_subfolder(root, _SRC_NAMES)
        dst = _find_subfolder(root, _DST_NAMES)
        val_src = _find_subfolder(root, _VAL_SRC_NAMES)
        val_dst = _find_subfolder(root, _VAL_DST_NAMES)
        mask = _find_subfolder(root, _MASK_NAMES)

        model_dir = os.path.join(root, "model")

        self._set_path(self.src_dir_input, src)
        self._set_path(self.dst_dir_input, dst)
        self._set_path(self.val_src_dir_input, val_src)
        self._set_path(self.val_dst_dir_input, val_dst)
        self._set_path(self.mask_dir_input, mask)
        self._set_path(self.model_folder_input, model_dir)

        lines = []
        if src:
            lines.append(f"<span style='color:#16A34A'>\u2713</span> src: {os.path.basename(src)}/")
        else:
            lines.append("<span style='color:#EF4444'>\u2717 Missing src/ folder</span>")
        if dst:
            lines.append(f"<span style='color:#16A34A'>\u2713</span> dst: {os.path.basename(dst)}/")
        else:
            lines.append("<span style='color:#EF4444'>\u2717 Missing dst/ folder</span>")
        if val_src:
            lines.append(f"<span style='color:#16A34A'>\u2713</span> val_src: {os.path.basename(val_src)}/")
        if val_dst:
            lines.append(f"<span style='color:#16A34A'>\u2713</span> val_dst: {os.path.basename(val_dst)}/")
        if mask:
            lines.append(f"<span style='color:#16A34A'>\u2713</span> mask: {os.path.basename(mask)}/")
        lines.append(f"<span style='color:#6b7280'>\u2192 Output: model/</span>")

        self._data_status_label.setText("<br>".join(lines))
