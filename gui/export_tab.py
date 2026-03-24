from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox,
)
from PySide6.QtCore import Qt


class ExportTabMixin:
    """Mixin that creates the Export tab (Tab 3)."""

    def _create_export_tab(self):
        """Tab 5: Export -- convert checkpoint for VFX apps."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addStretch()

        title = QLabel("Export Model")
        title.setStyleSheet("font-size: 14pt; font-weight: 600;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(4)

        info = QLabel(
            "Convert your trained checkpoint for use in compositing applications.\n"
            "The model is exported from the Output Folder set in the Training tab.")
        info.setAlignment(Qt.AlignCenter)
        info.setWordWrap(True)
        info.setStyleSheet("color: #8b919d;")
        layout.addWidget(info)
        layout.addSpacing(20)

        self.convert_flame_btn = QPushButton("Export for Flame / After Effects")
        self.convert_flame_btn.setToolTip(
            "Convert checkpoint to ONNX format.\n\n"
            "  Flame — Load directly into ML Timewarp / ML Inference nodes\n"
            "  After Effects — Use with ONNX-compatible plugins\n\n"
            "Creates: model.onnx + model.json sidecar")
        self.convert_flame_btn.clicked.connect(lambda: self._start_conversion('flame'))
        layout.addWidget(self.convert_flame_btn)

        self.convert_nuke_btn = QPushButton("Export for Nuke")
        self.convert_nuke_btn.setToolTip(
            "Convert checkpoint to TorchScript format.\n\n"
            "Creates: model.pt + helper.nk\n"
            "Open the .nk in Nuke, run CatFileCreator to produce\n"
            "the final .cat file for the Inference node.")
        self.convert_nuke_btn.clicked.connect(lambda: self._start_conversion('nuke'))
        layout.addWidget(self.convert_nuke_btn)

        layout.addSpacing(10)

        self.copy_before_convert_check = QCheckBox("Copy checkpoint to export subfolder first")
        self.copy_before_convert_check.setChecked(True)
        self.copy_before_convert_check.setToolTip(
            "Copy the checkpoint into an 'exports/' subfolder before converting.\n"
            "Keeps exported models separate from training checkpoints.")
        layout.addWidget(self.copy_before_convert_check, alignment=Qt.AlignCenter)

        layout.addStretch()
        self.tabs.addTab(tab, "Export")
