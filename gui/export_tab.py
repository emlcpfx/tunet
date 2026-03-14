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

        info = QLabel("Export your trained model checkpoint for use in\nthird-party compositing applications.")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        layout.addSpacing(15)

        self.convert_flame_btn = QPushButton("Export for Flame / After Effects")
        self.convert_flame_btn.setToolTip(
            "Convert checkpoint to ONNX for Autodesk Flame or Adobe After Effects.")
        self.convert_flame_btn.clicked.connect(lambda: self._start_conversion('flame'))
        layout.addWidget(self.convert_flame_btn)

        self.convert_nuke_btn = QPushButton("Export for Nuke")
        self.convert_nuke_btn.setToolTip(
            "Convert checkpoint for Foundry Nuke. A .nk script file is also generated.")
        self.convert_nuke_btn.clicked.connect(lambda: self._start_conversion('nuke'))
        layout.addWidget(self.convert_nuke_btn)

        self.copy_before_convert_check = QCheckBox("Copy checkpoint to export subfolder first")
        self.copy_before_convert_check.setChecked(True)
        self.copy_before_convert_check.setToolTip(
            "Copy checkpoint into a subfolder before converting. Keeps exports separate from training.")
        self.copy_before_convert_check.setContentsMargins(0, 10, 0, 0)
        layout.addWidget(self.copy_before_convert_check, alignment=Qt.AlignCenter)

        layout.addStretch()
        self.tabs.addTab(tab, "Export")
