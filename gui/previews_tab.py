from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QPushButton,
)
from PySide6.QtCore import Qt

from .helpers import ZoomPanScrollArea


class PreviewsTabMixin:
    """Mixin that creates the Previews tab (Tab 2)."""

    def _create_previews_tab(self):
        """Tab 4: Previews -- training + validation preview with toggle."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # Toggle buttons
        toggle_layout = QHBoxLayout()
        self.preview_train_btn = QPushButton("Training Preview")
        self.preview_train_btn.setCheckable(True)
        self.preview_train_btn.setChecked(True)
        self.preview_train_btn.setToolTip("Show latest training preview (src / dst / model output).")
        self.preview_val_btn = QPushButton("Validation Preview")
        self.preview_val_btn.setCheckable(True)
        self.preview_val_btn.setToolTip("Show latest validation preview (unseen data).")

        self.preview_train_btn.clicked.connect(lambda: self._switch_preview('train'))
        self.preview_val_btn.clicked.connect(lambda: self._switch_preview('val'))
        toggle_layout.addWidget(self.preview_train_btn)
        toggle_layout.addWidget(self.preview_val_btn)
        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)

        # Scroll area (shared for both previews)
        self.scroll_area = ZoomPanScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.preview_label = QLabel("Waiting for preview image...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.zoom_changed.connect(self._on_preview_wheel_zoom)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # Zoom controls + diff amplify
        controls_layout = QHBoxLayout()
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["Fit", "50%", "100%", "200%"])
        self.zoom_combo.setToolTip("Zoom level. Mouse wheel to zoom, middle-click to pan.")
        self.zoom_combo.activated.connect(lambda: self._on_zoom_combo_changed(self.zoom_combo.currentText()))
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Zoom:"))
        controls_layout.addWidget(self.zoom_combo)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Diff Amplify:"))
        self.diff_amplify_slider = QSlider(Qt.Horizontal)
        self.diff_amplify_slider.setRange(1, 50)
        self.diff_amplify_slider.setValue(5)
        self.diff_amplify_slider.setFixedWidth(120)
        self.diff_amplify_slider.setToolTip("Amplification for the diff heatmap. Higher = more visible subtle differences.")
        self.diff_amplify_label = QLabel("5x")
        self.diff_amplify_slider.valueChanged.connect(lambda v: self.diff_amplify_label.setText(f"{v}x"))
        controls_layout.addWidget(self.diff_amplify_slider)
        controls_layout.addWidget(self.diff_amplify_label)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Column labels
        self.labels_container = QWidget()
        labels_layout = QHBoxLayout(self.labels_container)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        lbl_src = QLabel("src data")
        lbl_dst = QLabel("dst data")
        lbl_model = QLabel("Model result")
        lbl_src.setAlignment(Qt.AlignLeft)
        lbl_dst.setAlignment(Qt.AlignCenter)
        lbl_model.setAlignment(Qt.AlignRight)
        labels_layout.addWidget(lbl_src)
        labels_layout.addWidget(lbl_dst)
        labels_layout.addWidget(lbl_model)
        main_layout.addWidget(self.labels_container)

        # Track which preview is active
        self._active_preview = 'train'

        self.tabs.addTab(tab, "Previews")
