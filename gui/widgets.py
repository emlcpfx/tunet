import logging
import threading
import yaml

from PySide6.QtWidgets import (
    QScrollArea, QVBoxLayout, QWidget, QLabel, QPushButton,
)
from PySide6.QtCore import Qt, QObject, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QTextCursor


class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


class ProcessStreamReader(QObject):
    """Reads a subprocess stdout/stderr stream in a thread and emits lines."""
    new_text = Signal(str)

    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self):
        for line in iter(self.stream.readline, ''):
            self.new_text.emit(line)


class FileCopyWorker(QObject):
    """Copies a file in a background thread with progress reporting."""
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)
    _is_cancelled = False

    def run(self, source_path_str, dest_path_str):
        from pathlib import Path
        self._is_cancelled = False
        source_path = Path(source_path_str)
        dest_path = Path(dest_path_str)
        try:
            total_size = source_path.stat().st_size
            bytes_copied = 0
            chunk_size = 4096 * 4
            with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
                while True:
                    if self._is_cancelled:
                        self.error.emit("Copy operation cancelled.")
                        if dest_path.exists():
                            dest_path.unlink()
                        return
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    bytes_copied += len(chunk)
                    percent = int((bytes_copied / total_size) * 100)
                    self.progress.emit(percent)
            self.finished.emit(str(dest_path))
        except Exception as e:
            self.error.emit(f"File copy failed: {e}")

    def cancel(self):
        self._is_cancelled = True


class ZoomPanScrollArea(QScrollArea):
    """Scroll area with mouse-wheel zoom and middle-click pan."""
    zoom_changed = Signal(float)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.zoom_changed.emit(factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, '_panning', False) and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CollapsibleGroupBox(QWidget):
    """A section with a clickable arrow + title bar that expands/collapses content.

    Uses ▶/▼ arrows instead of a checkbox so it's clear this is a
    disclosure control, not an on/off toggle.

    Parameters
    ----------
    title : str
        Section heading shown in the title bar.
    description : str, optional
        Brief helper text shown below the title when expanded (for beginners).
    collapsed : bool
        Whether the section starts collapsed (default True).
    """

    def __init__(self, title="", description="", collapsed=True, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = not collapsed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 0)
        outer.setSpacing(0)

        # --- Clickable header row ---
        self._header = QPushButton(self._header_text())
        self._header.setProperty("cssClass", "collapse-header")
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(self._toggle)
        outer.addWidget(self._header)

        # --- Body (description + content), wrapped in a styled frame ---
        self._body = QWidget()
        self._body.setProperty("cssClass", "collapse-body")
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(10, 6, 10, 10)
        body_layout.setSpacing(4)

        self._desc_label = None
        if description:
            self._desc_label = QLabel(description)
            self._desc_label.setWordWrap(True)
            self._desc_label.setProperty("cssClass", "section-desc")
            body_layout.addWidget(self._desc_label)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self._content)

        outer.addWidget(self._body)
        self._body.setVisible(self._expanded)

    def _header_text(self):
        arrow = "\u25BC" if self._expanded else "\u25B6"
        return f"  {arrow}  {self._title}"

    def _toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._header.setText(self._header_text())

    def isExpanded(self):
        return self._expanded

    def contentLayout(self):
        """Return the layout where child widgets should be added."""
        return self._content_layout

    def setContentLayout(self, layout):
        """Replace the content area's layout."""
        old = self._content.layout()
        if old is not None:
            QWidget().setLayout(old)  # reparent to discard
        self._content.setLayout(layout)

    def title(self):
        return self._title


class QTextEditLogHandler(logging.Handler):
    """Logging handler that writes to a QTextEdit from any thread."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + '\n'
        QTimer.singleShot(0, lambda: self._append(msg))

    def _append(self, msg):
        self.text_widget.moveCursor(QTextCursor.End)
        self.text_widget.insertPlainText(msg)
