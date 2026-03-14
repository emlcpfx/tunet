import logging
import threading
import yaml

from PySide6.QtWidgets import QScrollArea
from PySide6.QtCore import Qt, QObject, Signal, QTimer
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
