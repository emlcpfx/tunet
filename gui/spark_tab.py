"""
gui/spark_tab.py — Spark Compute integration for the TuNet desktop app.

UX model: NOT a separate tab. The Spark workflow lives where the user is
already configuring the run — on the Training tab — via:

    [Training tab]                          [Bottom control panel]
    │                                       │ Load  Save  Monitor  ──────  Start  Stop
    │  ... existing config form ...         │                              ☁ Run on Spark ▾
    │                                       │  Spark Jobs (2) ▾  ←── pill that opens drawer
    │                                       │
    └───────────────────────────────────────┴──────────────────────────────────────────────
              ▼ slides up when pill clicked
    ┌────────────────────────────────────────────────────────────────────┐
    │ Spark Compute · 2 active                                  [Refresh]│
    │ ──────────────────────────────────────────────────────────────────│
    │ Name      Status     GPU         Started    Runtime   Actions     │
    │ tunet-x   running    g6e.4xlarge 11:23      4m 12s   Logs Cancel │
    │ ...                                                                │
    └────────────────────────────────────────────────────────────────────┘

Submit flow: "Run on Spark" pulls the live form via gather_config_from_ui(),
auto-detects checkpoints from output_dir, packs the input tarball, submits
via spark_launch.submit_job(), and opens a streaming log window.
"""

import os
import sys
import threading
import tempfile
import time
import traceback

import requests
import yaml

from PySide6.QtCore import Qt, QObject, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QLineEdit, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QDialog, QPlainTextEdit, QMessageBox, QFrame, QSizePolicy,
    QGraphicsOpacityEffect, QToolButton,
)
from PySide6.QtGui import QColor, QTextCursor, QFont

# Defer import — spark_launch / runpod_launch live at the project root.
# We import lazily inside methods so the module loads even before the user
# has installed the Spark deps.
def _sl():
    import spark_launch
    return spark_launch


# ── Status palette (matches existing app tokens) ─────────────────────────────

STATUS_COLORS = {
    'queued':       '#1c64f2',
    'provisioning': '#1c64f2',
    'running':      '#16A34A',
    'completed':    '#6b7280',
    'failed':       '#EF4444',
    'cancelled':    '#9ca3af',
}
ACTIVE_STATUSES = {'queued', 'provisioning', 'running'}


def _job_id(j):       return j.get('id') or j.get('jobId') or '?'
def _job_status(j):   return j.get('status') or '?'
def _job_instance(j): return j.get('instance_type_name') or j.get('instanceType') or '?'

def _job_label(j):
    """Friendly label — falls back to job-id prefix (Spark API doesn't echo names)."""
    name = j.get('name') or j.get('label')
    if name:
        return name
    jid = _job_id(j)
    return jid[:8] if len(jid) > 8 else jid

def _job_started(j):
    ts = (j.get('started_running_at') or j.get('started_provisioning_at')
          or j.get('created_at') or j.get('createdAt'))
    if not ts:
        return '--'
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime('%H:%M')
    except Exception:
        return ts[:5]

def _job_runtime(j):
    from datetime import datetime, timezone
    start = (j.get('started_running_at') or j.get('started_provisioning_at')
             or j.get('created_at') or j.get('createdAt'))
    end   = j.get('terminal_at') or j.get('finished_at') or j.get('completed_at')
    if not start:
        return '--'
    try:
        s = datetime.fromisoformat(start.replace('Z', '+00:00'))
        e = datetime.fromisoformat(end.replace('Z', '+00:00')) if end else datetime.now(timezone.utc)
        secs = max(0, int((e - s).total_seconds()))
        h, rem = divmod(secs, 3600)
        m, ss = divmod(rem, 60)
        if h:    return f'{h}h {m:02d}m'
        if m:    return f'{m}m {ss:02d}s'
        return f'{ss}s'
    except Exception:
        return '--'


# ── Background poller — emits signals on the GUI thread ──────────────────────

class _JobPoller(QObject):
    refreshed = Signal(list)
    error     = Signal(str)
    auth_required = Signal()

    def __init__(self, interval=10):
        super().__init__()
        self._interval = interval
        self._running  = False
        self._enabled  = True

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def kick(self):
        """Trigger an immediate refresh (off-thread)."""
        threading.Thread(target=self._fetch_once, daemon=True).start()

    def _fetch_once(self):
        if not self._enabled:
            return
        try:
            jobs = _sl().list_jobs()
            self.refreshed.emit(jobs)
        except SystemExit as e:
            # spark_launch sys.exit's on auth failure — don't kill the dashboard
            self._enabled = False
            self.error.emit(str(e))
            self.auth_required.emit()
        except Exception as e:
            self.error.emit(str(e))

    def _loop(self):
        while self._running:
            self._fetch_once()
            for _ in range(int(self._interval * 10)):
                if not self._running:
                    return
                time.sleep(0.1)


# ── Streaming log window ─────────────────────────────────────────────────────

class SparkLogWindow(QDialog):
    _line  = Signal(str)
    _ended = Signal()

    def __init__(self, jid, name='', parent=None):
        super().__init__(parent)
        self._jid  = jid
        self._name = name or jid
        self.setWindowTitle(f'Spark · Logs · {self._name}')
        self.resize(960, 640)
        self._stop = threading.Event()
        self._build_ui()
        self._line.connect(self._append)
        self._ended.connect(self._on_ended)
        threading.Thread(target=self._stream, daemon=True).start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        hdr = QHBoxLayout()
        title = QLabel(f'<b>{self._name}</b>')
        title.setStyleSheet('font-size:12pt;')
        hdr.addWidget(title)
        sub = QLabel(self._jid)
        sub.setStyleSheet('color:#6b7280; font-family:Consolas,monospace; font-size:9pt;')
        hdr.addWidget(sub)
        hdr.addStretch()

        self._cancel_btn = QPushButton('Cancel Job')
        self._cancel_btn.setFixedWidth(110)
        self._cancel_btn.setStyleSheet(
            'QPushButton { background:#fef2f2; color:#EF4444; border:1px solid #fecaca; '
            'border-radius:6px; padding:6px 12px; font-weight:600; }'
            'QPushButton:hover { background:#fee2e2; }'
            'QPushButton:disabled { background:#f9fafb; color:#d1d5db; border-color:#f3f4f6; }')
        self._cancel_btn.clicked.connect(self._do_cancel)
        hdr.addWidget(self._cancel_btn)

        close_btn = QPushButton('Close')
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        hdr.addWidget(close_btn)
        root.addLayout(hdr)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            'QPlainTextEdit { background:#0f172a; color:#cdd6f4; '
            'font-family:Consolas,"Fira Code",monospace; font-size:10pt; '
            'border-radius:8px; padding:10px; selection-background-color:#7c3aed; }')
        root.addWidget(self._log)

        self._status_bar = QLabel('Connecting to log stream...')
        self._status_bar.setStyleSheet('color:#6b7280; font-size:9pt;')
        root.addWidget(self._status_bar)

    def _stream(self):
        try:
            sl = _sl()
            tok = sl.get_token()
            url = f'{sl.SPARK_API}/api/compute/jobs/{self._jid}/logs/stream'
            with requests.get(url,
                              headers={'Authorization': f'Bearer {tok}',
                                       'Accept': 'text/event-stream'},
                              stream=True, timeout=None) as r:
                for line in r.iter_lines(decode_unicode=True):
                    if self._stop.is_set():
                        return
                    if not line or line.startswith(':'):
                        continue
                    if line.startswith('data: '):
                        self._line.emit(line[6:])
                    else:
                        self._line.emit(line)
        except Exception as e:
            self._line.emit(f'\n[stream error] {e}')
        finally:
            self._ended.emit()

    def _append(self, text):
        self._log.appendPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _on_ended(self):
        self._status_bar.setText('Stream closed.')
        self._cancel_btn.setEnabled(False)

    def _do_cancel(self):
        if QMessageBox.question(self, 'Cancel Job',
                                f'Send SIGTERM to "{self._name}"?\n\n'
                                f'Worst case you lose ≤500 steps since the last save.',
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            _sl().cancel_job(self._jid)
            self._append('\n[cancel requested]')
            self._cancel_btn.setEnabled(False)
        except SystemExit as e:
            QMessageBox.warning(self, 'Cancel failed', str(e))

    def closeEvent(self, ev):
        self._stop.set()
        super().closeEvent(ev)


# ── Drawer: collapsible job dashboard ────────────────────────────────────────

class SparkJobDrawer(QFrame):
    """
    Slide-up drawer showing the live Spark jobs table. Hosted as a child of
    the main window's left layout, below the existing splitter+control panel.

    Visibility is toggled by the SparkJobsPill in the bottom bar.
    """
    log_requested = Signal(str, str)   # job_id, name
    jobs_count_changed = Signal(int, int)  # total, active

    DRAWER_HEIGHT = 240

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('sparkDrawer')
        self.setStyleSheet("""
            QFrame#sparkDrawer {
                background: #ffffff;
                border-top: 1px solid #e5e7eb;
                border-bottom: 1px solid #e5e7eb;
            }
        """)
        self.setMaximumHeight(0)   # collapsed by default
        self._jobs = []

        self.poller = _JobPoller(interval=10)
        self.poller.refreshed.connect(self._on_refresh)
        self.poller.error.connect(self._on_error)

        self._build_ui()
        self.poller.start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        hdr = QHBoxLayout()
        title = QLabel('Spark Compute')
        title.setStyleSheet('font-weight:700; font-size:11pt; color:#111827;')
        hdr.addWidget(title)

        self._summary_label = QLabel('connecting…')
        self._summary_label.setStyleSheet('color:#6b7280; font-size:9pt; padding-left:10px;')
        hdr.addWidget(self._summary_label)

        hdr.addStretch()

        self._refresh_btn = QPushButton('Refresh')
        self._refresh_btn.setFixedWidth(88)
        self._refresh_btn.clicked.connect(self.poller.kick)
        hdr.addWidget(self._refresh_btn)
        root.addLayout(hdr)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(['Name', 'Status', 'GPU', 'Started', 'Runtime', 'Actions'])
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 5):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.Fixed)
        self._table.setColumnWidth(5, 200)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setStyleSheet("""
            QTableWidget { border:1px solid #e5e7eb; border-radius:8px; background:white; font-size:10pt; }
            QTableWidget::item { padding:6px 10px; }
            QHeaderView::section { background:#F9FAFB; border:none; border-bottom:1px solid #e5e7eb;
                                   padding:6px 10px; font-weight:600; color:#374151; font-size:9pt; }
            QTableWidget::item:alternate { background:#F9FAFB; }
        """)
        root.addWidget(self._table)

    # ── Public API ──────────────────────────────────────────────────────────
    def is_open(self):
        return self.maximumHeight() > 0

    def open_drawer(self):
        self.setMaximumHeight(self.DRAWER_HEIGHT)

    def close_drawer(self):
        self.setMaximumHeight(0)

    def toggle(self):
        if self.is_open():
            self.close_drawer()
        else:
            self.open_drawer()
            self.poller.kick()

    def jobs_count(self):
        active = [j for j in self._jobs if _job_status(j) in ACTIVE_STATUSES]
        return len(self._jobs), len(active)

    # ── Slots ───────────────────────────────────────────────────────────────
    def _on_refresh(self, jobs):
        self._jobs = jobs
        total, active = self.jobs_count()
        self._summary_label.setText(
            f'· {active} active · {total} total · updated {time.strftime("%H:%M:%S")}')

        self._table.setRowCount(0)
        for row, j in enumerate(jobs):
            self._table.insertRow(row)
            self._table.setRowHeight(row, 38)
            status = _job_status(j)

            def _cell(text, align=Qt.AlignLeft | Qt.AlignVCenter):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment(align)
                return it

            self._table.setItem(row, 0, _cell(_job_label(j)))

            st = _cell(status, Qt.AlignCenter | Qt.AlignVCenter)
            st.setForeground(QColor(STATUS_COLORS.get(status, '#374151')))
            self._table.setItem(row, 1, st)
            self._table.setItem(row, 2, _cell(_job_instance(j)))
            self._table.setItem(row, 3, _cell(_job_started(j)))
            self._table.setItem(row, 4, _cell(_job_runtime(j)))

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            btn_layout.setSpacing(4)

            is_active = status in ACTIVE_STATUSES
            jid = _job_id(j); name = _job_label(j)

            logs_btn = QPushButton('Logs')
            logs_btn.setFixedWidth(60)
            logs_btn.setStyleSheet(
                'QPushButton { color:#7E3AF2; border:1px solid #e9d5ff; border-radius:4px; '
                'padding:3px 6px; font-size:9pt; background:white; }'
                'QPushButton:hover { background:#F7F4FC; }')
            logs_btn.clicked.connect(lambda _, jid=jid, name=name: self.log_requested.emit(jid, name))

            cancel_btn = QPushButton('Cancel')
            cancel_btn.setFixedWidth(60)
            cancel_btn.setEnabled(is_active)
            cancel_btn.setStyleSheet(
                'QPushButton { color:#EF4444; border:1px solid #fecaca; border-radius:4px; '
                'padding:3px 6px; font-size:9pt; background:white; }'
                'QPushButton:hover { background:#fef2f2; }'
                'QPushButton:disabled { color:#d1d5db; border-color:#f3f4f6; background:#fafafa; }')
            cancel_btn.clicked.connect(lambda _, jid=jid, name=name: self._do_cancel(jid, name))

            output_btn = QPushButton('Output')
            output_btn.setFixedWidth(64)
            output_btn.setStyleSheet(
                'QPushButton { color:#374151; border:1px solid #e5e7eb; border-radius:4px; '
                'padding:3px 6px; font-size:9pt; background:white; }'
                'QPushButton:hover { border-color:#ae69f4; color:#ae69f4; }')
            output_btn.clicked.connect(lambda _, jid=jid: self._open_output(jid))

            for b in (logs_btn, cancel_btn, output_btn):
                btn_layout.addWidget(b)
            self._table.setCellWidget(row, 5, btn_widget)

        # Notify pill so it can update its label/color
        self.jobs_count_changed.emit(total, active)

    def _on_error(self, msg):
        self._summary_label.setText(f'⚠ {msg[:80]}')

    def _do_cancel(self, jid, name):
        if QMessageBox.question(self, 'Cancel Spark Job',
                                f'Cancel "{name}"?\n\nSends SIGTERM. Worst case you lose ≤500 steps.',
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        def _go():
            try:
                _sl().cancel_job(jid)
                self.poller.kick()
            except Exception as e:
                self._on_error(str(e))
        threading.Thread(target=_go, daemon=True).start()

    def _open_output(self, jid):
        def _go():
            try:
                detail = _sl().get_job(jid)
                url = (detail.get('output') or {}).get('shareSyncBaseUrl')
                if url:
                    import webbrowser
                    webbrowser.open(url)
                else:
                    path = detail.get('output_share_sync_path') or detail.get('outputShareSyncPath')
                    QMessageBox.information(self, 'Output',
                        f'ShareSync path:\n{path}\n\nOpen via Spark web UI or WebDAV.')
            except Exception as e:
                QMessageBox.warning(self, 'Output', str(e))
        threading.Thread(target=_go, daemon=True).start()


# ── Bottom-bar pill button ───────────────────────────────────────────────────

class SparkJobsPill(QPushButton):
    """
    Compact status pill in the bottom control panel:
        [☁ Spark Jobs · 2 active ▾]

    Click to toggle the drawer. Updates its label/color via update_count().
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._total = 0
        self._active = 0
        self._open = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(32)
        self._refresh_label()
        self._refresh_style()

    def update_count(self, total, active):
        self._total = total
        self._active = active
        self._refresh_label()
        self._refresh_style()

    def set_open(self, opened):
        self._open = opened
        self._refresh_label()

    def _refresh_label(self):
        chev = '▴' if self._open else '▾'
        if self._active > 0:
            self.setText(f'☁ Spark · {self._active} active  {chev}')
        elif self._total > 0:
            self.setText(f'☁ Spark · {self._total} jobs  {chev}')
        else:
            self.setText(f'☁ Spark Jobs  {chev}')

    def _refresh_style(self):
        if self._active > 0:
            # Active state — green dot via background
            self.setStyleSheet("""
                QPushButton {
                    background: #f0fdf4;
                    color: #16A34A;
                    border: 1px solid #bbf7d0;
                    border-radius: 16px;
                    padding: 0 14px;
                    font-weight: 600;
                    font-size: 10pt;
                }
                QPushButton:hover { background: #dcfce7; border-color: #86efac; }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: #F7F4FC;
                    color: #7E3AF2;
                    border: 1px solid #e9d5ff;
                    border-radius: 16px;
                    padding: 0 14px;
                    font-weight: 600;
                    font-size: 10pt;
                }
                QPushButton:hover { background: #ede9fe; border-color: #c084fc; }
            """)


# ── "Run on Spark" CTA + GPU dropdown ────────────────────────────────────────

class SparkLaunchButton(QWidget):
    """
    Composite widget that is the primary submit CTA:
        [☁ Run on Spark]  [▾ l40s ($)]

    The GPU dropdown is sticky (saved to settings). Clicking the main button
    triggers SparkPanelMixin._submit_to_spark() on the parent window.
    """
    submit_requested = Signal(str)   # gpu key

    GPU_LABELS = [
        ('a10',          'a10  · 1× A10 24GB · cheap'),
        ('l4',           'l4   · 1× L4 24GB'),
        ('l40s',         'l40s · 1× L40S 48GB · recommended'),
        ('a10x4',        'a10x4 · 4× A10 24GB · multi-GPU'),
        ('l40sx4',       'l40sx4 · 4× L40S 48GB · multi-GPU'),
        ('rtxpro6000',   'rtxpro6000 · 1× RTX PRO 6000 96GB · fastest'),
        ('rtxpro6000x8', 'rtxpro6000x8 · 8× RTX PRO 6000 · ludicrous $$'),
        ('t4',           't4   · 1× T4 16GB · slow'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.btn = QPushButton('☁  Run on Spark')
        self.btn.setMinimumWidth(170)
        self.btn.setMinimumHeight(36)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #ae69f4, stop:1 #c084fc);
                color: white; border: none;
                border-top-left-radius: 8px; border-bottom-left-radius: 8px;
                border-top-right-radius: 0; border-bottom-right-radius: 0;
                font-weight: 700; font-size: 10.5pt; padding: 0 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #9333ea, stop:1 #ae69f4);
            }
            QPushButton:disabled {
                background: #e9d5ff; color: #f3e8ff;
            }
        """)
        self.btn.clicked.connect(self._on_click)
        layout.addWidget(self.btn)

        self.gpu_combo = QComboBox()
        for k, label in self.GPU_LABELS:
            self.gpu_combo.addItem(label, k)
        # Default: l40s
        for i in range(self.gpu_combo.count()):
            if self.gpu_combo.itemData(i) == 'l40s':
                self.gpu_combo.setCurrentIndex(i)
                break
        self.gpu_combo.setMinimumHeight(36)
        self.gpu_combo.setMinimumWidth(220)
        self.gpu_combo.setStyleSheet("""
            QComboBox {
                background: #6C2BD9; color: white;
                border: none;
                border-top-left-radius: 0; border-bottom-left-radius: 0;
                border-top-right-radius: 8px; border-bottom-right-radius: 8px;
                padding: 0 12px; font-weight: 600; font-size: 10pt;
            }
            QComboBox:hover { background: #581c87; }
            QComboBox::drop-down { border: none; width: 22px; }
            QComboBox::down-arrow { image: none; }
            QComboBox QAbstractItemView {
                background: white; color: #111827;
                border: 1px solid #e5e7eb; border-radius: 8px;
                padding: 4px; selection-background-color: #F7F4FC;
                selection-color: #7E3AF2; font-weight: 500;
            }
        """)
        layout.addWidget(self.gpu_combo)

    def _on_click(self):
        gpu = self.gpu_combo.currentData()
        self.submit_requested.emit(gpu)

    def set_busy(self, busy):
        if busy:
            self.btn.setText('☁  Submitting…')
            self.btn.setEnabled(False)
            self.gpu_combo.setEnabled(False)
        else:
            self.btn.setText('☁  Run on Spark')
            self.btn.setEnabled(True)
            self.gpu_combo.setEnabled(True)

    def selected_gpu(self):
        return self.gpu_combo.currentData()

    def set_selected_gpu(self, gpu_key):
        for i in range(self.gpu_combo.count()):
            if self.gpu_combo.itemData(i) == gpu_key:
                self.gpu_combo.setCurrentIndex(i)
                return


# ── Submit-progress toast ────────────────────────────────────────────────────

class SubmitProgressDialog(QDialog):
    """
    Lightweight non-modal progress dialog shown during pack+upload.
    Closes automatically when submit completes; can be cancelled.
    """
    line  = Signal(str)
    done  = Signal(str, str)   # job_id, sharesync_url (both may be empty)
    fail  = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Submitting to Spark')
        self.resize(540, 320)
        self.setModal(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        self._title = QLabel('☁  Packing your training run…')
        self._title.setStyleSheet('font-weight:700; font-size:12pt; color:#111827;')
        root.addWidget(self._title)

        self._sub = QLabel('Source · data · config · checkpoints')
        self._sub.setStyleSheet('color:#6b7280; font-size:9pt;')
        root.addWidget(self._sub)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            'QPlainTextEdit { background:#1e1e2e; color:#cdd6f4; '
            'font-family:Consolas,monospace; font-size:9.5pt; '
            'border-radius:6px; padding:8px; }')
        root.addWidget(self._log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._close_btn = QPushButton('Close')
        self._close_btn.setEnabled(False)
        self._close_btn.setFixedWidth(80)
        self._close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

        self.line.connect(self._on_line)
        self.done.connect(self._on_done)
        self.fail.connect(self._on_fail)

    def _on_line(self, text):
        self._log.appendPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _on_done(self, jid, sharesync):
        self._title.setText('✓  Submitted to Spark')
        self._title.setStyleSheet('font-weight:700; font-size:12pt; color:#16A34A;')
        if jid:
            self._sub.setText(f'Job ID: {jid}')
        self._close_btn.setEnabled(True)

    def _on_fail(self, msg):
        self._title.setText('✗  Submit failed')
        self._title.setStyleSheet('font-weight:700; font-size:12pt; color:#EF4444;')
        self._sub.setText(msg[:120])
        self._close_btn.setEnabled(True)


# ── Mixin — wire everything into MainWindow ──────────────────────────────────

class SparkPanelMixin:
    """
    Mixin attached to MainWindow. Provides:
      - self._spark_drawer       — SparkJobDrawer instance
      - self._spark_pill         — SparkJobsPill instance
      - self._spark_launch_btn   — SparkLaunchButton instance

    Call from __init__ AFTER tabs are created and AFTER _create_control_panel
    has run — _build_spark_panel() injects widgets into the existing
    control-panel layout and adds the drawer to the left layout.
    """

    SPARK_SETTINGS_KEY = 'spark_panel'

    def _build_spark_panel(self, control_panel_layout, left_layout, drawer_index=1):
        """
        control_panel_layout : QHBoxLayout returned by _create_control_panel()
        left_layout          : QVBoxLayout — the parent we add the drawer to
        drawer_index         : where in left_layout to insert the drawer
        """
        # --- Launch CTA + GPU combo (right side of bottom bar) ---
        self._spark_launch_btn = SparkLaunchButton(self)
        self._spark_launch_btn.submit_requested.connect(self._submit_to_spark)
        # Insert before the existing Start Training button
        # control_panel_layout currently is: Load Save Monitor [stretch] Start Stop
        # We want:                            Load Save Monitor [stretch] [Pill] Start Stop ⌥CTA
        # Strategy: append after Stop, separated by a vertical line.
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet('color:#e5e7eb; margin: 0 4px;')
        sep.setFixedHeight(28)
        control_panel_layout.addWidget(sep)
        control_panel_layout.addWidget(self._spark_launch_btn)

        # --- Pill (left of stretch, so it lives near "secondary" actions) ---
        self._spark_pill = SparkJobsPill(self)
        self._spark_pill.clicked.connect(self._toggle_spark_drawer)
        # Insert at index 3, just before the addStretch.
        # control_panel_layout layout order = Load(0) Save(1) Monitor(2) [stretch](3) Start(4) Stop(5)
        # Insert at position 3 → before stretch.
        control_panel_layout.insertWidget(3, self._spark_pill)

        # --- Drawer (below the splitter, above the control panel) ---
        # left_layout currently has: splitter, control_panel
        # We want:                   splitter, drawer, control_panel
        self._spark_drawer = SparkJobDrawer(self)
        self._spark_drawer.log_requested.connect(self._open_spark_logs)
        self._spark_drawer.jobs_count_changed.connect(self._spark_pill.update_count)
        self._spark_drawer.poller.auth_required.connect(self._on_spark_auth_failed)

        # left_layout.count() === 2 at this point (splitter, control_panel)
        # Insert drawer at the requested index (between splitter and control_panel)
        left_layout.insertWidget(drawer_index, self._spark_drawer)

        # Restore saved GPU choice
        self._restore_spark_settings()

    # ── Drawer toggle ───────────────────────────────────────────────────────
    def _toggle_spark_drawer(self):
        self._spark_drawer.toggle()
        self._spark_pill.set_open(self._spark_drawer.is_open())

    def _open_spark_logs(self, jid, name):
        win = SparkLogWindow(jid, name=name, parent=self)
        win.show()

    def _on_spark_auth_failed(self):
        # Visual signal in the pill
        self._spark_pill.setText('☁ Spark · auth needed')

    # ── Submit ──────────────────────────────────────────────────────────────
    def _submit_to_spark(self, gpu_key):
        """Pull the live config from the form and ship it."""
        # Lazy-import deps — avoids a hard requirement at app launch
        try:
            sl = _sl()
        except ImportError as e:
            QMessageBox.critical(self, 'Spark', f'spark_launch.py not importable:\n{e}')
            return

        # Validate config: gather and ensure src/dst/output_dir set
        try:
            config = self.gather_config_from_ui()
        except Exception as e:
            QMessageBox.critical(self, 'Config error', f'Could not build config:\n{e}')
            return

        data = config.get('data') or {}
        if not data.get('src_dir') or not data.get('dst_dir'):
            QMessageBox.warning(self, 'Missing data dirs',
                'Set src_dir and dst_dir in the Data tab first.')
            return
        if not data.get('output_dir'):
            QMessageBox.warning(self, 'Missing output_dir',
                'Set output_dir in the Data tab first.')
            return

        # Confirm for expensive SKUs
        if gpu_key in ('rtxpro6000x8', 'l40sx4'):
            if QMessageBox.question(self, 'Expensive GPU',
                f'You selected {gpu_key} — this is a multi-GPU instance and bills accordingly.\n\n'
                f'Proceed with submission?',
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        self._save_spark_settings({'gpu': gpu_key})

        # Build progress dialog
        progress = SubmitProgressDialog(self)
        progress.show()

        self._spark_launch_btn.set_busy(True)

        def _redirect_print(msg):
            progress.line.emit(msg)

        def _run():
            class _StreamRedirect:
                def write(self, t):
                    if t.strip():
                        _redirect_print(t.rstrip('\n'))
                def flush(self): pass

            old_stdout = sys.stdout
            sys.stdout = _StreamRedirect()
            try:
                # 1. Rewrite paths for /input + /output, collect uploads
                data_uploads = {}
                pod_config, job_name, remote_output = sl.rewrite_config_for_spark(
                    config, data_uploads)

                # 2. Pack tarball
                tunet_root = os.path.dirname(os.path.abspath(__file__))
                # spark_tab.py lives at gui/spark_tab.py — go up one
                tunet_root = os.path.dirname(tunet_root)

                cfg_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.dump(pod_config, cfg_tmp, default_flow_style=False, sort_keys=False)
                cfg_tmp.close()
                tar_path = sl.build_input_tar(tunet_root, cfg_tmp.name, data_uploads)
                os.unlink(cfg_tmp.name)

                # 3. Submit
                instance_type = sl.GPU_TYPES[gpu_key]
                label = f'tunet-{job_name}'
                command = ['bash', '/input/spark_start.sh', remote_output, sl.CONFIG_REMOTE]

                print(f'[spark] Submitting {label} on {instance_type}…')
                resp = sl.submit_job(label, instance_type, sl.DEFAULT_IMAGE, command,
                                     idle_hold_seconds=0)
                jid = resp.get('jobId') or resp.get('id') or ''
                upload_url = (resp.get('input') or {}).get('uploadUrl')
                if not jid or not upload_url:
                    raise RuntimeError(f'Missing jobId/uploadUrl in submit response: {resp}')
                print(f'[spark] Job ID: {jid}')

                # 4. Upload tarball
                sl.upload_tarball(upload_url, tar_path)
                try:
                    os.unlink(tar_path)
                except Exception:
                    pass

                ss = (resp.get('output') or {}).get('shareSyncBaseUrl') or ''
                progress.done.emit(jid, ss)

                # 5. Pop drawer + open log window — back on GUI thread
                QTimer.singleShot(0, lambda jid=jid, name=label:
                                       self._on_spark_after_submit(jid, name))

            except SystemExit as e:
                progress.fail.emit(str(e))
            except Exception as e:
                tb = traceback.format_exc()
                _redirect_print(tb)
                progress.fail.emit(str(e))
            finally:
                sys.stdout = old_stdout
                # Re-enable CTA on the GUI thread
                QTimer.singleShot(0, lambda: self._spark_launch_btn.set_busy(False))

        threading.Thread(target=_run, daemon=True).start()

    def _on_spark_after_submit(self, jid, name):
        # Refresh job list, open drawer, open logs
        self._spark_drawer.poller.kick()
        if not self._spark_drawer.is_open():
            self._spark_drawer.open_drawer()
            self._spark_pill.set_open(True)
        # Auto-open log window
        win = SparkLogWindow(jid, name=name, parent=self)
        win.show()

    # ── Settings persistence ────────────────────────────────────────────────
    def _restore_spark_settings(self):
        try:
            import json
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '_spark_panel.json')
            path = os.path.abspath(path)
            if os.path.isfile(path):
                with open(path) as f:
                    s = json.load(f)
                gpu = s.get('gpu')
                if gpu:
                    self._spark_launch_btn.set_selected_gpu(gpu)
        except Exception:
            pass

    def _save_spark_settings(self, settings):
        try:
            import json
            path = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '_spark_panel.json'))
            existing = {}
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        existing = json.load(f) or {}
                except Exception:
                    pass
            existing.update(settings)
            with open(path, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception:
            pass
