"""
Spark Compute v1 Dashboard — submit jobs, watch logs, cancel.

Targets the Spark Compute v1 API (auto-prepare jobs, ShareSync I/O, SSE logs):
    POST   /api/compute/jobs                   submit
    GET    /api/compute/jobs                   list
    GET    /api/compute/jobs/:id               status
    POST   /api/compute/jobs/:id/cancel        SIGTERM
    GET    /api/compute/jobs/:id/logs/stream   SSE log stream
    GET    /api/compute/skus                   eligible instance types

This replaces the older workstation-based dashboard. Compute v1 is closer
to RunPod's model: one HTTP submit, the agent owns the EC2 lifecycle,
container runs in foreground, /output streams to ShareSync as it's written.

Usage:
    python spark_dashboard.py
"""

import json
import os
import sys
import threading
import time

import requests
import yaml
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QAbstractItemView, QFrame,
    QDialog, QFormLayout, QLineEdit, QComboBox,
    QPlainTextEdit, QFileDialog, QGroupBox, QSizePolicy,
)
from PySide6.QtGui import QColor, QFont, QTextCursor, QIcon, QPixmap, QPainter, QPen, QBrush, QPainterPath

import spark_launch as sl

SPARK_API = sl.SPARK_API


# ── Background poller ─────────────────────────────────────────────────────────

class JobPoller(QObject):
    refreshed = Signal(list)
    error     = Signal(str)

    def __init__(self, interval=10):
        super().__init__()
        self._interval = interval
        self._running  = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                self.refreshed.emit(sl.list_jobs())
            except SystemExit as e:
                self.error.emit(str(e))
            except Exception as e:
                self.error.emit(str(e))
            time.sleep(self._interval)


# ── Status helpers ────────────────────────────────────────────────────────────

# Spark job statuses observed: queued, provisioning, running, completed, failed, cancelled
STATUS_COLORS = {
    'queued':       '#1c64f2',
    'provisioning': '#1c64f2',
    'running':      '#16A34A',
    'completed':    '#6b7280',
    'failed':       '#EF4444',
    'cancelled':    '#9ca3af',
}

ACTIVE_STATUSES = {'queued', 'provisioning', 'running'}


def job_status(j):    return j.get('status') or '?'
def job_id(j):        return j.get('id') or j.get('jobId') or '?'
def job_instance(j):  return j.get('instance_type_name') or j.get('instanceType') or '?'

def job_name(j):
    # Spark API doesn't echo name back in list/detail response. Fall back to
    # a short job-id prefix so the table is still useful.
    n = j.get('name') or j.get('label')
    if n:
        return n
    jid = job_id(j)
    return jid[:8] if len(jid) > 8 else jid


def job_created(j):
    ts = j.get('created_at') or j.get('createdAt') or ''
    if not ts:
        return '--'
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime('%m/%d %H:%M')
    except Exception:
        return ts[:16]


def job_runtime(j):
    """Elapsed wall-clock — prefers started_running_at to skip provision time."""
    from datetime import datetime, timezone
    start = (j.get('started_running_at')
             or j.get('started_provisioning_at')
             or j.get('created_at')
             or j.get('createdAt'))
    end   = j.get('terminal_at') or j.get('finished_at') or j.get('completed_at')
    if not start:
        return '--'
    try:
        s = datetime.fromisoformat(start.replace('Z', '+00:00'))
        e = datetime.fromisoformat(end.replace('Z', '+00:00')) if end else datetime.now(timezone.utc)
        secs = max(0, int((e - s).total_seconds()))
        h, rem = divmod(secs, 3600)
        m, ss = divmod(rem, 60)
        if h:
            return f'{h}h {m:02d}m'
        if m:
            return f'{m}m {ss:02d}s'
        return f'{ss}s'
    except Exception:
        return '--'


# ── Log Window (SSE stream) ───────────────────────────────────────────────────

class LogWindow(QDialog):
    _line  = Signal(str)
    _ended = Signal()

    def __init__(self, jid, name='', parent=None):
        super().__init__(parent)
        self._jid = jid
        self.setWindowTitle(f'Logs — {name or jid}')
        self.resize(900, 600)
        self._stop = threading.Event()
        self._build_ui()
        self._line.connect(self._append)
        self._ended.connect(self._on_ended)
        threading.Thread(target=self._stream, daemon=True).start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f'<b>Job:</b> {self._jid}'))
        hdr.addStretch()
        cancel_btn = QPushButton('Cancel Job')
        cancel_btn.setStyleSheet(
            'QPushButton { background:#fef2f2; color:#EF4444; border:1px solid #fecaca; border-radius:6px; padding:4px 10px; font-size:12px; }'
            'QPushButton:hover { background:#fee2e2; }')
        cancel_btn.clicked.connect(self._do_cancel)
        hdr.addWidget(cancel_btn)
        close_btn = QPushButton('Close'); close_btn.setFixedWidth(70)
        close_btn.clicked.connect(self.accept)
        hdr.addWidget(close_btn)
        root.addLayout(hdr)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            'QPlainTextEdit { background:#0f172a; color:#cdd6f4; '
            'font-family:Consolas,monospace; font-size:11px; border-radius:6px; padding:8px; }')
        root.addWidget(self._log)

    def _stream(self):
        try:
            tok = sl.get_token()
            url = f'{SPARK_API}/api/compute/jobs/{self._jid}/logs/stream'
            with requests.get(url,
                              headers={'Authorization': f'Bearer {tok}',
                                       'Accept': 'text/event-stream'},
                              stream=True, timeout=None) as r:
                for line in r.iter_lines(decode_unicode=True):
                    if self._stop.is_set():
                        return
                    if not line:
                        continue
                    if line.startswith(':'):
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
        self._log.appendPlainText('\n[stream closed]')

    def _do_cancel(self):
        if QMessageBox.question(self, 'Cancel Job',
                                f'Send SIGTERM to {self._jid}?\nWorst case you lose ≤500 steps.',
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            sl.cancel_job(self._jid)
            self._append('\n[cancel requested]')
        except SystemExit as e:
            QMessageBox.warning(self, 'Cancel failed', str(e))

    def closeEvent(self, ev):
        self._stop.set()
        super().closeEvent(ev)


# ── Launch Dialog ─────────────────────────────────────────────────────────────

_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'spark_dashboard_settings.json')

def _load_settings():
    try:
        with open(_SETTINGS_FILE) as f: return json.load(f)
    except Exception: return {}

def _save_settings(s):
    try:
        with open(_SETTINGS_FILE, 'w') as f: json.dump(s, f, indent=2)
    except Exception: pass


class _LogStream(QObject):
    line = Signal(str)
    def write(self, text):
        if text.strip(): self.line.emit(text.rstrip('\n'))
    def flush(self): pass


class LaunchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Submit Compute Job')
        self.resize(640, 620)
        self._settings = _load_settings()
        self._submitting = False
        self._build_ui()
        self._restore()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        cfg_group = QGroupBox('Job Configuration')
        cfg_group.setStyleSheet('QGroupBox { font-weight:600; }')
        form = QFormLayout(cfg_group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(12); form.setVerticalSpacing(8)

        cfg_row = QHBoxLayout()
        self._cfg_edit = QLineEdit(); self._cfg_edit.setPlaceholderText('Path to config.yaml')
        self._cfg_edit.textChanged.connect(self._on_config_changed)
        cfg_row.addWidget(self._cfg_edit)
        browse = QPushButton('Browse'); browse.setFixedWidth(70)
        browse.clicked.connect(lambda: self._cfg_edit.setText(
            QFileDialog.getOpenFileName(self,'Config YAML','','YAML (*.yaml *.yml)')[0] or self._cfg_edit.text()))
        cfg_row.addWidget(browse)
        form.addRow('Config YAML:', cfg_row)

        self._gpu_combo = QComboBox()
        for k, sku in sl.GPU_TYPES.items():
            self._gpu_combo.addItem(f'{k}  ({sku})', k)
        form.addRow('GPU:', self._gpu_combo)

        self._image_edit = QLineEdit(sl.DEFAULT_IMAGE)
        form.addRow('Image:', self._image_edit)

        self._idle_edit = QLineEdit('0')
        self._idle_edit.setPlaceholderText('seconds (0 = stop immediately)')
        form.addRow('Idle hold (s):', self._idle_edit)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText('(auto: tunet-<job_name>)')
        form.addRow('Job name:', self._name_edit)

        self._data_info = QLabel('— will be read from config src_dir / dst_dir —')
        self._data_info.setStyleSheet('color:#6b7280; font-size:11px; font-style:italic;')
        form.addRow('Data:', self._data_info)

        ckpt_row = QHBoxLayout()
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._ckpt_info = QLabel(''); self._ckpt_info.setStyleSheet('color:#6b7280; font-size:11px;')
        ckpt_row.addWidget(self._ckpt_combo); ckpt_row.addWidget(self._ckpt_info)
        form.addRow('Checkpoint:', ckpt_row)

        root.addWidget(cfg_group)

        btn_row = QHBoxLayout()
        self._submit_btn = QPushButton('Submit Job')
        self._submit_btn.setFixedHeight(34)
        self._submit_btn.setStyleSheet(
            'QPushButton { background:#ae69f4; color:white; border:none; border-radius:6px; font-weight:600; font-size:13px; }'
            'QPushButton:hover { background:#9b50e8; }'
            'QPushButton:disabled { background:#d8b4fe; color:#f3e8ff; }')
        self._submit_btn.clicked.connect(self._do_submit)
        self._close_btn = QPushButton('Close'); self._close_btn.setFixedHeight(34); self._close_btn.setFixedWidth(80)
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._submit_btn); btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

        log_group = QGroupBox('Submit Log'); log_group.setStyleSheet('QGroupBox { font-weight:600; }')
        log_vbox = QVBoxLayout(log_group)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); self._log.setMinimumHeight(180)
        self._log.setStyleSheet('QPlainTextEdit { background:#1e1e2e; color:#cdd6f4; font-family:Consolas,monospace; font-size:12px; border-radius:6px; padding:8px; }')
        log_vbox.addWidget(self._log)
        root.addWidget(log_group)

    def _on_config_changed(self, path):
        if not os.path.isfile(path): return
        try:
            with open(path) as f: cfg = yaml.safe_load(f)
            data_sect = cfg.get('data') or {}
            src = data_sect.get('src_dir',''); dst = data_sect.get('dst_dir','')
            parts = []
            if src: parts.append(f'src: {os.path.basename(src.rstrip("/\\"))}')
            if dst: parts.append(f'dst: {os.path.basename(dst.rstrip("/\\"))}')
            if parts:
                self._data_info.setText('Auto-upload: ' + '  |  '.join(parts))
                self._data_info.setStyleSheet('color:#16A34A; font-size:11px;')

            out_dir = data_sect.get('output_dir','')
            self._ckpt_combo.clear()
            self._ckpt_combo.addItem('Auto (latest checkpoint, or fresh)', None)
            self._ckpt_combo.addItem('Train fresh (ignore checkpoints)', '__fresh__')
            if out_dir and os.path.isdir(out_dir):
                import runpod_launch as rl
                ckpts = rl.scan_checkpoints(out_dir)
                for ckpt_path, is_latest in ckpts:
                    label = os.path.basename(ckpt_path)
                    size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
                    self._ckpt_combo.addItem(
                        f'{label}{" [latest]" if is_latest else ""}  ({size_mb:.0f} MB)',
                        ckpt_path)
                self._ckpt_info.setText(
                    f'{len(ckpts)} checkpoint{"s" if len(ckpts)!=1 else ""} found'
                    if ckpts else 'no .pth found — will train fresh')
        except Exception:
            pass

    def _restore(self):
        s = self._settings
        if s.get('config'): self._cfg_edit.setText(s['config'])
        if s.get('gpu'):
            for i in range(self._gpu_combo.count()):
                if self._gpu_combo.itemData(i) == s['gpu']:
                    self._gpu_combo.setCurrentIndex(i); break
        if s.get('image'): self._image_edit.setText(s['image'])
        if s.get('idle_hold') is not None: self._idle_edit.setText(str(s['idle_hold']))

    def _log_line(self, text):
        self._log.appendPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _do_submit(self):
        if self._submitting: return
        config_path = self._cfg_edit.text().strip()
        if not config_path or not os.path.isfile(config_path):
            QMessageBox.warning(self, 'No Config', 'Select a valid config.yaml.'); return
        gpu = self._gpu_combo.currentData()
        image = self._image_edit.text().strip() or sl.DEFAULT_IMAGE
        try:
            idle_hold = int(self._idle_edit.text().strip() or '0')
        except ValueError:
            idle_hold = 0
        name = self._name_edit.text().strip() or None
        resume = self._ckpt_combo.currentData()

        _save_settings({'config': config_path, 'gpu': gpu, 'image': image,
                        'idle_hold': idle_hold})

        self._submitting = True
        self._submit_btn.setEnabled(False)
        self._close_btn.setText('Working...'); self._close_btn.setEnabled(False)
        self._log.clear()

        stream = _LogStream(); stream.line.connect(self._log_line)

        def _run():
            old_stdout = sys.stdout
            sys.stdout = stream
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                data_uploads = {}
                resume_pth = None if resume in (None, '__fresh__') else resume
                pod_config, job_name_, remote_output = sl.rewrite_config_for_spark(
                    config, data_uploads, resume_pth=resume_pth)
                if resume == '__fresh__':
                    data_uploads = {k: v for k, v in data_uploads.items()
                                    if not (k.endswith('.pth') or k.endswith('.log'))}

                import tempfile
                tunet_root = os.path.dirname(os.path.abspath(__file__))
                cfg_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.dump(pod_config, cfg_tmp, default_flow_style=False, sort_keys=False)
                cfg_tmp.close()
                tar_path = sl.build_input_tar(tunet_root, cfg_tmp.name, data_uploads)
                os.unlink(cfg_tmp.name)

                instance_type = sl.GPU_TYPES[gpu]
                label = name or f'tunet-{job_name_}'
                command = ['bash', '/input/spark_start.sh', remote_output, sl.CONFIG_REMOTE]

                print(f'[spark] Submitting {label} on {instance_type}...')
                resp = sl.submit_job(label, instance_type, image, command, idle_hold)
                jid = resp.get('jobId') or resp.get('id')
                upload_url = (resp.get('input') or {}).get('uploadUrl')
                print(f'[spark] Job ID: {jid}')

                sl.upload_tarball(upload_url, tar_path)
                os.unlink(tar_path)
                print(f'\n✓ Submitted. Click "Logs" in the dashboard to stream output.')
                ss = (resp.get('output') or {}).get('shareSyncBaseUrl') or resp.get('outputShareSyncPath')
                if ss:
                    print(f'  ShareSync: {ss}')

            except SystemExit as e:
                print(f'\n{e}')
            except Exception as e:
                print(f'\nERROR: {e}')
                import traceback; traceback.print_exc(file=stream)
            finally:
                sys.stdout = old_stdout
                self._submitting = False
                self._submit_btn.setEnabled(True)
                self._close_btn.setText('Close'); self._close_btn.setEnabled(True)

        threading.Thread(target=_run, daemon=True).start()


# ── Main Window ───────────────────────────────────────────────────────────────

COLS = ['Name', 'Status', 'GPU', 'Started', 'Runtime', 'Actions']


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TuNet — Spark Compute Dashboard')
        self.resize(1100, 560)
        self._jobs = []
        self._build_ui()
        self._poller = JobPoller(interval=10)
        self._poller.refreshed.connect(self._on_refresh)
        self._poller.error.connect(self._on_error)
        self._poller.start()
        QTimer.singleShot(100, lambda: threading.Thread(
            target=self._first_fetch, daemon=True).start())

    def _first_fetch(self):
        try:
            self._poller.refreshed.emit(sl.list_jobs())
        except SystemExit as e:
            self._poller.error.emit(str(e))
        except Exception as e:
            self._poller.error.emit(str(e))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        hdr = QHBoxLayout()
        title = QLabel('Spark Compute')
        title.setFont(QFont('Plus Jakarta Sans', 16, QFont.Bold))
        hdr.addWidget(title)
        hdr.addStretch()

        self._status_label = QLabel('Refreshing...')
        self._status_label.setStyleSheet('color:#6b7280; font-size:12px;')
        hdr.addWidget(self._status_label)

        submit_btn = QPushButton('+ Submit Job')
        submit_btn.setFixedWidth(120)
        submit_btn.setStyleSheet(
            'QPushButton { background:#ae69f4; color:white; border:none; border-radius:6px; padding:5px 12px; font-size:12px; font-weight:600; }'
            'QPushButton:hover { background:#9b50e8; }')
        submit_btn.clicked.connect(self._open_launch)
        hdr.addWidget(submit_btn)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(lambda: threading.Thread(
            target=self._first_fetch, daemon=True).start())
        hdr.addWidget(refresh_btn)
        layout.addLayout(hdr)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setStyleSheet('color:#e5e7eb;')
        layout.addWidget(line)

        self._table = QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(COLS)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(COLS) - 1):
            self._table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(len(COLS) - 1, QHeaderView.Fixed)
        self._table.setColumnWidth(len(COLS) - 1, 240)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setStyleSheet("""
            QTableWidget { border:1px solid #e5e7eb; border-radius:8px; background:white; font-size:13px; }
            QTableWidget::item { padding:8px 12px; }
            QHeaderView::section { background:#F9FAFB; border:none; border-bottom:1px solid #e5e7eb; padding:8px 12px; font-weight:600; color:#374151; }
            QTableWidget::item:alternate { background:#F9FAFB; }
        """)
        layout.addWidget(self._table)

        self._footer = QLabel('No jobs yet.')
        self._footer.setStyleSheet('color:#6b7280; font-size:12px;')
        layout.addWidget(self._footer)

        self.setStyleSheet("""
            QMainWindow, QWidget { background:#F9FAFB; color:#111827; font-family:'Plus Jakarta Sans','Segoe UI',sans-serif; }
            QPushButton { background:white; border:1px solid #e5e7eb; border-radius:6px; padding:5px 12px; font-size:12px; color:#374151; }
            QPushButton:hover { border-color:#ae69f4; color:#ae69f4; }
            QPushButton:disabled { color:#d1d5db; border-color:#f3f4f6; }
            QPushButton#danger { color:#EF4444; border-color:#fecaca; }
            QPushButton#danger:hover { background:#fef2f2; }
            QPushButton#watch { color:#ae69f4; border-color:#e9d5ff; }
            QPushButton#watch:hover { background:#F7F4FC; }
        """)

    def _on_refresh(self, jobs):
        self._jobs = jobs
        self._table.setRowCount(0)
        active = [j for j in jobs if job_status(j) in ACTIVE_STATUSES]
        self._status_label.setText(
            f'Last updated: {time.strftime("%H:%M:%S")}  ·  {len(active)} active')

        for row, j in enumerate(jobs):
            self._table.insertRow(row)
            self._table.setRowHeight(row, 44)
            status = job_status(j)

            def cell(text, align=Qt.AlignLeft | Qt.AlignVCenter):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(align)
                return item

            self._table.setItem(row, 0, cell(job_name(j)))

            status_item = cell(status, Qt.AlignCenter | Qt.AlignVCenter)
            status_item.setForeground(QColor(STATUS_COLORS.get(status, '#374151')))
            self._table.setItem(row, 1, status_item)

            self._table.setItem(row, 2, cell(job_instance(j)))
            self._table.setItem(row, 3, cell(job_created(j)))
            self._table.setItem(row, 4, cell(job_runtime(j)))

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(8, 4, 8, 4)
            btn_layout.setSpacing(6)

            is_active = status in ACTIVE_STATUSES

            logs_btn = QPushButton('Logs')
            logs_btn.setObjectName('watch')
            logs_btn.setFixedWidth(60)
            logs_btn.clicked.connect(lambda _, jj=j: self._open_logs(jj))

            cancel_btn = QPushButton('Cancel')
            cancel_btn.setObjectName('danger')
            cancel_btn.setFixedWidth(60)
            cancel_btn.setEnabled(is_active)
            cancel_btn.clicked.connect(lambda _, jj=j: self._do_cancel(jj))

            sharesync_btn = QPushButton('Output')
            sharesync_btn.setFixedWidth(70)
            sharesync_btn.clicked.connect(lambda _, jj=j: self._open_sharesync(jj))

            for b in (logs_btn, cancel_btn, sharesync_btn):
                btn_layout.addWidget(b)
            self._table.setCellWidget(row, 5, btn_widget)

        self._footer.setText(
            f'{len(jobs)} job{"s" if len(jobs) != 1 else ""}  ·  '
            f'{len(active)} active')

    def _on_error(self, msg):
        self._status_label.setText(f'Error: {msg[:100]}')

    def _open_logs(self, j):
        win = LogWindow(job_id(j), name=job_name(j), parent=self)
        win.show()

    def _do_cancel(self, j):
        jid = job_id(j); name = job_name(j)
        if QMessageBox.question(self, 'Cancel Job',
                                f'Cancel "{name}" ({jid})?\n\nSends SIGTERM. Worst case you lose ≤500 steps.',
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        def _go():
            try:
                sl.cancel_job(jid)
                self._first_fetch()
            except SystemExit as e:
                self._poller.error.emit(str(e))
        threading.Thread(target=_go, daemon=True).start()

    def _open_sharesync(self, j):
        def _go():
            try:
                detail = sl.get_job(job_id(j))
                url = (detail.get('output') or {}).get('shareSyncBaseUrl')
                if not url:
                    path = detail.get('output_share_sync_path') or detail.get('outputShareSyncPath')
                    QMessageBox.information(self, 'Output',
                        f'ShareSync path:\n{path}\n\nOpen via Spark web UI or WebDAV.')
                    return
                import webbrowser
                webbrowser.open(url)
            except Exception as e:
                QMessageBox.warning(self, 'Output', str(e))
        threading.Thread(target=_go, daemon=True).start()

    def _open_launch(self):
        dlg = LaunchDialog(self)
        dlg.exec()
        threading.Thread(target=self._first_fetch, daemon=True).start()


# ── App icon ──────────────────────────────────────────────────────────────────

def _make_icon():
    size = 256; px = QPixmap(size, size); px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    purple = QColor('#7B3FE4'); cloud_color = QColor('#c4a8f5'); stroke = 10; radius = 48
    p.setPen(QPen(purple, stroke)); p.setBrush(QBrush(QColor('white')))
    bg = QPainterPath(); bg.addRoundedRect(stroke/2, stroke/2, size-stroke, size-stroke, radius, radius)
    p.drawPath(bg)
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(cloud_color))
    cloud = QPainterPath(); cloud.addRect(52, 178, 154, 42)
    for cx, cy, r in [(82,168,34),(118,152,42),(158,160,36),(60,185,26),(198,183,24)]:
        bump = QPainterPath(); bump.addEllipse(cx-r, cy-r, r*2, r*2); cloud = cloud.united(bump)
    clip = QPainterPath(); clip.addRoundedRect(stroke, stroke, size-stroke*2, size-stroke*2, radius-2, radius-2)
    p.setClipPath(clip); p.drawPath(cloud); p.setClipping(False)
    p.setPen(QPen(purple)); font = QFont('Arial', 1, QFont.Weight.Bold); font.setPixelSize(int(size*0.72))
    p.setFont(font); p.drawText(px.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, 'T')
    p.end(); return QIcon(px)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('TuNet Spark Compute')
    app.setWindowIcon(_make_icon())
    win = Dashboard()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
