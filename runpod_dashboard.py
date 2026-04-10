"""
RunPod Dashboard — list running pods, kill them, watch training, launch jobs.

Usage:
    python runpod_dashboard.py
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import requests
import yaml
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QAbstractItemView, QFrame, QSizePolicy,
    QDialog, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QFileDialog, QGroupBox, QScrollArea,
)
from PySide6.QtGui import QColor, QFont, QTextCursor, QIcon, QPixmap, QPainter, QPen, QBrush, QPainterPath

# ── RunPod API ────────────────────────────────────────────────────────────────

RUNPOD_API  = 'https://api.runpod.io/graphql'
MONITOR_PORT = 8080

LIST_PODS_Q = """
query {
  myself {
    pods {
      id
      name
      desiredStatus
      costPerHr
      gpuCount
      imageName
      uptimeSeconds
      lastStartedAt
      runtime {
        ports { ip isIpPublic privatePort publicPort type }
        gpus  { id gpuUtilPercent }
      }
      machine { gpuDisplayName }
    }
  }
}
"""

TERMINATE_Q = """
mutation TerminatePod($input: PodTerminateInput!) {
  podTerminate(input: $input)
}
"""

STOP_Q = """
mutation StopPod($input: PodStopInput!) {
  podStop(input: $input) { id desiredStatus }
}
"""


def _api_key():
    return os.environ.get('RUNPOD_API_KEY') or os.environ.get('runpodapi') or ''


def gql(query, variables=None):
    resp = requests.post(
        RUNPOD_API,
        json={'query': query, 'variables': variables or {}},
        headers={'Authorization': f'Bearer {_api_key()}'},
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f'API {resp.status_code}: {resp.text[:300]}')
    data = resp.json()
    if 'errors' in data:
        raise RuntimeError(json.dumps(data['errors']))
    return data['data']


def list_pods():
    data = gql(LIST_PODS_Q)
    return data['myself']['pods']


def terminate_pod(pod_id):
    gql(TERMINATE_Q, {'input': {'podId': pod_id}})


def stop_pod(pod_id):
    gql(STOP_Q, {'input': {'podId': pod_id}})


def get_ssh(pod):
    rt = pod.get('runtime') or {}
    for p in rt.get('ports') or []:
        if p.get('privatePort') == 22 and p.get('isIpPublic'):
            return p['ip'], p['publicPort']
    return None, None


def _pod_uptime_seconds(pod):
    """Compute uptime from lastStartedAt ISO timestamp (more reliable than uptimeSeconds)."""
    import datetime
    started = pod.get('lastStartedAt')
    if started:
        try:
            # RunPod returns e.g. "2026-04-09T18:00:00.000Z"
            ts = started.rstrip('Z')
            dt = datetime.datetime.fromisoformat(ts).replace(tzinfo=datetime.timezone.utc)
            return max(0, (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds())
        except Exception:
            pass
    return pod.get('uptimeSeconds') or 0


def fmt_uptime(secs):
    if not secs:
        return '--'
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    if h:
        return f'{h}h {m:02d}m'
    return f'{m}m {s:02d}s'


def fmt_cost(pod):
    cph = pod.get('costPerHr') or 0
    up  = _pod_uptime_seconds(pod)
    spent = cph * up / 3600
    return f'${cph:.2f}/hr  (${spent:.2f} spent)'


# ── Background poller ─────────────────────────────────────────────────────────

class Poller(QObject):
    refreshed = Signal(list)
    error     = Signal(str)

    def __init__(self, interval=15):
        super().__init__()
        self._interval = interval
        self._running  = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                pods = list_pods()
                self.refreshed.emit(pods)
            except Exception as e:
                self.error.emit(str(e))
            time.sleep(self._interval)


# ── Monitor launcher ──────────────────────────────────────────────────────────

SSH_KEY = os.path.expanduser('~/.ssh/runpod')


def open_pod_console(pod):
    """Open a new terminal window with an interactive SSH session into the pod."""
    host, port = get_ssh(pod)
    if not host:
        QMessageBox.warning(None, 'No SSH', 'Pod has no public SSH port yet.')
        return

    ssh_args = ['ssh', '-o', 'StrictHostKeyChecking=no',
                '-i', SSH_KEY, '-p', str(port), f'root@{host}']

    try:
        subprocess.Popen(['wt', '--'] + ssh_args)
    except FileNotFoundError:
        # Fall back to cmd
        subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k'] + ssh_args)


def launch_monitor_for_pod(pod):
    """
    SSH-tails the training.log from the pod into a temp file,
    then opens training_monitor.py watching that file.
    Retries until training.log appears (deps may still be installing).
    """
    host, port = get_ssh(pod)
    if not host:
        QMessageBox.warning(None, 'No SSH', 'Pod has no public SSH port yet.')
        return

    pod_id  = pod['id']
    tmp_log = os.path.join(tempfile.gettempdir(), f'tunet_{pod_id}.log')

    # Open the monitor immediately — it will show "waiting" until log data arrives
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_monitor.py')
    subprocess.Popen([sys.executable, script, '--log_file', tmp_log])

    def _tail():
        # Single persistent SSH connection — the shell loop runs on the pod side.
        # No inner quotes so Windows doesn't mangle them (pod paths have no spaces).
        remote_cmd = (
            "until LOG=$(find /workspace/output -name training.log 2>/dev/null | head -1)"
            " && test -n \"$LOG\"; do sleep 3; done;"
            " cat $LOG; tail -f -n 0 $LOG"
        )
        with open(tmp_log, 'w') as out:
            subprocess.Popen(
                ['ssh', '-o', 'StrictHostKeyChecking=no',
                 '-i', SSH_KEY, '-p', str(port), f'root@{host}', remote_cmd],
                stdout=out, stderr=subprocess.DEVNULL,
            )

    threading.Thread(target=_tail, daemon=True).start()


# ── Training analysis (ported from training_monitor.py) ──────────────────────

def _compute_analysis(steps, losses):
    """Return analysis dict or None if not enough data."""
    if not steps or len(steps) < 10:
        return None
    current_epoch = steps[-1]
    if current_epoch < 5:
        return None

    window_epochs = min(20, current_epoch * 0.5)
    cutoff = current_epoch - window_epochs
    pairs = [(s, l) for s, l in zip(steps, losses) if s >= cutoff]
    if len(pairs) < 10:
        return None
    w_steps, w_losses = zip(*pairs)
    w_steps, w_losses = list(w_steps), list(w_losses)

    # EMA smoothing
    smoothed, alpha, last = [], 0.95, w_losses[0]
    for v in w_losses:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)

    # Linear regression slope (relative)
    n = len(smoothed)
    xm, ym = sum(w_steps) / n, sum(smoothed) / n
    ss_xy = sum((x - xm) * (y - ym) for x, y in zip(w_steps, smoothed))
    ss_xx = sum((x - xm) ** 2 for x in w_steps)
    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    rel_slope = slope / smoothed[-1] if smoothed[-1] > 0 else 0

    # Half-window improvement %
    mid = len(smoothed) // 2
    f_avg = sum(smoothed[:mid]) / mid
    s_avg = sum(smoothed[mid:]) / (n - mid)
    pct = ((s_avg - f_avg) / f_avg * 100) if f_avg > 0 else 0

    # Best epoch
    best_idx = losses.index(min(losses))
    best_epoch = steps[best_idx]
    since_best = current_epoch - best_epoch

    if rel_slope < -0.005:   trend, tc = 'Improving', '#22c55e'
    elif rel_slope > 0.005:  trend, tc = 'Diverging', '#ef4444'
    else:                    trend, tc = 'Flat',      '#f59e0b'

    ic = '#22c55e' if pct < -1 else ('#ef4444' if pct > 1 else '#f59e0b')

    if since_best < 10:    pc, pl = f'Best {since_best:.0f}ep ago', '#22c55e'
    elif since_best < 30:  pc, pl = f'Best {since_best:.0f}ep ago', '#f59e0b'
    else:                  pc, pl = f'Best {since_best:.0f}ep ago', '#ef4444'

    if   rel_slope < -0.005 and since_best < 20: rec, rc = 'Training well',       '#22c55e'
    elif rel_slope > 0.01:                        rec, rc = 'Diverging - check LR','#ef4444'
    elif since_best > 50:                         rec, rc = 'Consider stopping',   '#ef4444'
    elif since_best > 30 or abs(rel_slope) < 0.001: rec, rc = 'Plateau - may stop','#f59e0b'
    elif rel_slope < -0.001:                      rec, rc = 'Slow progress',       '#f59e0b'
    else:                                         rec, rc = 'Stable',              '#6b7280'

    return dict(trend=trend, trend_color=tc,
                pct=f'{pct:+.1f}%', pct_color=ic,
                plateau=pc, plateau_color=pl,
                rec=rec, rec_color=rc,
                best_epoch=best_epoch)


# ── Training stats bar widget ─────────────────────────────────────────────────

_MONO = "font-family: 'Consolas', 'Fira Code', monospace;"

class TrainingStatsBar(QWidget):
    _data_ready  = Signal(object, object)
    _fetch_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pod_id   = None
        self._pod_name = '--'
        self._timer    = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(30_000)
        self._data_ready.connect(self._apply)
        self._fetch_error.connect(self._on_fetch_error)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Object name so stylesheet scopes only this widget, not children
        self.setObjectName('statsBar')
        self.setStyleSheet('''
            #statsBar {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
            QLabel { background: transparent; color: #f8fafc; }
        ''')

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 10, 16, 10)
        root.setSpacing(0)

        # ── Row 1: analysis ───────────────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(0)

        def dim(text):
            l = QLabel(text)
            l.setStyleSheet('color:#475569; font-size:11px;')
            return l

        row1.addWidget(dim('Analysis'))
        row1.addSpacing(16)

        self._analysis_fields = {}
        analysis_defs = [
            ('trend',   'Trend'),
            ('pct',     'Recent Change'),
            ('plateau', 'Plateau Check'),
            ('rec',     'Status'),
        ]
        for i, (key, caption) in enumerate(analysis_defs):
            if i > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet('color: #1e293b;')
                sep.setFixedWidth(1)
                row1.addSpacing(16)
                row1.addWidget(sep)
                row1.addSpacing(16)
            cap = dim(caption)
            val = QLabel('--')
            val.setStyleSheet('color:#475569; font-size:11px; font-weight:600;')
            row1.addWidget(cap)
            row1.addSpacing(5)
            row1.addWidget(val)
            self._analysis_fields[key] = val

        row1.addStretch()
        root.addLayout(row1)

        # ── Separator ─────────────────────────────────────────────────────────
        root.addSpacing(8)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet('background: #1e293b; border: none;')
        line.setFixedHeight(1)
        root.addWidget(line)
        root.addSpacing(8)

        # ── Row 2: metrics ────────────────────────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(0)

        self._metric_vals = {}
        metric_defs = [
            ('run',        'Run',        False),
            ('epoch',      'Epoch',      True),
            ('loss',       'Loss',       True),
            ('best_loss',  'Best Loss',  True),
            ('best_ep',    'Best @Ep',   True),
            ('val_loss',   'Val Loss',   True),
            ('val_best',   'Val Best',   True),
            ('lpips',      'LPIPS',      True),
            ('lpips_best', 'LPIPS Best', True),
            ('psnr',       'PSNR (dB)',  True),
            ('ssim',       'SSIM',       True),
            ('pts',        'Data Pts',   True),
            ('step_time',  'Step Time',  True),
        ]
        for i, (key, caption, is_mono) in enumerate(metric_defs):
            if i > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet('color: #1e293b;')
                sep.setFixedWidth(1)
                row2.addSpacing(16)
                row2.addWidget(sep)
                row2.addSpacing(16)

            col = QVBoxLayout()
            col.setSpacing(2)

            cap = QLabel(caption)
            cap.setStyleSheet('color:#475569; font-size:10px; font-weight:500; letter-spacing: 0.05em;')

            val = QLabel('--')
            mono_part = _MONO if is_mono else ''
            val.setStyleSheet(f'color:#f8fafc; font-size:12px; font-weight:600; {mono_part}')

            col.addWidget(cap)
            col.addWidget(val)
            row2.addLayout(col)
            self._metric_vals[key] = val

        row2.addStretch()
        root.addLayout(row2)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_pod(self, pod_id, pod_name=''):
        self._pod_id   = pod_id
        self._pod_name = pod_name or pod_id or '--'
        # Immediately show run name; other fields fill in when poll succeeds
        self._metric_vals['run'].setText(self._pod_name)
        self._metric_vals['run'].setStyleSheet('color:#a78bfa; font-size:12px; font-weight:700;')
        for key in ('trend', 'pct', 'plateau', 'rec'):
            self._analysis_fields[key].setText('Connecting...')
            self._analysis_fields[key].setStyleSheet('color:#475569; font-size:11px; font-weight:600;')
        self._poll()

    def clear(self):
        self._pod_id = None
        dim = 'color:#475569; font-size:11px; font-weight:600;'
        for v in self._analysis_fields.values():
            v.setText('--')
            v.setStyleSheet(dim)
        for v in self._metric_vals.values():
            v.setText('--')

    # ── Polling ───────────────────────────────────────────────────────────────
    def _poll(self):
        if not self._pod_id:
            return
        pod_id = self._pod_id
        def _fetch():
            url = f'https://{pod_id}-{MONITOR_PORT}.proxy.runpod.net'
            try:
                status  = requests.get(f'{url}/api/status',  timeout=5).json()
                metrics = requests.get(f'{url}/api/metrics', timeout=5).json()
                self._data_ready.emit(status, metrics)
            except Exception as e:
                self._fetch_error.emit(str(e))
        threading.Thread(target=_fetch, daemon=True).start()

    # ── Apply data (main thread) ──────────────────────────────────────────────
    def _apply(self, status, metrics):
        steps      = metrics.get('steps',     [])
        losses     = metrics.get('l1',        [])
        val_l1     = metrics.get('val_l1',    [])
        val_psnr   = metrics.get('val_psnr',  [])
        val_ssim   = metrics.get('val_ssim',  [])
        lpips_list = metrics.get('lpips',     [])

        def fmt(v, d=5):
            return f'{v:.{d}f}' if v is not None else '--'

        best_ep = '--'
        if steps and losses:
            bi = losses.index(min(losses))
            best_ep = f'{steps[bi]:.1f}'

        run_name = self._pod_name
        self._metric_vals['run'       ].setText(run_name)
        self._metric_vals['run'       ].setStyleSheet(
            'color:#a78bfa; font-size:12px; font-weight:700;')
        self._metric_vals['epoch'     ].setText(f'{steps[-1]:.2f}' if steps else '--')
        self._metric_vals['loss'      ].setText(fmt(status.get('loss')))
        self._metric_vals['best_loss' ].setText(fmt(status.get('best_loss')))
        self._metric_vals['best_ep'   ].setText(best_ep)
        self._metric_vals['val_loss'  ].setText(fmt(val_l1[-1])      if val_l1      else '--')
        self._metric_vals['val_best'  ].setText(fmt(min(val_l1))     if val_l1      else '--')
        self._metric_vals['lpips'     ].setText(fmt(lpips_list[-1])  if lpips_list  else '--')
        self._metric_vals['lpips_best'].setText(fmt(min(lpips_list)) if lpips_list  else '--')
        self._metric_vals['psnr'      ].setText(f'{val_psnr[-1]:.2f}' if val_psnr  else '--')
        self._metric_vals['ssim'      ].setText(f'{val_ssim[-1]:.4f}' if val_ssim  else '--')
        self._metric_vals['pts'       ].setText(str(len(steps))      if steps       else '--')
        st = status.get('step_time_s')
        self._metric_vals['step_time' ].setText(f'{st:.3f}s'         if st          else '--')

        # Analysis
        analysis = _compute_analysis(steps, losses)
        if analysis:
            for key in ('trend', 'pct', 'plateau', 'rec'):
                v = self._analysis_fields[key]
                v.setText(analysis[key])
                v.setStyleSheet(
                    f'color:{analysis[key+"_color"]}; font-size:11px; font-weight:600;')
        else:
            for v in self._analysis_fields.values():
                v.setText('Collecting...')
                v.setStyleSheet('color:#475569; font-size:11px; font-weight:600;')

    def _on_fetch_error(self, _err):
        for key in ('trend', 'pct', 'plateau', 'rec'):
            self._analysis_fields[key].setText('Unreachable')
            self._analysis_fields[key].setStyleSheet('color:#ef4444; font-size:11px; font-weight:600;')


# ── Main Window ───────────────────────────────────────────────────────────────

COLS = ['Name', 'GPU', 'Status', 'Uptime', 'Cost', 'GPU Util', 'Actions']

STATUS_COLORS = {
    'RUNNING': '#16A34A',
    'EXITED':  '#6b7280',
    'DEAD':    '#EF4444',
    'FAILED':  '#EF4444',
}


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TuNet — RunPod Dashboard')
        self.resize(1000, 500)
        self._pods = []
        self._build_ui()
        self._poller = Poller(interval=15)
        self._poller.refreshed.connect(self._on_refresh)
        self._poller.error.connect(self._on_error)
        self._poller.start()
        QTimer.singleShot(100, lambda: threading.Thread(
            target=self._first_fetch, daemon=True).start())

    def _first_fetch(self):
        try:
            pods = list_pods()
            self._poller.refreshed.emit(pods)
        except Exception as e:
            self._poller.error.emit(str(e))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = QLabel('RunPod Dashboard')
        title.setFont(QFont('Plus Jakarta Sans', 16, QFont.Bold))
        hdr.addWidget(title)
        hdr.addStretch()

        self._status_label = QLabel('Refreshing...')
        self._status_label.setStyleSheet('color: #6b7280; font-size: 12px;')
        hdr.addWidget(self._status_label)

        launch_btn = QPushButton('+ Launch Job')
        launch_btn.setFixedWidth(110)
        launch_btn.setStyleSheet(
            'QPushButton { background: #ae69f4; color: white; border: none; border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: 600; }'
            'QPushButton:hover { background: #9b50e8; }'
        )
        launch_btn.clicked.connect(self._open_launch)
        hdr.addWidget(launch_btn)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(lambda: threading.Thread(
            target=self._first_fetch, daemon=True).start())
        hdr.addWidget(refresh_btn)
        layout.addLayout(hdr)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet('color: #e5e7eb;')
        layout.addWidget(line)

        # Table
        self._table = QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(COLS)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        self._table.setColumnWidth(6, 430)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setStyleSheet("""
            QTableWidget { border: 1px solid #e5e7eb; border-radius: 8px; background: white; font-size: 13px; }
            QTableWidget::item { padding: 8px 12px; }
            QHeaderView::section { background: #F9FAFB; border: none; border-bottom: 1px solid #e5e7eb; padding: 8px 12px; font-weight: 600; color: #374151; }
            QTableWidget::item:alternate { background: #F9FAFB; }
        """)
        layout.addWidget(self._table)

        # Stats bar
        self._stats_bar = TrainingStatsBar()
        layout.addWidget(self._stats_bar)

        self._table.selectionModel().selectionChanged.connect(self._on_row_selected)

        # Footer
        self._footer = QLabel('No pods found.')
        self._footer.setStyleSheet('color: #6b7280; font-size: 12px;')
        layout.addWidget(self._footer)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #F9FAFB; color: #111827; font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif; }
            QPushButton { background: white; border: 1px solid #e5e7eb; border-radius: 6px; padding: 5px 12px; font-size: 12px; color: #374151; }
            QPushButton:hover { border-color: #ae69f4; color: #ae69f4; }
            QPushButton#danger { color: #EF4444; border-color: #fecaca; }
            QPushButton#danger:hover { background: #fef2f2; }
            QPushButton#watch { color: #ae69f4; border-color: #e9d5ff; }
            QPushButton#watch:hover { background: #F7F4FC; }
            QPushButton#export { color: #0891b2; border-color: #bae6fd; }
            QPushButton#export:hover { background: #f0f9ff; }
        """)

    def _on_refresh(self, pods):
        self._pods = pods
        self._table.setRowCount(0)

        running = [p for p in pods if p.get('desiredStatus') == 'RUNNING']
        self._status_label.setText(f'Last updated: {time.strftime("%H:%M:%S")}  ·  {len(running)} running')

        for row, pod in enumerate(pods):
            self._table.insertRow(row)
            self._table.setRowHeight(row, 52)

            status   = pod.get('desiredStatus', '?')
            gpu_name = (pod.get('machine') or {}).get('gpuDisplayName', '?')
            gpus     = (pod.get('runtime') or {}).get('gpus') or []
            gpu_util = f"{gpus[0]['gpuUtilPercent']}%" if gpus and gpus[0].get('gpuUtilPercent') is not None else '--'

            def cell(text, align=Qt.AlignLeft | Qt.AlignVCenter):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(align)
                return item

            self._table.setItem(row, 0, cell(pod.get('name', pod['id'])))
            self._table.setItem(row, 1, cell(f"{pod.get('gpuCount',1)}× {gpu_name}"))

            status_item = cell(status, Qt.AlignCenter | Qt.AlignVCenter)
            status_item.setForeground(QColor(STATUS_COLORS.get(status, '#374151')))
            self._table.setItem(row, 2, status_item)

            self._table.setItem(row, 3, cell(fmt_uptime(_pod_uptime_seconds(pod)), Qt.AlignRight | Qt.AlignVCenter))
            self._table.setItem(row, 4, cell(fmt_cost(pod)))
            self._table.setItem(row, 5, cell(gpu_util, Qt.AlignCenter | Qt.AlignVCenter))

            # Action buttons
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(8, 4, 8, 4)
            btn_layout.setSpacing(6)

            watch_btn = QPushButton('Watch')
            watch_btn.setObjectName('watch')
            watch_btn.setFixedWidth(60)
            watch_btn.setEnabled(status == 'RUNNING')
            watch_btn.clicked.connect(lambda _, p=pod: launch_monitor_for_pod(p))

            console_btn = QPushButton('Console')
            console_btn.setFixedWidth(68)
            console_btn.setEnabled(status == 'RUNNING')
            console_btn.clicked.connect(lambda _, p=pod: open_pod_console(p))

            export_btn = QPushButton('Export')
            export_btn.setObjectName('export')
            export_btn.setFixedWidth(60)
            export_btn.setEnabled(status == 'RUNNING')
            export_btn.clicked.connect(lambda _, p=pod: self._do_export(p))

            dl_btn = QPushButton('Download')
            dl_btn.setFixedWidth(78)
            dl_btn.setEnabled(status == 'RUNNING')
            dl_btn.clicked.connect(lambda _, p=pod: self._do_download_pth(p))

            stop_btn = QPushButton('Stop')
            stop_btn.setFixedWidth(50)
            stop_btn.setEnabled(status == 'RUNNING')
            stop_btn.clicked.connect(lambda _, p=pod: self._do_stop(p))

            kill_btn = QPushButton('Terminate')
            kill_btn.setObjectName('danger')
            kill_btn.setFixedWidth(80)
            kill_btn.clicked.connect(lambda _, p=pod: self._do_terminate(p))

            btn_layout.addWidget(watch_btn)
            btn_layout.addWidget(console_btn)
            btn_layout.addWidget(export_btn)
            btn_layout.addWidget(dl_btn)
            btn_layout.addWidget(stop_btn)
            btn_layout.addWidget(kill_btn)
            self._table.setCellWidget(row, 6, btn_widget)

        # Auto-select first running pod for stats bar
        for row, pod in enumerate(pods):
            if pod.get('desiredStatus') == 'RUNNING':
                self._table.selectRow(row)
                break
        else:
            self._stats_bar.clear()

        count = len(pods)
        total_cph = sum(p.get('costPerHr') or 0 for p in pods if p.get('desiredStatus') == 'RUNNING')
        self._footer.setText(
            f'{count} pod{"s" if count != 1 else ""}  ·  '
            f'${total_cph:.2f}/hr running'
        )

    def _on_row_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row < len(self._pods):
            pod = self._pods[row]
            if pod.get('desiredStatus') == 'RUNNING':
                name = pod.get('name', pod['id'])
                # Use just the part after "tunet-" if present
                short = name.replace('tunet-', '', 1) if name.startswith('tunet-') else name
                self._stats_bar.set_pod(pod['id'], short)
            else:
                self._stats_bar.clear()

    def _open_launch(self):
        dlg = LaunchDialog(self)
        dlg.exec()
        # Refresh pod list after dialog closes in case a new pod was created
        threading.Thread(target=self._first_fetch, daemon=True).start()

    def _on_error(self, msg):
        self._status_label.setText(f'Error: {msg[:80]}')

    def _do_stop(self, pod):
        reply = QMessageBox.question(
            self, 'Stop Pod',
            f'Stop training on "{pod.get("name", pod["id"])}"?\n\nPod stays alive for downloading.',
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            threading.Thread(target=stop_pod, args=(pod['id'],), daemon=True).start()

    def _do_terminate(self, pod):
        reply = QMessageBox.question(
            self, 'Terminate Pod',
            f'TERMINATE "{pod.get("name", pod["id"])}"?\n\nThis stops billing immediately. Make sure you\'ve downloaded your checkpoints.',
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            def _term():
                terminate_pod(pod['id'])
                time.sleep(2)
                self._first_fetch()
            threading.Thread(target=_term, daemon=True).start()

    def _do_export(self, pod):
        dlg = ExportDialog(pod, self)
        dlg.exec()

    def _do_download_pth(self, pod):
        dlg = DownloadPthDialog(pod, self)
        dlg.exec()



# ── Launch settings persistence ───────────────────────────────────────────────

_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runpod_dashboard_settings.json')

def _load_settings():
    try:
        with open(_SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_settings(s):
    try:
        with open(_SETTINGS_FILE, 'w') as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass


# ── Download .pth Dialog ─────────────────────────────────────────────────────

class DownloadPthDialog(QDialog):
    """Download checkpoint .pth files from the pod via SCP."""

    def __init__(self, pod, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Download Checkpoint')
        self.resize(560, 400)
        self._pod = pod
        self._running = False
        self._build_ui()
        self._populate_defaults()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)

        self._remote_dir_edit = QLineEdit()
        self._remote_dir_edit.setPlaceholderText('/workspace/output/<job>')
        form.addRow('Remote output dir:', self._remote_dir_edit)

        self._dest_edit = QLineEdit()
        self._dest_edit.setPlaceholderText('Local folder to save .pth files')
        browse_dest = QPushButton('…')
        browse_dest.setFixedWidth(28)
        browse_dest.clicked.connect(self._browse_dest)
        dest_row = QHBoxLayout()
        dest_row.addWidget(self._dest_edit)
        dest_row.addWidget(browse_dest)
        form.addRow('Save to:', dest_row)

        root.addLayout(form)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(200)
        self._log.setStyleSheet(
            'QPlainTextEdit { background: #111827; color: #d1fae5; font-family: Consolas, monospace; font-size: 11px; border-radius: 6px; }'
        )
        root.addWidget(self._log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._dl_btn = QPushButton('Download')
        self._dl_btn.setFixedWidth(110)
        self._dl_btn.setStyleSheet(
            'QPushButton { background: #7c3aed; color: white; border: none; border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: 600; }'
            'QPushButton:hover { background: #6d28d9; }'
            'QPushButton:disabled { background: #a78bfa; color: white; }'
        )
        self._dl_btn.clicked.connect(self._run)
        self._close_btn = QPushButton('Close')
        self._close_btn.setFixedWidth(80)
        self._close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._dl_btn)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

    def _populate_defaults(self):
        pod_name = self._pod.get('name', '')
        job = pod_name.replace('tunet-', '', 1) if pod_name.startswith('tunet-') else pod_name
        self._remote_dir_edit.setText(f'/workspace/output/{job}')
        s = _load_settings()
        self._dest_edit.setText(s.get('download_pth_dest', ''))

    def _browse_dest(self):
        d = QFileDialog.getExistingDirectory(self, 'Select download folder', self._dest_edit.text() or os.path.expanduser('~'))
        if d:
            self._dest_edit.setText(d)

    def _log_line(self, text):
        self._log.appendPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _run(self):
        if self._running:
            return
        remote_dir = self._remote_dir_edit.text().strip().rstrip('/')
        local_dest = self._dest_edit.text().strip()
        if not remote_dir:
            QMessageBox.warning(self, 'Missing', 'Enter the remote output directory.')
            return
        if not local_dest:
            QMessageBox.warning(self, 'Missing', 'Choose a local folder to save files.')
            return

        host, port = get_ssh(self._pod)
        if not host:
            QMessageBox.warning(self, 'No SSH', 'Pod has no public SSH port.')
            return

        os.makedirs(local_dest, exist_ok=True)
        s = _load_settings()
        s['download_pth_dest'] = local_dest
        _save_settings(s)

        self._running = True
        self._dl_btn.setEnabled(False)
        self._log.clear()

        stream = _LogStream()
        stream.line.connect(self._log_line)

        def _work():
            old_stdout = sys.stdout
            sys.stdout = stream
            try:
                import subprocess as _sp
                rl = _import_launch()
                key = SSH_KEY

                # List .pth files in the remote dir
                print(f'[download] Listing checkpoints in {remote_dir}...')
                r = rl.ssh(host, port, key,
                           f'find {remote_dir} -maxdepth 1 -name "*.pth" -type f',
                           check=False, capture=True)
                remote_files = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                if not remote_files:
                    print(f'[download] No .pth files found in {remote_dir}')
                    return

                for rpath in sorted(remote_files):
                    fname = os.path.basename(rpath)
                    lpath = os.path.join(local_dest, fname)
                    size_r = rl.ssh(host, port, key,
                                    f'stat -c%s {rpath}', check=False, capture=True)
                    size_mb = int(size_r.stdout.strip() or 0) / 1024 / 1024
                    print(f'[download] {fname}  ({size_mb:.1f} MB) → {lpath}')
                    result = _sp.run(
                        ['scp', '-o', 'StrictHostKeyChecking=no',
                         '-o', 'BatchMode=yes',
                         '-o', 'ConnectTimeout=15',
                         '-i', key,
                         '-P', str(port),
                         f'root@{host}:{rpath}', lpath],
                        capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        print(f'[download] ERROR: {result.stderr.strip()}')
                    else:
                        print(f'[download] ✓ {fname}')

                print(f'\n[download] Done! Files saved to: {local_dest}')

            except Exception as e:
                print(f'[download] ERROR: {e}')
            finally:
                sys.stdout = old_stdout
                self._running = False
                self._dl_btn.setEnabled(True)

        threading.Thread(target=_work, daemon=True).start()


# ── Export Dialog ────────────────────────────────────────────────────────────

class ExportDialog(QDialog):
    """SSH into pod, run flame_exporter.py on latest .pth, download .onnx + .json."""

    def __init__(self, pod, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Model')
        self.resize(560, 460)
        self._pod = pod
        self._running = False
        self._build_ui()
        self._populate_defaults()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)

        self._pth_edit = QLineEdit()
        self._pth_edit.setPlaceholderText('/workspace/output/<job>/model_tunet_latest.pth')
        browse_pth = QPushButton('…')
        browse_pth.setFixedWidth(28)
        browse_pth.clicked.connect(self._browse_pth)
        pth_row = QHBoxLayout()
        pth_row.addWidget(self._pth_edit)
        pth_row.addWidget(browse_pth)
        form.addRow('Remote .pth path:', pth_row)

        self._dest_edit = QLineEdit()
        self._dest_edit.setPlaceholderText('Local folder to save .onnx / .json')
        browse_dest = QPushButton('…')
        browse_dest.setFixedWidth(28)
        browse_dest.clicked.connect(self._browse_dest)
        dest_row = QHBoxLayout()
        dest_row.addWidget(self._dest_edit)
        dest_row.addWidget(browse_dest)
        form.addRow('Save to:', dest_row)

        root.addLayout(form)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(240)
        self._log.setStyleSheet(
            'QPlainTextEdit { background: #111827; color: #d1fae5; font-family: Consolas, monospace; font-size: 11px; border-radius: 6px; }'
        )
        root.addWidget(self._log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._export_btn = QPushButton('Export + Download')
        self._export_btn.setFixedWidth(150)
        self._export_btn.setStyleSheet(
            'QPushButton { background: #0891b2; color: white; border: none; border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: 600; }'
            'QPushButton:hover { background: #0e7490; }'
            'QPushButton:disabled { background: #a5f3fc; color: white; }'
        )
        self._export_btn.clicked.connect(self._run)
        self._close_btn = QPushButton('Close')
        self._close_btn.setFixedWidth(80)
        self._close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._export_btn)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

    def _populate_defaults(self):
        # Pre-fill remote path with the most common pattern
        pod_name = self._pod.get('name', '')
        # job name is everything after "tunet-"
        job = pod_name.replace('tunet-', '', 1) if pod_name.startswith('tunet-') else pod_name
        self._pth_edit.setText(f'/workspace/output/{job}/model_tunet_latest.pth')

        # Default local dest from settings
        s = _load_settings()
        self._dest_edit.setText(s.get('export_dest', ''))

    def _browse_pth(self):
        # Can't browse remote; let user type. This is a no-op placeholder.
        pass

    def _browse_dest(self):
        d = QFileDialog.getExistingDirectory(self, 'Select download folder', self._dest_edit.text() or os.path.expanduser('~'))
        if d:
            self._dest_edit.setText(d)

    def _log_line(self, text):
        self._log.appendPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _run(self):
        if self._running:
            return
        remote_pth = self._pth_edit.text().strip()
        local_dest = self._dest_edit.text().strip()
        if not remote_pth:
            QMessageBox.warning(self, 'Missing', 'Enter the remote .pth path.')
            return
        if not local_dest:
            QMessageBox.warning(self, 'Missing', 'Choose a local folder to save exports.')
            return

        host, port = get_ssh(self._pod)
        if not host:
            QMessageBox.warning(self, 'No SSH', 'Pod has no public SSH port.')
            return

        os.makedirs(local_dest, exist_ok=True)

        # Save dest for next time
        s = _load_settings()
        s['export_dest'] = local_dest
        _save_settings(s)

        self._running = True
        self._export_btn.setEnabled(False)
        self._log.clear()

        stream = _LogStream()
        stream.line.connect(self._log_line)

        def _work():
            old_stdout = sys.stdout
            sys.stdout = stream
            try:
                rl = _import_launch()
                key = SSH_KEY

                remote_dir = os.path.dirname(remote_pth).replace('\\', '/')
                remote_export_dir = remote_dir + '/exports/flame'

                # 1. Run exporter on pod
                print(f'[export] Running flame_exporter.py on pod...')
                export_cmd = (
                    f'cd /workspace/tunet && '
                    f'python exporters/flame_exporter.py '
                    f'--checkpoint {remote_pth} '
                    f'--output_dir {remote_export_dir}'
                )
                r = rl.ssh(host, port, key, export_cmd, check=False, capture=True, timeout=300)
                output = (r.stdout + r.stderr).strip()
                for line in output.splitlines():
                    print(f'  {line}')
                if r.returncode != 0:
                    print(f'[export] ERROR: exporter exited {r.returncode}')
                    return

                # 2. List exported files
                print(f'[export] Listing exports...')
                r = rl.ssh(host, port, key,
                           f'find {remote_export_dir} -maxdepth 1 -type f',
                           check=False, capture=True)
                remote_files = [l.strip() for l in r.stdout.splitlines()
                                if l.strip() and any(l.strip().endswith(e) for e in ('.onnx', '.json'))]
                if not remote_files:
                    print(f'[export] No .onnx/.json files found in {remote_export_dir}')
                    return

                # 3. Download each file
                import subprocess as _sp
                for rpath in remote_files:
                    fname = os.path.basename(rpath)
                    lpath = os.path.join(local_dest, fname)
                    size_r = rl.ssh(host, port, key,
                                    f'stat -c%s {rpath}', check=False, capture=True)
                    size_mb = int(size_r.stdout.strip() or 0) / 1024 / 1024
                    print(f'[download] {fname}  ({size_mb:.1f} MB) → {lpath}')
                    result = _sp.run(
                        ['scp', '-o', 'StrictHostKeyChecking=no',
                         '-o', 'BatchMode=yes',
                         '-o', 'ConnectTimeout=15',
                         '-i', key,
                         '-P', str(port),
                         f'root@{host}:{rpath}', lpath],
                        capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        print(f'[download] ERROR: {result.stderr.strip()}')

                print(f'\n[export] Done! Files saved to: {local_dest}')

            except Exception as e:
                print(f'[export] ERROR: {e}')
            finally:
                sys.stdout = old_stdout
                self._running = False
                self._export_btn.setEnabled(True)

        threading.Thread(target=_work, daemon=True).start()


# ── Launch Dialog ─────────────────────────────────────────────────────────────

# Import launch helpers lazily to avoid hard dependency at module load
def _import_launch():
    import importlib.util, sys as _sys
    name = 'runpod_launch'
    if name in _sys.modules:
        return _sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runpod_launch.py'))
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _LogStream(QObject):
    """Captures print() output from a background thread and emits it as a Qt signal."""
    line = Signal(str)

    def write(self, text):
        if text.strip():
            self.line.emit(text.rstrip('\n'))

    def flush(self):
        pass


class LaunchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Launch Training Job')
        self.resize(620, 680)
        self._settings = _load_settings()
        self._running = False
        self._build_ui()
        self._restore()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ── Config section ────────────────────────────────────────────────────
        cfg_group = QGroupBox('Job Configuration')
        cfg_group.setStyleSheet('QGroupBox { font-weight: 600; }')
        form = QFormLayout(cfg_group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # Config file
        cfg_row = QHBoxLayout()
        self._cfg_edit = QLineEdit()
        self._cfg_edit.setPlaceholderText('Path to config.yaml')
        self._cfg_edit.textChanged.connect(self._on_config_changed)
        cfg_row.addWidget(self._cfg_edit)
        cfg_browse = QPushButton('Browse')
        cfg_browse.setFixedWidth(70)
        cfg_browse.clicked.connect(self._browse_config)
        cfg_row.addWidget(cfg_browse)
        form.addRow('Config YAML:', cfg_row)

        # Pod name
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText('Auto from config')
        form.addRow('Pod Name:', self._name_edit)

        # GPU type
        rl = _import_launch()
        self._gpu_combo = QComboBox()
        for k, v in rl.GPU_TYPES.items():
            self._gpu_combo.addItem(f'{v}  [{k}]', k)
        form.addRow('GPU:', self._gpu_combo)

        # Disk / Volume
        disk_row = QHBoxLayout()
        self._disk_spin = QSpinBox()
        self._disk_spin.setRange(10, 500)
        self._disk_spin.setValue(50)
        self._disk_spin.setSuffix(' GB')
        disk_row.addWidget(self._disk_spin)
        disk_row.addWidget(QLabel('  Volume:'))
        self._vol_spin = QSpinBox()
        self._vol_spin.setRange(10, 1000)
        self._vol_spin.setValue(100)
        self._vol_spin.setSuffix(' GB')
        disk_row.addWidget(self._vol_spin)
        disk_row.addStretch()
        form.addRow('Container Disk:', disk_row)

        # Data info label (auto-detected from config)
        self._data_info = QLabel('— will be read from config src_dir / dst_dir —')
        self._data_info.setStyleSheet('color: #6b7280; font-size: 11px; font-style: italic;')
        form.addRow('Data:', self._data_info)

        # Checkpoint picker
        ckpt_row = QHBoxLayout()
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ckpt_row.addWidget(self._ckpt_combo)
        self._ckpt_info = QLabel('')
        self._ckpt_info.setStyleSheet('color: #6b7280; font-size: 11px;')
        ckpt_row.addWidget(self._ckpt_info)
        form.addRow('Checkpoint:', ckpt_row)

        # Options
        opts_row = QHBoxLayout()
        self._skip_code_cb = QCheckBox('Skip code upload')
        self._terminate_cb = QCheckBox('Terminate when done')
        opts_row.addWidget(self._skip_code_cb)
        opts_row.addWidget(self._terminate_cb)
        opts_row.addStretch()
        form.addRow('Options:', opts_row)

        root.addWidget(cfg_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._launch_btn = QPushButton('Launch Job')
        self._launch_btn.setFixedHeight(34)
        self._launch_btn.setStyleSheet(
            'QPushButton { background: #ae69f4; color: white; border: none; border-radius: 6px; font-weight: 600; font-size: 13px; }'
            'QPushButton:hover { background: #9b50e8; }'
            'QPushButton:disabled { background: #d8b4fe; color: #f3e8ff; }'
        )
        self._launch_btn.clicked.connect(self._do_launch)
        btn_row.addWidget(self._launch_btn)

        self._cancel_btn = QPushButton('Close')
        self._cancel_btn.setFixedHeight(34)
        self._cancel_btn.setFixedWidth(80)
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)
        root.addLayout(btn_row)

        # ── Log ───────────────────────────────────────────────────────────────
        log_group = QGroupBox('Launch Log')
        log_group.setStyleSheet('QGroupBox { font-weight: 600; }')
        log_vbox = QVBoxLayout(log_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(200)
        self._log.setStyleSheet(
            'QPlainTextEdit { background: #1e1e2e; color: #cdd6f4; font-family: Consolas, monospace; font-size: 12px; border-radius: 6px; padding: 8px; }'
        )
        log_vbox.addWidget(self._log)
        root.addWidget(log_group)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Config YAML', '', 'YAML files (*.yaml *.yml)')
        if path:
            self._cfg_edit.setText(path)

    def _on_config_changed(self, path):
        if not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            data_sect = cfg.get('data') or {}

            # Auto-fill pod name placeholder
            out_dir = data_sect.get('output_dir', '')
            if out_dir and not self._name_edit.text():
                stem = os.path.basename(out_dir.rstrip('/\\'))
                if stem:
                    self._name_edit.setPlaceholderText(f'tunet-{stem}')

            # Show detected data dirs
            src = data_sect.get('src_dir', '')
            dst = data_sect.get('dst_dir', '')
            parts = []
            if src:
                src_name = os.path.basename(src.rstrip('/\\'))
                parts.append(f'src: {src_name}')
            if dst:
                dst_name = os.path.basename(dst.rstrip('/\\'))
                parts.append(f'dst: {dst_name}')
            if parts:
                self._data_info.setText('Auto-upload: ' + '  |  '.join(parts))
                self._data_info.setStyleSheet('color: #16A34A; font-size: 11px;')
            else:
                self._data_info.setText('No src_dir / dst_dir found in config')
                self._data_info.setStyleSheet('color: #6b7280; font-size: 11px; font-style: italic;')

            # Populate checkpoint picker from output_dir
            self._ckpt_combo.clear()
            self._ckpt_combo.addItem('Auto (latest checkpoint, or fresh)', None)
            if out_dir and os.path.isdir(out_dir):
                rl = _import_launch()
                ckpts = rl.scan_checkpoints(out_dir)
                for ckpt_path, is_latest in ckpts:
                    label = os.path.basename(ckpt_path)
                    size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
                    tag = ' [latest]' if is_latest else ''
                    self._ckpt_combo.addItem(f'{label}{tag}  ({size_mb:.0f} MB)', ckpt_path)
                if ckpts:
                    self._ckpt_info.setText(f'{len(ckpts)} checkpoint{"s" if len(ckpts) != 1 else ""} found')
                else:
                    self._ckpt_info.setText('no .pth found — will train fresh')
            else:
                self._ckpt_info.setText('')
        except Exception:
            pass

    def _restore(self):
        s = self._settings
        if s.get('config'):
            self._cfg_edit.setText(s['config'])
        if s.get('pod_name'):
            self._name_edit.setText(s['pod_name'])
        if s.get('gpu'):
            rl = _import_launch()
            keys = list(rl.GPU_TYPES.keys())
            if s['gpu'] in keys:
                self._gpu_combo.setCurrentIndex(keys.index(s['gpu']))
        self._disk_spin.setValue(s.get('disk', 50))
        self._vol_spin.setValue(s.get('volume', 100))
        self._skip_code_cb.setChecked(s.get('skip_code', False))
        self._terminate_cb.setChecked(s.get('terminate', False))

    def _persist(self):
        self._settings.update({
            'config':    self._cfg_edit.text(),
            'pod_name':  self._name_edit.text(),
            'gpu':       self._gpu_combo.currentData(),
            'disk':      self._disk_spin.value(),
            'volume':    self._vol_spin.value(),
            'skip_code': self._skip_code_cb.isChecked(),
            'terminate': self._terminate_cb.isChecked(),
        })
        _save_settings(self._settings)

    def _log_line(self, text):
        self._log.appendPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    # ── Launch ────────────────────────────────────────────────────────────────

    def _do_launch(self):
        if self._running:
            return

        config_path = self._cfg_edit.text().strip()
        if not config_path or not os.path.isfile(config_path):
            QMessageBox.warning(self, 'No Config', 'Please select a valid config.yaml file.')
            return

        self._persist()
        self._running = True
        self._launch_btn.setEnabled(False)
        self._cancel_btn.setText('Running...')
        self._cancel_btn.setEnabled(False)
        self._log.clear()

        gpu_key     = self._gpu_combo.currentData()
        pod_name    = self._name_edit.text().strip() or None
        disk_gb     = self._disk_spin.value()
        vol_gb      = self._vol_spin.value()
        skip_code   = self._skip_code_cb.isChecked()
        terminate   = self._terminate_cb.isChecked()
        resume_pth  = self._ckpt_combo.currentData()   # None = auto

        # Log stream captures print() from the launch thread
        self._log_stream = _LogStream()
        self._log_stream.line.connect(self._log_line)

        def _run():
            import copy
            rl = _import_launch()
            api_key = rl.os.environ.get('RUNPOD_API_KEY') or rl.os.environ.get('runpodapi')
            if not api_key:
                self._log_stream.line.emit('ERROR: RUNPOD_API_KEY not set in environment / .env')
                self._finish(False)
                return

            old_stdout = sys.stdout
            sys.stdout = self._log_stream
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                data_uploads = {}
                pod_config, job_name = rl.rewrite_config_for_pod(config, data_uploads, resume_pth=resume_pth)
                name = pod_name or f'tunet-{job_name}'
                gpu_type = rl.GPU_TYPES[gpu_key]

                key_path = rl.find_ssh_key()
                print(f'[ssh] Using key: {key_path}')
                print(f'\n[runpod] Creating pod: {name}')
                print(f'         GPU    : {gpu_type}')
                print(f'         Image  : {rl.DEFAULT_IMAGE}')
                print(f'         Disk   : {disk_gb}GB container + {vol_gb}GB volume\n')

                pod = rl.create_pod(api_key, name, gpu_type, rl.DEFAULT_IMAGE, disk_gb, vol_gb)
                pod_id = pod['id']
                print(f'[runpod] Pod created: {pod_id}  (${pod.get("costPerHr", "?")}/hr)')

                pod_info, host, port = rl.wait_for_pod(api_key, pod_id)
                rl.wait_for_ssh(host, port, key_path)

                # ── Confirm we're on the right machine ───────────────────────
                r = rl.ssh(host, port, key_path,
                           'hostname',
                           check=False, capture=True)
                print(f'[verify] connected to: {r.stdout.strip() or "(no output)"}')

                def verify(remote_path, label):
                    r = rl.ssh(host, port, key_path,
                               f'ls -lah {remote_path} 2>&1 | head -3', check=False, capture=True)
                    print(f'[verify] {label}: {r.stdout.strip() or r.stderr.strip() or "?"}')

                if not skip_code:
                    tunet_root = os.path.dirname(os.path.abspath(__file__))
                    print(f'\n[upload] Syncing tunet source → {rl.TUNET_REMOTE}')
                    rl.ssh(host, port, key_path, f'mkdir -p {rl.TUNET_REMOTE}')
                    rl.rsync(
                        tunet_root + '/',
                        host, port, key_path,
                        rl.TUNET_REMOTE,
                        exclude=['__pycache__', '*.pyc', '.git', 'node_modules',
                                 'Spark', '_archive', '*.pth', '*.onnx',
                                 'tunet_session.yaml'],
                    )
                    verify(rl.TUNET_REMOTE, 'tunet source')

                if data_uploads:
                    remote_dirs = {os.path.dirname(p) for p in data_uploads.values()}
                    for d in remote_dirs:
                        rl.ssh(host, port, key_path, f'mkdir -p "{d}"')
                    for lp, rp in data_uploads.items():
                        if os.path.isdir(lp):
                            print(f'\n[upload] {os.path.basename(lp)} → {rp}')
                            rl.rsync(lp + '/', host, port, key_path, rp)
                            r = rl.ssh(host, port, key_path,
                                       f'du -sh {rp} 2>/dev/null; find {rp} -type f | wc -l',
                                       check=False, capture=True)
                            print(f'[verify] {rp}: {r.stdout.strip()}')
                        else:
                            size_mb = os.path.getsize(lp) / 1024 / 1024
                            print(f'\n[upload] {os.path.basename(lp)} ({size_mb:.0f} MB) → {rp}')
                            rl.scp(lp, host, port, key_path, rp)
                            verify(rp, os.path.basename(lp))

                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
                    yaml.dump(pod_config, tf, default_flow_style=False, sort_keys=False)
                    tmp_cfg = tf.name
                print(f'\n[upload] Config → {rl.CONFIG_REMOTE}')
                rl.scp(tmp_cfg, host, port, key_path, rl.CONFIG_REMOTE)
                os.unlink(tmp_cfg)
                verify(rl.CONFIG_REMOTE, 'config.yaml')

                start_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runpod_start.sh')
                rl.scp(start_sh, host, port, key_path, '/workspace/runpod_start.sh')
                rl.ssh(host, port, key_path, 'chmod +x /workspace/runpod_start.sh')
                verify('/workspace/runpod_start.sh', 'start script')

                print(f'\n[launch] Workspace contents:')
                r = rl.ssh(host, port, key_path, 'ls -lah /workspace/', check=False, capture=True)
                for line in r.stdout.strip().splitlines():
                    print(f'         {line}')

                output_dir = f'{rl.OUTPUT_REMOTE}/{job_name}'
                monitor_url = f'https://{pod_id}-{rl.MONITOR_PORT}.proxy.runpod.net'
                launch_cmd = (
                    f'nohup bash /workspace/runpod_start.sh '
                    f'"{output_dir}" "{rl.CONFIG_REMOTE}" "{rl.MONITOR_PORT}" '
                    f'> /workspace/launch.log 2>&1 &'
                )
                rl.ssh(host, port, key_path, launch_cmd)
                print(f'[launch] Start script fired — waiting 5s for launch.log...')
                time.sleep(5)
                r = rl.ssh(host, port, key_path,
                           'head -20 /workspace/launch.log 2>/dev/null || echo "(launch.log not yet written)"',
                           check=False, capture=True)
                print(f'[launch.log]\n{r.stdout.strip()}')

                print(f'\n✓ Training started!')
                print(f'  Pod ID     : {pod_id}')
                print(f'  Output     : {output_dir}')
                print(f'  Monitor    : {monitor_url}')
                print(f'  SSH        : ssh -i ~/.ssh/runpod -p {port} root@{host}')
                self._finish(True)

            except Exception as e:
                print(f'\nERROR: {e}')
                self._finish(False)
            finally:
                sys.stdout = old_stdout

        threading.Thread(target=_run, daemon=True).start()

    def _finish(self, success):
        self._running = False
        self._launch_btn.setEnabled(True)
        self._cancel_btn.setText('Close')
        self._cancel_btn.setEnabled(True)
        if success:
            self._log_line('\n── Done ──')


def _make_dashboard_icon():
    """Same base as tunet.py's icon but with a cloud shape behind the T."""
    size   = 256
    px     = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p      = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    purple      = QColor('#7B3FE4')
    cloud_color = QColor('#c4a8f5')   # soft lavender cloud
    stroke      = 10
    radius      = 48

    # ── White rounded-rect background ────────────────────────────────────────
    p.setPen(QPen(purple, stroke))
    p.setBrush(QBrush(QColor('white')))
    bg = QPainterPath()
    bg.addRoundedRect(stroke / 2, stroke / 2, size - stroke, size - stroke, radius, radius)
    p.drawPath(bg)

    # ── Cloud (drawn before T so T sits on top) ───────────────────────────────
    # Five overlapping circles + a flat-bottom rectangle form a classic cloud.
    # Positioned in the lower-centre, peeking out from behind the T.
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(cloud_color))

    cloud = QPainterPath()
    # flat base rect
    cloud.addRect(52, 178, 154, 42)
    # bumps (cx, cy, r)
    for cx, cy, r in [
        (82,  168, 34),
        (118, 152, 42),
        (158, 160, 36),
        (60,  185, 26),
        (198, 183, 24),
    ]:
        bump = QPainterPath()
        bump.addEllipse(cx - r, cy - r, r * 2, r * 2)
        cloud = cloud.united(bump)

    # Clip cloud to stay inside the rounded-rect card
    clip = QPainterPath()
    clip.addRoundedRect(stroke, stroke, size - stroke * 2, size - stroke * 2, radius - 2, radius - 2)
    p.setClipPath(clip)
    p.drawPath(cloud)
    p.setClipping(False)

    # ── Purple "T" ────────────────────────────────────────────────────────────
    p.setPen(QPen(purple))
    font = QFont('Arial', 1, QFont.Weight.Bold)
    font.setPixelSize(int(size * 0.72))
    p.setFont(font)
    p.drawText(px.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, 'T')

    p.end()
    return QIcon(px)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('TuNet RunPod Dashboard')
    app.setWindowIcon(_make_dashboard_icon())
    win = Dashboard()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
