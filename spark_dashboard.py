"""
Spark Cloud Studio Dashboard — list workstations, start/stop/delete, watch training.

Usage:
    python spark_dashboard.py

Credentials (from .env):
    spark_email   / SPARK_EMAIL
    spark_pass    / SPARK_PASSWORD
    SPARK_PROXY_BASE  — proxy URL pattern for monitor API, e.g.
                        https://{id}-8080.your-domain.com
                        Ask Walt for the real format.
"""

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

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QAbstractItemView, QFrame, QSizePolicy,
    QDialog, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QFileDialog, QGroupBox,
)
from PySide6.QtGui import QColor, QFont, QTextCursor, QIcon, QPixmap, QPainter, QPen, QBrush, QPainterPath

# ── Spark REST API ────────────────────────────────────────────────────────────

SPARK_API    = 'https://api.prod.aapse1.sparkcloud.studio'
MONITOR_PORT = 8080

# os_ready values observed: -2 = stopped.
# Ask Walt for the full lifecycle values (starting, running, etc.)
OS_READY_STOPPED = -2
OS_READY_RUNNING = 1   # assumed — confirm with Walt

# Proxy URL for monitor_api.py running on the workstation.
# Ask Walt for the real pattern. Placeholders {id} and {port} are replaced at runtime.
# Example: 'https://{id}-{port}.proxy.sparkcloud.studio'
SPARK_PROXY_BASE = os.environ.get('SPARK_PROXY_BASE', '')


def _creds():
    email = os.environ.get('SPARK_EMAIL') or os.environ.get('spark_email') or ''
    pw    = os.environ.get('SPARK_PASSWORD') or os.environ.get('spark_pass') or ''
    if not email or not pw:
        raise RuntimeError('Set spark_email and spark_pass in .env')
    return email, pw


# ── Token management (lazy, auto-refresh) ────────────────────────────────────

_token = None
_token_expires_at = 0


def _jwt_expiry(token):
    try:
        import base64 as _b64
        payload = json.loads(_b64.b64decode(token.split('.')[1] + '=='))
        return payload.get('exp', 0) * 1000
    except Exception:
        return 0


def get_token():
    global _token, _token_expires_at
    if _token and _token_expires_at > time.time() * 1000 + 60_000:
        return _token
    email, pw = _creds()
    resp = requests.post(f'{SPARK_API}/auth/login',
                         json={'email': email, 'password': pw}, timeout=15)
    if not resp.ok:
        raise RuntimeError(f'Spark auth failed: HTTP {resp.status_code}')
    data = resp.json()
    tok = data.get('token') or data.get('access_token')
    if not tok:
        raise RuntimeError('Spark auth: no token in response')
    _token = tok
    _token_expires_at = _jwt_expiry(tok)
    return _token


def spark(method, path, body=None):
    tok = get_token()
    resp = requests.request(
        method, f'{SPARK_API}{path}',
        headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/json'},
        json=body,
        timeout=20,
    )
    if not resp.ok:
        raise RuntimeError(f'Spark API HTTP {resp.status_code}: {resp.text[:200]}')
    return resp.json()


# ── Workstation helpers ───────────────────────────────────────────────────────

def list_workstations():
    data = spark('GET', '/api/workstations')
    return data.get('data', {}).get('data', {}).get('data', [])


def get_workstation(ws_id):
    data = spark('GET', f'/api/workstations/{ws_id}')
    return data.get('data')


def start_workstation(ws_id):
    spark('POST', f'/api/workstations/{ws_id}/start')


def stop_workstation(ws_id):
    spark('POST', f'/api/workstations/{ws_id}/stop')


def delete_workstation(ws_id):
    spark('DELETE', f'/api/workstations/{ws_id}')


def get_ssh_details(ws):
    """
    TODO: ask Walt which API endpoint returns SSH host/port for a running workstation.
    The /api/workstations/{id} response doesn't include IP or SSH port yet.
    Expected something like: GET /api/workstations/{id}/access → { public_ip, ssh_port, ssh_user }
    """
    # placeholder — uncomment and fill in once Walt provides the endpoint
    # data = spark('GET', f'/api/workstations/{ws["id"]}/access')
    # return data.get('public_ip'), data.get('ssh_port', 22), data.get('ssh_user', 'ubuntu')
    return None, None, None


def ws_status(ws):
    r = ws.get('os_ready', -99)
    if r == OS_READY_RUNNING:  return 'RUNNING'
    if r == OS_READY_STOPPED:  return 'STOPPED'
    if r == 0:                  return 'STARTING'
    return f'os_ready={r}'


def is_running(ws):
    return ws.get('os_ready') == OS_READY_RUNNING


def fmt_uptime(ws):
    # Spark doesn't give uptime directly — we'd need to track start time ourselves
    return '--'


def fmt_cost(ws):
    cph = ws.get('cost') or 0
    if is_running(ws):
        return f'${cph:.2f}/hr  (running)'
    return f'${cph:.2f}/hr'


def monitor_url(ws_id):
    if not SPARK_PROXY_BASE:
        return None
    return SPARK_PROXY_BASE.replace('{id}', str(ws_id)).replace('{port}', str(MONITOR_PORT))


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
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                wss = list_workstations()
                self.refreshed.emit(wss)
            except Exception as e:
                self.error.emit(str(e))
            time.sleep(self._interval)


# ── Training stats bar (same widget as runpod_dashboard, URL source differs) ──

_MONO = "font-family: 'Consolas', 'Fira Code', monospace;"


def _compute_analysis(steps, losses):
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
    smoothed, alpha, last = [], 0.95, w_losses[0]
    for v in w_losses:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    n = len(smoothed)
    xm, ym = sum(w_steps) / n, sum(smoothed) / n
    ss_xy = sum((x - xm) * (y - ym) for x, y in zip(w_steps, smoothed))
    ss_xx = sum((x - xm) ** 2 for x in w_steps)
    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    rel_slope = slope / smoothed[-1] if smoothed[-1] > 0 else 0
    mid = len(smoothed) // 2
    f_avg = sum(smoothed[:mid]) / mid
    s_avg = sum(smoothed[mid:]) / (n - mid)
    pct = ((s_avg - f_avg) / f_avg * 100) if f_avg > 0 else 0
    best_idx = losses.index(min(losses))
    best_epoch = steps[best_idx]
    since_best = current_epoch - best_epoch
    if rel_slope < -0.005:   trend, tc = 'Improving', '#22c55e'
    elif rel_slope > 0.005:  trend, tc = 'Diverging',  '#ef4444'
    else:                    trend, tc = 'Flat',        '#f59e0b'
    ic = '#22c55e' if pct < -1 else ('#ef4444' if pct > 1 else '#f59e0b')
    if since_best < 10:    pc, pl = f'Best {since_best:.0f}ep ago', '#22c55e'
    elif since_best < 30:  pc, pl = f'Best {since_best:.0f}ep ago', '#f59e0b'
    else:                  pc, pl = f'Best {since_best:.0f}ep ago', '#ef4444'
    if   rel_slope < -0.005 and since_best < 20: rec, rc = 'Training well',        '#22c55e'
    elif rel_slope > 0.01:                        rec, rc = 'Diverging - check LR', '#ef4444'
    elif since_best > 50:                         rec, rc = 'Consider stopping',    '#ef4444'
    elif since_best > 30 or abs(rel_slope) < 0.001: rec, rc = 'Plateau - may stop','#f59e0b'
    elif rel_slope < -0.001:                      rec, rc = 'Slow progress',        '#f59e0b'
    else:                                         rec, rc = 'Stable',               '#6b7280'
    return dict(trend=trend, trend_color=tc,
                pct=f'{pct:+.1f}%', pct_color=ic,
                plateau=pc, plateau_color=pl,
                rec=rec, rec_color=rc,
                best_epoch=best_epoch)


class TrainingStatsBar(QWidget):
    _data_ready  = Signal(object, object)
    _fetch_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ws_id   = None
        self._ws_name = '--'
        self._timer   = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(30_000)
        self._data_ready.connect(self._apply)
        self._fetch_error.connect(self._on_fetch_error)
        self._build_ui()

    def _build_ui(self):
        self.setObjectName('statsBar')
        self.setStyleSheet('''
            #statsBar { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; }
            QLabel { background: transparent; color: #f8fafc; }
        ''')
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 10, 16, 10)
        root.setSpacing(0)

        def dim(text):
            l = QLabel(text)
            l.setStyleSheet('color:#475569; font-size:11px;')
            return l

        row1 = QHBoxLayout()
        row1.setSpacing(0)
        row1.addWidget(dim('Analysis'))
        row1.addSpacing(16)
        self._analysis_fields = {}
        for i, (key, caption) in enumerate([('trend','Trend'),('pct','Recent Change'),('plateau','Plateau Check'),('rec','Status')]):
            if i > 0:
                sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setStyleSheet('color:#1e293b;'); sep.setFixedWidth(1)
                row1.addSpacing(16); row1.addWidget(sep); row1.addSpacing(16)
            val = QLabel('--')
            val.setStyleSheet('color:#475569; font-size:11px; font-weight:600;')
            row1.addWidget(dim(caption)); row1.addSpacing(5); row1.addWidget(val)
            self._analysis_fields[key] = val
        row1.addStretch()
        root.addLayout(row1)
        root.addSpacing(8)
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setStyleSheet('background:#1e293b; border:none;'); line.setFixedHeight(1)
        root.addWidget(line); root.addSpacing(8)

        row2 = QHBoxLayout(); row2.setSpacing(0)
        self._metric_vals = {}
        for i, (key, caption, mono) in enumerate([
            ('run','Run',False),('epoch','Epoch',True),('loss','Loss',True),
            ('best_loss','Best Loss',True),('best_ep','Best @Ep',True),
            ('val_loss','Val Loss',True),('val_best','Val Best',True),
            ('lpips','LPIPS',True),('lpips_best','LPIPS Best',True),
            ('psnr','PSNR (dB)',True),('ssim','SSIM',True),
            ('pts','Data Pts',True),('step_time','Step Time',True),
        ]):
            if i > 0:
                sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setStyleSheet('color:#1e293b;'); sep.setFixedWidth(1)
                row2.addSpacing(16); row2.addWidget(sep); row2.addSpacing(16)
            col = QVBoxLayout(); col.setSpacing(2)
            cap = QLabel(caption); cap.setStyleSheet('color:#475569; font-size:10px; font-weight:500; letter-spacing:0.05em;')
            val = QLabel('--'); val.setStyleSheet(f'color:#f8fafc; font-size:12px; font-weight:600; {_MONO if mono else ""}')
            col.addWidget(cap); col.addWidget(val)
            row2.addLayout(col); self._metric_vals[key] = val
        row2.addStretch()
        root.addLayout(row2)

    def set_ws(self, ws_id, ws_name=''):
        self._ws_id   = ws_id
        self._ws_name = ws_name or str(ws_id) or '--'
        self._metric_vals['run'].setText(self._ws_name)
        self._metric_vals['run'].setStyleSheet('color:#a78bfa; font-size:12px; font-weight:700;')
        for key in ('trend','pct','plateau','rec'):
            self._analysis_fields[key].setText('Connecting...')
            self._analysis_fields[key].setStyleSheet('color:#475569; font-size:11px; font-weight:600;')
        self._poll()

    def clear(self):
        self._ws_id = None
        dim = 'color:#475569; font-size:11px; font-weight:600;'
        for v in self._analysis_fields.values(): v.setText('--'); v.setStyleSheet(dim)
        for v in self._metric_vals.values(): v.setText('--')

    def _poll(self):
        if not self._ws_id:
            return
        ws_id = self._ws_id
        url = monitor_url(ws_id)
        if not url:
            self._fetch_error.emit('SPARK_PROXY_BASE not set — ask Walt for proxy URL format')
            return
        def _fetch():
            try:
                status  = requests.get(f'{url}/api/status',  timeout=5).json()
                metrics = requests.get(f'{url}/api/metrics', timeout=5).json()
                self._data_ready.emit(status, metrics)
            except Exception as e:
                self._fetch_error.emit(str(e))
        threading.Thread(target=_fetch, daemon=True).start()

    def _apply(self, status, metrics):
        steps = metrics.get('steps', []); losses = metrics.get('l1', [])
        val_l1 = metrics.get('val_l1', []); val_psnr = metrics.get('val_psnr', [])
        val_ssim = metrics.get('val_ssim', []); lpips_list = metrics.get('lpips', [])
        def fmt(v, d=5): return f'{v:.{d}f}' if v is not None else '--'
        best_ep = f'{steps[losses.index(min(losses))]:.1f}' if steps and losses else '--'
        self._metric_vals['run'].setText(self._ws_name)
        self._metric_vals['run'].setStyleSheet('color:#a78bfa; font-size:12px; font-weight:700;')
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
        self._metric_vals['step_time' ].setText(f'{st:.3f}s' if st else '--')
        analysis = _compute_analysis(steps, losses)
        if analysis:
            for key in ('trend','pct','plateau','rec'):
                v = self._analysis_fields[key]
                v.setText(analysis[key])
                v.setStyleSheet(f'color:{analysis[key+"_color"]}; font-size:11px; font-weight:600;')
        else:
            for v in self._analysis_fields.values():
                v.setText('Collecting...'); v.setStyleSheet('color:#475569; font-size:11px; font-weight:600;')

    def _on_fetch_error(self, err):
        for key in ('trend','pct','plateau','rec'):
            self._analysis_fields[key].setText('Unreachable')
            self._analysis_fields[key].setStyleSheet('color:#ef4444; font-size:11px; font-weight:600;')


# ── SSH helpers (same as runpod_launch.py) ────────────────────────────────────

SSH_KEY = os.path.expanduser('~/.ssh/id_rsa')   # Spark uses standard key — adjust if needed


def open_ws_console(ws):
    host, port, user = get_ssh_details(ws)
    if not host:
        QMessageBox.warning(None, 'No SSH', 'SSH details not available.\nAsk Walt for the /api/workstations/{id}/access endpoint.')
        return
    ssh_args = ['ssh', '-o', 'StrictHostKeyChecking=no', '-i', SSH_KEY,
                '-p', str(port), f'{user}@{host}']
    try:
        subprocess.Popen(['wt', '--'] + ssh_args)
    except FileNotFoundError:
        subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k'] + ssh_args)


def launch_monitor_for_ws(ws):
    host, port, user = get_ssh_details(ws)
    if not host:
        QMessageBox.warning(None, 'No SSH', 'SSH details not available.\nAsk Walt for the /api/workstations/{id}/access endpoint.')
        return
    ws_id  = ws['id']
    tmp_log = os.path.join(tempfile.gettempdir(), f'tunet_spark_{ws_id}.log')
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_monitor.py')
    subprocess.Popen([sys.executable, script, '--log_file', tmp_log])
    def _tail():
        remote_cmd = (
            "until LOG=$(find /workspace/output -name training.log 2>/dev/null | head -1)"
            " && test -n \"$LOG\"; do sleep 3; done;"
            " cat $LOG; tail -f -n 0 $LOG"
        )
        with open(tmp_log, 'w') as out:
            subprocess.Popen(['ssh', '-o', 'StrictHostKeyChecking=no', '-i', SSH_KEY,
                              '-p', str(port), f'{user}@{host}', remote_cmd],
                             stdout=out, stderr=subprocess.DEVNULL)
    threading.Thread(target=_tail, daemon=True).start()


# ── Main Window ───────────────────────────────────────────────────────────────

COLS = ['Name', 'GPU', 'RAM', 'Status', 'Cost', 'Actions']

STATUS_COLORS = {
    'RUNNING':  '#16A34A',
    'STOPPED':  '#6b7280',
    'STARTING': '#1c64f2',
}


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TuNet — Spark Dashboard')
        self.resize(1050, 520)
        self._workstations = []
        self._build_ui()
        self._poller = Poller(interval=20)
        self._poller.refreshed.connect(self._on_refresh)
        self._poller.error.connect(self._on_error)
        self._poller.start()
        QTimer.singleShot(100, lambda: threading.Thread(
            target=self._first_fetch, daemon=True).start())

    def _first_fetch(self):
        try:
            wss = list_workstations()
            self._poller.refreshed.emit(wss)
        except Exception as e:
            self._poller.error.emit(str(e))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        hdr = QHBoxLayout()
        title = QLabel('Spark Dashboard')
        title.setFont(QFont('Plus Jakarta Sans', 16, QFont.Bold))
        hdr.addWidget(title)
        hdr.addStretch()

        self._status_label = QLabel('Refreshing...')
        self._status_label.setStyleSheet('color:#6b7280; font-size:12px;')
        hdr.addWidget(self._status_label)

        launch_btn = QPushButton('+ Launch Job')
        launch_btn.setFixedWidth(110)
        launch_btn.setStyleSheet(
            'QPushButton { background:#ae69f4; color:white; border:none; border-radius:6px; padding:5px 12px; font-size:12px; font-weight:600; }'
            'QPushButton:hover { background:#9b50e8; }')
        launch_btn.clicked.connect(self._open_launch)
        hdr.addWidget(launch_btn)

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
        self._table.setColumnWidth(len(COLS) - 1, 380)
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

        self._stats_bar = TrainingStatsBar()
        layout.addWidget(self._stats_bar)
        self._table.selectionModel().selectionChanged.connect(self._on_row_selected)

        self._footer = QLabel('No workstations found.')
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

    def _on_refresh(self, wss):
        self._workstations = wss
        self._table.setRowCount(0)
        running = [w for w in wss if is_running(w)]
        self._status_label.setText(
            f'Last updated: {time.strftime("%H:%M:%S")}  ·  {len(running)} running')

        for row, ws in enumerate(wss):
            self._table.insertRow(row)
            self._table.setRowHeight(row, 52)
            status = ws_status(ws)

            def cell(text, align=Qt.AlignLeft | Qt.AlignVCenter):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(align)
                return item

            self._table.setItem(row, 0, cell(ws.get('workstation_name', ws['id'])))
            self._table.setItem(row, 1, cell(ws.get('gpu', '?')))
            self._table.setItem(row, 2, cell(ws.get('ram', '?')))

            status_item = cell(status, Qt.AlignCenter | Qt.AlignVCenter)
            status_item.setForeground(QColor(STATUS_COLORS.get(status, '#374151')))
            self._table.setItem(row, 3, status_item)
            self._table.setItem(row, 4, cell(fmt_cost(ws)))

            # Action buttons
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(8, 4, 8, 4)
            btn_layout.setSpacing(6)

            running_now = is_running(ws)

            watch_btn = QPushButton('Watch')
            watch_btn.setObjectName('watch')
            watch_btn.setFixedWidth(60)
            watch_btn.setEnabled(running_now)
            watch_btn.clicked.connect(lambda _, w=ws: launch_monitor_for_ws(w))

            console_btn = QPushButton('Console')
            console_btn.setFixedWidth(68)
            console_btn.setEnabled(running_now)
            console_btn.clicked.connect(lambda _, w=ws: open_ws_console(w))

            dl_btn = QPushButton('Download')
            dl_btn.setFixedWidth(78)
            dl_btn.setEnabled(running_now)
            dl_btn.clicked.connect(lambda _, w=ws: self._do_download(w))

            start_btn = QPushButton('Start')
            start_btn.setFixedWidth(52)
            start_btn.setEnabled(not running_now)
            start_btn.clicked.connect(lambda _, w=ws: self._do_start(w))

            stop_btn = QPushButton('Stop')
            stop_btn.setFixedWidth(50)
            stop_btn.setEnabled(running_now)
            stop_btn.clicked.connect(lambda _, w=ws: self._do_stop(w))

            del_btn = QPushButton('Delete')
            del_btn.setObjectName('danger')
            del_btn.setFixedWidth(60)
            del_btn.clicked.connect(lambda _, w=ws: self._do_delete(w))

            for b in (watch_btn, console_btn, dl_btn, start_btn, stop_btn, del_btn):
                btn_layout.addWidget(b)
            self._table.setCellWidget(row, 5, btn_widget)

        # Auto-select first running workstation for stats bar
        for row, ws in enumerate(wss):
            if is_running(ws):
                self._table.selectRow(row)
                break
        else:
            self._stats_bar.clear()

        total_cph = sum(w.get('cost', 0) for w in wss if is_running(w))
        self._footer.setText(
            f'{len(wss)} workstation{"s" if len(wss) != 1 else ""}  ·  '
            f'${total_cph:.2f}/hr running')

    def _on_row_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row < len(self._workstations):
            ws = self._workstations[row]
            if is_running(ws):
                name = ws.get('workstation_name', str(ws['id']))
                self._stats_bar.set_ws(ws['id'], name)
            else:
                self._stats_bar.clear()

    def _on_error(self, msg):
        self._status_label.setText(f'Error: {msg[:80]}')

    def _do_start(self, ws):
        name = ws.get('workstation_name', ws['id'])
        reply = QMessageBox.question(self, 'Start Workstation',
            f'Start "{name}"?\nThis will begin billing at ${ws.get("cost", "?"):.2f}/hr.',
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            def _go():
                try:
                    start_workstation(ws['id'])
                    time.sleep(3)
                    self._first_fetch()
                except Exception as e:
                    self._poller.error.emit(str(e))
            threading.Thread(target=_go, daemon=True).start()

    def _do_stop(self, ws):
        name = ws.get('workstation_name', ws['id'])
        reply = QMessageBox.question(self, 'Stop Workstation',
            f'Stop "{name}"?\nBilling pauses. Data is preserved.',
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            def _go():
                try:
                    stop_workstation(ws['id'])
                    time.sleep(3)
                    self._first_fetch()
                except Exception as e:
                    self._poller.error.emit(str(e))
            threading.Thread(target=_go, daemon=True).start()

    def _do_delete(self, ws):
        name = ws.get('workstation_name', ws['id'])
        reply = QMessageBox.question(self, 'Delete Workstation',
            f'DELETE "{name}"?\n\nThis permanently removes all data.',
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            def _go():
                try:
                    delete_workstation(ws['id'])
                    time.sleep(2)
                    self._first_fetch()
                except Exception as e:
                    self._poller.error.emit(str(e))
            threading.Thread(target=_go, daemon=True).start()

    def _do_download(self, ws):
        host, port, user = get_ssh_details(ws)
        if not host:
            QMessageBox.information(self, 'SSH Not Available',
                'Download requires SSH access.\n'
                'Ask Walt for the /api/workstations/{id}/access endpoint.')
            return
        dlg = DownloadDialog(ws, host, port, user, self)
        dlg.exec()

    def _open_launch(self):
        dlg = LaunchDialog(self._workstations, self)
        dlg.exec()
        threading.Thread(target=self._first_fetch, daemon=True).start()


# ── Download Dialog ───────────────────────────────────────────────────────────

class DownloadDialog(QDialog):
    def __init__(self, ws, host, port, user, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Download Checkpoint')
        self.resize(540, 380)
        self._ws = ws; self._host = host; self._port = port; self._user = user
        self._running = False
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)
        form = QFormLayout(); form.setSpacing(8)
        self._remote_edit = QLineEdit('/workspace/output')
        form.addRow('Remote output dir:', self._remote_edit)
        self._dest_edit = QLineEdit()
        browse = QPushButton('…'); browse.setFixedWidth(28)
        browse.clicked.connect(lambda: self._dest_edit.setText(
            QFileDialog.getExistingDirectory(self, 'Save to', self._dest_edit.text() or os.path.expanduser('~')) or self._dest_edit.text()))
        dest_row = QHBoxLayout(); dest_row.addWidget(self._dest_edit); dest_row.addWidget(browse)
        form.addRow('Save to:', dest_row)
        root.addLayout(form)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); self._log.setMinimumHeight(180)
        self._log.setStyleSheet('QPlainTextEdit { background:#111827; color:#d1fae5; font-family:Consolas,monospace; font-size:11px; border-radius:6px; }')
        root.addWidget(self._log)
        btn_row = QHBoxLayout(); btn_row.addStretch()
        self._dl_btn = QPushButton('Download'); self._dl_btn.setFixedWidth(110)
        self._dl_btn.setStyleSheet('QPushButton { background:#7c3aed; color:white; border:none; border-radius:6px; padding:5px 12px; font-weight:600; } QPushButton:hover { background:#6d28d9; }')
        self._dl_btn.clicked.connect(self._run)
        close_btn = QPushButton('Close'); close_btn.setFixedWidth(80); close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._dl_btn); btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _run(self):
        if self._running: return
        remote = self._remote_edit.text().strip(); dest = self._dest_edit.text().strip()
        if not remote or not dest:
            QMessageBox.warning(self, 'Missing', 'Fill in both fields.'); return
        os.makedirs(dest, exist_ok=True)
        self._running = True; self._dl_btn.setEnabled(False); self._log.clear()
        def _work():
            try:
                result = subprocess.run(
                    ['scp', '-r', '-o', 'StrictHostKeyChecking=no', '-i', SSH_KEY,
                     '-P', str(self._port), f'{self._user}@{self._host}:{remote}/*.pth', dest],
                    capture_output=True, text=True)
                self._log.appendPlainText(result.stdout or result.stderr or 'Done.')
            except Exception as e:
                self._log.appendPlainText(f'ERROR: {e}')
            finally:
                self._running = False; self._dl_btn.setEnabled(True)
        threading.Thread(target=_work, daemon=True).start()


# ── Launch Dialog ─────────────────────────────────────────────────────────────

_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spark_dashboard_settings.json')

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
    def __init__(self, workstations, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Launch Training Job')
        self.resize(600, 580)
        self._workstations = workstations
        self._settings = _load_settings()
        self._running = False
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

        # Config file
        cfg_row = QHBoxLayout()
        self._cfg_edit = QLineEdit(); self._cfg_edit.setPlaceholderText('Path to config.yaml')
        self._cfg_edit.textChanged.connect(self._on_config_changed)
        cfg_row.addWidget(self._cfg_edit)
        browse = QPushButton('Browse'); browse.setFixedWidth(70)
        browse.clicked.connect(lambda: self._cfg_edit.setText(
            QFileDialog.getOpenFileName(self,'Config YAML','','YAML (*.yaml *.yml)')[0] or self._cfg_edit.text()))
        cfg_row.addWidget(browse)
        form.addRow('Config YAML:', cfg_row)

        # Workstation selector
        self._ws_combo = QComboBox(); self._ws_combo.setMinimumWidth(380)
        self._populate_ws_combo()
        form.addRow('Workstation:', self._ws_combo)

        # Data info
        self._data_info = QLabel('— will be read from config src_dir / dst_dir —')
        self._data_info.setStyleSheet('color:#6b7280; font-size:11px; font-style:italic;')
        form.addRow('Data:', self._data_info)

        # Checkpoint
        ckpt_row = QHBoxLayout()
        self._ckpt_combo = QComboBox(); self._ckpt_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._ckpt_info = QLabel(''); self._ckpt_info.setStyleSheet('color:#6b7280; font-size:11px;')
        ckpt_row.addWidget(self._ckpt_combo); ckpt_row.addWidget(self._ckpt_info)
        form.addRow('Checkpoint:', ckpt_row)

        # Options
        opts_row = QHBoxLayout()
        self._skip_code_cb = QCheckBox('Skip code upload')
        self._terminate_cb = QCheckBox('Stop workstation when done')
        opts_row.addWidget(self._skip_code_cb); opts_row.addWidget(self._terminate_cb); opts_row.addStretch()
        form.addRow('Options:', opts_row)

        root.addWidget(cfg_group)

        # SSH note
        note = QLabel('⚠  SSH details endpoint not yet known — ask Walt for /api/workstations/{id}/access.\nLaunch will fail at the SSH step until that\'s resolved.')
        note.setStyleSheet('color:#D97706; font-size:11px; padding:6px 0;')
        note.setWordWrap(True)
        root.addWidget(note)

        btn_row = QHBoxLayout()
        self._launch_btn = QPushButton('Launch Job')
        self._launch_btn.setFixedHeight(34)
        self._launch_btn.setStyleSheet(
            'QPushButton { background:#ae69f4; color:white; border:none; border-radius:6px; font-weight:600; font-size:13px; }'
            'QPushButton:hover { background:#9b50e8; }'
            'QPushButton:disabled { background:#d8b4fe; color:#f3e8ff; }')
        self._launch_btn.clicked.connect(self._do_launch)
        self._close_btn = QPushButton('Close'); self._close_btn.setFixedHeight(34); self._close_btn.setFixedWidth(80)
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._launch_btn); btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

        log_group = QGroupBox('Launch Log'); log_group.setStyleSheet('QGroupBox { font-weight:600; }')
        log_vbox = QVBoxLayout(log_group)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); self._log.setMinimumHeight(180)
        self._log.setStyleSheet('QPlainTextEdit { background:#1e1e2e; color:#cdd6f4; font-family:Consolas,monospace; font-size:12px; border-radius:6px; padding:8px; }')
        log_vbox.addWidget(self._log)
        root.addWidget(log_group)

    def _populate_ws_combo(self):
        self._ws_combo.clear()
        stopped = [w for w in self._workstations if not is_running(w)]
        running = [w for w in self._workstations if is_running(w)]
        for w in stopped + running:
            label = f"{w.get('workstation_name', w['id'])}  —  {w.get('gpu','?')}  ({ws_status(w)})  ${w.get('cost',0):.2f}/hr"
            self._ws_combo.addItem(label, w['id'])

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
            if out_dir and os.path.isdir(out_dir):
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location('rl', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runpod_launch.py'))
                rl = _ilu.module_from_spec(spec); spec.loader.exec_module(rl)
                ckpts = rl.scan_checkpoints(out_dir)
                for ckpt_path, is_latest in ckpts:
                    label = os.path.basename(ckpt_path)
                    size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
                    self._ckpt_combo.addItem(f'{label}{" [latest]" if is_latest else ""}  ({size_mb:.0f} MB)', ckpt_path)
                self._ckpt_info.setText(f'{len(ckpts)} checkpoint{"s" if len(ckpts)!=1 else ""} found' if ckpts else 'no .pth found — will train fresh')
        except Exception: pass

    def _restore(self):
        s = self._settings
        if s.get('config'): self._cfg_edit.setText(s['config'])
        if s.get('ws_id'):
            for i in range(self._ws_combo.count()):
                if self._ws_combo.itemData(i) == s['ws_id']:
                    self._ws_combo.setCurrentIndex(i); break
        self._skip_code_cb.setChecked(s.get('skip_code', False))
        self._terminate_cb.setChecked(s.get('terminate', False))

    def _log_line(self, text):
        self._log.appendPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _do_launch(self):
        if self._running: return
        config_path = self._cfg_edit.text().strip()
        if not config_path or not os.path.isfile(config_path):
            QMessageBox.warning(self, 'No Config', 'Select a valid config.yaml.'); return
        ws_id = self._ws_combo.currentData()
        if ws_id is None:
            QMessageBox.warning(self, 'No Workstation', 'Select a workstation.'); return
        ws = next((w for w in self._workstations if w['id'] == ws_id), None)
        _save_settings({'config': config_path, 'ws_id': ws_id,
                        'skip_code': self._skip_code_cb.isChecked(),
                        'terminate': self._terminate_cb.isChecked()})
        self._running = True
        self._launch_btn.setEnabled(False)
        self._close_btn.setText('Running...'); self._close_btn.setEnabled(False)
        self._log.clear()

        stream = _LogStream(); stream.line.connect(self._log_line)
        skip_code = self._skip_code_cb.isChecked()
        terminate = self._terminate_cb.isChecked()
        resume_pth = self._ckpt_combo.currentData()

        def _run():
            import sys as _sys, copy, tempfile as _tmp
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location('rl', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runpod_launch.py'))
            rl = _ilu.module_from_spec(spec); spec.loader.exec_module(rl)

            old = _sys.stdout; _sys.stdout = stream
            try:
                # Start workstation if stopped
                if ws and not is_running(ws):
                    print(f'[spark] Starting workstation {ws_id}...')
                    start_workstation(ws_id)
                    deadline = time.time() + ws.get('instance_startup_time', 900) + 120
                    while time.time() < deadline:
                        time.sleep(10)
                        w = get_workstation(ws_id)
                        print(f'[spark] os_ready={w.get("os_ready")}...')
                        if is_running(w): break
                    else:
                        print('[spark] ERROR: workstation did not start in time.'); return

                # Get SSH details
                host, port, user = get_ssh_details({'id': ws_id})
                if not host:
                    print('[spark] ERROR: SSH details not available.')
                    print('        Ask Walt for GET /api/workstations/{id}/access')
                    print('        Expected response: { public_ip, ssh_port, ssh_user }')
                    return

                print(f'[ssh] Connecting to {user}@{host}:{port}')
                with open(config_path) as f: config = yaml.safe_load(f)
                data_uploads = {}
                pod_config, job_name = rl.rewrite_config_for_pod(config, data_uploads, resume_pth=resume_pth)

                key_path = SSH_KEY
                rl.wait_for_ssh(host, port, key_path)

                if not skip_code:
                    tunet_root = os.path.dirname(os.path.abspath(__file__))
                    print(f'[upload] Syncing tunet source → {rl.TUNET_REMOTE}')
                    rl.ssh(host, port, key_path, f'mkdir -p {rl.TUNET_REMOTE}')
                    rl.rsync(tunet_root + '/', host, port, key_path, rl.TUNET_REMOTE,
                             exclude=['__pycache__','*.pyc','.git','node_modules','Spark','_archive','*.pth','*.onnx','tunet_session.yaml','tunet-web'])

                if data_uploads:
                    for d in {os.path.dirname(p) for p in data_uploads.values()}:
                        rl.ssh(host, port, key_path, f'mkdir -p "{d}"')
                    for lp, rp in data_uploads.items():
                        if os.path.isdir(lp):
                            print(f'[upload] {lp} → {rp}')
                            rl.rsync(lp + '/', host, port, key_path, rp)
                        else:
                            print(f'[upload] {os.path.basename(lp)} → {rp}')
                            rl.scp(lp, host, port, key_path, rp)

                with _tmp.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
                    yaml.dump(pod_config, tf, default_flow_style=False, sort_keys=False)
                    tmp_cfg = tf.name
                print(f'[upload] Config → {rl.CONFIG_REMOTE}')
                rl.scp(tmp_cfg, host, port, key_path, rl.CONFIG_REMOTE)
                os.unlink(tmp_cfg)

                start_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runpod_start.sh')
                rl.scp(start_sh, host, port, key_path, '/workspace/runpod_start.sh')
                rl.ssh(host, port, key_path, 'chmod +x /workspace/runpod_start.sh')

                output_dir = f'{rl.OUTPUT_REMOTE}/{job_name}'
                launch_cmd = (f'nohup bash /workspace/runpod_start.sh '
                              f'"{output_dir}" "{rl.CONFIG_REMOTE}" "{rl.MONITOR_PORT}" '
                              f'> /workspace/launch.log 2>&1 &')
                rl.ssh(host, port, key_path, launch_cmd)

                print(f'\n✓ Training started!')
                print(f'  Workstation : {ws_id}')
                print(f'  Output      : {output_dir}')
                murl = monitor_url(ws_id)
                if murl: print(f'  Monitor     : {murl}')
                print(f'  SSH         : ssh -i {key_path} -p {port} {user}@{host}')

                if terminate:
                    print('\n[spark] Waiting for training to finish before stopping...')
                    # Poll until training.log contains completion marker
                    log_path = f'{output_dir}/training.log'
                    deadline = time.time() + 86400
                    while time.time() < deadline:
                        time.sleep(60)
                        r = rl.ssh(host, port, key_path,
                                   f'grep -c "Reached max_steps\\|Training complete" {log_path} 2>/dev/null',
                                   check=False, capture=True)
                        if r.stdout.strip() not in ('', '0'):
                            print('[spark] Training finished. Stopping workstation...')
                            stop_workstation(ws_id)
                            break

            except Exception as e:
                print(f'\nERROR: {e}')
            finally:
                _sys.stdout = old
                self._running = False
                self._launch_btn.setEnabled(True)
                self._close_btn.setText('Close'); self._close_btn.setEnabled(True)

        threading.Thread(target=_run, daemon=True).start()


# ── App icon (reused from runpod_dashboard) ───────────────────────────────────

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
    app.setApplicationName('TuNet Spark Dashboard')
    app.setWindowIcon(_make_icon())
    win = Dashboard()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
