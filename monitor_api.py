"""
TuNet Monitor API — runs on the RunPod pod alongside train.py.

Serves training metrics, logs, and preview images over HTTP so the
Spark web UI can poll them from anywhere.

Usage:
    python monitor_api.py --output_dir /workspace/output --port 8080

Endpoints:
    GET  /api/status      — current step, loss, elapsed, ETA, status
    GET  /api/metrics     — full loss history for chart rendering
    GET  /api/logs        — last N lines of training.log
    GET  /api/preview     — training_preview.jpg (binary)
    GET  /api/val_preview — val_preview.jpg (binary)
    POST /api/stop        — write .stop_training sentinel
    GET  /health          — liveness check
"""

import argparse
import json
import os
import re
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock

# ── Log parsing (same patterns as training_monitor.py) ──────────────────────

TRAIN_PAT  = re.compile(r'Epoch\[(\d+)\]\s*Step\[(\d+)\].*?\b(L1|L2|BCE\+Dice):([\d.]+)')
LPIPS_PAT  = re.compile(r'LPIPS:([\d.]+)')
TIME_PAT   = re.compile(r'T/Step:([\d.]+)s')
ITER_PAT   = re.compile(r'\((\d+)/(\d+)\)')
VAL_PAT    = re.compile(r'Val Epoch\[(\d+)\]\s*Step\[(\d+)\].*?Val_(L1|L2|BCE\+Dice):([\d.]+)')
VAL_LP_PAT = re.compile(r'Val_LPIPS:([\d.]+)')
PSNR_PAT   = re.compile(r'PSNR:([\d.]+)dB')
SSIM_PAT   = re.compile(r'SSIM:([\d.]+)')

MAX_LOG_LINES = 500
MAX_METRICS   = 50_000


class TrainingState:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.lock = Lock()

        # metrics
        self.steps       = deque(maxlen=MAX_METRICS)
        self.l1          = deque(maxlen=MAX_METRICS)
        self.lpips       = deque(maxlen=MAX_METRICS)
        self.val_steps   = deque(maxlen=MAX_METRICS)
        self.val_l1      = deque(maxlen=MAX_METRICS)
        self.val_lpips   = deque(maxlen=MAX_METRICS)
        self.val_psnr    = deque(maxlen=MAX_METRICS)
        self.val_ssim    = deque(maxlen=MAX_METRICS)

        self.best_l1        = float('inf')
        self.best_l1_step   = 0
        self.has_lpips      = False
        self.has_val        = False
        self.loss_label     = 'L1'
        self.last_step_time = None   # seconds per step (most recent T/Step value)

        # status
        self.start_time   = time.time()
        self.current_step = 0
        self.current_loss = None
        self.is_running   = True     # flipped to False when log file goes stale

        # log tail
        self.log_lines = deque(maxlen=MAX_LOG_LINES)

        # file positions
        self._log_pos     = 0
        self._log_mtime   = 0

    @property
    def log_path(self):
        return os.path.join(self.output_dir, 'training.log')

    @property
    def stop_path(self):
        return os.path.join(self.output_dir, '.stop_training')

    @property
    def preview_path(self):
        return os.path.join(self.output_dir, 'training_preview.jpg')

    @property
    def val_preview_path(self):
        return os.path.join(self.output_dir, 'val_preview.jpg')

    def poll(self):
        """Read any new log content. Called periodically by the server."""
        path = self.log_path
        if not os.path.exists(path):
            return

        try:
            mtime = os.path.getmtime(path)
            with self.lock:
                # Detect if training has gone stale (no log update in 120s)
                if mtime == self._log_mtime and self.is_running:
                    if time.time() - self._last_seen > 120:
                        self.is_running = False

                if mtime == self._log_mtime:
                    return

                self._log_mtime = mtime
                self._last_seen = time.time()
                self.is_running = True

                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self._log_pos)
                    new_text = f.read()
                    self._log_pos = f.tell()

                if new_text:
                    self._parse(new_text)
        except Exception:
            pass

    def _parse(self, text):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                self.log_lines.append(stripped)

            m = TRAIN_PAT.search(line)
            if m:
                epoch = int(m.group(1))
                detected_label = m.group(3)
                if not self.steps:
                    self.loss_label = detected_label
                l1_val = float(m.group(4))

                lp_m = LPIPS_PAT.search(line)
                lpips_val = float(lp_m.group(1)) if lp_m else None

                t_m = TIME_PAT.search(line)
                if t_m:
                    self.last_step_time = float(t_m.group(1))

                it_m = ITER_PAT.search(line)
                if it_m:
                    step_in_ep = int(it_m.group(1))
                    total      = int(it_m.group(2))
                    x = epoch - 1 + step_in_ep / total
                    self.current_step = (epoch - 1) * total + step_in_ep
                else:
                    x = epoch
                    self.current_step = epoch

                self.steps.append(x)
                self.l1.append(l1_val)
                self.current_loss = l1_val

                if l1_val < self.best_l1:
                    self.best_l1      = l1_val
                    self.best_l1_step = self.current_step

                if lpips_val is not None:
                    self.lpips.append(lpips_val)
                    self.has_lpips = True
                elif self.has_lpips:
                    self.lpips.append(self.lpips[-1] if self.lpips else 0)
                continue

            v = VAL_PAT.search(line)
            if v:
                self.has_val = True
                self.val_steps.append(int(v.group(1)))
                self.val_l1.append(float(v.group(4)))
                vl = VAL_LP_PAT.search(line)
                if vl:
                    self.val_lpips.append(float(vl.group(1)))
                psnr = PSNR_PAT.search(line)
                if psnr:
                    self.val_psnr.append(float(psnr.group(1)))
                ssim = SSIM_PAT.search(line)
                if ssim:
                    self.val_ssim.append(float(ssim.group(1)))

    def status_dict(self):
        elapsed = int(time.time() - self.start_time)
        eta_s = None
        if self.last_step_time and self.steps:
            # Can't know max_steps without config; report step/s instead
            pass

        with self.lock:
            return {
                'running':      self.is_running,
                'step':         self.current_step,
                'loss':         round(self.current_loss, 6) if self.current_loss is not None else None,
                'loss_label':   self.loss_label,
                'best_loss':    round(self.best_l1, 6) if self.best_l1 != float('inf') else None,
                'best_step':    self.best_l1_step,
                'elapsed_s':    elapsed,
                'elapsed':      _fmt_duration(elapsed),
                'step_time_s':  round(self.last_step_time, 3) if self.last_step_time else None,
                'has_lpips':    self.has_lpips,
                'has_val':      self.has_val,
                'stop_file':    os.path.exists(self.stop_path),
            }

    def metrics_dict(self):
        with self.lock:
            d = {
                'steps':      list(self.steps),
                'l1':         list(self.l1),
                'loss_label': self.loss_label,
            }
            if self.has_lpips:
                d['lpips'] = list(self.lpips)
            if self.has_val:
                d['val_steps'] = list(self.val_steps)
                d['val_l1']    = list(self.val_l1)
                if self.val_lpips:
                    d['val_lpips'] = list(self.val_lpips)
                if self.val_psnr:
                    d['val_psnr'] = list(self.val_psnr)
                if self.val_ssim:
                    d['val_ssim'] = list(self.val_ssim)
            return d

    def logs_dict(self, n=200):
        with self.lock:
            lines = list(self.log_lines)[-n:]
        return {'lines': lines}


def _fmt_duration(s):
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# ── HTTP Handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    state: TrainingState = None  # set at startup

    def log_message(self, *args):
        pass  # suppress per-request logs

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_image(self, path):
        if not os.path.exists(path):
            self._send_json({'error': 'not found'}, 404)
            return
        with open(path, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

    def do_GET(self):
        self.state.poll()
        p = self.path.split('?')[0]
        if p == '/health':
            self._send_json({'ok': True})
        elif p == '/api/status':
            self._send_json(self.state.status_dict())
        elif p == '/api/metrics':
            self._send_json(self.state.metrics_dict())
        elif p == '/api/logs':
            self._send_json(self.state.logs_dict())
        elif p == '/api/preview':
            self._send_image(self.state.preview_path)
        elif p == '/api/val_preview':
            self._send_image(self.state.val_preview_path)
        else:
            self._send_json({'error': 'not found'}, 404)

    def do_POST(self):
        if self.path == '/api/stop':
            try:
                open(self.state.stop_path, 'w').close()
                self._send_json({'ok': True, 'message': 'Stop file written'})
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        else:
            self._send_json({'error': 'not found'}, 404)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', required=True, help='Training output directory (contains training.log)')
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    Handler.state = TrainingState(args.output_dir)
    # Pre-read any existing log content (resumed run)
    Handler.state._last_seen = time.time()
    Handler.state.poll()

    server = HTTPServer(('0.0.0.0', args.port), Handler)
    print(f'[monitor_api] Serving on port {args.port} — output_dir={args.output_dir}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
