#!/bin/bash
# ============================================================
# TuNet RunPod startup script
# Runs inside the pod container. Called by runpod_launch.py
# via SSH after upload.
#
# Args:  $1 = output_dir (e.g. /workspace/output/my_job)
#        $2 = config path (e.g. /workspace/config.yaml)
#        $3 = monitor port (default 8080)
# ============================================================

set -e

OUTPUT_DIR="${1:-/workspace/output}"
CONFIG="${2:-/workspace/config.yaml}"
MONITOR_PORT="${3:-8080}"
TUNET_DIR="/workspace/tunet"

# ── Auto-terminate on idle ───────────────────────────────────
# If training finishes or crashes, terminate the pod after this
# many minutes so it doesn't bill indefinitely.
IDLE_TIMEOUT_MINUTES=30

_auto_terminate() {
    echo "[watchdog] Training ended (exit $1). Waiting ${IDLE_TIMEOUT_MINUTES}m before auto-terminating pod..."
    sleep $(( IDLE_TIMEOUT_MINUTES * 60 ))
    echo "[watchdog] Terminating pod ${RUNPOD_POD_ID}..."
    # RunPod pods can self-terminate via the metadata API
    curl -s --max-time 10 \
        -X POST "https://api.runpod.io/v2/pod/${RUNPOD_POD_ID}/stop" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" 2>/dev/null || true
    # Fallback: halt the machine (pod billing stops when container exits)
    kill 1
}

echo "=== TuNet RunPod Start ==="
echo "  output_dir   : $OUTPUT_DIR"
echo "  config       : $CONFIG"
echo "  monitor_port : $MONITOR_PORT"
echo "  tunet_dir    : $TUNET_DIR"
echo ""

# ── 1. Install dependencies (idempotent) ────────────────────
echo "[setup] Installing dependencies..."
pip install -q \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -q \
    albumentations==2.0.8 albucore==0.0.24 \
    opencv-python-headless==4.12.0.88 \
    lpips==0.1.4 \
    numpy==2.2.6 scipy==1.16.3 \
    pillow==12.0.0 \
    OpenEXR==3.2.4 \
    pyyaml==6.0.3 tqdm==4.67.1 \
    coloredlogs==15.0.1 humanfriendly==10.0 \
    onnx==1.20.0 onnxruntime-gpu==1.20.0

echo "[setup] Done."

# ── 2. Create output dir ─────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── 3. Start monitor API in background ──────────────────────
echo "[monitor] Starting monitor_api on port $MONITOR_PORT..."
cd "$TUNET_DIR"
nohup python monitor_api.py \
    --output_dir "$OUTPUT_DIR" \
    --port "$MONITOR_PORT" \
    > "$OUTPUT_DIR/monitor.log" 2>&1 &
MONITOR_PID=$!
echo "[monitor] PID $MONITOR_PID"
sleep 2

# Confirm it started
if kill -0 $MONITOR_PID 2>/dev/null; then
    echo "[monitor] Running — https://${RUNPOD_POD_ID}-${MONITOR_PORT}.proxy.runpod.net"
else
    echo "[monitor] WARNING: monitor_api failed to start, check $OUTPUT_DIR/monitor.log"
fi

# ── 4. Run training ──────────────────────────────────────────
echo ""
echo "[train] Starting train.py..."
export OPENCV_IO_ENABLE_OPENEXR=1
cd "$TUNET_DIR"
python train.py --config "$CONFIG" --stop-file "$OUTPUT_DIR/.stop_training"
TRAIN_EXIT=$?

echo ""
echo "[train] Finished with exit code $TRAIN_EXIT"

# ── 5. Signal monitor that training is done ──────────────────
touch "$OUTPUT_DIR/training.log" 2>/dev/null || true

# ── 6. Auto-terminate after idle timeout ─────────────────────
_auto_terminate $TRAIN_EXIT
