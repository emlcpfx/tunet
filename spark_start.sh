#!/bin/bash
# ============================================================
# TuNet Spark Compute v1 startup script
# Runs inside the agent-managed container. The Spark agent owns
# the container lifetime: when this script exits, /output/ is
# uploaded to ShareSync and the instance enters idle-hold.
#
# Args:  $1 = output_dir   (e.g. /output/my_job)
#        $2 = config path  (e.g. /input/config.yaml)
# ============================================================

set -e

OUTPUT_DIR="${1:-/output}"
CONFIG="${2:-/input/config.yaml}"
TUNET_DIR="/input/tunet"

echo "=== TuNet Spark Compute Start ==="
echo "  output_dir : $OUTPUT_DIR"
echo "  config     : $CONFIG"
echo "  tunet_dir  : $TUNET_DIR"
echo ""

# ── 1. Pick the newest available Python ─────────────────────
# The runpod/pytorch image ships both 3.10 (as `python3`) and 3.12
# (where bare `pip` is bound). Our pinned scipy/numpy require ≥3.11,
# so detect explicitly. Walt's doc, changelog item #6.
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "[setup] Using Python: $PY ($($PY --version))"

# ── 2. Install dependencies ─────────────────────────────────
echo "[setup] Installing dependencies..."
$PY -m pip install -q \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

$PY -m pip install -q \
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

# ── 3. Create output dir ─────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── 3b. Seed from staged resume checkpoint, if any ──────────
# spark_launch.py stages resume .pth + training.log under
# /input/output/<job>/ inside the tarball; copy them into the
# real /output/<job>/ so train.py picks them up on first run.
JOB_NAME=$(basename "$OUTPUT_DIR")
STAGED="/input/output/$JOB_NAME"
if [ -d "$STAGED" ]; then
    echo "[seed] Copying staged files from $STAGED → $OUTPUT_DIR"
    cp -an "$STAGED"/* "$OUTPUT_DIR"/ 2>/dev/null || true
fi

# ── 4. Run training (foreground — agent owns lifetime) ──────
# No nohup, no monitor_api, no auto-terminate. The Spark agent
# uploads /output/ when this process exits.
#
# BENCHMARK_STEPS env var opts into a short calibration run: train.py runs
# warmup + N steps, logs STEP_RATE, exits. Used by tunet-web to baseline
# the cost estimator. Skip the file-write the regular run does.
echo ""
echo "[train] Starting train.py..."
export OPENCV_IO_ENABLE_OPENEXR=1
cd "$TUNET_DIR"

BENCH_ARGS=""
if [ -n "$BENCHMARK_STEPS" ] && [ "$BENCHMARK_STEPS" -gt 0 ] 2>/dev/null; then
    BENCH_ARGS="--benchmark-steps $BENCHMARK_STEPS"
    if [ -n "$BENCHMARK_WARMUP" ] && [ "$BENCHMARK_WARMUP" -gt 0 ] 2>/dev/null; then
        BENCH_ARGS="$BENCH_ARGS --benchmark-warmup $BENCHMARK_WARMUP"
    fi
    echo "[bench] BENCHMARK MODE: $BENCH_ARGS"
fi

$PY train.py --config "$CONFIG" --stop-file "$OUTPUT_DIR/.stop_training" $BENCH_ARGS
echo "[train] Finished with exit $?"
