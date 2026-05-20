#!/bin/bash
# ============================================================
# TuNet Spark ONNX-export job entrypoint.
#
# Runs a *much* lighter pipeline than spark_start.sh: no
# albumentations / lpips / opencv / scipy / OpenEXR — we only
# need torch + onnx to rebuild the model and export it.
#
# Args (inherited from the submit `command`):
#   $1  output_dir   (e.g. /output/babies_resume_v7_instant)
#   $2  config path  (unused — kept for arg-shape parity with spark_start.sh)
#
# Env (set by the export-onnx submit route):
#   TUNET_BEARER     Bearer for ShareSync WebDAV (short-lived JWT). NOT
#                    `SPARK_TOKEN` — Spark reserves the SPARK_ prefix.
#   CHECKPOINT_URL   Full WebDAV URL of the .pth to export
#   CKPT_PREFIX      Optional filename prefix override
#   EPOCH_OVERRIDE   Optional epoch number override
# ============================================================

set -e

OUTPUT_DIR="${1:-/output/export}"
TUNET_DIR="/input/tunet"

echo "=== TuNet Spark ONNX Export ==="
echo "  output_dir     : $OUTPUT_DIR"
echo "  checkpoint_url : ${CHECKPOINT_URL:-<unset>}"
echo "  tunet_dir      : $TUNET_DIR"
echo ""

if [ -z "$CHECKPOINT_URL" ] || [ -z "$TUNET_BEARER" ]; then
    echo "[fatal] CHECKPOINT_URL and TUNET_BEARER must be set in env" >&2
    exit 2
fi

# ── 1. Pick Python (image ships torch 2.7.1 on python3.10 + python3.12) ──
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "[setup] Python: $PY ($($PY --version))"

# ── 2. Install ONLY what export needs on top of the image ─────────────
# runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204 already ships torch 2.7.1,
# numpy, pyyaml. We just need onnx for torch.onnx.export's writer backend.
echo "[setup] Installing onnx..."
$PY -m pip install -q onnx==1.20.0

# ── 3. Fetch the .pth from ShareSync ─────────────────────────────────
mkdir -p /tmp/export
CKPT_PATH=/tmp/export/checkpoint.pth
echo "[fetch] Downloading checkpoint..."
# -f: fail on HTTP error, -L: follow redirects, -sS: silent + show errors
curl -fL -sS -o "$CKPT_PATH" \
    -H "Authorization: Bearer $TUNET_BEARER" \
    "$CHECKPOINT_URL"
CKPT_SIZE=$(stat -c%s "$CKPT_PATH" 2>/dev/null || stat -f%z "$CKPT_PATH")
echo "[fetch] Got $CKPT_SIZE bytes -> $CKPT_PATH"

# ── 4. Run the export script ─────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cd "$TUNET_DIR"

EXTRA_ARGS=""
if [ -n "$CKPT_PREFIX" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --ckpt-prefix $CKPT_PREFIX"
fi
if [ -n "$EPOCH_OVERRIDE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --epoch $EPOCH_OVERRIDE"
fi

echo ""
echo "[export] Running spark_export_onnx.py..."
$PY scripts/spark_export_onnx.py \
    --checkpoint "$CKPT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS
echo "[export] Finished with exit $?"
