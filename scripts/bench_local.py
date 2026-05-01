#!/usr/bin/env python
"""
Local-GPU throughput benchmark for the cost estimator.

Sweeps (resolution × batch_size × config) on the local GPU, calls
train.py with --benchmark-steps, and grep STEP_RATE: out of the logs.

Run from the tunet repo root in the `tunet` conda env:
    conda activate tunet
    python scripts/bench_local.py --src /path/to/src --dst /path/to/dst

Default sweep: 256/512px × bs2/4/8 × {ref, ref+lpips, porter}.
1024px is included only if the source images are >= 1024 in both dims.

Each cell writes a temp config + invokes train.py as a subprocess.
Output is appended to scripts/bench_results.jsonl so you can re-run and
collect more data without losing prior measurements.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
TRAIN_PY      = REPO_ROOT / "train.py"
RESULTS_PATH  = REPO_ROOT / "scripts" / "bench_results.jsonl"
STEP_RATE_RE  = re.compile(r"STEP_RATE:\s*([\d.]+)\s*step/sec")

DEFAULT_RES     = [256, 512, 1024]
DEFAULT_BATCHES = [2, 4, 8]

# Same scenarios as the cost-grid script. Reference is the calibration
# baseline; the rest let us validate settingsMultiplier().
SCENARIOS = {
    "ref":       dict(model_size_dims=64,  model_type="unet", loss="l1"),
    "ref+lpips": dict(model_size_dims=64,  model_type="unet", loss="l1+lpips"),
    "porter":    dict(model_size_dims=128, model_type="msrn", loss="l1+lpips"),
}

def build_config(src_dir: Path, dst_dir: Path, output_dir: Path,
                 resolution: int, batch_size: int, scenario: dict) -> dict:
    """Mirror what tunet-web/.../benchmark/route.ts produces."""
    return {
        "data": {
            "src_dir":        str(src_dir),
            "dst_dir":        str(dst_dir),
            "output_dir":     str(output_dir),
            "resolution":     resolution,
            "overlap_factor": 0.25,
            "color_space":    "srgb",
            "mask_dir":       None,
            "val_src_dir":    None,
            "val_dst_dir":    None,
        },
        "mask": {
            "use_mask_loss":        False,
            "mask_weight":          10.0,
            "use_mask_input":       False,
            "use_auto_mask":        False,
            "auto_mask_gamma":      1.0,
            "skip_empty_patches":   False,
            "skip_empty_threshold": 3.0,
        },
        "model": {
            "model_type":       scenario["model_type"],
            "model_size_dims":  scenario["model_size_dims"],
            "recurrence_steps": 2,
        },
        "training": {
            "loss":                   scenario["loss"],
            "lr":                     1e-4,
            "lr_scheduler":           "none",
            "lambda_lpips":           0.2,
            "l1_weight":              0.5,
            "l2_weight":              0.5,
            "lpips_weight":           0.2,
            "use_amp":                True,
            "batch_size":             batch_size,
            "iterations_per_epoch":   1000,
            "max_steps":              0,
            "progressive_resolution": False,
        },
        "augmentations": {},
        "logging": {
            "log_interval":           50,
            "preview_batch_interval": 0,
            "preview_refresh_rate":   0,
            "val_interval":           0,
            "diff_amplify":           1.0,
        },
        "saving":         {"keep_last_checkpoints": 0},
        "early_stopping": {"enabled": False},
        "auto_export":    {"auto_export_interval": 0,
                           "auto_export_flame": False,
                           "auto_export_nuke":  False},
        "dataloader":     {"num_workers": 2, "prefetch_factor": 2},
    }

def get_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"

def src_dims(src_dir: Path) -> tuple[int, int] | None:
    """Probe first src image to see if 1024px patches are even valid."""
    try:
        from PIL import Image
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            for p in src_dir.glob(ext):
                with Image.open(p) as im:
                    return im.size
    except Exception:
        pass
    # EXR — quick header probe via OpenEXR; bail on any issue
    try:
        import OpenEXR
        for p in src_dir.glob("*.exr"):
            ex = OpenEXR.InputFile(str(p))
            dw = ex.header()["dataWindow"]
            return (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    except Exception:
        pass
    return None

def run_cell(src_dir: Path, dst_dir: Path,
             resolution: int, batch_size: int, scenario_name: str, scenario: dict,
             warmup: int, steps: int) -> dict:
    """Invoke train.py once for a single benchmark cell."""
    import yaml
    work = Path(tempfile.mkdtemp(prefix="bench-local-"))
    try:
        out_dir = work / "output"
        out_dir.mkdir()
        cfg = build_config(src_dir, dst_dir, out_dir, resolution, batch_size, scenario)
        # tunet expects a "base" config sibling — easiest is to drop it next
        # to the user config and write a no-op base. base/base.yaml already
        # exists in repo; train.py looks for it relative to the user config
        # dir AND the repo root, so the repo-root copy is fine.
        cfg_path = work / "bench.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        env = os.environ.copy()
        env["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        cmd = [
            sys.executable, str(TRAIN_PY),
            "--config",          str(cfg_path),
            "--benchmark-steps", str(steps),
            "--benchmark-warmup", str(warmup),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, cwd=str(REPO_ROOT))
        elapsed = time.time() - t0

        # train.py logs through `logging`, which goes to stderr.
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = STEP_RATE_RE.search(combined)
        rate = float(m.group(1)) if m else None

        # capture a small error blurb so failed cells are useful
        err_tail = ""
        if rate is None:
            tail_lines = [l for l in combined.splitlines()
                          if l.strip() and ("error" in l.lower() or "traceback" in l.lower()
                                             or "valueerror" in l.lower() or "runtimeerror" in l.lower())]
            err_tail = " | ".join(tail_lines[-3:])[:300]

        return {
            "ts":        datetime.now().isoformat(timespec="seconds"),
            "gpu":       get_gpu_name(),
            "resolution": resolution,
            "batch":     batch_size,
            "scenario":  scenario_name,
            "config":    scenario,
            "step_rate": rate,
            "elapsed_s": round(elapsed, 1),
            "rc":        proc.returncode,
            "error":     err_tail or None,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="src/ directory of training pairs")
    p.add_argument("--dst", required=True, help="dst/ directory")
    p.add_argument("--res", type=int, nargs="+", default=DEFAULT_RES,
                   help=f"resolutions to sweep (default: {DEFAULT_RES})")
    p.add_argument("--batch", type=int, nargs="+", default=DEFAULT_BATCHES,
                   help=f"batch sizes to sweep (default: {DEFAULT_BATCHES})")
    p.add_argument("--scenarios", nargs="+", default=["ref"],
                   choices=list(SCENARIOS),
                   help=f"which scenarios to run. Default: ref. Available: {list(SCENARIOS)}")
    p.add_argument("--steps", type=int, default=200,
                   help="timed steps per cell (default 200)")
    p.add_argument("--warmup", type=int, default=20,
                   help="warmup steps to discard (default 20)")
    p.add_argument("--results", default=str(RESULTS_PATH),
                   help="JSONL file to append results to")
    args = p.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    if not src_dir.is_dir() or not dst_dir.is_dir():
        sys.exit(f"src or dst not found: {src_dir} / {dst_dir}")

    # Filter resolutions if source is too small
    dims = src_dims(src_dir)
    if dims:
        w, h = dims
        too_big = [r for r in args.res if r > min(w, h)]
        if too_big:
            print(f"⚠ source is {w}×{h}; dropping resolutions {too_big} (need pixels ≥ res)")
            args.res = [r for r in args.res if r <= min(w, h)]

    cells = [(r, b, s) for s in args.scenarios for r in args.res for b in args.batch]
    print(f"▸ Local benchmark: {len(cells)} cells on {get_gpu_name()}")
    print(f"  src={src_dir}")
    print(f"  dst={dst_dir}")
    print(f"  resolutions={args.res}  batch_sizes={args.batch}  scenarios={args.scenarios}")
    print()

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("a", encoding="utf-8")

    for i, (res, bs, scen) in enumerate(cells, 1):
        tag = f"[{i:2d}/{len(cells)}] {scen:<10}  {res:>4}px  bs{bs}"
        print(f"{tag}  …", end=" ", flush=True)
        result = run_cell(src_dir, dst_dir, res, bs, scen, SCENARIOS[scen],
                          args.warmup, args.steps)
        f.write(json.dumps(result) + "\n")
        f.flush()
        if result["step_rate"] is not None:
            samp = result["step_rate"] * bs
            print(f"✓ {result['step_rate']:.2f} step/s  ({samp:.1f} samp/s)  in {result['elapsed_s']}s")
        else:
            print(f"✗ failed in {result['elapsed_s']}s  rc={result['rc']}")
            if result["error"]: print(f"      {result['error']}")

    f.close()
    print()
    print(f"Done. Results appended to {out}")

if __name__ == "__main__":
    main()
