#!/usr/bin/env python
"""
Read scripts/bench_results.jsonl + the inline cloud measurements and
back-fit empirical curves for resolutionPenalty(), batchMultiplier(), and
settingsMultiplier() in tunet-web/src/lib/spark-presets.ts.

This is *analysis*, not a benchmark. It produces:
  - A fitted formula per axis (with the residuals) printed to stdout
  - A suggested patch for spark-presets.ts (also printed; copy-paste manually)
  - Optional: append cloud measurements to bench_results.jsonl so the file
    is a single source of truth across machines

Run:  python scripts/calibrate_cost_model.py [--include-cloud]
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
JSONL = REPO / "scripts" / "bench_results.jsonl"
# Committed cloud snapshots — survive across machines (the per-machine
# bench_results.jsonl is gitignored). All matching scripts/bench_cloud_*.jsonl
# files load automatically. Add a new file when capturing new measurements
# instead of editing prior ones.
CLOUD_SNAPSHOT_GLOB = "bench_cloud_*.jsonl"

# Cloud measurements live in scripts/bench_cloud_2026-05-01.jsonl —
# committed alongside this script so they're available across machines
# (per-machine bench_results.jsonl is gitignored).

def load_jsonl(p: Path):
    if not p.exists(): return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out

def filter_valid(rows):
    """Drop cells that swapped to system RAM (step_rate < 0.5 of expected) or have None rate."""
    return [r for r in rows if r.get("step_rate") and r["step_rate"] > 0.1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-cloud", action="store_true",
                    help="Append cloud measurements into bench_results.jsonl too")
    args = ap.parse_args()

    # Local jsonl: per-machine, accumulates over time. Includes A/B labels
    # that we want to filter — only keep the unlabeled / "baseline-pre-opt"
    # cells when fitting the current cost model. opt2/opt3/opt4 sweeps all
    # share the same (gpu, res, batch, scenario) keys but reflect *changes*
    # to train.py we want to A/B against the baseline, not feed into the fit.
    raw_local = filter_valid(load_jsonl(JSONL))
    skip_labels = {'opt2-fused-adamw', 'opt3-torch-compile', 'opt3-test1', 'opt3-no-triton-fallback', 'opt4-test', 'opt4-test-512'}
    local_rows = [r for r in raw_local if r.get('label') not in skip_labels]
    if len(raw_local) != len(local_rows):
        print(f"Filtered {len(raw_local) - len(local_rows)} A/B-labeled rows from {JSONL.name} (kept {len(local_rows)})")

    # Per-key keep the latest timestamp so opt-experiment rows that snuck
    # past the label filter (or runs of the same cell at different times)
    # don't double-count the GPU.
    by_key: dict = {}
    for r in local_rows:
        k = (r["gpu"], r["resolution"], r["batch"], r["scenario"])
        if k not in by_key or (r.get("ts", "") > by_key[k].get("ts", "")):
            by_key[k] = r
    rows = list(by_key.values())
    print(f"Loaded {len(rows)} dedup'd cells from {JSONL.name}")

    # Cloud snapshots: committed, authoritative. Newer files override older
    # for the same key (e.g. if we re-measure L4/512/bs2 with torch.compile
    # enabled, the 05-02 entry replaces the 05-01 one since current code
    # has compile on by default).
    seen = set(by_key.keys())
    snapshot_files = sorted((REPO / "scripts").glob(CLOUD_SNAPSHOT_GLOB))
    for f in snapshot_files:
        snap_rows = filter_valid(load_jsonl(f))
        # Within a single snapshot file, prefer torch_compile=true rows over
        # compile=false (since cloud now defaults to compile-on); also prefer
        # later timestamps. Build the snapshot's own per-key map first.
        snap_by_key: dict = {}
        for r in snap_rows:
            k = (r["gpu"], r["resolution"], r["batch"], r["scenario"])
            existing = snap_by_key.get(k)
            if not existing:
                snap_by_key[k] = r
            else:
                # Prefer compile-on
                if r.get("torch_compile") and not existing.get("torch_compile"):
                    snap_by_key[k] = r
                elif r.get("ts", "") > existing.get("ts", ""):
                    snap_by_key[k] = r
        added = 0
        for k, r in snap_by_key.items():
            if k not in seen:
                rows.append(r); seen.add(k); added += 1
            else:
                # Replace
                for i, existing in enumerate(rows):
                    if (existing["gpu"], existing["resolution"], existing["batch"], existing["scenario"]) == k:
                        rows[i] = r; break
        print(f"Loaded {len(snap_by_key)} cells from {f.name} ({added} new, {len(snap_by_key)-added} updates)")

    if args.include_cloud:
        # Legacy: was for piping CLOUD literal into JSONL. The snapshot
        # file is the source of truth now; this flag's a no-op.
        print("(--include-cloud is a no-op now; cloud data lives in the committed snapshot)")

    # ── Reference cell (per GPU) ─────────────────────────────────────────────
    # Reference is (ref scenario, 512px, batch=2). Used to measure the
    # baselineStepsPerSec(sku) value.
    refs = {}
    for r in rows:
        if r["scenario"] == "ref" and r["resolution"] == 512 and r["batch"] == 2:
            refs[r["gpu"]] = r["step_rate"]

    print()
    print("=== baselineStepsPerSec (reference cell: ref / 512px / bs=2) ===")
    for gpu, rate in sorted(refs.items()):
        print(f"  {gpu:<25}  {rate:>6.2f} step/sec")

    # ── Fit resolutionPenalty(res) ──────────────────────────────────────────
    # Compute: at fixed (gpu, scenario, batch), step_rate(res) / step_rate(512).
    # Average across batches and GPUs to get an empirical curve.
    print()
    print("=== resolutionPenalty(res) fit ===")
    print("  (penalty = step_rate(512) / step_rate(res); model wants this so")
    print("   higher penalty = slower at that res)")
    by_res = defaultdict(list)
    for r in rows:
        if r["scenario"] != "ref": continue
        ref_at_512 = next((s["step_rate"] for s in rows
                           if s["gpu"] == r["gpu"] and s["scenario"] == "ref"
                           and s["resolution"] == 512 and s["batch"] == r["batch"]), None)
        if not ref_at_512: continue
        by_res[r["resolution"]].append(r["step_rate"] / ref_at_512)

    print(f"  {'res':>5}  {'speedup vs 512':>15}   {'penalty':>8}    {'(res/512)² model':>18}    {'samples':>8}")
    for res in sorted(by_res):
        speeds = by_res[res]
        med = median(speeds)
        penalty = 1 / med if med else float('inf')
        model_pred = (res/512)**2
        print(f"  {res:>5}  {med:>15.3f}   {penalty:>8.3f}    {model_pred:>18.3f}    {len(speeds):>8d}")

    # Fit penalty(res) = a + b*(res/512)² where a + b = 1.0 (so 512 → 1.0)
    # Using two unknowns and the data at 256 + 512 + (1024 if any).
    points = []
    for res, speeds in by_res.items():
        med = median(speeds)
        if med:
            points.append((res, 1/med))
    # Constrain f(512)=1.0; solve f(res) = a + (1-a)*(res/512)²
    # → 1.0 = a + (1-a)*1 ✓ always true, doesn't constrain. Need second point.
    # Use 256: f(256) = a + (1-a)*0.25
    # → measured_penalty_256 = a + 0.25 - 0.25a → a = (m - 0.25) / 0.75
    pen_256 = next((p for r, p in points if r == 256), None)
    if pen_256 is not None:
        a = (pen_256 - 0.25) / 0.75
        b = 1 - a
        print(f"\n  Fitted penalty(res) = {a:.3f} + {b:.3f} × (res/512)²")
        print(f"    256 → {a + b*0.25:.3f} (measured {pen_256:.3f})")
        print(f"    512 → {a + b*1.00:.3f}")
        if any(r == 1024 for r, _ in points):
            pen_1024 = next(p for r, p in points if r == 1024)
            print(f"    1024 → {a + b*4.00:.3f} (measured {pen_1024:.3f})")
        else:
            print(f"    1024 → {a + b*4.00:.3f} (not measured)")
    else:
        a, b = 0.4, 0.6
        print(f"  No 256px data — defaulting to a={a}, b={b}")

    # ── Fit batchMultiplier(bs) ──────────────────────────────────────────────
    # Per-step penalty as batch grows. Compare step_rate(bs) / step_rate(2)
    # at fixed (gpu, scenario, res).
    print()
    print("=== batchMultiplier(bs) fit ===")
    print("  (penalty = step_rate(bs=2) / step_rate(bs); current model: √(bs/2))")
    by_bs = defaultdict(list)
    for r in rows:
        if r["scenario"] != "ref": continue
        ref_at_2 = next((s["step_rate"] for s in rows
                         if s["gpu"] == r["gpu"] and s["scenario"] == "ref"
                         and s["resolution"] == r["resolution"] and s["batch"] == 2), None)
        if not ref_at_2: continue
        by_bs[r["batch"]].append(r["step_rate"] / ref_at_2)

    print(f"  {'batch':>5}  {'speedup vs bs=2':>16}   {'penalty':>8}    {'√(bs/2) model':>15}    {'samples':>8}")
    for bs in sorted(by_bs):
        speeds = by_bs[bs]
        med = median(speeds)
        penalty = 1 / med if med else float('inf')
        model_pred = math.sqrt(bs/2)
        print(f"  {bs:>5}  {med:>16.3f}   {penalty:>8.3f}    {model_pred:>15.3f}    {len(speeds):>8d}")

    # Fit penalty(bs) = (bs/2)^k. Solve for k from each measured bs.
    ks = []
    for bs, speeds in by_bs.items():
        if bs <= 2: continue
        med = median(speeds)
        if not med: continue
        penalty = 1 / med
        # penalty = (bs/2)^k  →  k = log(penalty) / log(bs/2)
        k = math.log(penalty) / math.log(bs/2)
        ks.append(k)
        print(f"    bs={bs}: implied exponent k = {k:.3f}")
    if ks:
        k_med = median(ks)
        print(f"  → median k = {k_med:.3f}  (current model uses 0.5; quadratic would be 1.0)")

    # ── Fit settingsMultiplier (porter vs ref) ───────────────────────────────
    print()
    print("=== settingsMultiplier (porter scenario penalty over ref) ===")
    print("  porter = msrn + dim=128 + l1+lpips (vs ref = unet + dim=64 + l1)")
    print("  current model: 1.7 (dim128) × 1.5 (msrn) × 1.25 (lpips) = 3.19×")
    porter_factors = []
    for r in rows:
        if r["scenario"] != "porter": continue
        ref = next((s for s in rows
                    if s["gpu"] == r["gpu"] and s["scenario"] == "ref"
                    and s["resolution"] == r["resolution"] and s["batch"] == r["batch"]), None)
        if not ref: continue
        factor = ref["step_rate"] / r["step_rate"]
        porter_factors.append((r["gpu"], r["resolution"], r["batch"], factor))
    for gpu, res, bs, f in porter_factors:
        print(f"  {gpu:<25} {res:>4}px bs={bs:<2}  porter is {f:.2f}× slower")
    if porter_factors:
        med_factor = median(f for _, _, _, f in porter_factors)
        print(f"  → median porter penalty: {med_factor:.2f}× (current model predicts 3.19×)")
        # Decompose: dim_128_penalty = ?  msrn_penalty × lpips_penalty.
        # Without isolating the components we can't fit each separately.
        # Approximation: keep msrn=1.5 and lpips=1.25, solve for dim:
        # measured = dim × 1.5 × 1.25  →  dim = measured / 1.875
        dim_implied = med_factor / (1.5 * 1.25)
        # Implied exponent: dim_penalty = (128/64)^k → k = log(dim_implied)/log(2)
        k_dim = math.log(dim_implied) / math.log(2) if dim_implied > 0 else 0
        print(f"  → assuming msrn=1.5, lpips=1.25 hold, dim=128 penalty ≈ {dim_implied:.2f}×")
        print(f"     → exponent k for sizeMult(dim) = (dim/64)^k: k ≈ {k_dim:.3f}")
        print(f"     (current model uses k=log₂(1.7)≈0.766; this run suggests {k_dim:.3f})")

    # ── Suggested patch for spark-presets.ts ─────────────────────────────────
    print()
    print("=== Suggested updates to tunet-web/src/lib/spark-presets.ts ===")
    print()
    if pen_256 is not None:
        print(f"  // resolutionPenalty(res) — fitted to {sum(len(v) for v in by_res.values())} cells")
        print(f"  function resolutionPenalty(res: number): number {{")
        print(f"    return {a:.3f} + {b:.3f} * Math.pow(res / 512, 2)")
        print(f"  }}")
        print()
    if ks:
        print(f"  // batchMultiplier(bs) — fitted exponent {k_med:.3f}")
        print(f"  function batchMultiplier(bs: number): number {{")
        print(f"    return Math.pow(bs / 2, {k_med:.3f})")
        print(f"  }}")
        print()
    if porter_factors:
        print(f"  // settingsMultiplier dim term — fitted exponent {k_dim:.3f}")
        print(f"  const sizeMult = Math.min(4, Math.pow(dims/64, {k_dim:.3f}))")

    # ── baselineStepsPerSec for cloud GPUs ───────────────────────────────────
    print()
    print("=== baselineStepsPerSec(sku) values ===")
    sku_map = {
        "Tesla T4":             ("g4dn", "T4"),
        "NVIDIA L4":            ("g6.",  "L4"),
        "NVIDIA A10":           ("g5",   "A10"),
        "NVIDIA A10G":          ("g5",   "A10 (cloud A10G variant)"),
        "NVIDIA L40S":          ("g6e",  "L40S"),
        "NVIDIA RTX PRO 6000":  ("g7e",  "RTX PRO 6000"),
        "NVIDIA GeForce RTX 3090": ("(local)", "RTX 3090 — not on Spark"),
    }
    for gpu, rate in sorted(refs.items()):
        prefix, label = sku_map.get(gpu, ("?", gpu))
        print(f"  if (sku.startsWith('{prefix}')) return {rate:.2f}  // {label}")

if __name__ == "__main__":
    main()
