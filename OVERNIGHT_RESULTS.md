# Overnight Run — 2026-05-01 → 2026-05-02

Goal: calibrate the cost estimator with real benchmark data, then test
three optimizations to `train.py` and commit each separately so any can
be reverted with a single `git revert`.

## TL;DR — what changed

**Cost estimator is now calibrated from 17 real benchmark cells** (9 RTX
3090 local + 8 cloud T4/L4/A10). The old "guess" multipliers were
materially wrong:

| Axis | Old model | New (fitted) | Reality check |
|---|---|---|---|
| res 256→512 | 4× speedup | **2.04× speedup** | over-predicted by ~2× |
| dim 64→128 | 1.7× slower | **2.54× slower** | under-predicted by ~50% |
| batch 2→8 | 2.0× per-step | **2.39× per-step** | under-predicted |
| Porter scenario | 3.19× ref | **4.76× ref** | under-predicted by ~50% |

**Optimizations applied to `train.py`:** 1 of 4 measurably worked locally;
all 4 produced a useful learning. Net step-rate impact at 256/bs2: **+3.3%**.

## Commits (rollback point at top, newest at bottom)

```
0f1f907  Add scripts/bench_local.py                          ← rollback to here
d14f2b2  Cost model: empirical calibration from 17 cells     ← cost-model only, no train changes
97de4d7  train.py: enable fused AdamW on CUDA                ← 1.5–3.3% local win
b1ed932  train.py: torch.compile model on CUDA when Triton…  ← can't validate locally; cloud will benefit
ed78709  bench_local: add --label and --cell-timeout flags
```

To revert any single optimization: `git revert <hash>`. They are
independent of each other.

## Optimization results (all measured on RTX 3090 / Porter EXR data, 200 timed steps)

| Cell | Baseline | Opt 2 (fused AdamW) | Opt 3 (compile) | Opt 4 (channels_last) |
|---|---|---|---|---|
| ref / 256 / bs2 | 16.73 step/s | **17.29** (+3.3%) | skipped (no Triton) | ❌ 10.07 (-40%) |
| ref / 256 / bs4 | 15.59 | **16.04** (+2.9%) | skipped | not run |
| ref / 512 / bs2 | 8.46 | **8.60** (+1.7%) | skipped | ❌ 3.12 (-64%) |
| ref / 512 / bs4 | 5.45 | **5.53** (+1.5%) | skipped | not run |

### Opt 1 — Dataloader workers + prefetch — SKIPPED

Couldn't apply locally. `train.py:635` *forces `num_workers=0` on Windows*
(spawn-multiprocessing hangs were fixed by killing it). On Linux Spark
agents the existing `auto_detect_num_workers()` already picks 6–8 workers
based on VRAM/CPU/resolution, which is roughly optimal — there's no
obvious headroom without architectural changes.

**No commit.** Documented as a no-op for local runs.

### Opt 2 — Fused AdamW — APPLIED ✅

`torch.optim.AdamW(..., fused=True)` — merges per-parameter update kernels
into one CUDA kernel. **Measurable win: +1.5% to +3.3%** depending on cell,
biggest at small workloads where the optimizer step is a bigger fraction
of iter time. Free, no risk — `fused=True` falls back to the regular
implementation on non-CUDA devices.

Applied at both AdamW construction sites (primary + checkpoint-load
fallback in case of corrupted resume).

**Commit `97de4d7`.**

### Opt 3 — `torch.compile(mode='reduce-overhead')` — APPLIED but UNVALIDATED ⚠️

The biggest potential win — published torch docs claim 5–25% on small
models where kernel-launch overhead dominates. Our calibration data
showed exactly that profile (resolution penalty has a ~13% fixed
overhead floor; the small-res benchmark over-predicted by 1.6×, classic
kernel-launch-bound symptom).

**Cannot validate locally:** Windows + standard CUDA PyTorch wheel
doesn't ship Triton (no Triton wheel exists for Windows). Without
Triton, the inductor backend crashes on first forward pass with
`TritonMissing`.

The committed implementation:
- Probes for `import triton` at compile time, skips cleanly with a log
  message if absent (instead of crashing on first forward)
- Falls back to eager if `torch.compile()` itself raises
- Opt-out: `TUNET_DISABLE_COMPILE=1`

**Should validate on the next cloud benchmark run** (Spark agents are
Linux + have Triton).

**Latent bugs fixed along the way (also committed):**
- `if model and optimizer and scaler:` was using truthiness, which
  delegates to `__len__` — torch.compile's wrapper raises `TypeError:
  UNet does not support len()`. Switched to explicit `is not None`.
- `state_dict()` saves now unwrap both DDP `.module` and torch.compile
  `._orig_mod` so saved checkpoints are plain UNet/MSRN state dicts —
  loadable from any wrapper combination.

**Commit `b1ed932`.**

### Opt 4 — `channels_last` memory format — REVERTED ❌

Tried converting model + inputs to NHWC for Ampere tensor-core peak
throughput. Made things **40–64% slower**:

| Cell | Baseline | channels_last |
|---|---|---|
| 256 / bs2 | 17.29 | 10.07 (**-40%**) |
| 512 / bs2 | 8.60 | 3.12 (**-64%**) |

**Why it failed:** UNet has `GroupNorm` (used in MSRN) and skip-connection
`torch.cat` ops that don't have NHWC implementations. PyTorch falls back
to NCHW at every such op, paying memory-layout conversion cost on every
layer instead of running native NHWC end-to-end.

**Lesson:** channels_last only wins when the *entire* compute graph has
NHWC kernels. Mixed graphs are strictly worse. Worth revisiting if we
ever swap GroupNorm out for BatchNorm, but not today.

**Reverted, not committed.**

## Cost-estimator calibration (the main work)

`tunet-web/src/lib/spark-presets.ts` was recalibrated from theoretical
multipliers to **empirical fits**. Methodology and raw data:
[`tunet-web/benchmark.md`](tunet-web/benchmark.md).

### What changed in the formulas

```
// resolutionPenalty(res)
OLD:  (res/512)²
NEW:  0.13 + 0.87 × (res/512)²       // 13% fixed-overhead floor

// settingsMultiplier sizeMult
OLD:  1.7^log2(dims/64)              // exponent 0.77
NEW:  (dims/64)^1.34                 // exponent 1.34, capped at 8×

// settingsMultiplier batchMult
OLD:  √(bs/2)                        // exponent 0.5
NEW:  (bs/2)^0.61                    // exponent 0.61, fitted from data
```

### baselineStepsPerSec (real measurements where available)

| GPU | SKU | step/sec | Source |
|---|---|---|---|
| T4 | g4dn.xlarge | **4.46** | Cloud bench, real EXR (was 4.09 from synthetic) |
| L4 | g6.2xlarge | **6.79** | Cloud bench, real EXR |
| A10 | g5.xlarge | **8.51** | Cloud bench, real EXR |
| L40S | g6e.8xlarge | 12 (guess) | Spark dropped cheaper variant; not yet measured |
| RTX PRO 6000 | g7e.2xlarge | 18 (guess) | Same — not yet measured |

**Local-only reference:** RTX 3090 → 8.17 step/sec. Reproduces the
business partner's 1.62 step/s on porter/512/bs2 to within **2%** —
calibration is reproducible across machines.

## Files added/modified

```
A   OVERNIGHT_RESULTS.md                           ← this file
A   scripts/bench_local.py                         ← local-GPU benchmark runner
A   scripts/calibrate_cost_model.py                ← curve-fitter
A   scripts/bench_cloud_2026-05-01.jsonl           ← committed cloud snapshot
M   train.py                                       ← Opts 2 + 3
M   tunet-web/src/lib/spark-presets.ts             ← calibrated formulas
M   tunet-web/benchmark.md                         ← updated methodology
M   .gitignore                                     ← gitignore per-machine bench_results.jsonl
```

`scripts/bench_results.jsonl` is gitignored — your machine's accumulated
measurements stay local. The committed `bench_cloud_2026-05-01.jsonl` is
the cross-machine snapshot.

## Things still TBD

1. **L40S + RTX PRO 6000 calibration on the new SKUs.** The Spark
   allow-list dropped `g6e.4xlarge` and `g7e.xlarge`; the eligible
   replacements (`g6e.8xlarge`, `g7e.2xlarge`) are +67% / +31% per hour
   for the same GPU. Numbers in `baselineStepsPerSec` for these are still
   educated guesses. Re-run the cloud benchmark once Spark capacity is
   loose enough to actually schedule them.

2. **1024px row of the matrix.** Porter dataset is 1000×1000 so 1024px
   patches can't be sampled (every file rejected). Need a ≥1024px shot to
   validate the resolution-penalty fit at the high end.

3. **`torch.compile` validation.** Need a cloud benchmark cell with
   Triton present (any Linux Spark agent) to measure the actual speedup.
   Code is already in place; just needs the run.

4. **VRAM thrash detection.** The Porter scenario at higher batch sizes
   silently swaps to system RAM on 24 GB GPUs (3090, A10, L4) — visible
   as step_rate dropping below 0.5. The benchmark logs this, but the
   cost estimator's UI doesn't warn the user before submit. Worth a
   "this will OOM/swap on <gpu>" pre-check in the new-job page.
