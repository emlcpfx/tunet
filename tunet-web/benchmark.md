# GPU Throughput Benchmarking & Cost Model

This is the working document for `tunet-web`'s training cost estimator.
The estimator turns user inputs (GPU, resolution, model size, loss, batch,
dataset size) into a runtime + cost range. It's only as good as the
calibration data behind it.

This file documents:
1. **Where the numbers in `spark-presets.ts` come from** — what's measured,
   what's a guess, and the date.
2. **How to re-run the calibration** — the `/demo/benchmark` page, the
   `--benchmark-steps` flag in `train.py`, and the Spark plumbing.
3. **Known limitations** — what the model gets right and where it lies.

If you're updating the calibration: edit this file in the same commit.

---

## The model

```
hours = steps / (baselineStepsPerSec(sku) / settingsMultiplier(opts) / resolutionPenalty)

steps              = user's max_steps cap, OR recommendedStepsForPairs(pairs) if 0
resolutionPenalty  = 0.13 + 0.87 × (res/512)²       ← empirical (was (res/512)²)
settingsMultiplier = sizeMult × modelTypeMult × lossMult × batchMult
sizeMult           = (dims/64)^1.34, capped at 8×   ← empirical (was 1.7^log2(dims/64))
batchMult          = (bs/2)^0.61                    ← empirical (was √(bs/2))
```

All formulas were re-fitted 2026-05-02 from 17 benchmark cells across
RTX 3090 / T4 / L4 / A10. Old multipliers were mostly theoretical and
turned out to be 30–60% off in various directions.

Three pieces in `tunet-web/src/lib/spark-presets.ts`:

| Function | Inputs | Returns |
|---|---|---|
| `baselineStepsPerSec(sku)` | GPU SKU | step/sec at *reference settings* (UNet, dim=64, 512px, batch=2, L1) |
| `settingsMultiplier(opts)` | model_size, model_type, loss, batch | per-step time penalty vs reference |
| `recommendedStepsForPairs(pairs)` | dataset frame count | typical converged step count |

The cost estimator returns a *range*, not a point: `low = central × 0.6`,
`high = central × 1.7`. This bracket is wide enough to cover real-world
variance (data-loader pressure, augmentation, content) without being
useless.

---

## Calibration status — 2026-05-02

### `baselineStepsPerSec(sku)`

| GPU | SKU | step/sec | Source | Confidence |
|---|---|---|---|---|
| T4 | `g4dn.xlarge` | **4.46** | Measured (Porter_0408_REDO real EXR) | High |
| L4 | `g6.2xlarge` | **6.79** | Measured (Porter_0408_REDO real EXR) | High |
| A10 | `g5.xlarge` | **8.51** | Measured (Porter_0408_REDO real EXR — A10G silicon) | High |
| L40S | `g6e.8xlarge` | 12 (guess) | Not measured — see "L40S/RTX PRO availability" below | Low |
| RTX PRO 6000 (Blackwell) | `g7e.2xlarge` | 18 (guess) | Not measured — same reason | Low |
| RTX 3090 | (local only) | **8.17** | Measured locally — anchor for cross-machine reproducibility | High |

The local 3090 reproduces the business partner's converged-run rate
(1.62 step/s @ porter/512/bs2) within 2% — calibration is reproducible.

**Reference settings** (must match `BENCH_FIXED` in
`src/app/api/spark/benchmark/route.ts` and what `settingsMultiplier`
treats as 1.0×):

- `model_type: unet`
- `model_size_dims: 64`
- `resolution: 512`
- `batch_size: 2`
- `loss: l1`
- `use_amp: true`
- `progressive_resolution: false`
- No augmentation

### `settingsMultiplier(opts)` — empirical fits 2026-05-02

Re-fitted from 9 RTX-3090 cells running both `ref` (UNet/dim=64/L1) and
`porter` (MSRN/dim=128/L1+LPIPS). The data + fit code: `scripts/calibrate_cost_model.py`.

| Factor | Multiplier | Justification |
|---|---|---|
| `model_size_dims` | `(dims/64)^1.34`, cap 8× | **Empirical.** porter/3090 measured 4.76× slower than ref at the same cell. Dividing out msrn=1.5× and lpips=1.25× → dim=128 ≈ 2.54×, implying exponent log₂(2.54)=1.34. dim=128 → 2.54×, dim=256 → 6.5×, dim=32 → 0.39×. Old curve (1.7^log2) was 50% too low. |
| `model_type=msrn` | × 1.5 | Theoretical. Couldn't isolate from porter measurements (porter is msrn AND dim=128 AND lpips). Held fixed during the fit; if msrn is actually 2× rather than 1.5×, the fitted dim exponent compensates. |
| `loss=l1+lpips` | × 1.25 | Theoretical. Same reason as msrn — couldn't isolate. |
| `loss=weighted` | × 1.30 | Theoretical. |
| `loss=bce+dice` | × 1.00 | Mask-only, no penalty. |
| `batch_size` | × `(bs/2)^0.61` | **Empirical.** Implied exponents from bs=4 (k=0.59) and bs=8 (k=0.63) measurements; using median k=0.61. Old `√(bs/2)` (k=0.5) under-predicted batch slowdown. bs=4 → 1.51×, bs=8 → 2.32×. |

### `resolutionPenalty(res)` — empirical fit 2026-05-02

```
penalty(res) = 0.13 + 0.87 × (res/512)²
```

12 cells fed the fit (T4 / L4 / A10 / 3090 cross-product). The
constant 0.13 captures fixed per-step overhead (kernel launches,
Python loop, dataloader baseline) that doesn't scale with pixel count;
the 0.87 captures the part that does. The pure `(res/512)²` model
over-predicted the 256px speedup by ~1.6×.

| res | New penalty | Old (res/512)² | Notes |
|---|---|---|---|
| 256 | 0.35 | 0.25 | Reality: only 2.0× speedup vs 512, not 4× |
| 512 | 1.00 | 1.00 | Reference |
| 1024 | 3.61 | 4.00 | Predicted only — not yet measured |

### `recommendedStepsForPairs(pairs)` — calibrated to 1 reference run

| Pairs (frames) | Recommended steps |
|---|---|
| ≤ 10 | 30,000 |
| 11–30 | 60,000 |
| 31–80 | **100,000** ← matches 3090 reference run |
| 81–200 | 130,000 |
| 200+ | 160,000 |

**Reference run:** 2026-04-09, RTX 3090 (consumer), Porter_0408_REDO, 457
slices (~50 frames). Plateau detector triggered at epoch 96, smoothed loss
0.003287 best @ epoch 88. `max_steps=30000` was set in config but training
ran past it (config edited mid-run or early-stopping overrode); final stop
at step 96000. This is the only converged-run data point we have. Foundry
CopyCat's published examples (10k epochs / 6 frames / batch 4 = 15k steps)
are consistent with our smaller tier but Foundry stops on visual match,
not loss plateau, so they underreport converged-run length.

---

## How to recalibrate

### Pre-flight

1. **Check Spark allow-list.** `GET /api/spark/skus` returns currently
   eligible SKUs. The list shifts. As of 2026-05-01 the cheap variants of
   L40S (`g6e.4xlarge`) and RTX PRO 6000 (`g7e.xlarge`) were dropped;
   `g6e.8xlarge` and `g7e.2xlarge` are the eligible single-GPU equivalents
   but cost +67% / +31% per hour respectively.
2. **Check your account concurrency limit.** Spark accounts have a
   per-account node cap (this account: **2 concurrent nodes** as of
   2026-05-02 — confirmed with Walt). If you submit N jobs at once,
   only the first 2 get provisioning slots; the rest sit in `queued`
   until Spark's `compute_provisioning_timeout_seconds` (~30 min) fires
   and reaps them as "Provisioning timed out". This error message
   *sounds* like AWS-out-of-capacity but is almost always the account
   cap. Throttle to ≤ 2 in flight, or wait for one to finish before
   sending the next. Stuck jobs (cancelled but still showing
   `provisioning`) hold a slot until Spark's watchdog reaps them, so
   one stuck job effectively halves your capacity.
3. **Pick a dataset.** EXRs at the resolution you actually train at.
   `/demo/benchmark` defaults to a synthetic 1280×1280 PNG dataset;
   prefer real data for I/O realism.

### Running the benchmark

The page lives at `/demo/benchmark`:

1. **Pick a folder** (optional — falls back to synthetic). Must have
   `src/` and `dst/` (or `input/` and `output/`) subfolders with matching
   filenames. EXRs need to be ≥ the largest patch resolution you select
   (e.g. 1024px patches require ≥1024×1024 source — Porter_0408_REDO is
   1000×1000, so only 256/512px are valid for that shot).
2. **Pick GPUs and the resolution × batch grid.**
3. **Run.** Each cell submits a Spark job that runs `train.py
   --benchmark-steps 200 --benchmark-warmup 20`, which:
   - Trains 20 warmup steps (discarded)
   - Times 200 steps
   - Logs `STEP_RATE: X.Y step/sec ...` and exits
4. The page polls each job's SSE log stream and grabs the rate as it
   appears. Once all cells are terminal, the **Calibrated baseline**
   panel emits a paste-able TS body for `baselineStepsPerSec()` plus a
   `BENCHMARK_MEASUREMENTS` JSON dump of the full surface.

### Code path

```
/demo/benchmark/page.tsx
  └ POST /api/spark/benchmark
        ├ stages dataset (synth → tmpdir, or user → /api/spark/upload-stage)
        ├ for each (gpu × resolution × batch) cell:
        │     ├ packInputTarball({ config: { resolution, batch_size, ... }, stageDir })
        │     ├ submitJob({ instanceType, env: { BENCHMARK_STEPS, BENCHMARK_WARMUP, ... } })
        │     └ uploadInputTarball(submitResp.input.uploadUrl, pack.buffer)
        └ returns runs[] = [{ gpuKey, jobId, resolution, batchSize }, ...]

spark_start.sh (in container):
  └ if BENCHMARK_STEPS > 0:
        $PY train.py --config ... --benchmark-steps $BENCHMARK_STEPS --benchmark-warmup $BENCHMARK_WARMUP

train.py (post-step block, line ~1330):
  └ when (global_step - start_step) == bench_warmup:
        torch.cuda.synchronize(); start clock
  └ when (global_step - bench_step_t0) >= bench_steps:
        torch.cuda.synchronize(); compute rate
        log "STEP_RATE: X.Y step/sec  (steps=N, elapsed=Ts, gpu=..., model=..., size=N, res=N, batch=N, loss=...)"
        exit cleanly
```

### Auto STEP_RATE from real runs

Every normal training job (no `--benchmark-steps` flag) also auto-emits one
`STEP_RATE` line at step ~80 (after 30-step warmup, timing the next 50).
Same wire format. So every real production run silently feeds the
calibration set — grep `STEP_RATE:` across job logs to harvest data points
in the wild.

---

## Known limitations

### Synthetic-data optimism
The benchmark falls back to synthetic 1280×1280 PNG pairs when no folder
is picked. PNG decode is much faster than EXR (no `parse-exr`, no
gamma-tonemap), so synthetic measurements over-estimate throughput when
the real workload is EXR-bound. The Porter dataset puts data-loader load
at ~7% of step time on a 3090 (`D:0.044 / T:0.628`); synthetic pushes it
to <1%. For T4-class GPUs this difference is small (compute-bound), for
A10+ it's ~10–15% optimistic.

### Unmeasured factors
- **`augmentations`**: not modeled at all. Heavy augmentation (color +
  affine + flip every step) can add ~20% to data-loader time.
- **`progressive_resolution`**: not modeled. Foundry claims up to 2×
  speedup; we ignore it and so over-estimate cost.
- **`use_auto_mask`** + **`skip_empty_patches`**: light extra compute,
  ignored.
- **Multi-GPU**: deliberately disabled. `spark_start.sh` runs `python
  train.py`, not `torchrun`, so DDP never engages. The web UI hides
  multi-GPU SKUs entirely. Re-enable once we wire torchrun + benchmark
  the actual scaling factor (rule of thumb: 0.6–0.8× linear).

### L40S / RTX PRO 6000 availability
2026-05-01: Spark dropped the cheaper host variants (`g6e.4xlarge`,
`g7e.xlarge`) from the allow-list. The remaining eligible single-GPU SKUs
(`g6e.8xlarge`, `g7e.2xlarge`) are +67% / +31% per hour respectively
because you're paying for unused host CPU/RAM. We haven't yet measured on
the new SKUs — the `baselineStepsPerSec` numbers for these GPUs are
guesses. Recalibrate as soon as cost is acceptable.

### Cost estimator confidence band
Returns `low = 0.6×, high = 1.7× central`. This is wide because:
- Real `step/sec` varies ~2× either way with content/augs/data-loader.
- `baselineStepsPerSec` is calibrated for 3 of 5 GPUs.
- `settingsMultiplier` is mostly theoretical except for `dims`.
- `recommendedStepsForPairs` is one reference run.

Tightening this band requires more measurements. Don't tighten without
more data.

---

## What still needs measuring

Priority order:

1. **L40S** (`g6e.8xlarge`) at reference settings. Note A10 is now the
   "RECOMMENDED" pick in the new-job UI (was L40S, before the SKU
   change made it +67% $/hr for the same silicon).
2. **RTX PRO 6000 Blackwell** (`g7e.2xlarge`) at reference settings.
3. **Resolution sweep at 1024px** to validate the empirical
   `0.13 + 0.87 × (res/512)²` fit at the high end. Need a shot
   ≥ 1024×1024 (Porter is 1000×1000, so it's invalid for 1024 patches).
4. **~~`torch.compile` validation on cloud Linux Spark agents~~** — done
   2026-05-02. L4/512/bs2 compile-on=7.23, compile-off=6.39 → **+13%**.
   Re-measure T4/A10 baselines (their numbers in `baselineStepsPerSec`
   predate the compile commit) so the estimator gets the production
   speedup credited too.
5. **Loss multipliers** — submit one cell at `l1+lpips`, one at
   `weighted`, compare to L1 baseline at the same GPU/res/batch. Right
   now those multipliers are theoretical (1.25× and 1.30×), couldn't
   isolate from porter.
6. **MSRN multiplier** — same: held fixed at 1.5× during the porter
   fit, never measured directly.

The infrastructure is ready (`/demo/benchmark` does all of this); we just
need to actually run it within the **account's 2-node concurrency cap**
so jobs don't waitlist past Spark's provisioning timeout. Submit ≤ 2
cells at a time, wait for the first batch to finish, then submit the
next. Watch out for stuck/cancelled jobs holding slots — those eat one
of the 2 concurrent allocations until Spark's watchdog reaps them.

---

## Reference data points

### 2026-05-02 — RTX 3090 local sweep, ref scenario (UNet, dim=64, L1)

`scripts/bench_local.py --src ...Porter_0408_REDO/input --dst ...output`

| res | bs=2 | bs=4 | bs=8 |
|---|---|---|---|
| 256 | 16.69 step/s | 14.28 | 9.91 |
| 512 | 8.17 | 5.00 | 2.94 |

### 2026-05-02 — RTX 3090 local sweep, porter scenario (MSRN, dim=128, L1+LPIPS)

| res | bs=2 | bs=4 | bs=8 |
|---|---|---|---|
| 256 | 4.73 | 3.00 | ⚠️ 0.04 (VRAM swap) |
| 512 | 1.62 | ⚠️ 0.04 (VRAM swap) | not run |

**The 3090 has 24 GB and porter (MSRN/128/LPIPS) is at the ceiling.**
Anything past 256/bs4 or 512/bs2 silently swaps to system RAM (PyTorch
falls back to unified memory, ~100–500× slower than VRAM-resident).
The same will happen on cloud A10/L4 (also 24 GB). L40S 48 GB and
RTX PRO 96 GB are the only cloud GPUs that can run porter past those
points.

### 2026-05-01 — T4 cloud sweep, ref scenario (Porter EXR via `/demo/benchmark`)

| res | bs=2 | bs=4 | bs=8 |
|---|---|---|---|
| 256 | 10.36 | 7.41 | 4.33 |
| 512 | 4.46 | 2.10 | (cancelled, queue timeout) |

The T4/512/bs4 → bs=4 is *slower in samples/sec* than bs=2 (8.4 vs 8.9),
i.e. T4 is past its compute saturation point at 512px even at the
smallest batch. Suggests T4 isn't worth picking past 512px even at
batch=2 — better to drop to L4 or up the GPU class.

### 2026-05-01 — Synthetic PNG dataset, /demo/benchmark (legacy, for context)
| GPU | resolution | batch | step/sec |
|---|---|---|---|
| T4 (Tesla T4) | 512 | 2 | 4.09 |
| L4 (NVIDIA L4) | 512 | 2 | 6.23 |

These were superseded by the real-EXR runs above. Kept for reference —
synthetic vs real-EXR shows ~9% difference for L4 (6.23 → 6.79), which
is run-to-run noise, suggesting our ref scenario isn't dataloader-bound
on these GPUs.

### 2026-05-01 — Porter_0408_REDO real EXR dataset (1000×1000), /demo/benchmark
| GPU | resolution | batch | step/sec | samples/sec |
|---|---|---|---|---|
| L4 | 256 | 8 | 6.61 | 52.9 |
| L4 | 512 | 2 | 6.79 | 13.6 |
| A10 (A10G) | 512 | 2 | 8.51 | 17.0 |

Note: the A10G in the cloud is the AWS variant of the A10, slightly
different silicon. Treat as A10 for cost-model purposes.

### 2026-05-02 — torch.compile A/B on Spark (L4 / 512px / bs=2 / Porter EXR)

Same cell, run twice with `TUNET_DISABLE_COMPILE` env var differing.
50-step warmup (extra-long to absorb torch.compile's ~1–3 min compile-
on-first-step cost), 400 timed steps.

| Variant | step/sec | samples/sec | Δ vs eager |
|---|---|---|---|
| compile-off (eager) | 6.39 | 12.8 | baseline |
| **compile-on (compiled)** | **7.23** | **14.5** | **+13.1%** |

L4 baseline in `baselineStepsPerSec` updated to 7.23 (was 6.79). T4
and A10 baselines were measured pre-compile and have not yet been
re-measured — production runs on those will be ~5–15% faster than the
estimator credits, until we re-benchmark.

### 2026-04-09 — RTX 3090 (consumer, Windows), Porter_0408_REDO converged run
- Config: `model_size_dims=128`, `model_type=msrn`, `resolution=512`,
  `batch_size=2`, `loss=l1+lpips`, `use_auto_mask=true`,
  `skip_empty_patches=true`
- `T/Step: 0.627s` → **1.595 step/sec**
- `D:0.044 T:0.001 C:0.578` → 7% data-loader, 92% compute, 1% transfer
- Trained 96 epochs × 1000 iter/epoch = **96,000 steps** to plateau
- Plateau detector best smoothed loss 0.003287 @ epoch 88, halted at
  epoch 96 (patience=8)
- Dataset: 457 usable slices from ~50 EXR frames

This is the **only converged-run data point** we have, and it's what
anchors `recommendedStepsForPairs`.

---

## train.py optimization attempts (2026-05-02)

Tested four optimizations against the baseline 3090 ref-scenario sweep
(256/512 × bs2/bs4). Each was its own commit so they can be reverted
individually with `git revert`.

| # | Optimization | Result | Commit | Notes |
|---|---|---|---|---|
| 1 | Dataloader workers + prefetch | **N/A locally; cloud re-test pending** | — | `train.py:635` forces `num_workers=0` on Windows (spawn-multiprocessing hang fix). On Spark, auto-detect was also clamping to 0 until 2026-05-18 because the container `/dev/shm` defaulted to 64 MB and bus-errored any DataLoader worker (safety net at `training/dataloader_utils.py:78-87`). Spark now defaults `/dev/shm` to 2 GiB with an optional `shmSize` submit param, so `auto_detect_num_workers()` will deliver 6–8 workers on cloud — worth re-measuring ref sweep on L4 (~10–20% expected at higher res). No commit. |
| 2 | Fused AdamW (CUDA) | **+1.5 to +3.3%** | `97de4d7` | `optim.AdamW(..., fused=True)`. Free win, no risk. Biggest at small workloads where the optimizer step is a bigger fraction of iter time. |
| 3 | `torch.compile(mode='reduce-overhead')` | **+13.1% on cloud L4** (validated 2026-05-02) | `b1ed932` | Can't run locally on Windows — Triton has no wheel. On cloud, L4/512/bs2 measured 6.39 step/s eager, 7.23 step/s compiled. Cost estimator's L4 baseline updated to reflect this. T4/A10 not yet re-measured. |
| 4 | `channels_last` memory format | **REGRESSION −40 to −64%** | reverted | UNet has GroupNorm + skip-connection `cat` ops that don't have NHWC kernels. PyTorch falls back to NCHW at every such op, paying memory-layout conversion cost on every layer. Mixed-format graphs are strictly worse. |

### Opt 2 measured impact (RTX 3090, ref scenario, Porter EXR)

| cell | baseline | fused AdamW | delta |
|---|---|---|---|
| 256/bs2 | 16.73 step/s | 17.29 | +3.3% |
| 256/bs4 | 15.59 | 16.04 | +2.9% |
| 512/bs2 | 8.46 | 8.60 | +1.7% |
| 512/bs4 | 5.45 | 5.53 | +1.5% |

### Latent bugs uncovered while implementing torch.compile (also fixed in `b1ed932`)

- **Truthiness on torch.compile-wrapped models raised TypeError.** The
  pattern `if model and optimizer and scaler:` in `train.py` falls
  through to `__len__` for `nn.Module`s, and torch.compile's
  `OptimizedModule` defines `__len__` to raise. Switched to explicit
  `is not None`. This bug existed but was latent before this work.
- **Checkpoint saves now unwrap both DDP and torch.compile.** Saved
  state_dict is a plain UNet/MSRN — loadable from any wrapper combo.
  Previously the DDP unwrap was conditional on `isinstance(model, DDP)`
  but didn't account for torch.compile's `_orig_mod`.

### Things to try next time

1. **CUDA Graphs** (`torch.cuda.CUDAGraph`) — explicit graph capture
   without inductor/Triton. Should work on Windows. Requires fixed input
   shapes (we have that after warmup). Bigger surgery than the others —
   needs separate graph per resolution if progressive_resolution is on.
2. **NVTX profile** — would tell us where the *actual* per-step time
   goes (kernel launch vs compute vs memcpy). Could reveal a single
   bottleneck worth attacking.
3. **`apex.optimizers.FusedAdam`** — sometimes faster than PyTorch's
   built-in fused. Adds a dependency though, only worth it if PyTorch's
   gives <2% (it gave 1.5–3.3% here, so probably not).
