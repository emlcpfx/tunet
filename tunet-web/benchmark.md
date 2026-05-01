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
resolutionPenalty  = (resolution / 512)²
settingsMultiplier = sizeMult × modelTypeMult × lossMult × batchMult
```

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

## Calibration status — 2026-05-01

### `baselineStepsPerSec(sku)`

| GPU | SKU | step/sec | Source | Confidence |
|---|---|---|---|---|
| T4 | `g4dn.xlarge` | **4.09** | Measured via `/demo/benchmark` (synthetic PNG dataset) | Medium — synth data, real EXR I/O may be ~10% slower |
| L4 | `g6.2xlarge` | **6.79** | Measured (Porter_0408_REDO real EXR dataset) | High |
| A10 | `g5.xlarge` | **8.51** | Measured (Porter_0408_REDO real EXR dataset) | High |
| L40S | `g6e.8xlarge` | 12 (guess) | Not measured — see "L40S/RTX PRO availability" below | Low |
| RTX PRO 6000 (Blackwell) | `g7e.2xlarge` | 18 (guess) | Not measured — same reason | Low |

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

### `settingsMultiplier(opts)` — back-fitted, not directly measured

| Factor | Multiplier | Justification |
|---|---|---|
| `model_size_dims` | `1.7^log2(dims/64)` capped at 4× | Empirical: 3090 reference run (dim=128) measured 1.6 step/sec; pure quadratic (4×) over-predicts by ~2.5×, this curve fits much better. dim=128 → 1.7×, dim=256 → 2.9×, dim=32 → 0.59× |
| `model_type=msrn` | × 1.5 | Theoretical: recurrent forward passes (`recurrence_steps=2` default) cost ~1.5× UNet. Untested. |
| `loss=l1+lpips` | × 1.25 | Theoretical: extra VGG forward + backward through frozen LPIPS. Untested. |
| `loss=weighted` | × 1.30 | Theoretical: L1 + L2 + LPIPS. Untested. |
| `loss=bce+dice` | × 1.00 | Mask-only, no penalty. |
| `batch_size` | × √(bs/2) | Sublinear. Bigger batch = longer step but more parallelism amortizes. Untested directly. |

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
2. **Check your account concurrency limit.** Spark has a hard
   provisioning timeout (~5 min). If you submit 45 jobs at once and the
   account can only run ~3 concurrently, the rest will time out. Throttle
   to ≤ 3 in flight at any time, or spread across resolutions/batches
   serially.
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

1. **L40S** (`g6e.8xlarge`) at reference settings. Highest-impact since
   it's the "RECOMMENDED" pick in the new-job UI.
2. **RTX PRO 6000** (`g7e.2xlarge`) at reference settings.
3. **Resolution sweep** on T4/L4/A10 at 256/1024 to validate
   `(res/512)²`. Need a shot ≥ 1024×1024 for the 1024 row.
4. **Batch sweep** at 4 and 8 to validate the `√(bs/2)` curve.
5. **Loss multipliers** — submit one cell at `l1+lpips`, one at
   `weighted`, compare to L1 baseline at the same GPU/res/batch.

The infrastructure is ready (`/demo/benchmark` does all of this); we just
need to actually run it without tripping Spark's provisioning timeout.
Submit ≤ 3 cells at a time, wait for them to finish, then submit the next
batch.

---

## Reference data points

### 2026-05-01 — Synthetic PNG dataset, /demo/benchmark
| GPU | resolution | batch | step/sec |
|---|---|---|---|
| T4 (Tesla T4) | 512 | 2 | 4.09 |
| L4 (NVIDIA L4) | 512 | 2 | 6.23 |

### 2026-05-01 — Porter_0408_REDO real EXR dataset (1000×1000), /demo/benchmark
| GPU | resolution | batch | step/sec | samples/sec |
|---|---|---|---|---|
| L4 | 256 | 8 | 6.61 | 52.9 |
| L4 | 512 | 2 | 6.79 | 13.6 |
| A10 (A10G) | 512 | 2 | 8.51 | 17.0 |

Note: the A10G in the cloud is the AWS variant of the A10, slightly
different silicon. Treat as A10 for cost-model purposes.

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
