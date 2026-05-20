Babies Lipstick on Glass.
 21 image pairs
 Epoch 10 proves that it's going to work
 E40 has better output for final.
 Going back with more image pairs -- should have done this at 10

Epoch	Wall-clock from start	Per-10-epoch chunk
1	23m 14s	(warmup epoch, includes one-time setup)
10	3h 50m	~3h 27m for epochs 1→10
20	7h 41m	~3h 51m for epochs 11→20
30	11h 31m	~3h 50m for epochs 21→30
38 (final)	14h 36m	~3h 5m for epochs 31→38

Steady-state pace: ~3 hours 50 minutes per 10 epochs, or roughly 23 minutes per epoch, completely consistent across the run.

---

## Hardware baseline: A10 (Linux) vs RTX 3090 (Windows)

Same training config (`069_005`), different machines. A10 run has more dataset pairs and `iterations_per_epoch=500`; 3090 run uses `iterations_per_epoch=1000`. Per-step is the apples-to-apples metric.

| Metric                  | 3090 (Windows, run 069_005, ep 39) | A10 (Linux, current run, ep 77) |
|-------------------------|------------------------------------|---------------------------------|
| T/Step (total)          | 1.380 s                            | 1.582 s                         |
| C (compute)             | 1.350 s                            | 0.726 s                         |
| D (loader)              | 0.015 s (blocking)                 | 4.3 s buffered (non-blocking)   |
| Non-compute overhead    | ~0.03 s                            | ~0.85 s                         |
| iterations_per_epoch    | 1000                               | 500                             |

### Takeaways
- **A10 compute is ~1.86× faster than 3090.** Much larger than the ~15% silicon delta; almost certainly attributable to `torch.compile` working on Linux but degrading or falling back on Windows (Triton-on-Windows is still rough), plus Linux cuDNN/AMP paths being 10–20% faster.
- **3090 wins total step time by ~13%** because the A10 has ~0.85 s/step of non-compute overhead the 3090 doesn't. Suspected sources on A10: auto-batch picking a different batch size, LPIPS forward and AMP unscale not being inside `C`, epoch-boundary work amortized into the per-step average.
- **Headroom on the A10:** if the 0.85 s/step overhead were eliminated, the A10 would run at ~0.75–0.85 s/step total, ~1.7× faster than the 3090. Worth profiling with `torch.profiler` to locate the gap.