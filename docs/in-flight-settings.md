# In-flight settings updates for live training jobs

**Status:** Design — not implemented. Tabled 2026-05-01.
**Context:** Spark Compute v1 has no "exec on running pod" RPC; the only
job-level controls are submit, cancel, and `idleHoldSeconds`. To change
settings during a run we have to either (a) have the trainer re-read state
on its own, or (b) cancel + relaunch, with checkpoint round-trip in between.

---

## What we want to change mid-run

Two categories, with very different mechanics:

| Category | Examples | Restart? |
|----------|----------|----------|
| **Soft** | `lr`, `max_steps`, `save_interval`, `idleHoldSeconds`, loss weights | No — apply live |
| **Hard** | `resolution`, `batch_size`, `model_size_dims`, `preset`, **Preview Filter threshold** | Yes — reshapes dataloader/model |

Preview Filter is the motivating case: changing the threshold changes which
patches are skipped, which changes the dataloader's iteration order, which
the trainer can't safely swap mid-epoch.

---

## Architecture

### 1. Control file in ShareSync output

Trainer polls `<output>/control.json` every N steps (default: 100). Schema:

```json
{
  "config_revision":         2,            // monotonic, trainer compares to last applied
  "lr":                      5e-5,
  "save_interval":           500,
  "max_steps":               30000,
  "loss_weights":            { "l1": 1.0, "lpips": 0.1 },
  "request_checkpoint_exit": false         // when true: save + sys.exit(0)
}
```

Trainer behavior (in [tunet.py](../tunet.py)):

- At the top of each train step, if `step % control_poll_interval == 0`,
  re-read `control.json`. If `config_revision` changed:
  - For each known soft key: assign to the running optimizer / scheduler /
    counters in place. Log `[control] applied revision=N: lr=…`.
  - For unknown keys: warn but continue.
- If `request_checkpoint_exit == true`:
  - Save a full checkpoint to `<output>/checkpoints/control_exit.pt`
    (model + optimizer + scheduler + step + epoch + RNG state).
  - Write `<output>/control_exit.json` with `{ exit_step, exit_reason: "control" }`.
  - `sys.exit(0)`.

The control file is the **source of truth** for the running job's settings.
On startup, the trainer also reads it once (after applying the initial
`config.yaml`) so a "resume" can carry forward the latest values.

### 2. Web API

```
POST /api/spark/jobs/:id/control
  Body: { lr?, save_interval?, max_steps?, loss_weights?, request_checkpoint_exit? }
  Behavior:
    - GET current control.json from ShareSync (404 → start from {})
    - Merge incoming fields, increment config_revision
    - PUT updated JSON back to ShareSync
    - Return the merged document
```

`config_revision` is incremented server-side, not client-side, so concurrent
edits don't collide on a stale value.

### 3. Hard-change flow (cancel + resume on a NEW Spark job)

Spark has no "relaunch on the same pod" — every submit is a fresh agent
boot. But the warm-pool cache means the *image pull* and *agent boot* are
typically much shorter on the second job. Estimated cost: 30s–2min of
overlap, vs. 3-5min for a true cold start.

UI flow when user changes a hard setting:

1. Show "This change requires restarting the job — your last checkpoint
   will be loaded. Estimated restart time: ~1 min."
2. POST `/api/spark/jobs/:id/control` with `request_checkpoint_exit: true`.
3. Poll job status — wait for `terminated` (or fall through to a hard cancel
   after 60s if the trainer is wedged).
4. POST `/api/spark/training-jobs` for a new job:
   - Same `instanceType`, same dataset (point at existing
     `output_share_sync_path` for `--resume-from`).
   - Updated config from the user's edits.
   - Stash `env.TUNET_RESUMED_FROM = <prior_job_id>` so the UI can show
     lineage.
5. Redirect to the new job's detail page with a "Resumed from <prior>" banner.

The new job's `control.json` is initialized from the prior job's last
control state plus the user's hard-change edits.

### 4. UI surface

On `/demo/jobs/[id]`:

- New "Settings" panel between the metric strip and the chart. Shows current
  effective settings (read from `control.json` if present, else `env`).
- Inline edit for soft knobs — debounced POST to `/api/spark/jobs/:id/control`.
- Modal-confirmed edit for hard knobs — explicit "Restart with new settings"
  button that triggers the cancel+resume flow.
- Job detail page header gets a "Resumed from <id>" link if `TUNET_RESUMED_FROM`
  is set.

---

## Open questions

1. **Should hard restarts be one logical job in the UI, or visibly two?**
   Spark has no grouping. Either we add a thin DB to track "job lineage"
   (cheap; we already have Supabase wired) or we infer the chain from
   `env.TUNET_RESUMED_FROM` and present a "Generation 3 of 3" header
   without persisting anything new.

2. **What if the trainer doesn't honor `request_checkpoint_exit` within the
   timeout?** Probably fall through to a hard cancel and log a warning.
   Worst case: the user loses up to `save_interval` steps — same as the
   current cancel button.

3. **Preview Filter is more invasive than other hard changes** — it
   regenerates the patch index. Need to make sure the `--resume-from`
   path still produces a coherent training continuation when the dataset
   reshuffles. Probably fine since the optimizer state is per-parameter,
   not per-batch, but worth a smoke test.

4. **Auth on the control endpoint** — we're not gating Spark routes today
   in the demo flow. Before this ships to real users, the control endpoint
   needs to verify the requester actually owns the job.

---

## Non-goals

- Live model architecture changes (model_size_dims, channel counts) — these
  are baked into the `state_dict` shape; resume from checkpoint is impossible
  without weight surgery. Treat as "start a new job" in the UI.
- True hot-swap of dataset (different src/dst folders entirely) — requires
  re-uploading data; just submit a new job.

---

## Estimated work

- Trainer-side control polling + checkpoint-exit: ~50 LOC in `tunet.py`.
- `POST /api/spark/jobs/:id/control` + ShareSync read/write: ~80 LOC.
- Resume submit path (reuse training-jobs route with `--resume-from`): ~40 LOC.
- UI panel (settings card + edit modals + restart confirmation): ~250 LOC.
- Telemetry / lineage display: ~30 LOC.

Total: half-day to a full day of work. The trainer-side changes are the
risky part — needs an end-to-end smoke test on a real Spark job before
shipping.
