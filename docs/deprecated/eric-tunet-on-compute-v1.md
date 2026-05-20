# Eric's TuNet on Spark Compute v1 — Adaptation Notes

**Audience**: Eric
**Author**: Walt
**Date started**: 2026-05-01
**Source repo**: `/Users/waltjones/tunetcloud/`  (your last-known-good RunPod setup)
**Goal**: prove your training workflow runs on Spark Compute v1, and hand you
a clean diff so you know exactly what changes when you port for real.

---

## API endpoints

All endpoints are on the production Spark backend:

| Purpose | Method | URL |
|---|---|---|
| **Authentication** | `POST` | `https://api.prod.aapse1.sparkcloud.studio/api/auth/login` |
| Submit job | `POST` | `https://api.prod.aapse1.sparkcloud.studio/api/compute/jobs` |
| Get job status | `GET` | `https://api.prod.aapse1.sparkcloud.studio/api/compute/jobs/:id` |
| Cancel job | `POST` | `https://api.prod.aapse1.sparkcloud.studio/api/compute/jobs/:id/cancel` |
| Stream logs (SSE) | `GET` | `https://api.prod.aapse1.sparkcloud.studio/api/compute/jobs/:id/logs/stream` |
| List eligible SKUs | `GET` | `https://api.prod.aapse1.sparkcloud.studio/api/compute/skus` |

ShareSync (file uploads/downloads) lives on a per-user host:

| Purpose | URL |
|---|---|
| WebDAV root | `https://<your-handle>.files.sparkcloud.studio/dav/spaces/...` |

The exact ShareSync upload URL for each job comes back in the submission
response — you don't need to construct it manually.

---

## Authentication

Spark uses bearer tokens. Get one with `POST /api/auth/login`:

```bash
SPARK_HOST="https://api.prod.aapse1.sparkcloud.studio"

LOGIN=$(curl -sS -X POST "$SPARK_HOST/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "eric@yourdomain.com",
    "password": "<your-password>"
  }')

# Response shape:
# { "resp": "Login Successful", "success": true, "token": "eyJhbGc..." }

TOKEN=$(echo "$LOGIN" | jq -r '.token')
echo "$TOKEN"   # paste into other terminals as needed
```

The token is a standard OIDC JWT. It expires after ~30 days; just re-login to
refresh. Use it as `Authorization: Bearer $TOKEN` on every Spark API call,
and *also* on every ShareSync WebDAV call (single sign-on — same token works
both places).

If you'd rather not embed credentials in scripts, you can also do the login
once interactively and stash the token in a file:

```bash
curl -sS -X POST "$SPARK_HOST/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"email\": \"$SPARK_EMAIL\", \"password\": \"$SPARK_PASSWORD\"}" \
  | jq -r '.token' > ~/.spark-token
chmod 600 ~/.spark-token

# In your scripts:
TOKEN=$(cat ~/.spark-token)
```

---

## TL;DR — what changes vs. RunPod

| Concern | RunPod (today) | Spark Compute v1 |
|---|---|---|
| Pod lifecycle | `runpod_launch.py` → GraphQL `podFindAndDeployOnDemand` | `POST /api/compute/jobs` (one HTTP call, agent does the rest) |
| File transfer | SSH + `tar` pipe + `scp` to `/workspace/` | One of two patterns: `inputPushMode: "auto-prepare"` (we give you `uploadUrl` + `exampleCurl`, you `tar | curl PUT`) **or** `inputShareSyncPath` (mount an existing ShareSync folder, no upload needed) |
| Container paths | `/workspace/{tunet,data,config.yaml,output/<job>}` | `/input/` (read-only mount of what you uploaded) and `/output/` (write here, files stream to ShareSync as they appear) |
| Process supervision | `nohup bash runpod_start.sh ... &` (so SSH disconnect doesn't kill it) | Container runs in foreground; agent owns lifetime. `nohup` is unnecessary and counter-productive (container exits when your foreground process exits, which is what triggers result upload + idle-hold). |
| Live training dashboard | `monitor_api.py` on port 8080 + RunPod proxy URL | SSE log stream `GET /api/compute/jobs/:id/logs/stream` (replays from start, then live-tails). Your existing regexes — `TRAIN_PAT`, `TIME_PAT`, `LPIPS_PAT`, etc. — apply unchanged on the client side. Preview JPGs (`training_preview.jpg`, `val_preview.jpg`) get streamed to ShareSync as you write them; UI fetches from ShareSync URLs. |
| Checkpoint sync | Background `sync_loop` polling SSH ls + scp every 120s | Streaming output uploader on the agent: every file written to `/output/` is `PUT`-ed to ShareSync as soon as it stops changing (~3–5s quiet window). No polling, no SSH. |
| Termination | `_auto_terminate` (sleep 30 min, then `kill 1`) | `idleHoldSeconds` parameter at submission time + reconciler cron. Set to `0` for "stop immediately when job exits", `300`–`1800` for "keep instance warm in case I want to resubmit." |
| Stop-during-training | Touch `.stop_training` via `POST /api/stop` on the pod's monitor_api | `POST /api/compute/jobs/:id/cancel` → SIGTERM. **v1 caveat**: this loses the soft-stop "finish current epoch and save final checkpoint" semantics your `--stop-file` flow gave you. Worst case you lose ≤`iterations_per_epoch` steps since the last save. We can wire the cancel signal to write your `.stop_training` sentinel into `/input/` instead in v1.1 if you want it back — let me know. |
| Multi-pod benchmark | `benchmark_all_gpus` spawns N threads, one pod per GPU | Loop submitting N independent jobs with different `instanceType` values. Each gets its own log stream. Same idea, just a different orchestration shape. |

---

## Per-file changes (running list — updated as the test progresses)

### `runpod_start.sh` → `spark_start.sh`

**Renamed** for clarity (this is no longer RunPod-specific). Functional changes:

1. **Removed `_auto_terminate()` function and the trailing call.**
   *Why*: the Spark agent owns the instance lifecycle. When `train.py` exits,
   the container exits, the agent uploads `/output/`, applies `idleHoldSeconds`,
   and stops the EC2 instance. Curling the RunPod stop API or `kill 1` from
   inside the container would just confuse the upload phase.

2. **Removed `monitor_api.py` startup block.**
   *Why*: no port exposure in compute v1, and SSE logs + ShareSync-streamed
   previews give the dashboard the same data without the in-pod webserver.
   See "Live training dashboard" row above.

3. **Removed `nohup ... &` wrapper around `train.py` launch.**
   *Why*: the agent runs the container in foreground and treats process exit
   as job completion. Backgrounding `train.py` would cause the container to
   exit immediately and the agent would think the job finished with whatever
   `nohup` returned (usually 0) before training even started.

4. **Path constants point at `/input/` and `/output/`** (instead of `/workspace/...`).
   *Why*: that's where the agent mounts your inputs (read-only) and your
   write directory (streamed to ShareSync). Your `runpod_launch.py` rewrote
   these paths client-side via `rewrite_config_for_pod`; we do the same thing,
   just with different prefixes.

5. **Removed `set -e` cascade for the `pip install` line.**
   *Why*: not actually changed yet — flagging in case it bites us. PyTorch
   wheels download takes ~3 min on first cold container start; if any single
   wheel times out, `set -e` will kill the whole script. We may switch to
   `pip install ... || pip install ...` retry pattern if we see flakes.

6. **Pinned to `python3.12` explicitly (was using whatever `python3` resolved to).**
   *Why* (surfaced over rounds 2 and 3 of the test): your
   `runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204` image ships **two
   Python interpreters**, and they're inconsistent with each other:
   - `python3` symlink resolves to **Python 3.10.12** (Ubuntu's stock)
   - `pip` binary lives in **`/usr/local/lib/python3.12/dist-packages/`** — i.e. it's bound to 3.12

   Two failures fell out of this:

   - **Round 2** (job `7c0a205f-4301-4cf2-8f62-7dc28bebc186`): we used
     bare `pip install pillow==12.0.0 …`. That installed into 3.12.
     We then ran `python3 train.py` (3.10) which couldn't see any of it
     → `ModuleNotFoundError: No module named 'PIL'`.

   - **Round 3** (job `db6d4b62-746a-4ab0-bac1-5406ac75c4ce`): we tried
     `python3 -m pip install …` to "just be consistent." That correctly
     targeted 3.10 — but your pinned `scipy==1.16.3` (and most likely
     `numpy==2.2.6`) **require Python ≥3.11**. Pip exited 1 with
     `ERROR: No matching distribution found for scipy==1.16.3`.

   So your version pins are implicitly assuming 3.12 (which is what
   the bare `pip` was using on RunPod). The fix that makes this robust:
   detect the newest available Python at the top of the script and use
   it explicitly throughout:

   ```bash
   PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
   $PY -m pip install …
   $PY train.py …
   ```

   On RunPod this is invisible because the bare `pip` already picked
   3.12. On any other base image (or any future runpod image change)
   it'll keep working as long as something ≥3.11 is installed.

   **You may also want to consider switching to a slimmer image** — the
   `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` official image has
   only one Python and is ~3GB instead of ~12GB. Trade-off: it doesn't
   pre-bake torch+opencv+lpips, so the cold-container `pip install` step
   is real work (~2-3 min for the torch wheels) — but on Spark Compute
   that's billable customer time (per-second), so it's worth running
   the comparison and seeing which net-out cheaper for your workload.

(More entries appended as testing reveals more deltas.)

### `runpod_launch.py` (no equivalent shipped this round)

Not adapting the launcher script in this round — it's overkill for the proof
of life. The minimal client equivalent for compute v1 is:

```bash
SPARK_HOST="https://api.prod.aapse1.sparkcloud.studio"

# 1. Submit
curl -X POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @submit.json
# Response includes: { jobId, input: { uploadUrl, exampleCurl }, output: { ... } }

# 2. Upload (response gives you the exact one-liner via input.exampleCurl)
tar -czf - -C /local/path . | curl -X PUT --data-binary @- \
  -H "Authorization: Bearer $TOKEN" "$INPUT_UPLOAD_URL"

# 3. Watch logs (SSE — replays from start, then live-tails)
curl -N "$SPARK_HOST/api/compute/jobs/$JOB_ID/logs/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: text/event-stream"
```

If you want a `spark_launch.py` that mirrors `runpod_launch.py` (auto-pack source,
auto-rewrite config paths, auto-tail logs, optional checkpoint download from
ShareSync at the end), that's a half-day port. Happy to write it once we've
proven the wire shape works. Functions to port directly with renames:

- `_pack_dir`        → identical
- `rewrite_config_for_pod`  → unchanged logic, just swap `/workspace/...` → `/input/...`
- `wait_for_pod`     → poll `GET /api/compute/jobs/:id` for `status` transitions
- `sync_loop`        → unnecessary (replaced by streaming output uploader)
- `download`         → fetch from ShareSync via WebDAV PROPFIND + GET, or just open the ShareSync space in Finder

### `monitor_api.py` (not used in compute v1)

Not removed from your repo — it'll keep working on RunPod. Not started
on Spark Compute, because the Spark UI consumes the SSE stream directly and
runs your existing log-parsing regexes (`TRAIN_PAT`, `TIME_PAT`, etc.)
client-side. Same data, no in-pod webserver.

### `train.py` (no changes)

Drop-in compatible. `--config` and `--stop-file` flags work exactly as today.

### Config (`base/base.yaml` → run-specific synthesized config)

Same shape as your `rewrite_config_for_pod` produces, just with `/workspace/...`
swapped for `/input/...` and `/output/...`. For the smoke test we override:

- `data.src_dir`     = `/input/data/src`
- `data.dst_dir`     = `/input/data/dst`
- `data.output_dir`  = `/output/eric-test`
- `training.max_steps` = 200
- `training.batch_size` = 1   (only have 3 fabricated pairs)
- `model.model_size_dims` = 64
- `logging.preview_batch_interval` = 0   (skip previews to keep test focused)

---

## Test plan

1. Fabricate 3 trivial 512×512 image pairs in `/tmp/tunet-eric-test/data/{src,dst}/`.
2. Pack `tunet/` source + `data/` + `config.yaml` + `spark_start.sh` into `spark-input.tar.gz`.
3. Submit `auto-prepare` job (image: `runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204` —
   your last-known-good image).
4. Upload tarball to the `uploadUrl` returned in the submission response.
5. Tail SSE logs.
6. Verify ShareSync at `/Compute Jobs/eric-tunet-test/`:
   - `training.log` (and matches the SSE stream)
   - At least one `.pth` checkpoint
   - `compute-job-<jobId>.log` archive (the agent's combined stdout/stderr archive)

## Test results (filled in after run)

**2026-05-01 16:00 (UTC+8)** — first attempt, job `2b8b4ecb-7d04-4067-8eb9-b9be3c146e21`:

- Submission OK. Tarball upload OK (HTTP 201, 809KB compressed).
- **Failed in `provisioning`** with `cancelled during pull-retry backoff (attempt 4)` after 233s. Root cause: warm-pool root volume was 30GB and already had multiple `pytorch/pytorch:*` and `nvidia/cuda:*` images cached from prior testing. Eric's `runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204` image is ~12GB compressed / ~20GB extracted. `docker pull` ran out of disk space mid-layer during partial-pull staging.
- **Two real findings** (one infra, one agent code):
  1. **Warm-pool root volumes need 200GB.** Resized.
  2. **Agent has no mid-pull disk-full self-heal.** Today's flow is: watermark check (prune if < 25% free) → start pull → on failure, exponential backoff up to 15min hard cap, no additional cleanup between retries. The retry just re-fails on the same condition. See "Agent ENOSPC mid-pull self-heal" below — fix shipped in v1.

**2026-05-01 16:11** — second attempt, job `7c0a205f-4301-4cf2-8f62-7dc28bebc186` (after 200GB resize):

- Upload OK, image pull OK (cached this time), input download OK, container started.
- **Failed** with exit 1 ~7s after `[setup] Done.`: `ModuleNotFoundError: No module named 'PIL'`.
- Cause: image's `python3` is 3.10 but bare `pip` is bound to 3.12. Bare `pip install pillow=…` landed packages in 3.12; `python3 train.py` (3.10) couldn't see them. See changelog item #6.

**2026-05-01 16:18** — third attempt, job `db6d4b62-746a-4ab0-bac1-5406ac75c4ce` (`pip → python3 -m pip` fix):

- **Failed** with `ERROR: No matching distribution found for scipy==1.16.3`.
- Cause: Eric's pinned `scipy==1.16.3` (and most likely `numpy==2.2.6`) require Python ≥3.11, so they were implicitly assuming 3.12 (which is what bare `pip` was using). `python3 -m pip` correctly used 3.10 → no compatible scipy. See changelog item #6.

**2026-05-01 16:23 — round 4, job `a8e4571f-c155-44ea-bd39-3fb005b11296`: GREEN** ✅

Submission → terminal in **1 min 47 sec** wall-clock (image cached, pip cache mostly warm).

| Phase | Wall-clock | Notes |
|---|---|---|
| Submit + tarball upload | 6.8s + 0.4s | 809 KB compressed input |
| Queued → provisioning | <5s | Job picked up immediately |
| Input download + extract | 3s | 808 KB tarball, 182 files |
| `pip install` | ~30s | torch wheels + albumentations etc. (mostly warm cache) |
| `nvidia-smi` GPU sanity | <1s | RTX PRO 6000 Blackwell, driver 580.105.08, CUDA 13.0 |
| Training (200 steps, batch=1, 512×512) | ~14s | T/Step 0.069s, ~14 steps/sec |
| Checkpoint saves (4 epoch + 1 latest) | ~5s | 377 MB each |
| Output upload to ShareSync | ~42s | 1.88 GB total, ~45 MB/s sustained |
| Container exit → terminal status | <2s | clean exit code 0 |

**ShareSync verification** (`/Compute Jobs/eric-tunet-test/`):

- `eric-test/training.log` (15 KB) — full Eric-format training log, ready for `monitor_api.py`-style regex parsing in Spark UI
- `eric-test/config_tunet_latest.pth` (377 MB) — final checkpoint
- `eric-test/config_tunet_epoch_000000001.pth` through `_000000004.pth` (377 MB each) — per-epoch checkpoints
- `spark-compute-a8e4571f-…log` (21 KB) — agent-side combined stdout/stderr archive with the header/footer summary
- (and the 3 earlier-attempt log archives — proof the log archival works on failure too)

**Loss curve** (from `training.log`, sanity-check that the model actually trained):

- Step 5: L1 0.0594
- Step 100: L1 0.0317
- Step 200: L1 0.0306 (Avg)

Model fit the trivial src→dst circle-shift+brightness mapping as expected.

---

## Agent / platform behavior notes

### Agent ENOSPC mid-pull self-heal — shipped in v1

If a `docker pull` runs out of disk space mid-layer (which can happen with
large customer images on a warm node that's been used for several jobs and
has a lot of cached image content), the agent now:

- Detects out-of-space conditions in the pull stderr.
- Runs an aggressive recovery pass — `docker system prune -af --volumes`
  plus a sweep of stale partial-pull staging directories.
- If recovery freed less than 1 GB → fails the job immediately with
  `error_code='disk_full'`. Doesn't loop.
- Otherwise retries the pull exactly once. If the second pull also fails
  with the same condition → `error_code='disk_full'`, no further loop.
- Non-disk pull failures (network, registry 5xx, etc.) keep their existing
  exponential-backoff behavior up to a 15-min ceiling.

**Practical implication for you:** the round-1 incident from this test (job
`2b8b4ecb-…`, 4 retries × 60s of guaranteed-failing pulls before manual
cancel) cannot recur. A submission against a full disk now either succeeds
(if recovery freed enough) or terminates with `disk_full` within ~30s.

### Instance type change on start — shipped in v1

**You can request a different `instanceType` on each `POST /api/compute/jobs`
call.** If the warm-pool member is currently sized differently (say it's a
`g7e.2xlarge` and you want a `g7e.4xlarge`), the server stops the warm
instance, runs `ModifyInstanceAttribute` to flip its EC2 type to whatever you
asked for, restarts it, and your job runs on the new shape. Verified end-to-end
with job `e0ec9caa-…`: warm `g7e.2xlarge` reshaped to `g7e.4xlarge` and ran
your `spark_start.sh` clean (16 vCPU / 128 GiB confirmed in the job log header
— that's the `g7e.4xlarge` shape, not the `g7e.2xlarge` we started with).

**Practical implications:**

- **First job after a reshape pays a one-time cost** of stop + AWS wait for
  stopped (~30–90s typical, occasionally up to 5 min on noisy AWS hardware) +
  attribute change (~5s) + start + agent boot (~45s). Plan on ~1–6 minutes
  from submit to "container running" for the first job at a new size.
- **Subsequent jobs at the same size** hit the warm path normally (zero
  reshape cost; idle-hold + warm-pool affinity covers your back-to-back-with-hold
  case at zero start time).
- **Held-affinity still requires a strict type match.** If you set
  `idleHoldSeconds=600` on a `g7e.2xlarge` job and then submit a `g7e.4xlarge`
  follow-up within the hold window, your follow-up DOESN'T get the held
  instance (it would defeat the point of the hold) — it'll fall through to the
  reshape path. This is the intended trade.
- **You're not paying for the reshape itself**, just for the EC2 time the
  instance is `running` after the reshape (same cost model as any warm-pool
  job today). The stop-modify-start sequence happens while the instance is
  `stopped` (no compute charges).
- **Eligible types are gated by an allowlist** that we expand as you tell us
  which sizes you actually want. Adding a SKU is a backend config update; ask
  Walt.

**Worth knowing for benchmark/multi-size sweeps:** if you `benchmark_all_gpus`-style
submit N jobs at N different sizes from a single warm pool, expect each
size-change to cost the reshape window. The most efficient ordering is "all
jobs at size A first, then all jobs at size B" — same logic as RunPod's
pod-creation overhead, just with different absolute numbers.

### Other v1.x candidates (planned, not in v1)

1. **Pre-pull free-space estimate.** Docker manifest API returns total
   compressed layer size. We could (a) sum that, (b) multiply by ~1.7 for
   typical extract overhead, (c) compare to actual free space, and (d) prune
   *before* the pull starts if the estimate doesn't fit. Avoids the
   half-pulled-then-fail trap entirely. Slightly more complex than today's
   reactive disk-full recovery; probably v1.2.

2. **Soft-stop semantics matching your `.stop_training` flow.** Today
   `POST .../cancel` sends SIGTERM; your existing UX touches a sentinel file
   so training finishes the current epoch and saves a final checkpoint
   before exiting cleanly. We can wire cancel to write your sentinel into
   `/input/` instead of (or alongside) the SIGTERM in v1.1 if you want the
   exact RunPod behavior preserved. Let me know.

3. **Preview image freshness tightening.** Current ShareSync upload window
   is ~3–5s quiet period after last write + upload roundtrip. Acceptable for
   most dashboards; if you want it tighter we can cut the quiet window.

---

## Open questions for Eric

1. **Soft-stop semantics** — keep your `.stop_training` UX in v1.1 (we wire cancel
   to write the sentinel) or accept hard SIGTERM (lose ≤500 steps, but no agent
   change required)?
2. **Preview image freshness** — current ShareSync upload window is ~3–5s quiet
   period after last write + upload roundtrip. Acceptable for the dashboard,
   or do you want it tighter?
3. **`spark_launch.py` wrapper** — worth writing, or do you prefer to drive
   the API directly from your existing tooling (the GUI in `spark_dashboard.py`,
   say)?
4. **Multi-GPU benchmark** — port `benchmark_all_gpus` to a multi-job submit
   loop now, or wait until v1.1?
