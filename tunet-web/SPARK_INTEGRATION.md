# tunet-web ↔ Spark Compute v1 Integration

**Audience:** Walt (Spark) and anyone picking up the web side.
**Status:** Working end-to-end against `api.prod.aapse1.sparkcloud.studio` as of 2026-05-01.
**Scope:** This doc describes only the Spark Compute v1 surface of `tunet-web/`.
The older RunPod/Supabase/Stripe layer (see `PRD.md`, `TECH_STACK.md`) is
unchanged and still wired through `(dashboard)/...` routes; the new Spark
flow lives under `/demo/...` and is auth-bypassed in dev.

---

## What it does

`/demo/dashboard` lists every Spark Compute job for the configured account.
`/demo/jobs/new` is a 4-step preset-driven submitter that:

1. Lets the user pick a **local project folder** (`<input webkitdirectory>`)
2. Auto-detects `src/`, `dst/`, `val_src/`, `val_dst/`, `mask/` subfolders by
   alias (so `input/` and `output/` work too — same alias rules as
   `gui/data_tab.py`)
3. Picks a preset (Beauty / Roto / Paintout / General — mirrors the desktop
   app) and a GPU
4. Uploads the picked frames to a server-side stage, packs them with the
   tunet source + a synthesized `config.yaml` + `spark_start.sh`, and ships
   the whole thing through Spark's `auto-prepare` flow

Once submitted, `/demo/jobs/[id]` shows a provisioning timeline (queued →
provisioning → pulling image → starting → running) inferred from the job's
timestamps, plus a live SSE log stream parsed from Spark's named `log`
events. There's a Cancel button that maps directly to
`POST /api/compute/jobs/:id/cancel`.

---

## Architecture in two pictures

### Submit flow

```
Browser                Next.js (Node runtime)              Spark
───────                ──────────────────────              ─────

[Pick folder]
  │ webkitdirectory scan
  │ → 234 .exr files, 1.2 GB
  ▼
[Click Start]
  │ POST /api/spark/upload-stage  (multipart, batched ≤200 MB each)
  ├─────────────────────────────► Stage to /tmp/tunet-stages/<id>/{src,dst}/
  │                                  │
  │ ←─── { stageId } ───────────────┘
  │
  │ POST /api/spark/training-jobs (text/event-stream response)
  ├─────────────────────────────► Validate
  │                               │
  │ ◄── data: {phase: validate, status: done}
  │                               │
  │                               ├─► Pack tarball:
  │                               │     tunet/ source (~760 KB)
  │                               │     + data/{src,dst}/ (1.2 GB)
  │                               │     + config.yaml
  │                               │     + spark_start.sh
  │ ◄── data: {phase: pack, files, kb, ms}
  │                               │
  │                               ├─► POST /api/compute/jobs ────────► Spark API
  │                               │   { name, instanceType, image,
  │                               │     command, inputPushMode: "auto-prepare",
  │                               │     idleHoldSeconds, env }
  │                               │   ◄─── { jobId, input.uploadUrl, ... }
  │ ◄── data: {phase: submit, jobId, ms}
  │                               │
  │                               ├─► PUT <uploadUrl>     ───────────► ShareSync WebDAV
  │                               │   (binary tarball body)
  │ ◄── data: {phase: upload, sentBytes, totalBytes}
  │                               │
  │                               └─► rm -rf /tmp/tunet-stages/<id>/
  │ ◄── data: {phase: done, jobId, output.shareSyncBaseUrl}
  │
  ▼
[Redirect to /demo/jobs/<jobId>]
```

### Job-detail flow

```
Browser                Next.js                        Spark
───────                ───────                        ─────

GET /demo/jobs/[id]
  │ (server component)
  ├──► getJob(id)  ────────────► GET /api/compute/jobs/:id
  │                              ◄── job detail JSON
  │ ◄── HTML with status header + metric strip
  ▼
[Hydrate ProvisioningTimeline]
  │ poll every 2.5s
  ├──► GET /api/spark/jobs/[id] ─► GET /api/compute/jobs/:id
  │                              ◄── current state
  │ classify failure phase from error_code / timestamps
  ▼
[Open EventSource]
  │ EventSource('/api/spark/jobs/[id]/logs')
  ├──► passthrough proxy ───────► GET /api/compute/jobs/:id/logs/stream
  │   ◄══════ event: log ═══════════════════════════════
  │           data: {ts, stream, phase, line}
  ▼
[Render lines, color stderr, copy/clear/auto-scroll]
```

---

## Where to look in the code

```
tunet-web/src/
├── lib/
│   ├── spark.ts              ← server-only Spark API client (auth, list, get,
│   │                           submit, cancel, upload, openLogStream)
│   ├── spark-types.ts        ← client-safe shapes + status helpers
│   ├── spark-packer.ts       ← server-only tarball builder (uses `tar` package)
│   └── spark-presets.ts      ← Beauty/Roto/Paintout/General + cost estimator
├── app/api/spark/
│   ├── jobs/
│   │   ├── route.ts                      GET list, POST smoke-test submit
│   │   └── [id]/
│   │       ├── route.ts                  GET detail
│   │       ├── cancel/route.ts           POST → SIGTERM
│   │       └── logs/route.ts             GET → SSE proxy (passthrough)
│   ├── skus/route.ts                     GET eligible SKUs
│   ├── upload-stage/route.ts             POST to stage user files server-side
│   └── training-jobs/route.ts            POST: pack + submit + upload (SSE)
├── app/demo/
│   ├── dashboard/page.tsx                live dashboard
│   ├── jobs/page.tsx                     list with filters
│   └── jobs/[id]/page.tsx                detail with timeline + log stream
└── components/spark/
    ├── status-badge.tsx                  Spark-aware (queued/provisioning/...)
    ├── log-stream-panel.tsx              EventSource client + JSON parser
    ├── provisioning-timeline.tsx         pre-log "what's happening" view
    ├── job-live-view.tsx                 wraps timeline + log panel
    ├── cancel-job-button.tsx             confirmation + POST cancel
    ├── submit-progress.tsx               modal overlay with phase steps
    └── folder-picker.tsx                 webkitdirectory + alias detection
```

---

## Live API observations

These are the shapes we actually see from the Spark API as of 2026-05-01.
Walt — flagging anything that surprised us:

### Auth

```
POST /auth/login
  Body:    { email, password }
  Returns: { token, ... }
```

The token is a standard JWT. The `exp` claim is read client-side to refresh
proactively. **Note:** `/api/auth/login` (with `/api/` prefix as written in
your doc) returns 404 — the bare `/auth/login` path is what works.

### Submit response

```
POST /api/compute/jobs
  Body:    { name, instanceType, image, command, inputPushMode: "auto-prepare",
             idleHoldSeconds, env }
  Returns: {
    jobId,           // camelCase
    status,
    imageDigest,
    outputShareSyncPath,
    output: { shareSyncPath, shareSyncSpaceName, shareSyncBaseUrl },
    input:  { shareSyncPath, uploadUrl, uploadMethod: "PUT", exampleCurl }
  }
```

### List / detail response

```
GET /api/compute/jobs           returns { jobs: [...] }
GET /api/compute/jobs/:id       returns single job

  Per-job fields (snake_case):
    id, status, image, command, env,
    output_share_sync_path,
    instance_type_name,                      ← e.g. 'g5.xlarge'
    error_code, error_message, exit_code,
    created_at, started_provisioning_at,
    last_agent_heartbeat_at,                 ← null on most failed jobs?
    started_running_at, terminal_at,
    cancel_requested_at,
    cuda_version, driver_version, gpu_name,  ← see note below
    log_archive_share_sync_path,
    idle_hold_seconds,
    input_share_sync_path,
```

**`name` is not echoed back** — neither in the submit response nor in the
list/detail response. We work around this by stashing the friendly name in
`env.TUNET_JOB_NAME` on submit and reading it back from there. Would be
cleaner if the API persisted and returned `name`. (We also stash
`TUNET_PRESET` and `TUNET_GPU` in env for the same reason — useful metadata
for our UI to show without a separate DB.)

**`gpu_name` / `cuda_version` / `driver_version` are populated even when
the container never started** (i.e. when `started_running_at` is null and
the job is failed). I think this comes from SKU metadata, not from
`nvidia-smi` inside the container. That's fine but it means we can't use
those fields as a signal that the container actually ran. We use
`error_code === 'container_nonzero_exit'` instead — that one only happens
post-pull.

### Log stream

```
GET /api/compute/jobs/:id/logs/stream
  Accept: text/event-stream
  Returns SSE with named events:

    event: log
    id: <iso-ts>-<stream>
    data: { "ts": "2026-05-01T16:43:25.375Z",
            "stream": "stdout"|"stderr",
            "phase":  "agent"|"container",
            "line":   "<actual log content>" }
```

The browser `EventSource` API listens on the default `'message'` event by
default, so a naive client misses everything. We listen on the named
`'log'` event explicitly. Walt's doc says "data: lines" which is technically
true but doesn't surface the actual JSON shape — worth adding to the doc
for next users.

The stream replays from the start of the job, then live-tails. Heartbeats
arrive as `:keepalive` SSE comments which the browser silently drops — good.

### Job lifecycle inference

The detail response doesn't have a single explicit "phase" field; we infer
from timestamps:

| Phase            | Heuristic                                                      |
|------------------|----------------------------------------------------------------|
| `queued`         | `status === 'queued'` and no `started_provisioning_at`         |
| `provisioning`   | `started_provisioning_at` set, no heartbeat or run-start       |
| `pulling`        | `last_agent_heartbeat_at` set, no `started_running_at`         |
| `starting`       | (collapses into pulling — agent doesn't report separately)     |
| `running`        | `started_running_at` set, no terminal                          |
| `done` (success) | `status === 'completed'`                                       |
| `cancelled`      | `cancel_requested_at` set                                      |
| `failed`         | `status === 'failed'`; phase classified by `error_code`        |

**Failure phase classification is the trickiest part.** We use:

- `error_code === 'container_nonzero_exit'` → failed in **Running** (container
  started, exited non-zero — typical training error)
- `error_code === 'disk_full'` or `'image-pull-failed'` → failed in **Pulling**
- otherwise fall back to whichever phase had a timestamp without the next one

Would be much cleaner if the API exposed a structured `phase_at_failure`
field. Until then this works.

### ShareSync WebDAV

We probed direct browser → ShareSync uploads (intent: bypass the Next.js
middleman for large datasets). Two findings:

1. The submit-issued `uploadUrl` works fine with our bearer token — that's
   how the Compute v1 docs say it should.
2. `MKCOL` on user-managed paths like `/Compute Inputs/<project>/` returns
   409 Conflict ("create container: error: precondition failed"). So
   arbitrary user folders aren't writable from outside the per-job
   namespace. **For now, we bundle the data into the per-job tarball.**

If you (Walt) plan to support user-managed staging folders for larger
datasets (1 GB+ EXR projects), that would need:

- A way to MKCOL a user space (or user-scoped pre-flight)
- A signed-URL or scoped-token mechanism so the browser can PUT directly
  without us proxying

For now the Next.js server proxies the upload (browser → Next.js → tar →
ShareSync). That's fine for dev/single-user but won't scale.

---

## What runs in the container

`spark_start.sh` (in the project root, packed into every tarball) is the
adapted version of `runpod_start.sh`:

```bash
#!/bin/bash
set -e

OUTPUT_DIR="${1:-/output}"
CONFIG="${2:-/input/config.yaml}"
TUNET_DIR="/input/tunet"

PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
$PY -m pip install -q torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q albumentations==2.0.8 ... etc

mkdir -p "$OUTPUT_DIR"
# Seed staged checkpoints if any (for resume)
[ -d "/input/output/$(basename "$OUTPUT_DIR")" ] && \
    cp -an "/input/output/$(basename "$OUTPUT_DIR")"/* "$OUTPUT_DIR"/ || true

cd "$TUNET_DIR"
$PY train.py --config "$CONFIG" --stop-file "$OUTPUT_DIR/.stop_training"
```

Per your doc: no `nohup`, no `monitor_api.py`, no auto-terminate. The
container runs `train.py` in foreground; when it exits the agent uploads
`/output/` to ShareSync and the instance enters idle-hold.

---

## Configuration

`.env.local` needs:

```
SPARK_EMAIL=...
SPARK_PASSWORD=...
SPARK_DEFAULT_REGION=us-east-1   # optional
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_placeholder   # demo bypass
```

That's it. No DB, no Stripe, no RunPod key needed for the Spark flow.

---

## Run it locally

```bash
cd tunet-web
npm install
npm run dev
# → http://localhost:3000   (auto-redirects to /demo/dashboard)
```

The first `tsc --noEmit` should be clean. Submitting a job costs real money
(it lands at Spark) — for testing, use `gpu: 't4'` (cheapest) and
`max_steps: 1` to abort early. Or set `image: 'busybox:latest'` and
`command: ['echo','hi']` against `/api/spark/jobs` directly to verify the
wire without provisioning a real GPU.

---

## Open questions for Walt

1. **`name` in submit response** — please add it to list/detail output so
   we don't have to round-trip through env vars.
2. **`phase` field on terminal jobs** — having Spark report which phase
   actually failed (rather than us inferring from `error_code`) would
   simplify the timeline UI a lot.
3. **User-managed ShareSync folders** — what's the recommended path for
   pre-staging datasets so they don't have to ride in every tarball? The
   `inputShareSyncPath` mode mentioned in the doc would help, but we
   couldn't find the field name in the submit body. Is it documented
   somewhere?
4. **CORS on ShareSync WebDAV** — direct browser PUT would let us drop the
   Next.js proxy step for large datasets. Do you support that already, or
   is it a planned addition?
5. **`event: status` SSE events** — your doc mentions `data:` events; we
   only see `event: log`. Are there other named events (status changes,
   resource samples) we should be listening for?
6. **Customizable `outputShareSyncPath`** — outputs currently land at
   `/Compute Jobs/<jobId>/...`, keyed by UUID. From a user's perspective,
   "where did my Porter_0408 model go?" requires looking up the jobId first.
   Could the submit body accept an `outputShareSyncPath` parameter (e.g.
   `/My Models/Porter_0408_REDO_v3/`) that overrides the default? Would
   let us organize outputs by project / friendly name without a separate
   move/rename step. Same question for `inputShareSyncPath` so users can
   pre-stage data under their own folder names.
7. **Job deletion / archival** — there's no `DELETE /api/compute/jobs/:id`
   that we've found. We're hiding terminal jobs from our UI via localStorage
   for now, but that just clutters Spark's account history. Is there a
   server-side way to actually purge a job (and ideally its ShareSync
   artifacts) from the account?
8. **ShareSync storage pricing + cleanup** — what's the rate? Is there an
   automatic TTL on `/Compute Jobs/<jobId>/` outputs, or do they live
   forever until manually deleted via WebDAV? A typical TuNet run produces
   ~2 GB of checkpoints; if users do hundreds of runs that adds up.

---

## What's NOT here (intentionally)

- **No persistence layer.** Spark is the source of truth for jobs; we don't
  cache anything in a DB. If the user closes their browser, they refresh
  the dashboard and pick up wherever the job is.
- **No auth.** This is the `/demo/*` routes only — Clerk is bypassed for
  local dev. The older `(dashboard)/*` routes still expect Clerk + Supabase
  but those aren't part of the Spark flow.
- **No billing.** This is your account submitting your jobs. Multi-user
  billing would layer on top of this when we wire Clerk back in for prod.
