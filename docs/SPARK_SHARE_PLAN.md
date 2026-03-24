# Tunet Spark Share - Product & Technical Plan

## Vision

A turnkey web service branded under CPFX/Spark that lets VFX artists upload source/target frames, pick a preset, and get a trained CopyCat-style model back — no local GPU, no Deadline, no command line. Powered by Spark's on-demand GPU infrastructure.

---

## 0. Existing Work: Simplified Preset UI (branch: `claude/simplified-preset-ui-xeFvu`)

A simplified, preset-driven local UI (`tunet_simple.py`) already exists and serves as the **design reference** for the Spark Share web interface. It was built for After Effects users who don't want to touch the full Tunet UI. Key elements to carry forward:

### What's Already Built (1,021 lines)

- **2 task presets** with auto-configured training parameters:

| Preset | Model | Loss | LR | Auto-Mask | Augmentations |
|--------|-------|------|----|-----------|---------------|
| **Beauty / Paint** | MSRN 128 | L1+LPIPS (0.1) | 5e-4 | Yes (weight 10.0) | HFlip |
| **Roto / Matte** | UNet 128 | BCE+Dice | 1e-4 | No | HFlip + Affine |

- **Auto-detection**: Scans `src/` folder, reads image dimensions, recommends resolution/model size/batch size based on image resolution and preset
- **Simple folder convention**: User points to a project folder containing `src/`, `dst/`, and `model/` is auto-created
- **One-click train/stop/inference**: No config files to write, no CLI
- **Live preview**: Refreshes training preview image every 3 seconds
- **Session persistence**: Remembers last preset and project folder
- **After Effects user guide** (`docs/after_effects_guide.md`): Step-by-step walkthrough written for compositors, not ML engineers

### What the Web Version Inherits

The Spark Share web UI should mirror this exact workflow:
1. **Same 2 presets** — the preset definitions (`PRESETS` dict) become the server-side preset configs
2. **Same auto-detection logic** — `recommend_settings()` runs server-side after upload
3. **Same folder convention** — `src/`, `dst/`, `model/`, `output/` structure on ShareSync storage
4. **Same "pick preset, point to data, hit train" simplicity** — just in a browser instead of PySide6
5. **AE guide adapted for web** — the existing guide's workflow section becomes the Spark Share tutorial, replacing "export from AE, run locally" with "export from AE, upload to Spark Share"

### What Changes for Cloud

| Local (tunet_simple.py) | Spark Share (web) |
|-------------------------|-------------------|
| Browse for local folder | Drag-and-drop upload or ShareSync path |
| Subprocess runs train.py | Worker node runs train.py |
| Preview via QPixmap timer | Preview via WebSocket image stream |
| Free (your own GPU) | From $1.65/hr (A10) with cost estimate before start |
| Single job at a time | Multiple concurrent jobs on separate nodes |
| No inference export | Auto-exports ONNX for Nuke/Flame/AE on completion |

---

## 1. User Experience

### Web Interface Flow

```
Login (Spark account) --> New Job --> Upload Data --> Pick Preset --> Review Cost --> Train --> Download Results
```

**Step-by-step:**

1. **Login** - Authenticate via existing Spark account (SSO/OAuth)
2. **New Job** - Name the job
3. **Upload Data**
   - Drag-and-drop zones for: Source frames (`src/`), Target frames (`dst/`), Validation frames (optional)
   - Accepts PNG, JPG, EXR, TIFF, BMP, WebP
   - Files go to ShareSync-mounted storage (no copies needed if already on Spark)
   - Or paste a ShareSync path if data is already in the cache
   - **Auto-detection kicks in**: image count, dimensions, recommended settings displayed
4. **Pick Preset**
   - **Simple mode** (default): Pick from 2 presets (Beauty/Paint, Roto/Matte) — each with description explaining the use case
   - **Advanced mode**: Expandable panel with full parameter control (resolution, model type, loss, batch size, LR, mask settings, augmentations)
   - Auto-detected settings shown (resolution, model capacity, batch size) — user can override
5. **Review Cost** - Clear cost estimate shown before submission
6. **Train** - Hit "Run" and monitor progress in the browser
7. **Download** - Get ONNX exports (ready to load in Nuke/Flame/AE), trained checkpoint (.pth), and preview images. User runs inference in their compositor.

### Presets (Simple Mode)

Two presets cover the real VFX pix2pix use cases:

| Preset | Description | Model | Loss | Use Case |
|--------|-------------|-------|------|----------|
| **Beauty / Paint** | Skin retouching, blemish removal, wire removal, rig removal, cleanup & paint fixes | MSRN 128 | L1+LPIPS | Auto-mask focuses on changed areas |
| **Roto / Matte** | Segmentation and matte extraction | UNet 128 | BCE+Dice | src=plate, dst=B&W matte |

Advanced users can still override model type, loss function, etc. in the Advanced Settings panel.

### Dashboard Features

- **Job Queue** - See all active, queued, and completed jobs
- **Live Progress** - Real-time loss curve, preview images refreshing during training
- **Cost Tracker** - Running cost per job and monthly total
- **History** - Re-run past jobs with tweaked settings
- **Download Center** - All outputs with export format options

---

## 2. Pricing

### GPU Tiers

Based on actual Spark Cloud Studio workstation pricing:

| Tier | GPU | VRAM | CPU | RAM | Rate | Best For |
|------|-----|------|-----|-----|------|----------|
| **Starter** | NVIDIA A10 | 24GB | 4-core Zen2 EPYC | 16GB | **$1.65/hr** | Most jobs (beauty, roto, color, cleanup) |
| **Mid-Tier** | NVIDIA L4 | 24GB | 8-core Zen3 EPYC | 32GB | **$2.59/hr** | Larger datasets, faster CPU preprocessing |
| **Performance** | NVIDIA L40S | 48GB | 16-core Zen3 EPYC | 128GB | **~$4.50/hr** | High-res (1024px), large models, multi-GPU |

### Cost Estimator (shown before "Run")

The web UI calculates estimated cost before submission:

```
Estimated cost = (GPU tier rate) x (estimated hours)

Estimated hours based on:
  - Number of frames
  - Resolution
  - Model complexity (UNet vs MSRN, hidden dims)
  - Max steps / iterations
```

**Example estimates displayed to user (Starter A10 tier):**
- "200 frames, Beauty/Paint preset (MSRN), ~6 hours --> ~$9.90"
- "500 frames, Beauty/Paint preset (MSRN), ~9 hours --> ~$14.85"
- "300 frames, Roto/Matte preset, ~6 hours --> ~$9.90"

### Recommended Training Duration

Instead of a raw "max budget" or "max steps" input, the UI presents a **recommended training duration** based on preset + dataset size. This is critical — setting a max budget is pointless if the model hasn't trained long enough to converge.

| Preset | Frames | Recommended Duration | Approx Steps | Approx Cost (A10) |
|--------|--------|---------------------|--------------|-------------------|
| Beauty / Paint | 100-200 | ~6 hours | 10,000-12,000 | ~$9.90 |
| Beauty / Paint | 200-500 | ~9 hours | 15,000-20,000 | ~$14.85 |
| Beauty / Paint | 500+ | ~12 hours | 20,000-30,000 | ~$19.80 |
| Roto / Matte | 100-200 | ~4.5 hours | 8,000-10,000 | ~$7.43 |
| Roto / Matte | 200-500 | ~7.5 hours | 15,000 | ~$12.38 |
| Roto / Matte | 500+ | ~10.5 hours | 20,000 | ~$17.33 |

> **These are placeholder estimates** — real values need benchmarking on Spark hardware. The recommendation engine should be tuned based on actual training results and user feedback.

Users can choose:
- **Recommended** (default) — duration/steps tuned to dataset size
- **Short** — ~40% of recommended (quick test, may underfit)
- **Long** — ~2x recommended (extra convergence, diminishing returns)
- **Custom** — set exact steps or hours

### Budget Controls

- Monthly spending alerts at configurable thresholds
- Organization-level spending caps

---

## 3. Architecture

### Core Loop

The entire system boils down to one thing: **the frontend builds a config YAML, sends it to a GPU node, and the node runs `train.py --config`**. Everything else is plumbing.

```
Frontend (browser)                    Backend                         GPU Node (Rocky 9)
──────────────────                    ───────                         ──────────────────

1. User picks preset,        ──►  2. API validates,           ──►  4. Node receives config.yaml
   uploads data to                    estimates cost,                  + ShareSync data path
   ShareSync                          writes config.yaml
                                      to job queue
3. User sees cost,           ◄──                                    5. Runs: train.py --config job.yaml
   hits "Train"
                                                                    6. Training runs, writes to output_dir:
                                                                       - checkpoints (.pth)
                                  ◄── WebSocket/polling ◄──           - training_preview.jpg (updates live)
7. Frontend shows:                                                     - loss_log.csv
   - Loss chart (from log)                                             - validation_preview.jpg
   - Preview images (live)
   - Cost ticker
                                                                    8. On completion/stop:
8. User clicks "Stop & Save"  ──►  signal ──────────────────►         - Final checkpoint saved
   or training hits max_steps                                          - Auto-export ONNX (Nuke/Flame/AE)
                                                                       - Node goes idle → auto-shutdown
9. User downloads:            ◄──  serves files from ShareSync
   - .pth checkpoint
   - .onnx exports
   - preview images
```

### What the Frontend Actually Does

The frontend is a **config builder + training monitor**. It does NOT run training or inference.

1. **Builds a config YAML** from preset selection + user overrides + auto-detected settings
2. **Uploads data** to ShareSync (or accepts a ShareSync path)
3. **Shows cost estimate** based on `recommend_settings()` output
4. **Monitors training** via:
   - **Loss chart**: Reads `loss_log.csv` from the output dir (same as `training_monitor.py` does locally)
   - **Preview images**: Polls `training_preview.jpg` and `validation_preview.jpg` from output dir
   - **Console log**: Streams stdout from the training process
5. **Downloads results** when done

### What the GPU Node Does

The node is dead simple — it's a Rocky 9 box with Tunet installed that:

1. Receives a config YAML + path to data on ShareSync
2. Runs `python train.py --config /path/to/job_config.yaml`
3. Writes all output to ShareSync (checkpoints, previews, logs)
4. On completion, runs ONNX export for Nuke/Flame/AE
5. Reports elapsed time for billing

That's it. The node doesn't need to know about the web frontend, billing, or user accounts. It just runs train.py.

### V1: No Inference Needed

Users download the trained model and run inference in their compositor:
- **Nuke**: Load the `.onnx` + generated `.nk` script
- **Flame**: Load the `.onnx` via ML toolkit
- **After Effects**: Load via ONNX plugin or Tunet local inference

Inference-as-a-service can be a future addition, but V1 ships without it.

### System Components

```
                        +------------------+
                        |   Web Frontend   |  (React/Vite)
                        |   spark.cpfx.io  |
                        +--------+---------+
                                 |
                                 | HTTPS / WebSocket
                                 |
                        +--------+---------+
                        |    API Server    |  (FastAPI)
                        |                  |
                        |  - Auth (Spark)  |
                        |  - Config builder|
                        |  - Job queue     |
                        |  - Cost calc     |
                        |  - File proxy    |
                        +--------+---------+
                                 |
                    +------------+------------+
                    |                         |
           +--------+--------+     +---------+---------+
           |   Job Scheduler  |     |     ShareSync      |
           |   (Deadline or   |     |   (File Storage)   |
           |    custom queue) |     |  data + outputs +   |
           +--------+---------+     |  previews + logs    |
                    |               +---------------------+
        +-----------+-----------+         ▲
        |           |           |         │ (reads/writes)
   +----+----+ +----+----+ +----+----+    │
   | GPU Node | | GPU Node | | GPU Node |──┘
   | train.py | | train.py | | train.py |
   +----------+ +----------+ +----------+
```

#### Frontend (React/Vite)
- Config builder UI (preset picker + advanced params)
- Training monitor: loss chart + preview images + console log
- File upload to ShareSync
- Job dashboard (active, queued, completed)
- Cost display

#### API Server (FastAPI)
- Generates config YAML from frontend request
- Queues jobs + manages node spin-up
- Proxies training progress files (previews, logs) from ShareSync to frontend via WebSocket
- Cost estimation using `recommend_settings()`
- Auth via Spark SSO

#### GPU Nodes (Rocky 9)
- Tunet installed (Python, PyTorch, CUDA)
- ShareSync mounted
- Runs `train.py --config` and writes to ShareSync
- Auto-exports ONNX on completion
- Reports elapsed time, goes idle

---

## 4. API (for Power Users)

REST API for programmatic submission:

```
POST   /api/v1/jobs              # Create training job (sends config YAML to queue)
GET    /api/v1/jobs              # List all jobs
GET    /api/v1/jobs/{id}         # Job status + details + cost so far
DELETE /api/v1/jobs/{id}         # Cancel/stop a job (sends SIGINT to train.py)
GET    /api/v1/jobs/{id}/logs    # Training console output
GET    /api/v1/jobs/{id}/loss    # Loss log CSV (for charting)
GET    /api/v1/jobs/{id}/preview # Latest training/validation preview image
POST   /api/v1/estimate          # Cost estimate from preset + frame count (no job created)
GET    /api/v1/presets            # List available presets (from shared PRESETS dict)
WS     /api/v1/jobs/{id}/stream  # WebSocket: live loss values + preview update notifications
```

**Example: Submit a job via curl**
```bash
curl -X POST https://spark.cpfx.io/api/v1/jobs \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hero_beauty_v2",
    "preset": "beauty",
    "data": {
      "src_path": "/sharesync/project/src/",
      "dst_path": "/sharesync/project/dst/",
      "val_src_path": "/sharesync/project/val_src/",
      "val_dst_path": "/sharesync/project/val_dst/"
    },
    "gpu_tier": "standard",
    "max_budget": 10.00,
    "max_hours": 8
  }'
```

**Example: Submit with advanced config**
```bash
curl -X POST https://spark.cpfx.io/api/v1/jobs \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -d '{
    "name": "detail_cleanup",
    "config": {
      "model": {"model_type": "msrn", "model_size_dims": 128, "recurrence_steps": 2},
      "training": {"loss": "l1+lpips", "lambda_lpips": 0.2, "batch_size": 4, "use_amp": true},
      "data": {"resolution": 512, "overlap_factor": 0.25}
    },
    "data": {
      "src_path": "/sharesync/project/src/",
      "dst_path": "/sharesync/project/dst/"
    },
    "gpu_tier": "performance"
  }'
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
**Goal: End-to-end proof of concept**

- [ ] **Merge simplified UI branch** (`claude/simplified-preset-ui-xeFvu`)
  - Merge into `multios`
  - Extract `PRESETS` and `recommend_settings()` into shared `tunet/presets.py`
  - Both local UIs (`tunet.py`, `tunet_simple.py`) import from shared module

- [ ] **Worker service** (`spark/worker.py`)
  - Receives config YAML from job queue
  - Runs `train.py --config job.yaml` (data + output on ShareSync)
  - train.py already writes previews, loss logs, checkpoints to output_dir
  - On completion/stop: runs ONNX export (Nuke/Flame/AE)
  - Reports elapsed time for billing

- [ ] **API server** (FastAPI)
  - Job submission endpoint (POST /api/v1/jobs)
  - Job status endpoint (GET /api/v1/jobs/{id})
  - Cost estimation endpoint (uses `recommend_settings()` from shared module)
  - Preset listing endpoint (serves `PRESETS` dict directly)
  - Basic auth (API key to start, Spark SSO later)
  - PostgreSQL schema for jobs table

- [ ] **Rocky 9 base image**
  - Install Python, PyTorch, CUDA, Tunet deps
  - Configure ShareSync mount
  - Test training end-to-end on Spark GPU node

- [ ] **Minimal web UI**
  - Login page
  - Job submission form mirroring `tunet_simple.py` flow: pick preset, upload data, see auto-detected settings
  - Job list with status
  - Cost display before submission

### Phase 2: Full Web Experience (Weeks 4-6)
**Goal: Polished drag-and-drop interface**

- [ ] **File upload system**
  - Chunked upload for large frame sequences
  - Upload to ShareSync-backed storage
  - Progress bars, drag-and-drop zones
  - Support direct ShareSync path input (skip upload)

- [ ] **Training monitor in browser**
  - Loss chart: reads loss_log.csv from ShareSync output dir (same data as local `training_monitor.py`)
  - Preview images: polls training_preview.jpg + validation_preview.jpg from output dir
  - Training/validation toggle (like local UI's Previews tab)
  - Console log streaming via WebSocket
  - Cost ticker: elapsed time x rate, updated live

- [ ] **Advanced configuration mode**
  - All Tunet parameters exposed in collapsible sections
  - Mirrors the local PySide6 UI options
  - Parameter validation with helpful tooltips

- [ ] **Job management**
  - Cancel running jobs (sends SIGINT → train.py saves checkpoint gracefully)
  - Clone/retry completed jobs with tweaked settings
  - Download results: checkpoint (.pth) + ONNX exports + preview images

### Phase 3: Multi-Job & Scaling (Weeks 7-9)
**Goal: Concurrent jobs, auto-scaling nodes**

- [ ] **Multi-job support**
  - Queue multiple jobs per user
  - Priority queue (FIFO by default)
  - Concurrent jobs on separate nodes

- [ ] **Node auto-scaling**
  - Spin up GPU nodes when jobs are queued
  - Spin down after configurable idle timeout
  - Node health monitoring and auto-restart
  - Support multiple GPU tiers (Standard/Performance/Multi-GPU)

- [ ] **Multi-GPU jobs**
  - DDP training across 2+ GPUs on a single node
  - `torchrun` integration in worker
  - Automatic batch size scaling

- [ ] **Billing integration**
  - Per-second billing (rounded to nearest minute)
  - Budget caps per job and per account
  - Monthly invoicing tied to Spark account
  - Usage dashboard with charts

### Phase 4: Production Polish (Weeks 10-12)
**Goal: Production-ready, onboarding flow**

- [ ] **Spark SSO integration**
  - Full OAuth flow with Spark accounts
  - Organization/team support
  - Role-based access (admin, user)

- [ ] **Onboarding wizard**
  - First-time user walkthrough
  - Sample dataset to try for free (first job free?)
  - Tutorial: "Train your first model in 5 minutes"

- [ ] **Notifications**
  - Email when job completes
  - Slack webhook integration (optional)
  - Browser push notifications

- [ ] **Inference-as-a-service** (future, not V1)
  - Upload frames + select a trained model → run on cloud GPU → download results
  - V1 users run inference locally in Nuke/Flame/AE using the exported ONNX model

- [ ] **Monitoring & ops**
  - Logging (structured, centralized)
  - Error alerting
  - Usage analytics
  - Node utilization metrics

---

## 6. New Files / Changes to Tunet

### New files to create:

```
tunet/
├── tunet_simple.py               # Existing: simplified local UI (from branch)
├── docs/after_effects_guide.md   # Existing: AE user guide (from branch)
├── presets.py                    # New: shared PRESETS + recommend_settings()
│                                 #   (extracted from tunet_simple.py, used by
│                                 #   local UI, API server, and worker)
├── spark/                        # New: Spark Share package
│   ├── __init__.py
│   ├── worker.py                 # GPU node worker (polls queue, runs training)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app
│   │   ├── routes/
│   │   │   ├── jobs.py           # Job CRUD endpoints
│   │   │   ├── auth.py           # Authentication
│   │   │   ├── presets.py        # Preset listing
│   │   │   └── estimate.py       # Cost estimation
│   │   ├── models.py             # Pydantic schemas
│   │   ├── database.py           # PostgreSQL connection
│   │   └── websocket.py          # Real-time streaming
│   ├── config.py                 # Spark-specific settings
│   ├── cost.py                   # Pricing/estimation logic
│   └── node_manager.py           # Node spin-up/down orchestration
├── web/                          # New: Frontend
│   ├── package.json
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Login.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── NewJob.tsx
│   │   │   ├── JobDetail.tsx
│   │   │   └── Settings.tsx
│   │   ├── components/
│   │   │   ├── FileUpload.tsx
│   │   │   ├── PresetPicker.tsx
│   │   │   ├── ConfigEditor.tsx
│   │   │   ├── CostEstimate.tsx
│   │   │   ├── TrainingProgress.tsx
│   │   │   └── PreviewViewer.tsx
│   │   └── ...
│   └── ...
└── docker/                       # New: Containerization
    ├── Dockerfile.worker         # GPU worker image (Rocky 9 + CUDA + Tunet)
    ├── Dockerfile.api            # API server image
    └── docker-compose.yml        # Local dev stack
```

### Existing files to leverage (from `claude/simplified-preset-ui-xeFvu` branch):

- **`tunet_simple.py`** — The `PRESETS` dict and `recommend_settings()` function are extracted into the API server's preset/config logic. The PySide6 UI continues working standalone for local users.
- **`docs/after_effects_guide.md`** — Adapted into the web onboarding tutorial. The "Step by Step" section becomes the basis for the Spark Share getting-started guide, with "run locally" steps replaced by "upload and train in browser."

### Changes to existing Tunet code:

- **`tunet_simple.py`**: Extract `PRESETS` dict and `recommend_settings()` into a shared module (`tunet/presets.py`) importable by both the local UI and the API server
- **`train.py`**: No changes needed for V1 — the worker just runs `train.py --config` as a subprocess, same as the local UI does
- **`utils/convert_nuke.py` / `convert_flame.py`**: Worker calls these after training completes to auto-export ONNX. Already work as standalone scripts.
- **`inference.py`**: No changes for V1 — users run inference locally in their compositor using the ONNX export
- No changes to the local PySide6 UIs — both `tunet.py` (full) and `tunet_simple.py` (preset-based) continue to work independently

---

## 7. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **API framework** | FastAPI | Python (same as Tunet), async, auto-docs, WebSocket support |
| **Frontend** | React + Vite | Simple, fast, widely known |
| **Database** | PostgreSQL | Reliable, good for job/billing data |
| **Job queue** | Redis | Fast, pub/sub for real-time, lightweight |
| **File storage** | ShareSync + S3 | ShareSync for Spark integration, S3 for uploads |
| **Worker base OS** | Rocky 9 | Walt's recommendation, enterprise Linux |
| **Containerization** | Docker | Reproducible GPU worker images |
| **Real-time updates** | WebSocket + Redis pub/sub | Low latency training progress |
| **Auth** | Spark SSO (OAuth2) | Seamless Spark onboarding |

---

## 8. Pricing Display (What Users See)

### Before Submission
```
+------------------------------------------+
|  Cost Estimate                           |
|                                          |
|  GPU: Starter (A10 24GB)     $1.65/hr   |
|  Frames: 200 src + 200 dst              |
|  Preset: Beauty (MSRN 128, 512px)       |
|  Est. duration: ~2.5 hours               |
|                                          |
|  Estimated cost:  $4.13                  |
|  Your max budget: $5.00                  |
|                                          |
|  [  Start Training  ]                   |
+------------------------------------------+
```

### During Training
```
+------------------------------------------+
|  Job: hero_beauty_v2          TRAINING   |
|                                          |
|  Step 1,247 / 5,000                      |
|  Loss: 0.0142 (val: 0.0189)             |
|  Elapsed: 1h 12m                         |
|  Cost so far: $1.98                      |
|  Est. remaining: ~1h 18m ($2.15)        |
|                                          |
|  [Preview] [Logs] [Stop & Save]         |
+------------------------------------------+
```

---

## 9. Onboarding Strategy

This service doubles as a **Spark onboarding funnel**:

1. VFX artist hears about Tunet (open source, social media, word of mouth)
2. They visit the Spark Share page, see "Train a CopyCat model in 5 minutes"
3. Sign up for Spark account (free)
4. First job free or $1 credit to try
5. They're now a Spark user with an account, file cache, and billing set up
6. Natural upsell to other Spark services (render farm, storage, collaboration)

### Existing Content to Repurpose

The **After Effects user guide** (`docs/after_effects_guide.md`) from the simplified UI branch is already written in non-technical language targeting compositors. Key sections to adapt for Spark Share:

- **"What Can It Do?"** table — maps directly to preset picker descriptions
- **"The Basic Idea"** diagram (src/ + dst/ --> model --> output/) — becomes the web landing page explainer
- **"Step by Step"** walkthrough — becomes the web onboarding wizard, replacing "run locally" with "upload to Spark"
- **"Choosing the Right Preset"** section — becomes the preset picker's expandable help text
- **"Troubleshooting"** table — becomes the FAQ/help page (minus GPU-specific local issues)
- **"GPU and Performance"** section — replaced by pricing tiers ("we handle the GPU, you pick a tier")

---

## 10. Open Questions for Walt / Spark Team

1. **Node provisioning**: Can nodes auto-spin-up via API, or is there a Deadline integration we should use?
2. **ShareSync mounting**: What's the mount path / config for GPU nodes to access the file cache?
3. **Billing integration**: Is there an existing Spark billing API, or do we build our own and reconcile?
4. **Domain/branding**: spark.cpfx.io? tunet.spark.io? Something else?
5. **GPU inventory**: What GPU types are available in the Spark pool? RTX 4090, A100, others?
6. **SSL/auth**: Does Spark have an existing OAuth2 provider we plug into?
7. **Deadline integration**: Should jobs go through Deadline, or a custom lighter-weight queue?
8. **First free job**: Good idea for onboarding? What's the cap?
