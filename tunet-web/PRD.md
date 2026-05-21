# Spark Flint — Product Requirements Document

## Overview

Spark Flint is a multi-user SaaS platform for launching and monitoring TuNet neural network
training jobs on RunPod GPU infrastructure. The platform operator owns one RunPod account,
bills users via Stripe credits, and marks up GPU compute to cover costs and generate margin.

---

## Goals

| Goal | Description |
|------|-------------|
| **Managed compute** | Operator runs one RunPod account. Users never touch RunPod directly. |
| **Self-service** | Users sign up, top up credits, launch jobs, and monitor training — no operator intervention required. |
| **Billing transparency** | Users see exactly what they're spending, per job and in aggregate. |
| **Spark aesthetic** | UI matches the Spark Cloud Studio design system (purple accent, clean SaaS). |

---

## User Roles

### End User
- Signs up with email/social
- Tops up credit balance via Stripe
- Uploads config YAML + optional checkpoint
- Uploads training data (src/dst as ZIP)
- Launches training jobs
- Monitors live training (loss curves, logs, preview images)
- Stops / terminates jobs
- Downloads trained models

### Admin (Operator)
- Views all users and jobs
- Views real RunPod costs vs platform revenue (margin report)
- Manually adjusts user credit balances
- Configures GPU pricing and markup
- Manages the tunet code bundle (upload new release)

---

## Core Features

### 1. Authentication
- Email + password sign-up/in via Clerk
- Google OAuth
- Email verification
- Password reset

### 2. Dashboard
- Summary cards: credit balance, total spend this month, active jobs, GPU hours used
- Active jobs list with live status badges
- Quick-launch button
- Recent job history (last 10)

### 3. Launch Job (`/jobs/new`)
- **Job name** (auto-generated from config, editable)
- **GPU selector** — card UI showing each GPU with VRAM, price/hr, availability tier
  - L40S (recommended), RTX 4090, A40, A100 80GB, A100 SXM4, RTX PRO 6000 Blackwell
- **Config upload** — drag-and-drop YAML, inline preview, validation
- **Training data** — upload src.zip and dst.zip (or skip if paths already in config)
- **Checkpoint** — upload .pth to resume from, or start fresh
- **Storage** — container disk (10–200 GB), volume (10–500 GB)
- **Credit estimate** — shows estimated cost for 1h / 4h / 8h
- Launch button — disabled if insufficient credits

### 4. Job Detail (`/jobs/[id]`)
- Status badge + elapsed time + current cost
- **Live loss chart** — L1 and LPIPS, train + val (via RunPod proxy to monitor_api)
- **Training preview images** — polled from monitor_api every 30s
- **Log viewer** — last 200 lines, auto-scroll, monospace
- **Actions**: Stop (keep pod alive), Terminate (end billing), Download model
- Auto-polls monitor_api every 10s while job is RUNNING

### 5. Jobs List (`/jobs`)
- Table: name, GPU, status, duration, cost, actions
- Filter by status (running / completed / terminated / failed)
- Pagination

### 6. Billing (`/billing`)
- Current credit balance (large, prominent)
- Top-up via Stripe Checkout: preset amounts ($10, $25, $50, $100) + custom
- Usage history: table of charges (per job, per hour) and top-ups
- Monthly spend chart

### 7. Settings (`/settings`)
- Display name
- Notification preferences (low balance warning email)
- Danger zone: delete account

### 8. Admin Panel (`/admin`)
- User list with balance and total spend
- All jobs across all users
- Margin report: RunPod cost vs platform revenue
- Credit adjustments (add/remove credits from any user)
- GPU pricing editor
- TuNet code bundle manager (upload new `.tar.gz` release)

---

## Billing Model

### Credits
- 1 credit = $0.01 USD (cents-based, stored as integers)
- Users buy credits in packs: $10 (1000cr), $25 (2500cr), $50 (5000cr), $100 (10000cr)
- No credits expire
- Credits deducted in real-time (every 5 minutes via Vercel Cron)
- Low-balance warning email at $5 remaining

### GPU Pricing (platform rate, user-facing)

| GPU | Platform $/hr | RunPod $/hr | Margin |
|-----|--------------|-------------|--------|
| RTX 4090 | $0.99 | ~$0.74 | ~34% |
| A40 | $1.05 | ~$0.79 | ~33% |
| L40S | $1.49 | ~$1.14 | ~31% |
| RTX PRO 6000 Blackwell | $2.59 | ~$1.99 | ~30% |
| A100 80GB PCIe | $3.89 | ~$2.99 | ~30% |
| A100 SXM4 80GB | $4.49 | ~$3.49 | ~29% |

### Auto-stop on low balance
- If user's credits drop below the equivalent of 15 minutes of compute, job is auto-stopped
- User is emailed with a prompt to top up and resume

---

## Pod Lifecycle

```
[User clicks Launch]
        │
        ▼
[API creates job record — status: pending]
        │
        ▼
[API uploads files to Supabase Storage → signed URLs]
        │
        ▼
[API calls RunPod podFindAndDeployOnDemand
  with startScript containing bootstrap.sh
  (downloads tunet code + config + data from URLs)]
        │
        ▼
[Pod status: provisioning → RUNNING]
        │
        ▼
[bootstrap.sh runs: downloads files, patches config, runs runpod_start.sh]
        │
        ▼
[monitor_api.py starts on :8080 (RunPod proxy network)]
        │
        ▼
[Web polls https://{podId}-8080.proxy.runpod.net/api/status every 10s]
        │
        ▼
[Vercel Cron (every 5min) charges credits based on costPerHr × markup]
        │
        ▼
[User stops or credits run out → podStop or podTerminate]
```

---

## Non-Goals (v1)

- Multi-GPU pods (always 1 GPU per job)
- Team/org accounts (single user per account)
- Scheduled/queued jobs
- Model versioning / model registry
- Custom Docker images per user
- SSH access for end users
- White-labeling

---

## Success Metrics

- Time from sign-up to first launched job: < 5 minutes
- Credit top-up to pod running: < 2 minutes
- Job detail page refresh latency: < 15 seconds
- Zero jobs that continue billing after user stops them
