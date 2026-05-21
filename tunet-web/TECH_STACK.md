# Spark Flint — Tech Stack & Architecture

## Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Framework | **Next.js 15** (App Router, TypeScript) | SSR + API routes in one deploy; Vercel-native |
| Auth | **Clerk** | Drop-in multi-user auth, session management, webhooks |
| Database | **Supabase** (Postgres) | Managed Postgres, Row-Level Security, Realtime, Storage |
| File Storage | **Supabase Storage** | Presigned upload URLs (client → bucket direct), signed download URLs |
| Payments | **Stripe** (Checkout + Webhooks) | One-time credit top-ups, no subscription complexity |
| Hosting | **Vercel** | Zero-ops, Cron jobs for billing tick, Edge middleware for auth |
| Styling | **Tailwind CSS v4** + CSS vars | Spark design tokens as CSS custom properties |
| Charts | **Recharts** | Lightweight, SSR-safe, good line chart support |
| GPU Infra | **RunPod** (GraphQL API) | Existing integration, proven |

---

## Architecture Diagram

```
Browser
  │
  ├─ GET/POST → Vercel (Next.js App Router)
  │               ├─ /app/*         — React pages (Clerk session)
  │               ├─ /api/pods/*    — RunPod GraphQL proxy
  │               ├─ /api/jobs/*    — Job CRUD + monitor proxy
  │               ├─ /api/billing/* — Stripe Checkout + webhook
  │               └─ /api/upload/*  — Supabase Storage presigned URLs
  │
  ├─ Direct upload → Supabase Storage (presigned PUT URL)
  │
  └─ Poll (every 10s) → /api/jobs/[id]/monitor
                              │
                              └─ Server proxies to:
                                 https://{podId}-8080.proxy.runpod.net/api/*
                                 (monitor_api.py running on pod)

Vercel Cron (*/5 * * * *)
  └─ /api/cron/billing-tick
       ├─ List all RUNNING jobs from DB
       ├─ For each job: charge credits = costPerHr × markup × elapsed_minutes / 60
       ├─ If balance < 15min credit: stop pod + notify user
       └─ Update job.accumulated_cost_cents

Stripe Webhook → /api/billing/webhook
  └─ payment_intent.succeeded → add credits to user balance
```

---

## Data Model (Supabase / Postgres)

### `users`
```sql
id            text PRIMARY KEY  -- Clerk user ID
email         text NOT NULL
name          text
credit_balance_cents  integer NOT NULL DEFAULT 0
is_admin      boolean NOT NULL DEFAULT false
created_at    timestamptz DEFAULT now()
```

### `jobs`
```sql
id                    uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id               text NOT NULL REFERENCES users(id)
name                  text NOT NULL
pod_id                text          -- RunPod pod ID (null until pod created)
status                text NOT NULL  -- pending|provisioning|running|stopped|failed|terminated
gpu_type_id           text NOT NULL  -- RunPod GPU type string
gpu_display_name      text
runpod_cost_per_hr    numeric(10,4)  -- actual RunPod rate (filled after pod created)
platform_cost_per_hr  numeric(10,4)  -- what we charge user
accumulated_cost_cents integer NOT NULL DEFAULT 0
billing_last_tick_at  timestamptz
config_path           text          -- Supabase Storage path
src_zip_path          text
dst_zip_path          text
checkpoint_path       text
container_disk_gb     integer NOT NULL DEFAULT 50
volume_gb             integer NOT NULL DEFAULT 100
started_at            timestamptz
ended_at              timestamptz
created_at            timestamptz DEFAULT now()
```

### `billing_events`
```sql
id                      uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id                 text NOT NULL REFERENCES users(id)
job_id                  uuid REFERENCES jobs(id)
type                    text NOT NULL  -- top_up|compute_charge|manual_adjustment|refund
amount_cents            integer NOT NULL  -- positive=credit added, negative=deducted
description             text
stripe_payment_intent   text
created_at              timestamptz DEFAULT now()
```

### `gpu_pricing`
```sql
gpu_type_id           text PRIMARY KEY  -- matches RunPod gpuTypeId
display_name          text NOT NULL
short_key             text NOT NULL
vram_gb               integer
platform_cost_per_hr  numeric(10,4) NOT NULL
runpod_cost_per_hr    numeric(10,4) NOT NULL
is_available          boolean NOT NULL DEFAULT true
tier                  text NOT NULL DEFAULT 'standard'  -- standard|recommended|pro|premium
sort_order            integer NOT NULL DEFAULT 99
```

---

## Pod Provisioning — No-SSH Architecture

### Problem
The existing `runpod_launch.py` uses SSH (rsync + scp) to upload code and config to the pod.
SSH is not usable from Vercel serverless functions (no persistent connection, 10s timeout on
hobby, 60s on Pro).

### Solution: `startScript` bootstrap
RunPod's `podFindAndDeployOnDemand` mutation accepts a `startScript` field (base64-encoded
bash). This runs before the container's default entrypoint and replaces the need for SSH upload.

The bootstrap script:
1. Downloads the tunet code tarball from Supabase Storage (admin-uploaded release bundle)
2. Downloads the user's `config.yaml` from a signed URL
3. Downloads `src.zip` / `dst.zip` / `checkpoint.pth` if provided
4. Patches the config to use downloaded paths
5. Calls the existing `runpod_start.sh` with the correct args

Signed URLs are generated with 24-hour TTL (plenty of time for the pod to bootstrap).

### Fallback
If RunPod's `startScript` field is unavailable, the `pending` job can be picked up by a local
worker process (`worker/job_worker.py`) that the operator runs on any machine with SSH access.
The worker polls the `jobs` table and executes the existing SSH-based flow. This is fully
transparent — the web platform works identically either way.

### Monitor Access
Once running, the pod exposes `monitor_api.py` on port 8080. RunPod's proxy network makes it
accessible at `https://{podId}-8080.proxy.runpod.net`. All monitor calls from the web UI
are proxied server-side through `/api/jobs/[id]/monitor?path=/api/status` to avoid CORS and
to enforce job ownership (users can only poll their own jobs).

---

## Billing Tick (Vercel Cron)

Runs every 5 minutes via `vercel.json` cron config.

```
For each RUNNING job:
  minutes_since_last_tick = (now - billing_last_tick_at) / 60
  charge_cents = ROUND(platform_cost_per_hr / 60 * minutes_since_last_tick * 100)
  
  UPDATE users SET credit_balance_cents -= charge_cents WHERE id = job.user_id
  INSERT INTO billing_events (type=compute_charge, amount=-charge_cents)
  UPDATE jobs SET accumulated_cost_cents += charge_cents, billing_last_tick_at = now()
  
  IF user.credit_balance_cents < (platform_cost_per_hr / 4 * 100):  -- < 15 min remaining
    podStop(job.pod_id)
    UPDATE jobs SET status = stopped
    send_low_balance_email(user)
```

The cron route is protected with a `CRON_SECRET` header that Vercel injects automatically.

---

## File Upload Flow

```
1. Browser → POST /api/upload/presign
   { fileName, fileType, jobId, role: 'config'|'src'|'dst'|'checkpoint' }
   
2. API → Supabase Storage.createSignedUploadUrl(path)
   Returns: { signedUrl, token, path }
   
3. Browser → PUT signedUrl  (direct to Supabase, bypasses Vercel entirely)
   Max sizes: config 1 MB, checkpoint 10 GB, src/dst ZIPs 10 GB each
   
4. Browser → POST /api/jobs  (with storage paths, triggers pod launch)
```

---

## Security

| Concern | Mitigation |
|---------|-----------|
| RunPod API key exposure | Stored in Vercel env var, never sent to browser |
| Supabase service role key | Server-only, never in client bundle |
| User data isolation | Supabase Storage paths scoped to `{userId}/{jobId}/...`; Row-Level Security on all tables |
| Monitor proxy | Verifies job ownership before proxying to RunPod proxy URL |
| Billing cron | Protected by `CRON_SECRET` Vercel env var |
| Stripe webhooks | Verified with `stripe.webhooks.constructEvent` using webhook secret |
| Admin routes | `is_admin` check in middleware before any `/admin/*` route |

---

## Audit — Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| RunPod `startScript` not in API | High | Fallback worker (`worker/job_worker.py`) using existing SSH flow |
| Large data upload (10 GB ZIPs) | Medium | Supabase Storage 50 GB free tier; direct browser upload bypasses Vercel limits |
| Vercel cron misfire → user overcharged | Medium | Idempotency via `billing_last_tick_at`; cron only charges elapsed since last tick |
| Pod terminates without billing update | Medium | Cron reconciles against RunPod pod list; dead pods get final charge + terminated status |
| Monitor proxy slow (RunPod proxy latency) | Low | 10s client poll, server-side proxy with 8s timeout; graceful "connecting..." state |
| Stripe webhook replay | Low | `stripe.webhooks.constructEvent` validates signature + timestamp |
| User spends more than balance | Low | Pre-launch check: balance must cover 1 hour; cron stops pod at 15-min threshold |
| Supabase Storage signed URL expires before pod downloads | Low | URLs are 24-hour TTL; pod bootstrap completes in < 30 minutes |

---

## Environment Variables

```bash
# Clerk
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
CLERK_WEBHOOK_SECRET=

# Supabase
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# RunPod
RUNPOD_API_KEY=

# Stripe
STRIPE_SECRET_KEY=
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=
STRIPE_WEBHOOK_SECRET=

# App
NEXT_PUBLIC_APP_URL=https://your-domain.vercel.app
CRON_SECRET=                    # Vercel injects this automatically for cron routes

# Storage
TUNET_CODE_BUNDLE_PATH=         # Supabase Storage path of tunet release tarball
                                # e.g. admin/tunet-latest.tar.gz
```

---

## Deployment Checklist

- [ ] Supabase project created, schema migrated, RLS policies applied
- [ ] Clerk app created, webhook endpoint configured (`/api/webhooks/clerk`)
- [ ] Stripe account, products created (credit packs), webhook configured
- [ ] RunPod account, API key in Vercel env vars
- [ ] Admin uploads tunet code tarball to Supabase Storage (`admin/tunet-latest.tar.gz`)
- [ ] `vercel.json` cron set to `*/5 * * * *`
- [ ] `CRON_SECRET` env var set in Vercel dashboard
- [ ] Test full flow: sign up → top up → launch job → monitor → terminate
