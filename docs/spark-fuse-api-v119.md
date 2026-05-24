# Spark Fuse API — customer guide (v1.19)

**Audience:** developers integrating with the Spark Fuse REST API for
the first time. This guide walks through getting an auth token,
submitting your first compute job, uploading input data, streaming
logs, polling status, cancelling, listing past jobs, designing
resumable jobs with checkpoints, and pulling outputs back from
ShareSync.

---

## 0. The 30-second mental model

A Spark Fuse job is **a Docker image + a command, run on a Spark Fuse
GPU you don't have to manage**.

You submit a job (`POST /api/compute/jobs`) with the image, the command,
and the instance type SKU you want. Spark Fuse then:

1. Picks compute, starts it, and pulls your image.
2. Runs your command inside the container with `/output/` mounted to a
   ShareSync-backed directory and (optionally) `/input/` mounted
   read-only from a folder you uploaded or pre-populated.
3. Streams logs back to you while it runs.
4. Persists everything your container wrote to `/output/` on ShareSync.
5. Releases the compute when it's done (with an optional idle-hold
   window if you want another job to land on the same warm compute
   with no cold-start).

Auth is a Spark bearer token. You get one from `POST /api/auth/login`
once and reuse it for every API call until it expires. The same token
also authenticates against ShareSync (the WebDAV server hosting your
inputs and outputs) — no token-exchange dance needed.

Base URL: `https://api.prod.aapse1.sparkcloud.studio` (production). The
rest of this guide assumes you've exported it:

```bash
export SPARK_HOST="https://api.prod.aapse1.sparkcloud.studio"
```

---

## 0.1 Compute modes: InstantCompute (`mode='instant'`) and SmartCompute (`mode='smart'`)

Spark Fuse has two compute modes you pick between on each job submit:

- **InstantCompute (`mode='instant'`, default).** The job lands on a
  warm GPU within a couple of seconds — or, if no warm compute is
  available for your SKU, ~3 min while Spark Fuse brings one up. Once
  running, the job never gets interrupted mid-flight. `idleHoldSeconds`
  and next-job-on-same-warm-compute affinity
  ([§12.3](#123-idle-hold--warm-pool)) are InstantCompute-only.
- **SmartCompute (`mode='smart'`).** The job runs on preemptible
  compute that's typically ~60% cheaper, but Spark Fuse may reclaim
  it mid-job with a short warning. Every submit is a fresh cold-start
  — no warm pool. See [§14](#14-smartcompute-mode-smart) for the deep
  dive (retry budget, interruption webhook, trade-offs). SmartCompute
  capacity is currently limited for some GPU SKUs at peak times — see
  the callout at the top of §14.

The rest of this guide assumes InstantCompute unless a section flags
SmartCompute specifically.

---

## 1. Authentication: getting a bearer token

Spark issues bearer tokens through `POST /api/auth/login`. You
authenticate with your Spark email + password and get back a JSON
response carrying the token to send on every subsequent API call.
**Do not call any underlying identity-provider endpoints directly** —
`/api/auth/login` is the only supported auth surface and the only one
that survives provider migrations.

### 1.1 Get a token

`POST /api/auth/login`

Request body (`application/json`):

```json
{
  "email": "you@yourcompany.example",
  "password": "..."
}
```

```bash
export SPARK_TOKEN=$(curl -sX POST "$SPARK_HOST/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$YOUR_SPARK_EMAIL\",
    \"password\": \"$YOUR_SPARK_PASSWORD\"
  }" | jq -r '.token')

echo "Got token? ${SPARK_TOKEN:0:20}..."
```

Successful response (HTTP 200):

```json
{
  "resp": "Login Successful",
  "success": true,
  "token": "eyJhbGciOiJSUzI1NiIsInR5cCIgOi...",
  "password_expired": false,
  "password_expires_in_days": 45,
  "requires_password_change": false
}
```


| Field                      | Type          | Meaning                                                                                                                                                                                                                    |
| -------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resp`                     | string        | Human-readable status message. `"Login Successful"` on success; on failure, a short reason ("Invalid email or password ..."). Safe to surface to end users.                                                                |
| `success`                  | boolean       | `true` if the credentials authenticated and the account is in good standing. **Always branch on `success === true`** before reading `token` — failed logins still return HTTP 200 with `success: false` and `token: null`. |
| `token`                    | string | null | The bearer token to put in `Authorization: Bearer <token>` on every subsequent call. `null` when `success` is `false`.                                                                                                     |
| `password_expired`         | boolean       | `true` if the user's password has hit its expiry deadline. The token is still issued, but `POST /api/auth/reset-password` should be the user's next step.                                                                  |
| `password_expires_in_days` | number | null | Days until the password expires (`null` if your tenant has password expiry disabled). Useful for a "your password expires in N days" UI nudge.                                                                             |
| `requires_password_change` | boolean       | `true` if the user must change their password before doing meaningful work (typically set after an admin-driven reset).                                                                                                    |


Failure cases — HTTP 200 with `success: false`:

```json
{
  "resp": "Invalid email or password. Please check your credentials and try again.",
  "success": false,
  "token": null
}
```

Other `resp` values you may see (verbatim from the server): account not
yet email-verified, account disabled by admin, generic upstream failure
(`"An unexpected error occurred. Please try again later."`). In all
cases, branch on `success === true` first and then surface `resp` to
the user if `false`.

### 1.2 Use the token

Every Spark Compute API call expects the token as an `Authorization: Bearer`
header:

```bash
curl -s "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" | jq '.'
```

The same token authenticates against ShareSync (the WebDAV server holding
your job inputs + outputs). No second login dance.

### 1.3 Token lifetime + refresh

Bearer tokens are short-lived (typically minutes). When yours expires
you'll get an HTTP 401 back from a `/api/compute/...` call; re-call
`POST /api/auth/login` with the same credentials to mint a new one.

There is **no separate refresh endpoint** — the login call is
cheap and is the canonical way to refresh. For long-running scripts
that submit many jobs, wrap your API calls in a retry-on-401 that
re-runs the login snippet above (sourcing credentials from your
secrets manager rather than env vars in production).

Treat any stored `SPARK_TOKEN` as transient. Do not persist tokens to
disk or commit them to source control.

### 1.4 Errors you might see


| HTTP | Endpoint           | Body             | Meaning                         | Likely cause                                                                                                    |
| ---- | ------------------ | ---------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 200  | `/api/auth/login`  | `success: false` | Login failed                    | Wrong email/password, account not email-verified, account disabled, or upstream identity failure. Check `resp`. |
| 401  | `/api/compute/...` | —                | Token expired or invalid        | Re-call `POST /api/auth/login` and retry the request.                                                           |
| 403  | `/api/compute/...` | —                | Token valid, wrong organisation | The job / resource you're touching belongs to a different org.                                                  |


---

## 2. Submitting a compute job

`POST /api/compute/jobs`

The minimum viable submission needs an image, a command, and an instance
type. Everything else has a sensible default.

### 2.1 Minimal example

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
    "command": ["bash", "-c", "nvidia-smi && echo hello-from-spark"],
    "instanceType": "g7e.2xlarge"
  }' | jq '.'
```

Response:

```json
{
  "jobId": "16fb6d80-0ac3-4d42-9c5d-80c02a26e5ae",
  "status": "queued",
  "imageDigest": "sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2",
  "outputShareSyncPath": "/Spark Fuse Jobs/16fb6d80-0ac3-4d42-9c5d-80c02a26e5ae/",
  "createdAt": "2026-05-05T09:12:32.338757+00:00",
  "output": {
    "shareSyncPath": "/Spark Fuse Jobs/16fb6d80-0ac3-4d42-9c5d-80c02a26e5ae/",
    "shareSyncSpaceName": null,
    "shareSyncBaseUrl": "https://your-org.files.sparkcloud.studio/dav/spaces/<space-id>/Spark%20Fuse%20Jobs/16fb6d80-.../"
  },
  "input": null
}
```

The `output.shareSyncBaseUrl` is where anything your container writes to
`/output/` ends up. See [§9](#9-retrieving-outputs-from-sharesync) for how
to read those files back.

### 2.2 Submit DTO — full field list


| Field                      | Type     | Required | Description                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `image`                    | string   | yes      | Container image reference (with optional tag or digest). Any **public** `linux/amd64` image works, including images you build and push yourself — see [§2.5](#25-using-your-own-docker-image). Pinning by digest is recommended for reproducibility. Private-registry auth is not yet available but is coming soon.                                                                        |
| `command`                  | string[] | yes      | The command to run inside the container. First element is the binary; rest are arguments.                                                                                                                                                                                                                                                                                                                                      |
| `instanceType`             | string   | yes      | Instance type SKU (e.g. `g7e.2xlarge`). Must be in the current allow-list — see [§8](#8-browsing-eligible-instance-types).                                                                                                                                                                                                                                                                                                     |
| `env`                      | object   | no       | Customer environment variables. `SPARK_*` keys are reserved by Spark Fuse and rejected.                                                                                                                                                                                                                                                                                                                                       |
| `startScriptB64`           | string   | no       | Base64-encoded bash script that runs after image pull and before `command`. Useful for cache-warming, light setup.                                                                                                                                                                                                                                                                                                             |
| `mode`                     | string   | no       | `instant` (warm-pool guaranteed-availability, the default) or `smart` (preemptible — opt-in, typically ~60% cheaper than `instant` but subject to reclamation; see [§14](#14-smartcompute-mode-smart)).                                                                                                                                                                                                               |
| `maxRetriesOnInterrupt`    | number   | no       | (`mode='smart'` only) How many times Spark Fuse may re-launch this job on fresh smart-mode compute if it reclaims the underlying compute mid-run. Default `1`; range `[0, 5]`. `0` = first interruption is terminal. Each retry consumes one tick from the budget; when the budget is exhausted, the job lands terminal `failed` with `errorCode='smartcompute_interrupted_no_retries_left'`. Ignored on `mode='instant'`. |
| `outputShareSyncPath`      | string   | no       | ShareSync logical path that outputs are written to. Defaults to `/Spark Fuse Jobs/{jobId}/`.                                                                                                                                                                                                                                                                                                                                   |
| `outputShareSyncSpaceName` | string   | no       | Name of a ShareSync Space (project) to write outputs into. Omit to use your Personal space. Spark Fuse validates the space exists at submission time and rejects with HTTP 400 if it doesn't.                                                                                                                                                                                                                                  |
| `inputShareSyncPath`       | string   | no       | Mount workflow: a ShareSync path you've already populated with input files. Spark Fuse PROPFINDs the path at submit time and rejects HTTP 400 if unreachable. Mutually exclusive with `inputPushMode`. See [§3.2](#32-mount-workflow-pre-populated-input-folder).                                                                                                                                                               |
| `inputShareSyncSpaceName`  | string   | no       | Name of the ShareSync Space containing your input path (or, with `inputPushMode='auto-prepare'`, the space Spark Fuse will allocate the input folder under). Omit to use your Personal space.                                                                                                                                                                                                                                  |
| `inputPushMode`            | string   | no       | `auto-prepare` triggers the push workflow: Spark Fuse allocates an input folder, returns a one-shot upload curl, and pulls + extracts the upload before running your command. See [§3.1](#31-push-workflow-auto-prepare).                                                                                                                                                                                                      |
| `webhookEndpointId`        | string   | no       | UUID of a webhook endpoint to fire job-state events to. Manage endpoints via `/api/compute/webhooks/*` (separate doc).                                                                                                                                                                                                                                                                                                         |
| `idleHoldSeconds`          | number   | no       | (`mode='instant'` only) After the job reaches a terminal state (succeeded / failed), how long the compute should idle before being released. During the hold, your org's next job lands on this same warm compute with no cold-start image pull and no provisioning latency. Defaults to a platform-tuned value (~600 s today); range `[0, max]`. **Cancel skips the hold** — cancelling releases the compute immediately.    |
| `shmSize`                  | string   | no       | Docker `--shm-size` override for the container's `/dev/shm` (POSIX shared memory tmpfs). Docker's built-in default is 64 MB, which crashes PyTorch `DataLoader` workers (`num_workers > 0`), HuggingFace Transformers training, and anything else using `multiprocessing.shared_memory` with `Bus error`. Spark Fuse defaults to **`2g` (2048 MiB)** — same as RunPod / Modal / Replicate / SageMaker — which is enough for typical batch sizes. Override when you need more (or less): format is a Docker size string matching `/^\d+[kmg]?$/` (lowercase suffix); examples: `'512m'`, `'2g'`, `'4g'`, `'16g'`. Maximum `'32g'`. Resolved value is echoed back on the submit response and on `GET /api/compute/jobs/{jobId}` as `shmSize` (with a snake_case alias `container_shm_size`). |
| `instanceHandle`           | string   | no       | (`mode='instant'` only) Handle returned by `POST /api/compute/instances/prepare` — routes this submission to a specific pre-warmed instance instead of going through the regular warm-pool path. When supplied, `instanceType` is snapped from the session and must match if also supplied. See [§13](#13-persistent-compute--sessions-mode-instant-only). Rejected on `mode='smart'`.                                                                                                                            |
| `tags`                     | string[] | no       | Customer-chosen opaque string tags. Up to 32 tags per job; each 1-64 chars matching `^[a-z0-9_\-:\.]+$`. Duplicates collapsed at submission. Echoed verbatim on every job response and on every `compute.job.*` webhook payload. Used to filter the list endpoint (`?tag=`, `?tagsAny=`) and to group billing line items. See [§2.3](#23-tagging-jobs). |
| `notifyOnFailure`          | boolean  | no       | Per-job override for the failure email — see [§14.7](#147-interruption-no-retries-left-email-notifications). Tri-state: omit (or pass `null`) to use the default for your org; pass `true` to force-send so this job emails even if your org's default is off; pass `false` to force-suppress so this job doesn't email even if your org's default is on. Echoed verbatim on every job response. **Current scope:** the only failure that triggers an email today is the smart-mode `smartcompute_interrupted_no_retries_left` terminal — in-container failures (nonzero container exit, image-pull errors, etc.) do not email regardless of this field. |
| `maxWallClockSeconds`        | number   | no       | **Opt-in only.** Hard ceiling on container wall-clock runtime, in seconds. Omit (the default) and Spark Fuse will **never** terminate your job on time-elapsed alone — your container runs until it exits, you cancel it, or the inactivity detector fires (see `containerInactivitySeconds` below). Supply an integer to opt in: when the container has been running for this many seconds, Spark Fuse SIGKILLs it and the job lands terminal `failed` with `errorCode='wallclock_exceeded'`. Range `[60, 86400]` (1 min to 24 h). Distinct from `containerInactivitySeconds`, which fires when your container is making no progress regardless of how long it has been running. Echoed back on every job response as `maxWallClockSeconds` (snake_case alias `max_wallclock_seconds`). |
| `containerInactivitySeconds` | number   | no       | Per-job override for the container-inactivity detector. Inactivity = **no** stdout/stderr lines emitted **AND** **no** CPU activity **AND** **no** GPU activity for the full window. When all three signals are quiet, Spark Fuse SIGTERMs your container (30 s grace, then SIGKILL) and the job lands terminal `failed` with `errorCode='container_inactive'`. The three-signal AND is intentional — training loops that don't log but keep the GPU busy do **not** trigger the detector. **Tri-state**: omit (the default) to use the Spark Fuse default (currently **1800 s = 30 min**); pass `0` to **disable** the detector entirely for this job (use sparingly — for legitimately long-quiet batch pipelines); pass a positive integer to override the default with your own threshold. Range `[60, 86400]` for positive values, or exactly `0`. Echoed back on every job response as `containerInactivitySeconds` (snake_case alias `container_inactivity_seconds`). |


### 2.3 Tagging jobs

Optional. Attach arbitrary string tags to a job at submission time. Tags
are echoed back on every job response, included in every
`compute.job.*` webhook payload, propagated onto every billing line
item the job produces, and support filtering via
`GET /api/compute/jobs?tag=...`.

Use cases:

- Track jobs by workflow, experiment, or external application.
- Group your jobs in your own dashboards (filter the list endpoint by
  tag).
- Receive workflow-branded failure emails or apply different retry
  policies for specific tags (where Spark has configured your account
  for it).
- Split your invoice / usage analytics by tag without parsing job
  descriptions.

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{
        "image": "docker.io/ericstudio/blender:4.2-cuda12.8",
        "command": ["blender", "-b", "/input/scene.blend", "-o", "/output/frame_####.png", "-f", "1"],
        "instanceType": "g7e.2xlarge",
        "tags": ["tunet", "tunet:training", "experiment_id:e3094"]
      }' | jq '.tags'
# -> ["tunet", "tunet:training", "experiment_id:e3094"]
```

Rules:

- Up to **32 tags** per job.
- Each tag is **1-64 characters**, matching `^[a-z0-9_\-:\.]+$` —
  lowercase letters, digits, underscore, hyphen, colon, period.
- Duplicates within a single submission are collapsed.
- The platform does not interpret tag semantics by default — they're
  opaque to Spark Fuse unless your org has been specifically configured
  (e.g. revenue-share or custom email branding by Spark).

#### Filtering the list endpoint

`GET /api/compute/jobs` accepts two mutually-exclusive query params:

- `?tag=foo` — repeatable, **AND** semantics. Row must carry **every**
  tag listed. Example:
  `GET /api/compute/jobs?tag=team_alpha&tag=training`.
- `?tagsAny=foo,bar` — comma-separated, **OR** semantics. Row must
  carry **at least one** of the tags. Example:
  `GET /api/compute/jobs?tagsAny=team_alpha,team_beta`.

Passing both returns HTTP 400 with `error_code='invalid_tags'`. Each
filter value is validated against the same regex/length rules as
submission tags.

#### Webhook payloads

Every `compute.job.*` event payload carries a top-level `tags` field:

```json
{
  "event": "compute.job.failed",
  "jobId": "09da25d2-0a45-4cbb-bc24-717a0e7eb300",
  "tags": ["team_alpha", "experiment_id:e3094"],
  "status": "failed",
  "errorCode": "image_pull_failed",
  "...": "rest of existing payload"
}
```

Receivers can branch on tag without a follow-up `GET /jobs/:id`.


### 2.4 Common errors


| HTTP                                                              | Meaning                                                                                                                                                                   |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 400 `instanceType ... is not eligible`                            | The SKU isn't in the current allow-list — call `GET /api/compute/skus` for the live list.                                                                                 |
| 400 `Output ShareSync space ... not found`                        | The space name you supplied doesn't exist or you don't have write access.                                                                                                 |
| 400 `Input ShareSync path ... not reachable`                      | (mount workflow) The path you supplied couldn't be PROPFINDed with your token. Check spelling + permissions.                                                              |
| 400 `inputShareSyncPath and inputPushMode are mutually exclusive` | Pick one.                                                                                                                                                                 |
| 400 `error_code='invalid_shm_size'`                               | `shmSize` didn't match `/^\d+[kmg]?$/` (e.g. `"big"`, `"4G"` with uppercase, missing unit number) or exceeded the 32g platform ceiling. Omit the field to use the 2g default, or correct the value.   |
| 400 `error_code='invalid_tags'`                                   | One or more `tags` failed validation: not an array of strings, more than 32 entries, individual tag empty / longer than 64 chars, or containing characters outside `[a-z0-9_\-:\.]`. Also raised on the list endpoint when `?tag` and `?tagsAny` are both supplied (mutually exclusive). See [§2.3](#23-tagging-jobs).                                                                |
| 400 `error_code='invalid_max_wallclock_seconds'`                  | `maxWallClockSeconds` was supplied but isn't an integer in `[60, 86400]`. Omit the field entirely to disable the wall-clock kill (the default), or correct the value.                                                                                                                                                                                                                                                                                                |
| 400 `error_code='invalid_container_inactivity_seconds'`           | `containerInactivitySeconds` was supplied but isn't an integer that is either exactly `0` (disable the detector) or in `[60, 86400]` (override the default). Omit the field entirely to use the Spark Fuse default.                                                                                                                                                                                                                                            |
| 403                                                               | Token invalid or for a different org.                                                                                                                                     |
| 503                                                               | All warm members are busy. Retry after a few seconds. (Rare.) |


### 2.5 Using your own Docker image

Spark Fuse can run any workload that fits the [container contract](#127-spark-fuse-is-for-linux-containers-only)
as **`image` + `command`** — including images you build from your
own code, third-party stacks (ComfyUI, Blender, Houdini, LoRAs, Wan, 
Qwen, custom training pipelines), or public Hub images you already use elsewhere.

#### Build and push

1. **Build for `linux/amd64`.** Spark Fuse hosts run Linux on x86_64.
   The platform validates your image reference against the registry
   manifest at submit time and rejects images that lack a
   **`linux/amd64`** entry (see [§12.7](#127-spark-fuse-is-for-linux-containers-only)).
   When building on Apple Silicon or another arm64 machine, pass an
   explicit platform:

   ```bash
   docker build --platform=linux/amd64 -t yourorg/my-app:v1 .
   docker push yourorg/my-app:v1
   ```

2. **Push to a registry Spark can pull without credentials.** v1 supports
   **public** images on Docker Hub, GHCR, and other registries that allow
   anonymous `docker pull`. Private-registry auth is not wired through job
   submission yet — if submit returns an image auth error, make the repo
   public for now or ask Spark about private-registry access for your org.

3. **Pin by digest when you care about reproducibility** (optional but
   recommended):

   ```json
   "image": "yourorg/my-app@sha256:abc123..."
   ```

#### Submit the job

Minimum fields are the same as any other job — `image`, `command`, and
`instanceType`:

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "yourorg/my-app:v1",
    "command": ["python3", "/app/run.py", "--out-dir", "/output/"],
    "instanceType": "g7e.2xlarge"
  }' | jq '.'
```

Inside the container:

- Write results to **`/output/`** (or `$SPARK_OUTPUT_DIR`) — that
  tree is persisted to ShareSync.
- Read inputs from **`/input/`** (or `$SPARK_INPUT_DIR`) when you used
  the [mount](#32-mount-workflow-pre-populated-input-folder) or
  [push](#31-push-workflow-auto-prepare) input workflows in [§3](#3-uploading-input-data).
- If your app needs more POSIX shared memory than the default, set
  `shmSize` (see the submit DTO table in [§2.2](#22-submit-dto--full-field-list)).

#### Three ways to supply code and assets

| Approach | When to use it |
| -------- | -------------- |
| **Bake into the image** | Dependencies, models, and entrypoints are stable; fastest cold start on warm nodes that already pulled your tag. |
| **`/input/` via ShareSync** | Change scripts, configs, or small assets every run without rebuilding (same pattern as `python3 /input/train.py`). |
| **`startScriptB64`** | Short host-side bash after pull, before `command` — cache warming, `git clone`, light setup. Runs on the compute host, not inside the container. |

You can combine them (e.g. fat image + per-run config in `/input/`).

#### What to expect

- **GPU:** Containers run with `--gpus all`. Your image must include a
  CUDA stack compatible with the instance GPU driver, or use an NVIDIA/CUDA
  base image and install your runtime on top.
- **Batch / headless:** Spark Fuse does not publish container ports. Design
  for CLI, API, or file-based I/O to `/output/`, not for browsing a web UI
  on a forwarded port.
- **Image size:** Large layers can take several minutes to pull; very large
  images on busy warm nodes can fail with `image_pull_failed` or `disk_full`
  if the node root volume is tight — pin a slim base image when you can.

Public third-party images work the same way — only the `image` and
`command` strings change:

```json
{
  "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
  "command": ["bash", "-c", "python3 /input/train.py"],
  "instanceType": "g7e.2xlarge",
  "inputPushMode": "auto-prepare"
}
```


### 2.8 Cost estimation (pre-submit)

`POST /api/compute/jobs/estimate` returns a cost quote for a given SKU
without actually submitting a job. Useful for pre-flight UI ("this run
will cost about $X").

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs/estimate" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{
        "instanceType": "g7e.2xlarge",
        "estimatedRuntimeSeconds": 3600,
        "idleHoldSeconds": 600
      }' | jq '.'
```

Response (abbreviated):

```json
{
  "instanceType": "g7e.2xlarge",
  "mode": "instant",
  "rate": {
    "billedPerSecondCents": "0.03305555",
    "billedPerHourUsd": "1.19"
  },
  "estimate": {
    "billableSeconds": 4200,
    "totalCents": "138.83",
    "totalUsd": "1.39"
  },
  "notes": [
    "Quotes are estimates — actual billed cost is computed from (started_running_at, terminal_at, idle_hold_seconds) at job-completion time and may differ from this projection by the gap between estimatedRuntimeSeconds and actual runtime."
  ]
}
```

Smart-mode (`mode: "smart"`) responses carry the same `rate` /
`estimate` shape but `notes[]` includes additional smart-mode-specific
entries — the worst-case-rate caveat, an idle-hold-not-applicable note,
and a per-attempt billing note (retries count separately).

Body fields:


| Field                     | Type   | Required | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `instanceType`            | string | yes      | Same allow-list as `POST /api/compute/jobs`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `mode`                    | string | no       | `instant` (default) or `smart`. Both modes quote against the worst-case hourly rate currently observed across every compute location the SKU is available in — Spark Fuse picks the actual landing location at launch time, so quoting the high end is honest. Real-world billed cost is typically lower. `rate.billedPerHourUsd` is the final customer rate for the requested mode — read it as-is. If smart-mode pricing data isn't yet observed for the SKU (rare; happens for a brief window after a new SKU is added to the catalog), smart-mode falls back to a discount-against-instant approximation.    |
| `estimatedRuntimeSeconds` | number | no       | Optional. When omitted, response carries rate-only output (no `estimate` block).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `idleHoldSeconds`         | number | no       | Optional. Used together with `estimatedRuntimeSeconds`; total = (runtime + hold) × rate.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |


Common errors:


| HTTP                                   | Meaning                                                                |
| -------------------------------------- | ---------------------------------------------------------------------- |
| 400 `instanceType ... is not eligible` | Same allow-list gate as job submission.                                |
| 404                                    | No pricing rows for that SKU in any compute location — data gap; please report. |


The quote is an *estimate*, not a quote-lock. Actual billed cost is
computed at job-completion time from the real (`started_running_at`,
`terminal_at`, `idle_hold_seconds`) tuple, so a job that finishes faster
than `estimatedRuntimeSeconds` is billed for less.

---

## 3. Uploading input data

Spark Compute mounts `/input/` (read-only) inside your container if — and
only if — you've supplied input data. There are two workflows.

### 3.1 Push workflow (`auto-prepare`)

You don't have any input files in ShareSync yet; you want the server to
allocate a destination and give you a one-shot upload command.

Submit with `inputPushMode='auto-prepare'`:

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
    "command": ["bash", "-c", "ls -la /input/ && python3 /input/run.py"],
    "instanceType": "g7e.2xlarge",
    "inputPushMode": "auto-prepare"
  }' | jq '.'
```

The response now contains an `input` block:

```json
{
  "jobId": "...",
  "status": "queued",
  "input": {
    "shareSyncPath": "/Spark Fuse Job Inputs/<jobId>/",
    "shareSyncSpaceName": null,
    "shareSyncBaseUrl": "https://your-org.files.sparkcloud.studio/dav/spaces/<space-id>/Spark%20Fuse%20Job%20Inputs/<jobId>/",
    "uploadUrl": "https://your-org.files.sparkcloud.studio/dav/spaces/<space-id>/Spark%20Fuse%20Job%20Inputs/<jobId>/spark-input.tar.gz",
    "uploadMethod": "PUT",
    "exampleCurl": "tar czf - . | curl -X PUT --data-binary @- -H \"Authorization: Bearer $SPARK_TOKEN\" 'https://your-org.files.sparkcloud.studio/.../spark-input.tar.gz'"
  },
  ...
}
```

To upload your input directory:

```bash
cd /path/to/your/input/folder
tar czf - . | curl -X PUT --data-binary @- \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  '<the input.uploadUrl from the response>'
```

Spark Fuse waits up to 5 minutes for the upload to land. Once it does,
it pulls the tarball into `/input/` and starts your container. Inside
the container, your files are at `/input/...` exactly as you laid them
out under the directory you ran `tar` from.

If you don't upload within the 5-minute window the job fails with
`error_code='input_download_failed'`.

### 3.2 Mount workflow (pre-populated input folder)

You already have files staged in a ShareSync folder (e.g. you've been
working in your ShareSync mount on macOS / Windows). Submit with
`inputShareSyncPath`:

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
    "command": ["bash", "-c", "python3 /input/run.py"],
    "instanceType": "g7e.2xlarge",
    "inputShareSyncPath": "/My Stuff/job-foo/"
  }' | jq '.'
```

Spark Fuse PROPFINDs the path with your token at submit time. If the
path isn't reachable, you get HTTP 400 immediately — no half-started
job. If it is reachable, the job queues and everything under that path
is pulled into `/input/` before your command runs.

You can also specify `inputShareSyncSpaceName` if the path lives in a
non-Personal Space:

```json
{
  "inputShareSyncPath": "/job-foo/inputs/",
  "inputShareSyncSpaceName": "Renders 2026"
}
```

### 3.3 No input

Omit both `inputShareSyncPath` and `inputPushMode`. `/input/` is not
mounted; your command runs without any input directory.

---

## 4. Streaming job logs (SSE)

`GET /api/compute/jobs/:jobId/logs/stream`

Server-Sent Events. Stays open while the job runs and pushes new
lines as they're produced. Closes when the job reaches a terminal
state.

### 4.1 With curl

```bash
curl -N "$SPARK_HOST/api/compute/jobs/$JOB_ID/logs/stream" \
  -H "Authorization: Bearer $SPARK_TOKEN"
```

Output (SSE wire format — `event:` lines name the event type, `data:` lines
carry a JSON payload):

```
event: queue.status
data: {"status":"queued","queuePosition":2,"estimatedStartSeconds":120,"done":false}

:keepalive

event: queue.status
data: {"status":"provisioning","queuePosition":0,"estimatedStartSeconds":0,"done":false}

event: queue.status
data: {"status":"running","queuePosition":null,"estimatedStartSeconds":null,"done":true}

event: log
id: 1714824601123-stdout
data: {"ts":"2026-05-05T09:12:34.123Z","stream":"stdout","line":"hello from spark","phase":"container"}

event: log
id: 1714824601456-stderr
data: {"ts":"2026-05-05T09:12:34.456Z","stream":"stderr","line":"GPU 0: NVIDIA L4","phase":"container"}

event: log
id: 1714824601999-stdout
data: {"ts":"2026-05-05T09:12:35.999Z","stream":"stdout","line":"job complete","phase":"container"}
```

Comment lines (lines starting with `:`) are SSE keepalives — your client
should ignore them. Real frames carry an `event:` (one of `log`,
`truncated`, or `queue.status`) and a `data:` JSON payload.

The `queue.status` frames are emitted while your job is still in
`queued` or `provisioning` — once per ~10 s — so you can show a "you're
#N in line, ETA ~M s" indicator without polling. They stop firing once
your job reaches `running` or terminal; the final frame in that
sequence carries `done: true`.

### 4.2 With JavaScript / EventSource

```javascript
const es = new EventSource(
  `${sparkHost}/api/compute/jobs/${jobId}/logs/stream`,
  { headers: { Authorization: `Bearer ${sparkToken}` } }
);

es.addEventListener('queue.status', (evt) => {
  const q = JSON.parse(evt.data);
  if (q.done) {
    hideQueueBadge();
  } else {
    showQueueBadge(q.queuePosition, q.estimatedStartSeconds);
  }
});

es.addEventListener('log', (evt) => {
  const line = JSON.parse(evt.data);
  console.log(`[${line.stream}]`, line.line);
});

es.addEventListener('truncated', (evt) => {
  console.warn('logs truncated:', JSON.parse(evt.data));
});

es.onerror = () => {
  // The server closed the stream — typically because the job terminated.
  // Poll GET /api/compute/jobs/:jobId for the final status + error_code.
  es.close();
};
```

> **Heads up:** the stream starts from the **moment you connect** — there
> is no historical replay of logs that landed before the SSE connection
> opened. If you want logs from job start, connect immediately after
> submission.

### 4.3 Log archive

After a job reaches a terminal state, the full log is archived to a
ShareSync `.log` file. The path is on the job row as
`log_archive_share_sync_path` (see [§5](#5-polling-job-status)). You can
GET it via your ShareSync bearer (which is the same Spark token).

---

## 5. Polling job status

`GET /api/compute/jobs/:jobId`

```bash
curl -s "$SPARK_HOST/api/compute/jobs/$JOB_ID" \
  -H "Authorization: Bearer $SPARK_TOKEN" | jq '.'
```

Returns the full job row (a `ComputeJob`) plus resolved `output` and
`input` URL blocks.

```json
{
  "id": "16fb6d80-...",
  "organisation_id": 13,
  "user_id": 42,
  "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
  "image_digest": "sha256:c16f4c749e...",
  "command": ["bash", "-c", "nvidia-smi && echo hello-from-spark"],
  "instance_type_name": "g7e.2xlarge",
  "mode": "instant",

  "status": "succeeded",
  "error_code": null,
  "error_message": null,
  "exit_code": 0,

  "created_at":               "2026-05-05T09:12:32.338+00:00",
  "started_provisioning_at":  "2026-05-05T09:12:35.412+00:00",
  "started_running_at":       "2026-05-05T09:13:48.917+00:00",
  "terminal_at":              "2026-05-05T09:14:11.183+00:00",
  "cancel_requested_at":      null,

  "cuda_version":   "13.0",
  "driver_version": "580.105.08",
  "gpu_name":       "NVIDIA L4",

  "log_archive_share_sync_path": "/Spark Fuse Jobs/16fb6d80-.../spark-fuse-16fb6d80-....log",
  "log_archive_uploaded_at":     "2026-05-05T09:14:13.501+00:00",

  "idle_hold_seconds": 600,
  "container_shm_size": "2g",
  "shmSize": "2g",

  "notify_on_failure": null,

  "max_wallclock_seconds": null,
  "maxWallClockSeconds": null,
  "container_inactivity_seconds": null,
  "containerInactivitySeconds": null,

  "output": {
    "shareSyncPath": "/Spark Fuse Jobs/16fb6d80-.../",
    "shareSyncSpaceName": null,
    "shareSyncBaseUrl": "https://your-org.files.sparkcloud.studio/dav/spaces/<id>/Spark%20Fuse%20Jobs/16fb6d80-.../"
  },
  "input": null
}
```

The status lifecycle is:

```
queued
  ↓ (job is picked up; compute starts; image pulls)
provisioning
  ↓ (container starts running)
running
  ↓ (container exits OR is killed on cancel/timeout)
succeeded | failed | cancelled | smartcompute-interrupted
```

`succeeded` means the container exited with code 0 (and outputs uploaded
without a fatal upload error). `failed` carries an `error_code` +
`error_message` describing why; common values are documented in
[§12.2](#122-error-codes).

> **Tip:** poll on a short interval while the job is non-terminal. 5
> seconds is plenty — most state transitions happen on roughly a 10 s
> cadence. For a more interactive experience, use the SSE log stream
> (§4) instead — it pushes log lines as they arrive and closes when
> the job ends.

---

## 6. Listing your jobs

`GET /api/compute/jobs`

Returns every job for **your organisation** (not just your user).
Useful for dashboards and bulk status views.

```bash
curl -s "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" | jq '.'
```

Response:

```json
{
  "jobs": [
    {
      "id": "16fb6d80-...",
      "image": "...",
      "instance_type_name": "g7e.2xlarge",
      "status": "succeeded",
      "created_at": "2026-05-05T09:12:32.338+00:00",
      "terminal_at": "2026-05-05T09:14:11.183+00:00",
      "exit_code": 0,
      "output": {
        "shareSyncPath": "/Spark Fuse Jobs/16fb6d80-.../",
        "shareSyncBaseUrl": null
      },
      "input": null
      ... (most other ComputeJob fields)
    },
    ...
  ]
}
```

> **Important — `shareSyncBaseUrl` is `null` in list responses.** Composing
> the URL requires a ShareSync Graph API roundtrip per job, which would
> blow up latency on a long list. Call `GET /api/compute/jobs/:jobId` for
> any specific job to pick up the resolved URL. The other URL fields
> (`shareSyncPath`, `shareSyncSpaceName`) are present in list responses.

> **Pagination:** the endpoint returns the full list. If you have many
> jobs and want server-side pagination, file a request.

### 6.1 Filtering by tag

The list endpoint supports two mutually-exclusive tag filters (see
[§2.3](#23-tagging-jobs) for tag rules):

```bash
# AND semantics — repeatable ?tag. Returns jobs that carry EVERY listed tag.
curl -s "$SPARK_HOST/api/compute/jobs?tag=team_alpha&tag=training" \
  -H "Authorization: Bearer $SPARK_TOKEN" | jq '.jobs | length'

# OR semantics — comma-separated ?tagsAny. Returns jobs that carry AT LEAST
# ONE of the listed tags.
curl -s "$SPARK_HOST/api/compute/jobs?tagsAny=team_alpha,team_beta" \
  -H "Authorization: Bearer $SPARK_TOKEN" | jq '.jobs | length'
```

Passing both returns HTTP 400 with `error_code='invalid_tags'`.

---

## 7. Cancelling a job

`POST /api/compute/jobs/:jobId/cancel`

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs/$JOB_ID/cancel" \
  -H "Authorization: Bearer $SPARK_TOKEN" | jq '.'
```

Cancel is idempotent. Cancelling a job that's already in a terminal state
returns the job row unchanged. Cancelling a queued job removes it from the
queue immediately. Cancelling a running job sends a SIGTERM to the
container (with a 30 s grace period, then SIGKILL if the container hasn't
exited).

The response is the same shape as `GET /api/compute/jobs/:jobId` — the
job's current state. The `status` field will be `cancelled` (or still
mid-transition; poll to confirm terminal).

> **Idle-hold behaviour:** cancel always **skips** any remaining idle-hold
> window. The compute is released as soon as the cancel is acknowledged.

---

## 8. Browsing eligible instance types

`GET /api/compute/skus`

Returns the current allow-list of instance types you can put in
`instanceType` on `POST /api/compute/jobs`.

```bash
curl -s "$SPARK_HOST/api/compute/skus" \
  -H "Authorization: Bearer $SPARK_TOKEN" | jq '.'
```

```json
{
  "skus": [
    "g4dn.xlarge",
    "g5.xlarge",
    "g6.2xlarge",
    "g6.4xlarge",
    "g7e.2xlarge",
    "g7e.4xlarge",
    "gr6.8xlarge"
    ...
  ]
}
```

The list is updated as we expand GPU coverage.

> **Heads up — catalog vs. capacity.** The SKUs on this list are the ones
> Spark Fuse will **accept** in submission. They are not a guarantee
> that capacity exists for every SKU at every moment. If a submission
> you'd expect to succeed times out waiting on capacity, retry with a
> nearby SKU.

---

## 9. Retrieving outputs from ShareSync

When your container writes to `/output/`, Spark Fuse persists everything
on the ShareSync path on the job row. Pull the files back via the
same WebDAV server using your Spark bearer.

### 9.1 List output files (PROPFIND)

```bash
curl -X PROPFIND \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Depth: 1" \
  "$OUTPUT_BASE_URL"
```

(`$OUTPUT_BASE_URL` is the `output.shareSyncBaseUrl` from the
`GET /api/compute/jobs/:jobId` response.) Returns an XML multistatus body
listing every entry in the directory.

### 9.2 Download a specific file

```bash
curl -O \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  "$OUTPUT_BASE_URL/results.tar.gz"
```

### 9.3 Pull everything

```bash
mkdir -p ./outputs && cd ./outputs
# Use any WebDAV client of your choice — the example below uses one
# (rclone) but cadaver, davix, curl, or your language's WebDAV
# library will all work the same way.
rclone copy ":webdav:" ./ --webdav-url "$OUTPUT_BASE_URL" \
  --webdav-vendor other --webdav-bearer-token "$SPARK_TOKEN"
```

> `**outputShareSyncPath` defaults to `/Spark Fuse Jobs/{jobId}/**` — if you
> don't override it on submit, every job's outputs land in a uniquely-
> namespaced directory. Override `outputShareSyncPath` on submit if you
> want a more meaningful folder name (e.g. `/Renders 2026/run-007/`).
>
> **Default folder rename note (2026-05-06):** the default was previously
> `/Compute Jobs/{jobId}/` — same shape, just the top-level folder name
> was renamed to match the Spark Fuse product brand. If you have past
> jobs whose archives landed in `/Compute Jobs/`, those archives are
> still there and still listed verbatim on each job's
> `output.shareSyncPath` (the resolved path is captured at submission
> time, so existing jobs are unaffected). Only **new** submissions
> with `outputShareSyncPath` omitted will use the new
> default. If you've been passing an explicit `outputShareSyncPath` on
> every submit, this change has no effect on you.

### 9.4 Log archive

In addition to outputs, the per-job log file (everything you'd see on
the SSE stream, plus platform-side provisioning logs) is uploaded to
`log_archive_share_sync_path`. Pull it the same way — prefer reading
the exact path from the `log_archive_share_sync_path` field on the job
rather than constructing the URL by hand, since the filename prefix
changed 2026-05-13 (`spark-compute-<jobId>.log` → `spark-fuse-<jobId>.log`)
and past jobs still carry the old name verbatim on their row:

```bash
curl -O \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  "$SHARESYNC_BASE_URL/spark-fuse-$JOB_ID.log"
```

---

## 10. Resumable jobs and checkpoints

For jobs that take long enough to be worth resuming on failure — long
training runs, multi-frame renders, big batch sweeps — `/output/` is
your durable scratch space. Anything your container writes there is
visible on ShareSync within seconds, so the next attempt of the
**same** `jobId` can pick up where the previous one left off.

When this matters:

- **SmartCompute jobs** (`mode='smart'`). The platform may reclaim the
  compute mid-run and re-launch your container from scratch. Without
  a checkpoint, retries restart from frame 0. With one, they don't.
- **Long instant-mode jobs** that you might want to resubmit by hand
  after a failure. Same pattern works — just resubmit with the same
  `outputShareSyncPath` and your prior progress is already there.

**The pattern, end to end:**

1. Write your output incrementally to `/output/` — one file per frame,
   per epoch, per shard, whatever your unit of work is. Writes are
   flushed to ShareSync continuously (around every 10 s), so partial
   progress survives a crash even mid-run.
2. After each unit of work completes, write a tiny checkpoint sentinel
   to `/output/` recording how far you've gotten — e.g.
   `/output/.checkpoint.json` with `{"next_frame": 17}`.
3. At container start, check for the sentinel. If it's there, resume
   from the recorded position; if not, start fresh.

A minimal wrapper script:

```bash
#!/usr/bin/env bash
set -euo pipefail

CKPT=/output/.checkpoint.json

if [[ -f "$CKPT" ]]; then
  echo "Resuming from checkpoint: $(cat "$CKPT")"
  RESUME_FROM=$(jq -r '.next_frame' "$CKPT")
else
  echo "Fresh start"
  RESUME_FROM=0
fi

for ((i = RESUME_FROM; i < 100; i++)); do
  python3 render.py --frame "$i" --out "/output/frame-$i.png"
  echo "{\"next_frame\": $((i+1))}" > "$CKPT"
done

rm -f "$CKPT"
echo "Done."
```

**Two things to know:**

- The `output.shareSyncPath` is **stable within a `jobId`** — every
  SmartCompute retry of the same job mounts the same path, so your
  checkpoint sentinel from the previous attempt is already there when
  the new container starts. No download phase, no race window.
- **A file that was mid-write at the moment of an interruption may
  exist at truncated length on ShareSync.** That's why the sentinel
  matters: it's your signal that `frame-N.png` was *finished*, not
  just *started*. If your workload isn't naturally append-only or
  idempotent, lean on the sentinel.

For the SmartCompute-specific deep dive on retry behaviour, end-of-job
drain semantics, and `output_*_failed` error codes, see
[§14.6](#146-retry-safe-containers-output-persistence--the-checkpoint-pattern).

---

## 11. End-to-end example

A complete copy-paste flow — get token, submit, upload input, watch logs,
pull outputs.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---- 0. Config -------------------------------------------------------
export SPARK_HOST="https://api.prod.aapse1.sparkcloud.studio"
export SPARK_EMAIL="you@yourcompany.example"
export SPARK_PASSWORD="..."             # use your secrets manager in real life
INPUT_DIR="./my-input-folder"           # local files to send up
IMAGE="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
COMMAND='bash -c "ls -la /input/ && python3 /input/job.py /output/"'
INSTANCE_TYPE="g7e.2xlarge"

# ---- 1. Auth ---------------------------------------------------------
LOGIN_RESPONSE=$(curl -sX POST "$SPARK_HOST/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$SPARK_EMAIL\",
    \"password\": \"$SPARK_PASSWORD\"
  }")
if [[ "$(echo "$LOGIN_RESPONSE" | jq -r '.success')" != "true" ]]; then
  echo "Login failed: $(echo "$LOGIN_RESPONSE" | jq -r '.resp')" >&2
  exit 1
fi
export SPARK_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.token')
echo "Got token: ${SPARK_TOKEN:0:20}..."

# ---- 2. Submit with auto-prepare input ------------------------------
SUBMIT_RESPONSE=$(curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$IMAGE\",
    \"command\": [\"bash\", \"-c\", \"ls -la /input/ && python3 /input/job.py /output/\"],
    \"instanceType\": \"$INSTANCE_TYPE\",
    \"inputPushMode\": \"auto-prepare\",
    \"outputShareSyncPath\": \"/Spark Fuse Jobs/end-to-end-example/\"
  }")
echo "$SUBMIT_RESPONSE" | jq '.'

JOB_ID=$(echo "$SUBMIT_RESPONSE"     | jq -r '.jobId')
UPLOAD_URL=$(echo "$SUBMIT_RESPONSE" | jq -r '.input.uploadUrl')
OUTPUT_URL=$(echo "$SUBMIT_RESPONSE" | jq -r '.output.shareSyncBaseUrl')

# ---- 3. Upload input -------------------------------------------------
( cd "$INPUT_DIR" && tar czf - . ) | curl -sX PUT --data-binary @- \
  -H "Authorization: Bearer $SPARK_TOKEN" "$UPLOAD_URL"
echo "Input uploaded to $UPLOAD_URL"

# ---- 4. Stream logs (blocks until job terminates) -------------------
echo "=== logs ==="
curl -N "$SPARK_HOST/api/compute/jobs/$JOB_ID/logs/stream" \
  -H "Authorization: Bearer $SPARK_TOKEN"
echo "=== /logs ==="

# ---- 5. Get final status --------------------------------------------
FINAL=$(curl -s "$SPARK_HOST/api/compute/jobs/$JOB_ID" \
  -H "Authorization: Bearer $SPARK_TOKEN")
echo "$FINAL" | jq '{status, exit_code, error_code, error_message, terminal_at}'

# ---- 6. Pull outputs (PROPFIND lists, then download each file) ------
mkdir -p ./out
echo "Outputs at: $OUTPUT_URL"
# Use your favourite WebDAV client. For one specific file:
# curl -O -H "Authorization: Bearer $SPARK_TOKEN" "$OUTPUT_URL/results.tar.gz"
```

---

## 12. Reference

### 12.1 Job status values


| `status`           | Meaning                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `queued`           | Submitted; waiting for compute to be picked.                                                 |
| `provisioning`     | Compute starting + image pulling.                                                            |
| `running`          | `docker run` dispatched; your command is executing.                                          |
| `succeeded`        | Container exited with code 0; outputs uploaded successfully.                                 |
| `failed`           | Something went wrong. Inspect `error_code` and `error_message`.                              |
| `cancelled`        | You called `POST /api/compute/jobs/:jobId/cancel` and the job was stopped before completing.                  |
| `smartcompute-interrupted` | (`mode='smart'` only) The platform reclaimed the smart-mode compute before the job finished. |


`succeeded`, `failed`, `cancelled`, `smartcompute-interrupted` are **terminal** —
the job will not transition again.

### 12.2 Error codes

When `status='failed'`, `error_code` carries one of the following values.
`error_message` adds human-readable context.


| `error_code`                       | Cause                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image_pull_failed`                | The image reference wasn't pullable (typo, registry auth, network).                                                                                                                                                                                                                                                                                                                                                                   |
| `disk_full`                        | The compute ran out of disk during image pull or container run.                                                                                                                                                                                                                                                                                                                                                                       |
| `input_download_failed`            | (auto-prepare) You didn't upload to the input URL within 5 minutes, or the upload was malformed.                                                                                                                                                                                                                                                                                                                                      |
| `output_upload_failed`             | Outputs couldn't be uploaded to ShareSync. The container may have produced valid outputs that simply didn't make it back. Retained for older job submissions; new submissions surface the `output_mount_*` codes below instead — see §14.6.                                                                                                                                       |
| `output_mount_failed`              | Spark Fuse couldn't bring up the `/output/` mount before the container started. The container never ran — semantically equivalent to a provisioning fault. Not customer-actionable; resubmit. See §14.6. |
| `output_drain_failed`              | The container ran to its terminal state, but the post-exit drain window expired with pending `/output/` writes still in flight to ShareSync. Some `/output/` files may not have reached ShareSync. Preserves the underlying terminal status (`succeeded`, `failed`, `smartcompute-interrupted`) by appending the code rather than replacing it. See §14.6. |
| `output_unmount_failed`            | A rare teardown failure releasing the per-job `/output/` mount after the container exited. Appended to the underlying terminal status rather than replacing it. Not customer-actionable; resubmit if any output files appear missing. See §14.6. |
| `container_nonzero_exit`           | Your command exited with a non-zero status. `exit_code` field carries the exit code.                                                                                                                                                                                                                                                                                                                                                  |
| `cancelled`                        | The job was cancelled via `POST /api/compute/jobs/:id/cancel` while running. `status` is projected to `cancelled` for cancel-driven failures — see §12.1.                                                                                                                                                                                                                                                                                                                                 |
| `agent_silent`                     | Spark Fuse stopped receiving heartbeats from the compute host running your container. A watchdog drove the job to terminal and released the compute. Usually a transient infrastructure event; resubmit.                                                                                                                                                                                                                              |
| `start_timeout`                    | The compute was assigned and started, but the container never reached `running` within the configured budget — typically a slow image pull or a host-side wedge. Distinct from `no_capacity` (capacity-driven). Resubmit; escalate to Spark if persistent.                                                                                                                                                                            |
| `no_capacity`                      | Spark Fuse retried for the full provisioning window and couldn't find an eligible warm compute for the requested SKU — every candidate was busy and no fresh allocation was possible. Retry shortly, or pick a different SKU from `GET /api/compute/skus`.                                                                                                                                                                            |
| `smartcompute_interrupted`                 | (`mode='smart'` only) Spark Fuse reclaimed the underlying compute and terminated the running container before the job finished. Surfaced as `status='smartcompute-interrupted'`. If `maxRetriesOnInterrupt > 0` and the budget is not yet exhausted, the job is automatically re-queued on fresh smart-mode compute instead of going terminal — so you'll typically only see this terminal `error_code` when the budget is exhausted. |
| `smartcompute_interrupted_no_retries_left` | (`mode='smart'` only) The smart-mode compute was reclaimed AND the configured `maxRetriesOnInterrupt` budget has been exhausted. Distinct from `smartcompute_interrupted` so you can pattern-match on "we ran out of retries" specifically. The job is terminal `failed`; resubmit with a higher `maxRetriesOnInterrupt`, switch to `mode='instant'`, or both.                                                                       |
| `instance_handle_invalid`          | The `instanceHandle` you submitted with the job pointed at a session that no longer routes traffic — released, expired, failed mid-prepare, or otherwise lost between submit and job-start. Submit a fresh `POST /api/compute/instances/prepare` and use the new handle. Distinct from the 400 `instance_handle_invalid` rejection at submit time: this one fires when the handle was valid at submit but went bad between submit and job-start. |
| `wallclock_exceeded`                       | Spark Fuse SIGKILLed the container because it exceeded the `maxWallClockSeconds` value you supplied at submission time. **Only fires when you explicitly opted in** by setting `maxWallClockSeconds` on the submission — jobs that omit the field never carry this `errorCode`. Resubmit with a higher `maxWallClockSeconds` (or omit the field entirely to remove the ceiling) if the runtime was legitimate. |
| `container_inactive`                       | The container-inactivity detector fired — your container emitted no stdout/stderr output **and** had no CPU activity **and** had no GPU activity for the entire `containerInactivitySeconds` window (default 30 min). Spark Fuse SIGTERMed the container (30 s grace, then SIGKILL). The three-signal AND means a training loop that's keeping the GPU busy without logging will **not** trigger this code — but a job blocked on a network call or stuck in an idle loop with no GPU work will. Resubmit with a higher `containerInactivitySeconds`, or pass `containerInactivitySeconds: 0` to disable the detector for jobs with legitimate long quiet periods. |


### 12.3 Idle-hold + warm-pool

After your job ends, the compute can stay running for `idleHoldSeconds`
before Spark Fuse releases it. During that window, your org's next
submission preferentially lands on this same warm compute — no cold-start
image pull, no provisioning latency, and it's the same `instance_type`. 
(Idle-hold is InstantCompute-only; SmartCompute
jobs don't honor `idleHoldSeconds`.)

If you submit two back-to-back jobs and want zero cold-start on the
second:

1. Submit job 1 with `idleHoldSeconds: 600` (or whatever covers the gap).
2. Wait for job 1 to reach a terminal state (`status='succeeded'` or
  `'failed'`).
3. Submit job 2 with the same `instanceType`. Within ~1–3 seconds it lands
  on the same warm compute.

### 12.4 Submit response shape (`CreateComputeJobResponse`)

```typescript
{
  jobId: string;            // UUID
  status: string;           // always 'queued' on the submit response
  imageDigest: string | null;  // resolved image digest, if pinning succeeded
  outputShareSyncPath: string;
  createdAt: string;        // ISO 8601 timestamp
  output: {
    shareSyncPath: string;
    shareSyncSpaceName: string | null;
    shareSyncBaseUrl: string | null;  // null on transient ShareSync resolve failures
  };
  input: {
    shareSyncPath: string;
    shareSyncSpaceName: string | null;
    shareSyncBaseUrl: string | null;
    uploadUrl?: string;       // present only on inputPushMode='auto-prepare'
    uploadMethod?: 'PUT';
    exampleCurl?: string;
  } | null;

  // Queue position + ETA. Approximate; treat as
  // order-of-magnitude hint, not an SLA.
  queuePosition?: number | null;        // 0=next-up, N>0=N ahead, null=N/A
  estimatedStartSeconds?: number | null;  // queuePosition * average provisioning time

  // Resolved Docker --shm-size for this container (Docker size string —
  // e.g. '2g'). Echo of either the submitted shmSize, the Spark Fuse
  // default (currently '2g'), or the hardcoded '2g' fallback when
  // neither is available.
  shmSize?: string;

  // Echo of the resolved per-job notifyOnFailure override (§2.2 +
  // §14.7). `null` when the submission omitted the field (defer to
  // your org's default); `true` / `false` when an explicit value
  // was supplied.
  notifyOnFailure: boolean | null;

  // Echo of the opt-in wall-clock kill ceiling (§2.2). `null` when
  // the submission omitted the field — Spark Fuse will NEVER kill
  // this job on time-elapsed alone. A positive integer when the
  // customer opted in. Same value surfaced on
  // GET /api/compute/jobs/{jobId}.
  maxWallClockSeconds: number | null;

  // Echo of the per-job container-inactivity threshold (§2.2).
  // `null` = the org default applies; `0` = customer disabled the
  // detector for this job; positive integer = customer override.
  containerInactivitySeconds: number | null;
}
```

### 12.5 Get-job / list / cancel response shape (`ComputeJobApiShape`)

The full job payload plus `output` and `input` URL blocks (and the
same `queuePosition` / `estimatedStartSeconds` pair as the submit
response — both `null` once the job is past `provisioning`). See
[§5](#5-polling-job-status) for an annotated example.

The resolved Docker `--shm-size` is returned in two forms for
convenience: `container_shm_size` (snake_case, always present) and
`shmSize` (camelCase alias, always present). Both carry the same
value.

The v1.18 lifecycle-limit fields are also returned in both forms:
`max_wallclock_seconds` / `maxWallClockSeconds` and
`container_inactivity_seconds` / `containerInactivitySeconds`. `null`
on `maxWallClockSeconds` is the load-bearing "no wall-clock kill"
sentinel — Spark Fuse never kills a job on time-elapsed alone unless
this field carries a positive integer.

### 12.6 Endpoint reference


| Method | Path                                   | Description                                                                  |
| ------ | -------------------------------------- | ---------------------------------------------------------------------------- |
| `POST` | `/api/compute/jobs`                    | [Submit a job](#2-submitting-a-compute-job).                                 |
| `GET`  | `/api/compute/jobs`                    | [List jobs for your org](#6-listing-your-jobs).                              |
| `GET`  | `/api/compute/jobs/:jobId`             | [Get a single job](#5-polling-job-status).                                   |
| `POST` | `/api/compute/jobs/:jobId/cancel`      | [Cancel a job](#7-cancelling-a-job).                                         |
| `GET`  | `/api/compute/jobs/:jobId/logs/stream` | [SSE log stream](#4-streaming-job-logs-sse).                                 |
| `GET`  | `/api/compute/skus`                    | [List eligible instance types](#8-browsing-eligible-instance-types).         |
| `POST` | `/api/compute/jobs/estimate`           | [Get a cost quote before submitting](#28-cost-estimation-pre-submit).        |
| `POST` | `/api/compute/instances/prepare`         | [Pre-start an Instant-mode compute instance + open a session](#131-prepare-an-instance). |
| `GET`  | `/api/compute/instances/:handle`         | [Poll a session's status](#132-poll-a-session).                              |
| `POST` | `/api/compute/instances/:handle/release` | [Explicitly tear down a session](#133-release-a-session).                  |


### 12.7 Spark Fuse is for Linux containers only

Spark Fuse runs **Linux** Docker containers on **Linux x86_64** hosts.
There is no Windows-compute path. At submit time Spark Fuse requires a 
**`linux/amd64`** manifest. That means:

- A normal **Linux** image built for amd64 — including images you build
  yourself ([§2.5](#25-using-your-own-docker-image)) — is supported.
- **`windows/amd64`** images (Windows containers) are **not** supported,
  even though the CPU architecture is also amd64.
- **`linux/arm64`** images are **not** supported on current SKUs, even
  though they are Linux containers. Rebuild with
  `--platform=linux/amd64` and push a multi-arch index or an amd64-only
  tag.

If submit fails image validation, the API error text will say the image
has no `linux/amd64` manifest or could not be pulled from the registry.

### 12.8 Limits + quotas

- **Submission rate:** soft limit per org (default ~10
  submissions/minute, tunable per-org by Spark). Exceeding returns
  HTTP 429 with `Retry-After`.
- **Idle hold:** range `[0, max]`; the max is currently in the
  low-thousands of seconds.
- **Image pull timeout:** ~15 min with retry. `disk_full` and
  `image_pull_failed` are surfaced as terminal `error_code` values
  rather than infinite retry.

### 12.9 Help, support, and feedback

- API issues, unexpected errors, missing features: ping your Spark
contact or open a ticket on the customer support portal.
- Documentation issues (this document): file a doc bug; we'd rather know.

---

## 13. Persistent compute / sessions (`mode='instant'` only)

Idle-hold ([§12.3](#123-idle-hold--warm-pool)) lets you ride a warm
instance forward from the *end* of one job into the start of the next.
Sessions let you do the same thing from the *beginning* — call `POST
/api/compute/instances/prepare` to pre-warm a Spark Fuse instance
before you have a job to submit, get back an `instanceHandle`, and
then route any number of sequential jobs to that same instance by
passing `instanceHandle` on `POST /api/compute/jobs`. Explicit
teardown is `POST /api/compute/instances/:handle/release`; otherwise
the session self-expires after the `holdSeconds` you set on prepare.

The pre-warmed compute is billed the same per-second rate as a job
running on the same SKU. You pay for the wall-clock time the instance
is held minus the time it spends actively running your jobs (those
windows are billed as normal `compute_instant` line items, not
double-counted). The idle slice is surfaced as a separate
`compute_instant_session` balance line so it's easy to attribute.

**Current limitation:** sessions are InstantCompute only. `mode='smart'`
on `/prepare` returns HTTP 400 `smart_mode_not_supported`.

### 13.1 Prepare an instance

`POST /api/compute/instances/prepare`

Request body (`application/json`):

| Field                | Type   | Required | Notes                                                                                                                                                                                                                                                                       |
| -------------------- | ------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `instanceType`       | string | yes      | Instance type SKU (e.g. `g7e.2xlarge`). Same allow-list as `POST /api/compute/jobs`.                                                                                                                                                                                       |
| `mode`               | string | no       | `instant` (default). Supplying `smart` returns HTTP 400 `smart_mode_not_supported`.                                                                                                                                                                                         |
| `holdSeconds`        | number | yes      | Maximum wall-clock seconds Spark Fuse will keep the instance held. Range `[1, max]` where the max defaults to 7200 (2 h) and is tunable per-org. The hold clock starts when the instance first reports ready — you don't pay for cold-start prep before that. See §13.4 for the math. |

```bash
curl -sX POST "$SPARK_HOST/api/compute/instances/prepare" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{
        "instanceType": "g7e.2xlarge",
        "holdSeconds": 1800
      }' | jq '.'
```

Response (`PrepareInstanceResponse`):

```json
{
  "instanceHandle": "b41f8a17-4c8a-46e7-8d5b-2c8e0d4f6f02",
  "status": "preparing",
  "instanceType": "g7e.2xlarge",
  "holdSeconds": 1800,
  "preparedAt": "2026-05-19T11:34:56.000Z",
  "readyAt": null,
  "releasedAt": null,
  "expiredAt": null,
  "failedAt": null,
  "errorCode": null,
  "errorMessage": null,
  "expiresAt": "2026-05-19T12:34:56.000Z",
  "firstJobId": null,
  "lastJobId": null
}
```

`status` starts as `preparing` and transitions to `ready` once the
held instance is idle and warm. Poll
`GET /api/compute/instances/:handle` (§13.2) until `status='ready'`
before you submit jobs that target the handle — submissions against a
`preparing` handle are accepted but won't actually run on the held
instance until the session is `ready`.

The preparing → ready window depends on what warm-pool candidate is
available for your SKU:

- **Exact SKU, already running** (same-org idle-hold hit, or another
  warm instance ready to go). Response often comes back with
  `status='ready'` already — no cold-start.
- **Exact SKU, stopped.** ~60–90 s typical: cold boot.
- **Near-match SKU.** 30 s – 6 min typical: Spark Fuse adjusts the
  instance configuration to match before the instance becomes
  available. The HTTP response always comes back fast; the customer
  just polls a little longer to see `ready`.

If the initial warm-pool candidate hits a capacity limit during
start-up, Spark Fuse silently retries against the next candidate (up
to 3 total attempts by default). The session stays `preparing` across
the fallback. Only when every candidate has been exhausted does the
session go terminal `failed` with `errorCode='no_warm_pool_capacity'`.

Non-capacity start-up failures (the instance went bad mid-prepare, the
prepare timed out, etc.) bypass the fallback and go straight to
terminal `status='failed'` with an `errorCode` and `errorMessage` you
can inspect on the next `GET /api/compute/instances/:handle`. See §13.2.

`expiresAt` is the wall-clock timestamp the session self-expires if
nothing else changes it. It's `prepare_started_at + holdSeconds` while
in `preparing`, then re-clocked to `ready_at + holdSeconds` once the
session reaches `ready`. Submitting a job re-clocks it again from the
job's terminal_at (idle-hold is re-armed when the job ends — same
behaviour as the post-job `idleHoldSeconds` window).

Common errors:

| HTTP | Meaning                                                                                                                                                                                                                                                                  |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 400  | `smart_mode_not_supported` (`mode='smart'`); `invalid_hold_seconds` (out of range); `instanceType ... is not eligible` (same allow-list gate as submit).                                                                                                                  |
| 503  | `no_warm_pool_capacity` — no eligible warm compute available right now. Retry shortly or submit a regular job (`POST /api/compute/jobs`), which can wait through the full provisioning timeout for capacity to free up.                                                  |
| 500  | `prepare_failed` — synchronous-portion failure before the prepare could be dispatched. Distinct from terminal `status='failed'` (§13.2), which surfaces asynchronous-portion failures. Resubmit; contact Spark if persistent.                                             |

### 13.2 Poll a session

`GET /api/compute/instances/:handle`

Returns the same `PrepareInstanceResponse` shape as §13.1, with
`status` reflecting the current session state. Possible values:

| `status`     | Meaning                                                                                                                                                                                                                                                                                |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preparing`  | The instance is starting / pulling the boot image / not yet idle-holding. Submissions targeting the handle are queued and held until `ready`.                                                                                              |
| `ready`      | The instance is idle-holding and waiting for work. Your next submission with `instanceHandle` lands on this instance with zero cold-start.                                                                                                                                                              |
| `running`    | A job submitted with `instanceHandle` is currently running on the instance. The session re-opens to `ready` when the job terminates (idle-hold is re-armed on terminal).                                                                                                          |
| `released`   | Terminal. Customer called `POST /api/compute/instances/:handle/release` (or the underlying instance went terminal for an out-of-band reason).                                                                                                                       |
| `expired`    | Terminal. The `holdSeconds` window elapsed without a release call and the instance was self-stopped.                                                                                                                                                                              |
| `failed`     | Terminal. Spark Fuse couldn't deliver on the handle — the selected instance was lost mid-prepare, `/prepare` timed out, etc. `errorCode` and `errorMessage` carry the diagnosis. Distinct from `released` (customer tear-down) so you can pattern-match for "retry". |

When `status='failed'`, the response carries `failedAt`, `errorCode`,
and `errorMessage`. `errorCode` is short + stable and pattern-matchable:

| `errorCode`            | Meaning                                                                                                                                                                                                                                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `no_warm_pool_capacity` | Spark Fuse tried every available warm-pool candidate (up to the per-session cap, default 3) and every one of them hit a capacity limit on start. Compute for this SKU is thin right now. Retry shortly, or submit a regular job (`POST /api/compute/jobs`) which can wait through the full provisioning timeout. |
| `reshape_failed`       | Spark Fuse tried to adjust the underlying instance configuration to match your requested SKU and the change was rejected (unsupported type, etc.). Not retried — resubmit `/prepare` and a different warm-pool candidate may be selected.                                                                          |
| `prepare_failed`       | Catch-all for non-capacity, non-reshape failures during the prepare lifecycle. Resubmit; contact Spark if persistent.                                                                                                                                                                                              |
| `prepare_timeout`      | The session sat in `preparing` for longer than the hard ceiling (default 10 min) without becoming ready. Boot-side wedge. Resubmit.                                                                                                                                                                                |
| `instance_lost`        | The underlying compute instance was lost or reassigned between `/prepare` and the instance reporting ready. Resubmit.                                                                                                                                                                                              |

Errors (the HTTP-level kind, distinct from terminal `failed` status):

| HTTP | Meaning                                                              |
| ---- | -------------------------------------------------------------------- |
| 404  | Handle doesn't exist or belongs to a different org.                  |

### 13.3 Release a session

`POST /api/compute/instances/:handle/release`

Explicit teardown. Idempotent — if the session is already terminal,
returns 200 with the current state.

```bash
curl -sX POST "$SPARK_HOST/api/compute/instances/$HANDLE/release" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-length: 0' | jq '.'
```

Response is the same `PrepareInstanceResponse` shape, projected with
`status='released'` once the underlying instance has finished stopping.

Errors:

| HTTP | Meaning                                                                                                                                                                                                                                 |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 404  | Handle doesn't exist or belongs to a different org.                                                                                                                                                                                     |
| 409  | A job is still running on the held instance — cancel it first (`POST /api/compute/jobs/:jobId/cancel`) or wait for it to terminate, then re-issue the release. Spark Fuse does not auto-cancel an in-flight job on release.                                                                                                            |

### 13.4 Billing model

While a session is open, you're paying for whatever fraction of the
wall-clock window Spark Fuse is holding compute for you. The math
splits into two line items per session:

- **`compute_instant`** — per-job lines, one per job submitted with
  this `instanceHandle`. Same rate, same billing math as a standalone
  InstantCompute job. Charged for the `(started_running_at, terminal_at)`
  window of each constituent job.
- **`compute_instant_session`** — the idle slice. Wall-clock window
  between `ready_at` and `released_at` / `expired_at` on the session
  row, minus the union of the constituent jobs' `(started_running_at,
  terminal_at)` windows. Same per-second rate as `compute_instant`
  against the same SKU — you're renting the whole instance, Spark Fuse
  just isn't running your code right now.

So a session that prepares a `g7e.2xlarge`, runs three 5-minute jobs
over 30 minutes, and gets released bills as three `compute_instant`
lines totaling 15 minutes plus one `compute_instant_session` line for
the remaining ~15 minutes. The preparing→ready window is **not** 
billed — the clock only starts at `ready_at`.

If the session ends in `expired` instead of `released` (you forgot to
call release), the math is the same; we don't charge a penalty for
expiry vs explicit release. If the session ends in `failed`, **no
billing line items are written** — Spark Fuse couldn't deliver on
the handle, so you pay nothing.

### 13.5 Pattern: interactive dev loop

When you're iterating on a job spec or container image and want to
avoid 30–90 s warm-pool selection on every submit:

```bash
# 1. Pre-warm an instance for the next hour.
HANDLE=$(curl -sX POST "$SPARK_HOST/api/compute/instances/prepare" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"instanceType": "g7e.2xlarge", "holdSeconds": 3600}' \
  | jq -r '.instanceHandle')

# 2. Poll until ready.
while true; do
  STATUS=$(curl -sX GET "$SPARK_HOST/api/compute/instances/$HANDLE" \
    -H "Authorization: Bearer $TOKEN" | jq -r '.status')
  case "$STATUS" in
    ready)               break ;;
    released|expired|failed)
                         echo "session terminal in $STATUS; bailing" >&2; exit 1 ;;
  esac
  sleep 2
done

# 3. Submit jobs against the handle. Each lands instantly.
for ITER in 1 2 3; do
  curl -sX POST "$SPARK_HOST/api/compute/jobs" \
    -H "Authorization: Bearer $TOKEN" \
    -H 'content-type: application/json' \
    -d "{
          \"image\": \"pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime\",
          \"command\": [\"python3\", \"/input/train.py\", \"--iter\", \"$ITER\"],
          \"instanceType\": \"g7e.2xlarge\",
          \"instanceHandle\": \"$HANDLE\",
          \"inputShareSyncPath\": \"/training-data/$ITER/\"
        }"
  # ... wait for terminal, inspect, edit code, loop ...
done

# 4. Done iterating. Stop paying for the held instance.
curl -sX POST "$SPARK_HOST/api/compute/instances/$HANDLE/release" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-length: 0'
```

---

## 14. SmartCompute — `mode='smart'`

> **Capacity availability — current limitation.** SmartCompute capacity
> for some GPU SKUs can be **extremely limited** at peak times.
> Practical consequences:
>
> - Submissions can sit in `queued` longer than instant-mode (sometimes
> several minutes before matching capacity is found).
> - `error_code='no_capacity'` is materially more likely on scarce SKUs
> than it is for instant-mode against the same SKU. Retry with a
> sibling SKU from `GET /api/compute/skus`, or fall back to
> `mode='instant'` for that submission.
> - Smart-mode estimate quotes ([§2.8](#28-cost-estimation-pre-submit))
> still return rate data for any SKU with observed pricing, even when
> no capacity is currently available — a quote is not a guarantee
> capacity exists right now.

`mode='smart'` runs your job on preemptible Spark Fuse capacity
instead of guaranteed-availability warm-pool compute. Compared to
`mode='instant'`:


|                       | `instant` (default)                                | `smart`                                                                           |
| --------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- |
| Underlying capacity   | Warm-pool guaranteed-availability                  | Preemptible (one-shot)                                                            |
| Typical price         | full instant-mode rate                             | typically ~60% off instant-mode (varies; quoted via the cost-estimation endpoint) |
| Cold-start            | ~2s on warm hit; ~60s if cold                      | ~3 min cold provisioning every time (no warm pool)                                |
| Idle-hold (warm-keep) | yes                                                | no                                                                                |
| Affinity routing      | yes (org-level next-job-on-same-warm-compute)      | no                                                                                |
| Mid-run interruption  | never                                              | possible — compute is reclaimed with a 2-minute warning                           |
| Best fit              | low-latency / interactive runs / multi-job warmups | bulk batch, long-running training, anything tolerant to a re-launch               |


### 14.1 Submitting a smart job

```bash
curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{
        "image": "ghcr.io/your-org/training:v3.2",
        "command": ["python3", "train.py"],
        "instanceType": "g7e.2xlarge",
        "mode": "smart",
        "maxRetriesOnInterrupt": 2
      }' | jq '.'
```

`maxRetriesOnInterrupt` (default `1`, range `[0, 5]`) controls how many
times Spark Fuse may re-launch the job on fresh smart-mode compute
when the compute is reclaimed:

- `0` — first interruption is terminal. Cheapest, lowest reliability.
- `1` — one retry. Default, good for most batch use.
- `2`+ — two or more retries. Use for long jobs that you'd rather have
finish even at the cost of two extra cold-starts.

### 14.2 Lifecycle when compute is reclaimed

The `compute.job.smartcompute-interrupted` webhook fires on **every**
interruption (recovered or terminal). The payload disambiguates with a
`willRetry` flag plus `retriesUsedOnInterrupt` / `maxRetriesOnInterrupt`
counters:

```json
{
  "type": "compute.job.smartcompute-interrupted",
  "data": {
    "jobId": "...",
    "status": "smartcompute-interrupted",
    "errorCode": "smartcompute_interrupted",
    "errorMessage": "...",
    "retriesUsedOnInterrupt": 1,
    "maxRetriesOnInterrupt": 2,
    "willRetry": true,
    "retryHistory": [
      {
        "attemptNumber": 1,
        "instanceId": "fuse-inst-0abc...",
        "startedRunningAt": "2026-05-20T14:01:23Z",
        "terminalAt":      "2026-05-20T14:23:11Z",
        "terminalStatus":  "smartcompute-interrupted",
        "errorCode":       "smartcompute_interrupted",
        "attemptedComputeCostUsdEstimate": "0.4567",
        "source": "agent"
      }
    ],
    "totalAttemptedComputeCostUsdEstimate": "0.4567"
  }
}
```

`retryHistory` and `totalAttemptedComputeCostUsdEstimate` fire on
EVERY terminal `compute.job.*` payload, not just smart-mode
interruptions — see [§14.5.1](#1451-retry_history-records) and
[§14.5.2](#1452-pre-billing-cost-vs-actual-billed-cost).

When `willRetry: true` the job's row flips back to `status='queued'` and
fresh smart-mode compute is launched (potentially in a different
location to maximize fulfillment chances). When
`willRetry: false` the job is terminal `failed` with
`error_code='smartcompute_interrupted_no_retries_left'`.

`retries_used_on_interrupt` on the job is bumped at the same time the
webhook fires, so customers polling `GET /api/compute/jobs/:jobId` see
the same counter the webhook payload carries — webhook integration is
a performance optimization, not a correctness requirement. See
[§14.5](#145-reading-retry-state-without-webhooks).

### 14.3 Pricing

`POST /api/compute/jobs/estimate` with `mode: "smart"` returns a
smart-mode quote that takes the worst-case smart-mode hourly rate
currently observed across every compute location the SKU is available
in (Spark Fuse picks the actual landing location at launch, so quoting
the high end is honest — real-world billed cost is typically lower).
`rate.billedPerHourUsd` is the final customer rate for the
requested mode — read it as-is. See [§2.8](#28-cost-estimation-pre-submit).

### 14.4 Trade-offs to consider before opting in

- **Capacity is currently thin for some SKUs.** See the callout at
the top of §14. Confirm the SKU you want is finding capacity in
your dev/test loop before you wire SmartCompute into a production
batch path.
- **Warm-pool affinity is unavailable.** Every smart job is a fresh
cold provision (~3 min) and does not honor `idleHoldSeconds` —
smart-mode compute is one-shot per job.
- **Interruption frequency depends on capacity.** For our usual SKUs
  a typical mid-day rate is single-digit % per job; off-peak it can
  be much lower. A multi-hour run is more likely to see at least one
  interruption than a 5-minute one.
- **Idempotency matters.** If your container is sensitive to being
  re-launched (no checkpoint, no resume logic), a smart-mode retry
  re-runs from scratch and you pay for both attempts. Use
  `maxRetriesOnInterrupt: 0` to opt out of retries, or design your
  container to checkpoint to `/output/` (see [§10](#10-resumable-jobs-and-checkpoints)
  for the pattern and [§14.6](#146-retry-safe-containers-output-persistence--the-checkpoint-pattern)
  for SmartCompute-specific details) so a retry can pick up where
  the prior attempt left off (the same `output.shareSyncPath` is
  reused across retries within the same `jobId`).

### 14.5 Reading retry state without webhooks

Webhooks ([§14.2](#142-lifecycle-when-compute-is-reclaimed)) are the
push-driven option; polling `GET /api/compute/jobs/:jobId` is the
equivalent pull-driven option and is fully sufficient for smart-mode
integration. The same retry counters are surfaced on the job:


| Field                       | Type          | Meaning                                                                                                                                                                                                      |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mode`                                       | string        | `"smart"` for smart-mode jobs; echoes the submit value.                                                                                                                                                      |
| `max_retries_on_interrupt`                   | number        | Retry budget set at submit time (the value you passed as `maxRetriesOnInterrupt`). Immutable.                                                                                                                |
| `retries_used_on_interrupt`                  | number        | How many retries have been consumed so far. `0` at submit; incremented atomically on every reclamation-driven re-queue.                                                                                |
| `status`                                     | string        | Cycles `queued → provisioning → running → smartcompute-interrupted → queued → …` on each retry, and lands terminal (`succeeded`, `failed`, `cancelled`) once the budget is exhausted or the container exits cleanly. |
| `error_code`                                 | string | null | While retries remain, transient `smartcompute_interrupted` is replaced by the next attempt's outcome. On terminal failure with no retries left, you'll see `smartcompute_interrupted_no_retries_left`.                       |
| `retry_history`                              | array         | Per-attempt audit trail. Append-only, ordered by attempt. Each element carries `attemptNumber` (1-indexed), `instanceId`, `startedRunningAt`, `terminalAt`, `terminalStatus`, `errorCode`, `attemptedComputeCostUsdEstimate`, `source`. See [§14.5.1](#1451-retry_history-records) below. |
| `total_attempted_compute_cost_usd_estimate`  | string | null | Rolling pre-billing sum of `retry_history[*].attemptedComputeCostUsdEstimate`. `null` until the first attempt completes. NOT an authoritative billed amount — see [§14.5.2](#1452-pre-billing-cost-vs-actual-billed-cost). |


A safe polling shape:

```bash
while true; do
  ROW=$(curl -s "$SPARK_HOST/api/compute/jobs/$JOB_ID" \
    -H "Authorization: Bearer $SPARK_TOKEN")
  STATUS=$(echo "$ROW" | jq -r '.status')
  USED=$(echo "$ROW" | jq -r '.retries_used_on_interrupt')
  MAX=$(echo "$ROW" | jq -r '.max_retries_on_interrupt')
  echo "[$(date +%H:%M:%S)] status=$STATUS retries=$USED/$MAX"
  case "$STATUS" in
    succeeded|failed|cancelled) break ;;
  esac
  sleep 5
done

echo "$ROW" | jq '{status, error_code, error_message, exit_code, retries_used_on_interrupt}'
```

The `status='smartcompute-interrupted'` window is short and may not show up on
a 5 s poll cadence — `status` flips back to `queued` for the retry
within ~1 s of the reclamation. Don't rely on `'smartcompute-interrupted'`
ever being observed by your poller; rely on the `retries_used_on_interrupt`
counter monotonically increasing instead.

#### 14.5.1 `retry_history` records

Each terminal transition (final-success, final-failure, cancel, AND
every interruption — whether recovered or budget-exhausted) appends a
record to `retry_history`. The array is bounded by
`maxRetriesOnInterrupt + 1` (one record per attempt that has ended),
so it stays short even for long-running smart jobs. Reading it is the
cheapest way to answer "what's the full story on this job?" in a
single request.

Per-record shape:

| Field                              | Type          | Meaning                                                                                                                                                            |
| ---------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `attemptNumber`                    | number        | 1-indexed. Equals `(retries_used_on_interrupt at attempt-start) + 1`; also `(index in array) + 1`. Carried explicitly so consumers don't depend on the array order. |
| `instanceId`                       | string | null | Spark Fuse instance id this attempt ran on. `null` for attempts that never bound to an instance (queue-time cancels, queue-insert failures).                                                                                                                                |
| `startedRunningAt`                 | string | null | ISO timestamp when the container reached `running`. `null` if the attempt never ran (e.g. cancelled while queued, provisioning-timeout failure).                   |
| `terminalAt`                       | string        | ISO timestamp when this attempt ended (terminal write OR re-queue point).                                                                                          |
| `terminalStatus`                   | string        | One of `succeeded`, `failed`, `smartcompute-interrupted`. (Never `cancelled` — when a cancel races a failure, the per-attempt history preserves the original `failed`, even though the top-level job status reports `cancelled`.) |
| `errorCode`                        | string | null | The `error_code` the attempt ended with. Mirrors the customer-facing values in [§12.2](#122-error-codes).                                                          |
| `attemptedComputeCostUsdEstimate`  | string        | Pre-billing estimate for THIS attempt's compute cost, formatted to 4 decimal places (e.g. `"0.4567"`). Computed as `cost_per_hour_usd × (terminalAt − startedRunningAt)` in hours; `"0.0000"` when the attempt never ran. |
| `source`                           | string        | Opaque diagnostic string identifying which subsystem wrote the record. Useful for operator dashboards / debugging; customers can ignore. |

Example mid-retry-budget row (after one recovered interruption):

```json
{
  "retries_used_on_interrupt": 1,
  "max_retries_on_interrupt": 2,
  "status": "provisioning",
  "retry_history": [
    {
      "attemptNumber": 1,
      "instanceId": "inst-0abc...",
      "startedRunningAt": "2026-05-20T14:01:23Z",
      "terminalAt":      "2026-05-20T14:23:11Z",
      "terminalStatus":  "smartcompute-interrupted",
      "errorCode":       "smartcompute_interrupted",
      "attemptedComputeCostUsdEstimate": "0.4567",
      "source": "agent"
    }
  ],
  "total_attempted_compute_cost_usd_estimate": "0.4567"
}
```

#### 14.5.2 Pre-billing cost vs actual billed cost

`total_attempted_compute_cost_usd_estimate` (and per-attempt
`attemptedComputeCostUsdEstimate`) are PRE-BILLING ESTIMATES surfaced
for visibility while a job is in flight. They are computed
deterministically from the same pricing data the pre-submit
`/api/compute/jobs/estimate` endpoint reads, so a quote returned at
submit time and the running estimate during retries should be in the
same order of magnitude.

For authoritative billed cost — what hits your invoice — a separate
billing entry is recorded on your balance ledger after the job lands
terminal. Use the balance ledger (not these estimate fields) for any
reconciliation, accounting, or revenue-share logic.

Webhook receivers also see both fields on every terminal
`compute.job.*` payload (`retryHistory` + `totalAttemptedComputeCostUsdEstimate`,
camelCased to match webhook payload conventions).

### 14.6 Retry-safe containers: `/output/` persistence + the checkpoint pattern

Spark Fuse exposes `/output/` inside your container as a **live
mount** of the job's ShareSync output collection.
Files your container writes are flushed back to ShareSync
incrementally — there is no "all-at-end upload" phase to wait on,
no separate post-exit upload window to budget, and no scratch tier
between `/output/` and ShareSync. The same path is mounted on every
retry attempt of the same `jobId`, so partial work from a reclaimed
attempt is visible to the next attempt under the same paths
**immediately at container start** (the next attempt's `/output/`
already contains the previous attempt's files; nothing to download,
nothing to seed).

How the mount behaves in practice:

- Write latency to `/output/` is local-disk fast — writes are
  absorbed synchronously by a local cache and flushed back to
  ShareSync asynchronously on a short interval (around 10 s).
- The local cache is sized per-job (a few GiB by default). If your
  workload writes faster than the cache can flush back to ShareSync,
  Spark Fuse applies **backpressure** — `write()` syscalls block —
  rather than silently dropping data. A container that's blocked on
  backpressure looks no different from one that's blocked on slow
  local-disk I/O.
- Read latency is roughly the same as `/input/`: files already
  present in the cache (e.g. ones the same container just wrote)
  return at local-disk speed; cold reads fetch from ShareSync once
  and then cache.

**End-of-job drain.** When your container exits (zero or non-zero)
or its compute is reclaimed in SmartCompute mode, Spark Fuse waits
up to a short drain window (around a minute) for any in-flight
writes to land in ShareSync before reporting the terminal status.
Three observable outcomes:

1. **Clean drain:** the terminal status fires with no
   `output_*_failed` code on it. All files visible to your
   container at exit are visible on ShareSync.
2. **Drain budget exhausted:** the terminal status carries
   `error_code='output_drain_failed'` (or it's appended to an
   existing terminal `error_code` if the container also exited
   non-zero — e.g. `container_nonzero_exit+output_drain_failed`).
   Some `/output/` files may not have reached ShareSync. The
   underlying status is preserved so you can still distinguish
   "container succeeded but upload was incomplete" from "container
   failed and upload was incomplete".
3. **Unmount failure:** rare — the per-job `/output/` mount couldn't
   be released after the container exited. Surfaces as
   `error_code='output_unmount_failed'` (appended, like
   `output_drain_failed`). Not customer-actionable; the underlying
   container status (succeeded / failed) is preserved so the
   appended code is purely informational.

**Pre-job mount failure.** If the `/output/` mount can't be brought
up in the first place, the container never runs and the job lands
terminal with `error_code='output_mount_failed'`. Semantically
equivalent to a provisioning fault; SmartCompute retry budgets do
not treat this as a reclamation (no retry credit consumed).

#### Retry-safe containers

Because smart-mode retries re-launch your container from scratch (no
in-place resume, no container state preservation), the only durable
state between attempts is what your container has already written to
`/output/`. The `output.shareSyncPath` is identical across retries
within the same `jobId`, so the previous attempt's partial work is
visible to the next attempt under the same paths.

A minimal retry-safe wrapper pattern:

```bash
#!/usr/bin/env bash
# Wrap your real workload so partial progress survives mid-run reclamation.
set -euo pipefail

CKPT=/output/.spark-checkpoint.json

if [[ -f "$CKPT" ]]; then
  echo "Resuming from checkpoint: $(cat "$CKPT")"
  RESUME_FROM=$(jq -r '.next_frame' "$CKPT")
else
  echo "Fresh start (no prior checkpoint)"
  RESUME_FROM=0
fi

# Your workload: long batch render, frame loop, etc.
for ((i = RESUME_FROM; i < 100; i++)); do
  python3 render.py --frame "$i" --out "/output/frame-$i.png"
  echo "{\"next_frame\": $((i+1))}" > "$CKPT"
done

rm -f "$CKPT"
echo "All frames complete."
```

How this plays with smart-mode:

1. First attempt: container starts, no `$CKPT` file exists, renders
   frames 0..N, gets reclaimed at frame N+1. Frame 0..N PNGs and a
   `$CKPT` pointing at N+1 are already on ShareSync (flushed
   incrementally by the live `/output/` mount).
2. Spark Fuse re-queues the job (status flips back to `queued`,
   `retries_used_on_interrupt` bumps), allocates fresh smart-mode
   compute (~3 min), and re-runs your `command`.
3. New container starts, sees `$CKPT`, resumes from frame N+1.
4. If a second reclamation hits before the run finishes and your
   `maxRetriesOnInterrupt` budget is exhausted, the job lands terminal
   `failed` with `error_code='smartcompute_interrupted_no_retries_left'` —
   but the partial work in `/output/` from both attempts is still
   available for you to resume manually (e.g. by resubmitting with
   `mode='instant'` to finish the tail with no further reclamation
   risk).

**What `/output/` guarantees across retries:**

- The path is stable within a `jobId` (same `output.shareSyncPath`
  reused for every retry).
- `/output/` is a live mount of the ShareSync collection;
  files written by a reclaimed attempt are visible to the next
  attempt under the same paths **at container start** — no
  download phase, no scratch tier, no race window where a partial
  upload could be invisible to the resumer.
- Writes are flushed back to ShareSync asynchronously (default
  10s write-back interval) and on container exit via the drain
  step described above. You do not have to wait for the container
  to exit cleanly to see partial output on ShareSync.
- **Plan for partial files:** if your workload isn't naturally
  append-only / idempotent, use a checkpoint sentinel (as above)
  to disambiguate "this file is from a prior attempt that may be
  partial" from "this is finished work". A file that was being
  written when the previous attempt was reclaimed may exist on
  ShareSync at truncated length; the checkpoint sentinel is your
  signal that the previous attempt actually finalized it.

### 14.7 Interruption-no-retries-left email notifications

When a smart-mode job lands terminal `failed` with
`error_code='smartcompute_interrupted_no_retries_left'` (every
attempt was reclaimed and the configured `maxRetriesOnInterrupt`
budget is exhausted), Spark Fuse sends the submitting user an
email summarizing the run. The signal is "you have a job that
didn't get to finish and Spark Fuse won't auto-retry it again,
here are the details and what to do next".

What the email contains:

- The `jobId`, terminal status, and final `error_code`.
- `retries_used_on_interrupt` / `max_retries_on_interrupt` and the
  full `retry_history` summary (start/end timestamps + per-attempt
  cost-estimate from [§14.5](#145-reading-retry-state-without-webhooks)).
- A direct link to the job in the customer console.
- A link to the partial output collection on ShareSync (the
  `/output/` contents from every attempt, persisted across retries
  per [§14.6](#146-retry-safe-containers-output-persistence--the-checkpoint-pattern)).

Controlling the email per job:

Pass the optional `notifyOnFailure` field on submit to override the
default for a single job. The override only affects the event
described above — the `smartcompute_interrupted_no_retries_left`
terminal on a smart-mode job. In-container failures (nonzero
container exit, image-pull errors, etc.) do not email today regardless
of `notifyOnFailure`; the field also has no effect on `mode='instant'`
jobs because they can't hit this event in the first place.

- `notifyOnFailure: true` — force-send. The email fires even if
  your org's default is "off". Use when you want the email for a
  particular run even if your org's default is suppressed.
- `notifyOnFailure: false` — force-suppress. The email does not
  fire even if your org's default is "on". Use for batch
  submissions where you're already polling for terminal state and
  don't want one email per job.
- Omit the field (or pass `null`) — defer to your org's default.
  This is the existing behaviour.

The resolved value is echoed back on the submit response (as
`notifyOnFailure`) and on every GET / list / cancel response (as
`notify_on_failure`, the snake_case alias) so you can confirm
what was stored.

### 14.8 Smart-mode end-to-end (curl)

A short walkthrough mirroring [§11](#11-end-to-end-example) but for
`mode='smart'`. Assumes `$SPARK_HOST` and `$SPARK_TOKEN` are set.

```bash
# ---- 1. Submit a smart job with two-retry budget ---------------------
JOB=$(curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "ghcr.io/your-org/batch-renderer:v3.2",
    "command": ["bash", "/app/render-with-checkpoint.sh"],
    "instanceType": "g7e.2xlarge",
    "mode": "smart",
    "maxRetriesOnInterrupt": 2
  }')
JOB_ID=$(echo "$JOB" | jq -r '.jobId')
echo "Submitted smart-mode job: $JOB_ID"

# ---- 2. Poll status + retry counter ---------------------------------
while true; do
  ROW=$(curl -s "$SPARK_HOST/api/compute/jobs/$JOB_ID" \
    -H "Authorization: Bearer $SPARK_TOKEN")
  STATUS=$(echo "$ROW" | jq -r '.status')
  USED=$(echo "$ROW" | jq -r '.retries_used_on_interrupt')
  MAX=$(echo "$ROW" | jq -r '.max_retries_on_interrupt')
  printf '[%s] status=%s retries=%s/%s\n' "$(date +%H:%M:%S)" \
    "$STATUS" "$USED" "$MAX"
  case "$STATUS" in
    succeeded|failed|cancelled) break ;;
  esac
  sleep 5
done

# ---- 3. Final outcome -----------------------------------------------
echo "$ROW" | jq '{
  status, exit_code, error_code, error_message,
  retries_used_on_interrupt, max_retries_on_interrupt, terminal_at
}'
```

Three possible terminal outcomes for the above:

- `status='succeeded'`, `error_code=null`, `retries_used_on_interrupt=0..2`
  — the container completed (possibly after one or two reclamation-driven
  re-launches that resumed from the checkpoint).
- `status='failed'`, `error_code='smartcompute_interrupted_no_retries_left'`,
  `retries_used_on_interrupt=2` — every attempt was reclaimed; partial
  output is still on ShareSync; consider resubmitting tail work with
  `mode='instant'`.
- `status='failed'`, `error_code` = anything else — same failure modes
  as instant-mode (image pull, container nonzero exit, etc.); not
  reclamation-related. Diagnose via `error_message`.

---

*Last updated 2026-05-22.*