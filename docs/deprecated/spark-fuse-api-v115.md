# Spark Fuse API — customer guide (v1.15)

**Audience:** developers integrating with the Spark Fuse REST API for the
first time. Walks through getting an auth token, submitting your first
compute job, uploading input data, streaming logs, polling status,
cancelling, listing past jobs, and pulling outputs back from ShareSync.

**Status:** Spark Fuse v1.15 is the current release (shipped 2026-05-15;
tracks the v1.0 GA from 2026-05-01 and the v1.1 SmartCompute /
cost-estimation drop). API surface is stable for the v1 series — future
minor versions will add endpoints + fields but won't remove anything
documented here. Operational caveats specific to v1.15 are flagged
inline (see e.g. the SmartCompute capacity note at the top of [§13](#13-smartcompute-mode-smart)).

---

## 0. The 30-second mental model

A Spark Fuse job is **a Docker image + a command, run on a Spark Fuse
GPU you don't have to manage**.

You submit a job (`POST /api/compute/jobs`) with the image, the command,
and the instance type SKU you want. The server:

1. Picks stopped warm-pool compute of the right shape (or reshapes a
  warm member that's close), starts it, and pulls your image.
2. Runs your command inside the container with `/output/` mounted to a
  ShareSync-backed directory and (optionally) `/input/` mounted read-only
   from a folder you uploaded or pre-populated.
3. Streams logs back to you while it runs.
4. Uploads anything you wrote to `/output/` to the ShareSync path of your
  choice.
5. Stops the compute when it's done (with an optional idle-hold window if
  you'll want another job to land on the same warm compute with no
   cold-start).

Auth is a Spark bearer token. You get one from `POST /api/auth/login`
once and reuse it for every API call until it expires. The same token
also authenticates against ShareSync (the WebDAV server hosting your
inputs and outputs) — no token-exchange dance needed.

Base URL: `https://api.prod.aapse1.sparkcloud.studio` (production). The rest
of this guide assumes you've exported it:

```bash
export SPARK_HOST="https://api.prod.aapse1.sparkcloud.studio"
```

---

## 0.1 Compute modes: InstantCompute (`mode='instant'`) and SmartCompute (`mode='smart'`)

Spark Fuse has two compute modes you pick between on each job submit:

- **InstantCompute (`mode='instant'`, default).** The job lands on a
warm GPU within a couple of seconds — or, if the warm pool is cold
for your SKU, ~3 min while the platform brings a fresh GPU up. Once
running, the job never gets interrupted mid-flight. `idleHoldSeconds`
and next-job-on-same-warm-compute affinity ([§11.3](#113-idle-hold--warm-pool))
are InstantCompute-only.
- **SmartCompute (`mode='smart'`).** The job runs on preemptible
Spark Fuse capacity that's typically ~60% cheaper, but the platform
may reclaim the underlying compute mid-job with a short warning.
Every submit is a fresh cold-start in v1.1 — no warm pool. See
[§13](#13-smartcompute-mode-smart) for the deep-dive (retry budget,
preemption webhook, trade-offs). **v1.15 caveat:** SmartCompute
capacity in v1.15 is sourced from only two data centers, so
availability for some GPU SKUs can be extremely limited — see the
callout at the top of §13. Spark Fuse v1.2 expands this by
2026-05-22.

The rest of this guide assumes InstantCompute unless a section flags
SmartCompute specifically.

---

## 1. Authentication: getting a bearer token

Spark issues bearer tokens through the platform's own login endpoint —
`POST /api/auth/login`. You authenticate with your Spark email + password
and get back a JSON response carrying the token to send on every
subsequent API call. **Do not call any underlying identity-provider
endpoints directly**; the platform endpoint is the only supported
auth surface and the only one that survives provider migrations.

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

There is **no separate refresh endpoint** in v1 — the login call is
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
| `image`                    | string   | yes      | Container image reference (with optional tag or digest). Public Docker Hub and (configured) private registries are supported. Pinning by digest is recommended for reproducibility.                                                                                                                                                                                                                                            |
| `command`                  | string[] | yes      | The command to run inside the container. First element is the binary; rest are arguments.                                                                                                                                                                                                                                                                                                                                      |
| `instanceType`             | string   | yes      | Instance type SKU (e.g. `g7e.2xlarge`). Must be in the current allow-list — see [§8](#8-browsing-eligible-instance-types).                                                                                                                                                                                                                                                                                                     |
| `env`                      | object   | no       | Customer environment variables. `SPARK_*` keys are reserved by the platform and rejected.                                                                                                                                                                                                                                                                                                                                      |
| `startScriptB64`           | string   | no       | Base64-encoded bash script that runs after image pull and before `command`. Useful for cache-warming, light setup.                                                                                                                                                                                                                                                                                                             |
| `availabilityRegion`       | string   | no       | Region code (e.g. `us-east-1`). v1 supports `us-east-1` only. Defaults to `us-east-1`.                                                                                                                                                                                                                                                                                                                                         |
| `mode`                     | string   | no       | `instant` (warm-pool guaranteed-availability, the default) or `smart` (preemptible — opt-in, typically ~60% cheaper than `instant` but subject to platform reclamation; see [§13](#13-smartcompute-mode-smart)).                                                                                                                                                                                                               |
| `maxRetriesOnInterrupt`    | number   | no       | (`mode='smart'` only) How many times the platform may re-launch this job on fresh smart-mode compute if the platform reclaims the underlying compute mid-run. Default `1`; range `[0, 5]`. `0` = first interruption is terminal. Each retry consumes one tick from the budget; when the budget is exhausted, the job lands terminal `failed` with `errorCode='spot_interrupted_no_retries_left'`. Ignored on `mode='instant'`. |
| `outputShareSyncPath`      | string   | no       | ShareSync logical path the platform writes outputs to. Defaults to `/Spark Fuse Jobs/{jobId}/`.                                                                                                                                                                                                                                                                                                                                |
| `outputShareSyncSpaceName` | string   | no       | Name of a ShareSync Space (project) to write outputs into. Omit to use your Personal space. The platform validates the space exists at submission time and rejects with HTTP 400 if it doesn't.                                                                                                                                                                                                                                |
| `inputShareSyncPath`       | string   | no       | Mount workflow: a ShareSync path you've already populated with input files. Server PROPFINDs at submit time and rejects HTTP 400 if unreachable. Mutually exclusive with `inputPushMode`. See [§3.2](#32-mount-workflow-pre-populated-input-folder).                                                                                                                                                                           |
| `inputShareSyncSpaceName`  | string   | no       | Name of the ShareSync Space containing your input path (or, with `inputPushMode='auto-prepare'`, the space the server will allocate the input folder under). Omit to use your Personal space.                                                                                                                                                                                                                                  |
| `inputPushMode`            | string   | no       | `auto-prepare` triggers the push workflow: the server allocates an input folder, returns a one-shot upload curl, and the agent downloads + extracts before running your command. See [§3.1](#31-push-workflow-auto-prepare).                                                                                                                                                                                                   |
| `webhookEndpointId`        | string   | no       | UUID of a webhook endpoint to fire job-state events to. Manage endpoints via `/api/compute/webhooks/*` (separate doc).                                                                                                                                                                                                                                                                                                         |
| `idleHoldSeconds`          | number   | no       | (`mode='instant'` only) After the job reaches a terminal state (succeeded / failed), how long the compute should idle before stopping. During the hold, your org's next job lands on this same warm compute with no cold-start. Defaults to a parameter-map value (~600 s today); range `[0, max]`. **Cancel skips the hold** — cancelling drops the compute immediately.                                                      |
| `shmSize`                  | string   | no       | Override the container's `/dev/shm` size. Positive integer optionally suffixed with lowercase `k`, `m`, or `g`; bare integer = bytes; max `32g`. Defaults to `2g` (platform default since 2026-05-18 — was 64 MB previously, which bus-errored any PyTorch DataLoader with `num_workers > 0`). Increase past `2g` only for very large batches / high resolutions that genuinely need more shared memory between DataLoader workers. Examples: `"4g"`, `"8g"`, `"512m"`, `"16g"`.                                                                                                                                       |


### 2.3 Common errors


| HTTP                                                              | Meaning                                                                                                                                                                   |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 400 `instanceType ... is not eligible`                            | The SKU isn't in the current allow-list — call `GET /api/compute/skus` for the live list.                                                                                 |
| 400 `Output ShareSync space ... not found`                        | The space name you supplied doesn't exist or you don't have write access.                                                                                                 |
| 400 `Input ShareSync path ... not reachable`                      | (mount workflow) The path you supplied couldn't be PROPFINDed with your token. Check spelling + permissions.                                                              |
| 400 `inputShareSyncPath and inputPushMode are mutually exclusive` | Pick one.                                                                                                                                                                 |
| 403                                                               | Token invalid or for a different org.                                                                                                                                     |
| 503                                                               | All warm members are busy AND no eligible warm member could be reshaped in time. Retry after a few seconds. (Rare; see the operator-side autoscaler item in our roadmap.) |


### 2.8 Cost estimation (pre-submit)

`POST /api/compute/jobs/estimate` returns a cost quote for a given SKU
without actually submitting a job. Useful for pre-flight UI ("this run
will cost about $X"). Available in v1.1.

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
  "availabilityRegion": "us-east-1",
  "mode": "instant",
  "rate": {
    "costPerHourUsd": "0.79333333",
    "markupMultiplier": 1.5,
    "billedPerSecondCents": "0.03305555",
    "billedPerHourUsd": "1.19"
  },
  "estimate": {
    "billableSeconds": 4200,
    "totalCents": "138.83",
    "totalUsd": "1.39"
  },
  "notes": [
    "Quoted rate is instant-mode only; smart-mode quotes are surfaced ...",
    "Markup multiplier 1.5x is the v1.0 scaffold value; subject to change ...",
    "Linux pricing only (compute v1 is Linux-container-only).",
    "Quotes are estimates — actual billed cost is computed from ..."
  ]
}
```

Body fields:


| Field                     | Type   | Required | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `instanceType`            | string | yes      | Same allow-list as `POST /api/compute/jobs`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `availabilityRegion`      | string | no       | Default `us-east-1` (the only quotable region in v1.1).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `mode`                    | string | no       | `instant` (default) or `smart`. Smart-mode quotes use the worst-case smart-mode hourly rate currently observed across every region/zone the SKU is available in (the platform picks the actual landing location at launch time, so quoting the high end is honest). Real-world billed cost is typically lower. The smart-mode markup is already baked into the quoted rate; `markupMultiplier` is reported as `1.0` on smart-mode responses to reflect this. If smart-mode pricing data isn't yet observed for the SKU (rare; happens for a brief window after a new SKU is added to the catalog), smart-mode falls back to a discount-against-instant approximation and `notes[]` flags the fallback. |
| `estimatedRuntimeSeconds` | number | no       | Optional. When omitted, response carries rate-only output (no `estimate` block).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `idleHoldSeconds`         | number | no       | Optional. Used together with `estimatedRuntimeSeconds`; total = (runtime + hold) × rate.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |


Common errors:


| HTTP                                   | Meaning                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| 400 `instanceType ... is not eligible` | Same allow-list gate as job submission.                                        |
| 404                                    | No pricing row for that SKU in the requested region — data gap; please report. |


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
    "shareSyncBaseUrl": "https://your-org.files.sparkcloud.studio/dav/spaces/<space-id>/Compute%20Job%20Inputs/<jobId>/",
    "uploadUrl": "https://your-org.files.sparkcloud.studio/dav/spaces/<space-id>/Compute%20Job%20Inputs/<jobId>/spark-input.tar.gz",
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

The agent waits up to 5 minutes for the upload to land. Once it does, the
agent downloads the tarball, extracts it to `/input/`, and starts your
container. Inside the container, your files are at `/input/...` exactly as
you laid them out under the directory you ran `tar` from.

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

The server PROPFINDs the path with your token at submit time. If the path
isn't reachable, you get HTTP 400 immediately — no half-started job. If
it is reachable, the job queues and the agent downloads everything under
that path into `/input/` before running your command.

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

Server-Sent Events. Stays open while the job runs and pushes new lines as
the agent ingests them. Closes when the job reaches a terminal state.

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

The `queue.status` frames (v1.1) are emitted while your job is still in
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

> **Heads up:** v1 streams from the **moment you connect** — there is no
> historical replay of logs that landed before the SSE connection opened.
> If you want logs from job start, connect immediately after submission.
> Resumable replay via `Last-Event-Id` is on the v1.1 roadmap.

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
  "availability_region": "us-east-1",
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
  ↓ (queue processor picks the job; compute starts; image pulls)
provisioning
  ↓ (docker run dispatched)
running
  ↓ (container exits OR agent kills it on cancel/timeout)
succeeded | failed | cancelled | spot-interrupted
```

`succeeded` means the container exited with code 0 (and outputs uploaded
without a fatal upload error). `failed` carries an `error_code` +
`error_message` describing why; common values are documented in
[§11.2](#112-error-codes).

> **Tip:** poll on a short interval while the job is non-terminal. 5
> seconds is plenty — most state transitions happen on a 10 s heartbeat
> cadence on the server side. For a more interactive experience, use the
> SSE log stream (§4) instead — it pushes log lines as they arrive and
> closes when the job ends.

---

## 6. Listing your jobs

`GET /api/compute/jobs`

Returns all `compute_jobs` rows for **your organisation** (not just your
user). Useful for dashboards and bulk status views.

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

> **Pagination:** v1 returns the full list. If you have many jobs and want
> server-side pagination, file a request — it's on the v1.x backlog.

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
> window. The compute is released as soon as the agent acknowledges the
> cancel.

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

The list is driven from a server-side parameter (`compute_eligible_skus`)
and is updated as the platform expands GPU coverage.

> **Heads up — catalog vs. capacity.** The SKUs on this list are the ones
> we'll **accept** in submission. They are not a guarantee that capacity
> exists for every SKU at every moment. v1.1 will surface "no capacity for
> this SKU right now" as a distinct `error_code` separate from
> `start_timeout`. If a submission you'd expect to succeed times out
> waiting on capacity, the v1.1 error-code split will tell you so directly;
> for now, retry with `g6` / `g7e` family alternates.

---

## 9. Retrieving outputs from ShareSync

When your container writes to `/output/`, the agent uploads everything to
the ShareSync path on the job row. Pull the files back via the same
WebDAV server using your Spark bearer.

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
# Pseudocode — use a WebDAV client like `cadaver` or `rclone` for full sync.
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
> `output.shareSyncPath` (we store the resolved path on the row at
> submission time, so existing jobs are unaffected). Only **new**
> submissions with `outputShareSyncPath` omitted will use the new
> default. If you've been passing an explicit `outputShareSyncPath` on
> every submit, this change has no effect on you.

### 9.4 Log archive

In addition to outputs, the per-job log file (everything you'd see on the
SSE stream, plus agent provisioning logs) is uploaded to
`log_archive_share_sync_path`. Pull it the same way — prefer reading the
exact path from the `log_archive_share_sync_path` field on the job rather
than constructing the URL by hand, since the filename prefix changed
2026-05-13 (`spark-compute-<jobId>.log` → `spark-fuse-<jobId>.log`) and
past jobs still carry the old name verbatim on their row:

```bash
curl -O \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  "$SHARESYNC_BASE_URL/spark-fuse-$JOB_ID.log"
```

---

## 10. End-to-end example

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

## 11. Reference

### 11.1 Job status values


| `status`           | Meaning                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `queued`           | Submitted; waiting for compute to be picked.                                                 |
| `provisioning`     | Compute starting + image pulling.                                                            |
| `running`          | `docker run` dispatched; your command is executing.                                          |
| `succeeded`        | Container exited with code 0; outputs uploaded successfully.                                 |
| `failed`           | Something went wrong. Inspect `error_code` and `error_message`.                              |
| `cancelled`        | You called `POST /api/compute/jobs/:jobId/cancel` and the agent honored it.                  |
| `spot-interrupted` | (`mode='smart'` only) The platform reclaimed the smart-mode compute before the job finished. |


`succeeded`, `failed`, `cancelled`, `spot-interrupted` are **terminal** —
the job will not transition again.

### 11.2 Error codes

When `status='failed'`, `error_code` carries one of the following values.
`error_message` adds human-readable context.


| `error_code`                       | Cause                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image_pull_failed`                | The image reference wasn't pullable (typo, registry auth, network).                                                                                                                                                                                                                                                                                                                                                                   |
| `disk_full`                        | The compute ran out of disk during image pull or container run.                                                                                                                                                                                                                                                                                                                                                                       |
| `input_download_failed`            | (auto-prepare) You didn't upload to the input URL within 5 minutes, or the upload was malformed.                                                                                                                                                                                                                                                                                                                                      |
| `output_upload_failed`             | Outputs couldn't be uploaded to ShareSync. The container may have produced valid outputs that simply didn't make it back.                                                                                                                                                                                                                                                                                                             |
| `container_nonzero_exit`           | Your command exited with a non-zero status. `exit_code` field carries the exit code.                                                                                                                                                                                                                                                                                                                                                  |
| `cancelled`                        | (legacy internal label — `status` is projected to `cancelled` for cancel-driven failures, see §11.1.)                                                                                                                                                                                                                                                                                                                                 |
| `agent_silent`                     | The agent stopped reporting back (OS crash, kernel panic, etc.). The watchdog drove the job to terminal and stopped the compute.                                                                                                                                                                                                                                                                                                      |
| `start_timeout`                    | The compute was assigned and started, but never reached `running` within the configured budget — agent stuck, image pull stuck, kernel hang, etc. Distinct from `no_capacity` (capacity-driven). Escalate or retry.                                                                                                                                                                                                                   |
| `no_capacity`                      | The platform retried for the full provisioning window and never found a warm member of the requested SKU shape — every eligible candidate was busy and the underlying infrastructure had no capacity for a fresh allocation on every attempt. Retry shortly, or pick a different SKU from `GET /api/compute/skus`.                                                                                                                    |
| `spot_interrupted`                 | (`mode='smart'` only) The platform received a preemption notice and terminated the running container before the job finished. Surfaced as `status='spot-interrupted'`. If `maxRetriesOnInterrupt > 0` and the budget is not yet exhausted, the platform automatically re-queues the job on fresh smart-mode compute instead of going terminal — so you'll typically only see this terminal `error_code` when the budget is exhausted. |
| `spot_interrupted_no_retries_left` | (`mode='smart'` only) The platform preempted the smart-mode compute AND the configured `maxRetriesOnInterrupt` budget has been exhausted. Distinct from `spot_interrupted` so you can pattern-match on "we ran out of retries" specifically. The job is terminal `failed`; resubmit with a higher `maxRetriesOnInterrupt`, switch to `mode='instant'`, or both.                                                                       |


### 11.3 Idle-hold + warm-pool

After your job ends, the compute can stay running for `idleHoldSeconds`
before the platform stops it. During that window, your org's next
submission preferentially lands on this same warm compute — no cold-start
image pull, no provisioning latency, and it's the same `instance_type` so
no reshape is needed. (Idle-hold is InstantCompute-only; SmartCompute
jobs don't honor `idleHoldSeconds`.)

If you submit two back-to-back jobs and want zero cold-start on the
second:

1. Submit job 1 with `idleHoldSeconds: 600` (or whatever covers the gap).
2. Wait for job 1 to reach a terminal state (`status='succeeded'` or
  `'failed'`).
3. Submit job 2 with the same `instanceType`. Within ~1–3 seconds it lands
  on the same warm compute.

### 11.4 Submit response shape (`CreateComputeJobResponse`)

```typescript
{
  jobId: string;            // compute_jobs.id (UUID)
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

  // v1.1 (d): queue position + ETA. Approximate; treat as
  // order-of-magnitude hint, not an SLA.
  queuePosition?: number | null;        // 0=next-up, N>0=N ahead, null=N/A
  estimatedStartSeconds?: number | null;  // queuePosition * average provisioning time
}
```

### 11.5 Get-job / list / cancel response shape (`ComputeJobApiShape`)

The full `compute_jobs` row plus `output` and `input` URL blocks (and,
in v1.1, the same `queuePosition` / `estimatedStartSeconds` pair as the
submit response — both `null` once the job is past `provisioning`). See
[§5](#5-polling-job-status) for an annotated example.

### 11.6 Endpoint reference


| Method | Path                                   | Description                                                                  |
| ------ | -------------------------------------- | ---------------------------------------------------------------------------- |
| `POST` | `/api/compute/jobs`                    | [Submit a job](#2-submitting-a-compute-job).                                 |
| `GET`  | `/api/compute/jobs`                    | [List jobs for your org](#6-listing-your-jobs).                              |
| `GET`  | `/api/compute/jobs/:jobId`             | [Get a single job](#5-polling-job-status).                                   |
| `POST` | `/api/compute/jobs/:jobId/cancel`      | [Cancel a job](#7-cancelling-a-job).                                         |
| `GET`  | `/api/compute/jobs/:jobId/logs/stream` | [SSE log stream](#4-streaming-job-logs-sse).                                 |
| `GET`  | `/api/compute/skus`                    | [List eligible instance types](#8-browsing-eligible-instance-types).         |
| `POST` | `/api/compute/jobs/estimate`           | [Get a cost quote before submitting](#28-cost-estimation-pre-submit) (v1.1). |


### 11.7 Spark Fuse is for Linux containers only (v1)

v1 runs Linux Docker containers on Linux hosts. There is no Windows-
compute path in v1; v2 will introduce one. The image you supply must be
linux/amd64 (or linux/arm64 for ARM-eligible SKUs once those are added).

### 11.8 Limits + quotas (v1)

- **Submission rate:** soft limit per org (parameter-map driven; default
~10 submissions/minute). Exceeding returns HTTP 429 with `Retry-After`
in v1.1.
- **Idle hold:** range `[0, max]`; the max is parameter-map driven
(currently in the low-thousands of seconds).
- **Image pull timeout:** ~15 min with retry. `disk_full` and
`image_pull_failed` are surfaced as terminal `error_code` values rather
than infinite retry.
- **Region:** v1 is `us-east-1` only.

### 11.9 Help, support, and feedback

- API issues, unexpected errors, missing features: ping your Spark
contact or open a ticket on the customer support portal.
- Documentation issues (this document): file a doc bug; we'd rather know.
- Roadmap visibility: ask about Spark Fuse v1.1 — that's the next
milestone and includes (among other things) full SmartCompute support,
fully-realized queue position + ETA in the SSE stream, the
cost-estimation endpoint, and force-cancel guarantees.

---

## 13. SmartCompute — `mode='smart'`

> **Capacity availability — v1.15 limitation, lifted in v1.2 by 2026-05-22.**
> In v1.15, SmartCompute is fulfilled from only two data centers, so
> availability for some GPU SKUs can be **extremely limited** at peak
> times. Practical consequences:
>
> - Submissions can sit in `queued` longer than instant-mode (sometimes
> several minutes before the platform finds matching capacity).
> - `error_code='no_capacity'` is materially more likely on scarce SKUs
> than it is for instant-mode against the same SKU. Retry with a
> sibling SKU from `GET /api/compute/skus`, or fall back to
> `mode='instant'` for that submission.
> - Smart-mode estimate quotes ([§2.8](#28-cost-estimation-pre-submit))
> still return rate data for any SKU with observed pricing, even when
> no capacity is currently available — a quote is not a guarantee
> capacity exists right now.
>
> Spark Fuse v1.2 (target 2026-05-22) expands SmartCompute capacity to
> additional data centers and is expected to alleviate this; the API
> contract for `mode='smart'` does not change in v1.2, so anything you
> build against this section will continue to work once capacity
> widens.

`mode='smart'` runs your job on preemptible Spark Fuse capacity
instead of guaranteed-availability warm-pool compute. Compared to
`mode='instant'`:


|                       | `instant` (default)                                | `smart`                                                                           |
| --------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- |
| Underlying capacity   | Warm-pool guaranteed-availability                  | Preemptible (one-shot)                                                            |
| Typical price         | full instant-mode rate                             | typically ~60% off instant-mode (varies; quoted via the cost-estimation endpoint) |
| Cold-start            | ~2 s on warm hit; ~3 min if reshape                | ~3 min cold provisioning every time (no warm pool in v1.1)                        |
| Idle-hold (warm-keep) | yes                                                | no                                                                                |
| Affinity routing      | yes (org-level next-job-on-same-warm-compute)      | no                                                                                |
| Mid-run interruption  | never (modulo agent crash)                         | possible — the platform reclaims capacity with a 2-minute warning                 |
| Best fit              | low-latency / interactive runs / multi-job warmups | bulk batch, long-running training, anything tolerant to a re-launch               |


### 13.1 Submitting a smart job

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
times the platform may re-launch the job on fresh smart-mode compute
when the platform preempts:

- `0` — first interruption is terminal. Cheapest, lowest reliability.
- `1` — one retry. Default, good for most batch use.
- `2`+ — two or more retries. Use for long jobs that you'd rather have
finish even at the cost of two extra cold-starts.

### 13.2 Lifecycle when the platform preempts

The `compute.job.spot-interrupted` webhook fires on **every**
interruption (recovered or terminal). The payload disambiguates with a
`willRetry` flag plus `retriesUsedOnInterrupt` / `maxRetriesOnInterrupt`
counters:

```json
{
  "type": "compute.job.spot-interrupted",
  "data": {
    "jobId": "...",
    "status": "spot-interrupted",
    "errorCode": "spot_interrupted",
    "errorMessage": "Preemption notice: action=terminate ...",
    "retriesUsedOnInterrupt": 1,
    "maxRetriesOnInterrupt": 2,
    "willRetry": true
  }
}
```

When `willRetry: true` the job's row flips back to `status='queued'` and
the platform launches fresh smart-mode compute (potentially in a
different region/zone to maximize fulfillment chances). When
`willRetry: false` the job is terminal `failed` with
`error_code='spot_interrupted_no_retries_left'`.

`retries_used_on_interrupt` on the `compute_jobs` row is bumped at the
same time the webhook fires, so customers polling `GET /api/compute/jobs/:jobId`
see the same counter the webhook payload carries — webhook integration
is a performance optimization, not a correctness requirement. See
[§13.5](#135-reading-retry-state-without-webhooks).

### 13.3 Pricing

`POST /api/compute/jobs/estimate` with `mode: "smart"` returns a
smart-mode quote that takes the worst-case smart-mode hourly rate
currently observed across every region/zone the SKU is available in
(the platform picks the actual landing location at launch, so quoting
the high end is honest — real-world billed cost is typically lower).
The smart-mode markup is already baked into the quoted rate; the
response carries `markupMultiplier: 1.0` for smart-mode to reflect
this. See [§2.8](#28-cost-estimation-pre-submit).

### 13.4 Trade-offs to consider before opting in

- **Two-data-center capacity in v1.15.** See the callout at the top
of §13. Capacity for some SKUs is currently very thin — confirm
the SKU you want is finding capacity in your dev/test loop before
you wire SmartCompute into a production batch path. v1.2
(2026-05-22) is the relief date.
- **Warm-pool affinity is unavailable.** Every smart job is a fresh
cold provision (~3 min) and does not honor `idleHoldSeconds` —
smart-mode compute is one-shot per job in v1.1.
- **Interruption frequency depends on capacity.** For our usual SKUs
a typical mid-day rate is single-digit % per job; off-peak it can
be much lower. A multi-hour run is more likely to see at least one
preemption than a 5-minute one.
- **Idempotency matters.** If your container is sensitive to being
re-launched (no checkpoint, no resume logic), a smart-mode retry
re-runs from scratch and you pay for both attempts. Use
`maxRetriesOnInterrupt: 0` to opt out of retries, or design your
container to checkpoint to `/output/` (see [§13.6](#136-retry-safe-containers-the-checkpoint-pattern))
so a retry can pick up where the prior attempt left off (the same
`output_share_sync_path` is reused across retries within the same
`jobId`).

### 13.5 Reading retry state without webhooks

Webhooks ([§13.2](#132-lifecycle-when-the-platform-preempts)) are the
push-driven option; polling `GET /api/compute/jobs/:jobId` is the
equivalent pull-driven option and is fully sufficient for smart-mode
integration. The same retry counters are surfaced on the row:


| Field                       | Type          | Meaning                                                                                                                                                                                                      |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mode`                      | string        | `"smart"` for smart-mode jobs; echoes the submit value.                                                                                                                                                      |
| `max_retries_on_interrupt`  | number        | Retry budget set at submit time (the value you passed as `maxRetriesOnInterrupt`). Immutable.                                                                                                                |
| `retries_used_on_interrupt` | number        | How many retries the platform has consumed so far. `0` at submit; incremented atomically on every preemption-driven re-queue.                                                                                |
| `status`                    | string        | Cycles `queued → provisioning → running → spot-interrupted → queued → …` on each retry, and lands terminal (`succeeded`, `failed`, `cancelled`) once the budget is exhausted or the container exits cleanly. |
| `error_code`                | string | null | While retries remain, transient `spot_interrupted` is replaced by the next attempt's outcome. On terminal failure with no retries left, you'll see `spot_interrupted_no_retries_left`.                       |


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

The `status='spot-interrupted'` window is short and may not show up on
a 5 s poll cadence — the row flips back to `queued` for the retry
within ~1 s of the preemption notice. Don't rely on `'spot-interrupted'`
ever being observed by your poller; rely on the `retries_used_on_interrupt`
counter monotonically increasing instead.

### 13.6 Retry-safe containers: the checkpoint pattern

Because smart-mode retries re-launch your container from scratch (no
in-place resume, no container state preservation), the only durable
state between attempts is what your container has already written to
`/output/`. The `output_share_sync_path` is identical across retries
within the same `jobId`, so the previous attempt's partial work is
visible to the next attempt under the same paths.

A minimal retry-safe wrapper pattern:

```bash
#!/usr/bin/env bash
# Wrap your real workload so partial progress survives preemption.
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
  frames 0..N, gets preempted at frame N+1. Frame 0..N PNGs and a
   `$CKPT` pointing at N+1 are already in `/output/` (uploaded
   incrementally by the agent's continuous upload loop).
2. Platform receives the preemption, re-queues the job (status flips
  back to `queued`, `retries_used_on_interrupt` bumps), allocates
   fresh smart-mode compute (~3 min), and re-runs your `command`.
3. New container starts, sees `$CKPT`, resumes from frame N+1.
4. If a second preemption hits before the run finishes and your
  `maxRetriesOnInterrupt` budget is exhausted, the job lands terminal
   `failed` with `error_code='spot_interrupted_no_retries_left'` —
   but the partial work in `/output/` from both attempts is still
   available for you to resume manually (e.g. by resubmitting with
   `mode='instant'` to finish the tail with no further preemption
   risk).

**What `/output/` guarantees across retries:**

- The path is stable within a `jobId` (same
`output_share_sync_path` row column reused for every retry).
- The agent uploads `/output/` incrementally during the run; you do
not have to wait for the container to exit cleanly to see partial
output on ShareSync.
- Files written by a preempted attempt are visible to the next
attempt under the same paths. **Plan for this:** if your workload
isn't naturally append-only / idempotent, use a checkpoint sentinel
(as above) to disambiguate "this file is from a prior attempt that
may be partial" from "this is finished work".

### 13.7 Smart-mode end-to-end (curl)

A short walkthrough mirroring [§10](#10-end-to-end-example) but for
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
— the container completed (possibly after one or two preemption-driven
re-launches that resumed from the checkpoint).
- `status='failed'`, `error_code='spot_interrupted_no_retries_left'`,
`retries_used_on_interrupt=2` — every attempt got preempted; partial
output is still on ShareSync; consider resubmitting tail work with
`mode='instant'`.
- `status='failed'`, `error_code` = anything else — same failure modes
as instant-mode (image pull, container nonzero exit, etc.); not
preemption-related. Diagnose via `error_message`.

---

*Last updated 2026-05-15 for Spark Fuse v1.15 (current release). Tracks
API v1.0 + v1.1 (q) SmartCompute + v1.15 (b) smart-mode estimate
pricing. Includes the v1.15 SmartCompute capacity caveat (§0.1 / §13 /
§13.4); the v1.2 capacity expansion targeted for 2026-05-22 will lift
that caveat without changing the `mode='smart'` API contract.
Vocabulary harmonized with the ComfyUI headless quickstart
(compute/GPU/warm-member language; no underlying-provider references).*