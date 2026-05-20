Quickstart: Headless ComfyUI on Spark Fuse
Audience: ComfyUI users moving from a local workstation (or a hosted Windows desktop) to running Comfy as a service on Spark Fuse. You already know your way around a ComfyUI workflow JSON; you don't need the GUI on the remote machine. You want to fire a workflow off from your concept / storyboard / animatic app, have it run on a Spark GPU, and read the output images back into your app as they're produced.
Total setup time is less than an hour with a working pipeline: auth → first smoke-test job → first real ComfyUI job → reading outputs back into your app. We've seen some early adopters burn through setup in less than five minutes.
Companion docs:
spark-compute-api.md: full Spark Fuse API reference. This quickstart skips most of the detail there; flip to the reference when you want the complete picture on a specific endpoint.

0. The 30-second mental model
ComfyUI on Spark Fuse is: submit one API call → the GPU picks up your job within seconds → ComfyUI runs headless on it → your outputs land in your ShareSync folder as they're rendered → your client app reads them out.
You're never "logged into a workstation" and you're never paying for idle time. Compute is allocated when you submit, runs for the length of your workflow (typically 10 s – 5 min for concept / storyboard shots), and shuts down. Next call gets a fresh allocation — or lands on the same compute with no cold-start if your previous job finished within the configurable "idle hold" window (more on that in §6).
The pieces:
your app  ──POST /api/compute/jobs──▶  Spark Fuse API
                                        │
                                        ▼
                                       GPU (g6 / g7e / etc.)
                                        │   docker pull comfyui
                                        │   docker run …  /input/ + /output/
                                        │
                                       outputs ──▶ ShareSync folder
                                                    │
                                                    ▼
your app  ◀──WebDAV reads───────────────────────────┘
Three things to keep in mind as you read the rest of this doc:
You're submitting a Docker container, not steering a GUI. The
workflow JSON you'd normally drag into the Comfy web UI becomes a /input/workflow.json file that your container reads on startup.
**/output/ inside the container is your customer-visible output
folder.** Anything your container writes there gets streamed to ShareSync as it's produced; your app reads from ShareSync.
Cost is "per call" not "per workstation hour". A 90-second
storyboard render on an RTX PRO 6000 GPU costs ~$0.07 today (g7e.2xlarge is ~$2.89/hr for InstantCompute, you pay for the seconds the container actually ran).

0.1 Two compute modes: InstantCompute and SmartCompute
Spark Fuse has two compute modes you pick between on each job submit:
InstantCompute (mode: "instant", default). The job lands on a warm GPU within a couple of seconds — or, if the warm pool is cold for your SKU, ~15 seconds while the platform brings a fresh GPU up. Once running, the job never gets interrupted mid-flight. This is the iteration-friendly mode: fast, predictable, the right default for concept / storyboard / animatic work where you want the result back now.
SmartCompute (mode: "smart"). The job runs on preemptible Spark Fuse capacity that's typically ~60% cheaper, but the platform may reclaim the underlying GPU mid-job with a short warning. Best for long batch work where a rare auto-retry is acceptable (multi- hour training, large frame sweeps you'll run overnight). Every submit is a fresh cold-start in v1 — no warm pool.
This whole doc uses InstantCompute (i.e. mode: "instant", or omit mode entirely — it's the default). It's the fastest path for the kind of work ComfyUI users typically do interactively, and the warm-pool / idle-hold behaviour you'll lean on in §6 is InstantCompute-only. When you want to read about SmartCompute, see spark-compute-api.md §13.

1. Prereqs
Before you can submit your first job you need three things.
1.1 A Spark account + your bearer token
You already have a Spark account from running workstations. Same login mints a bearer token for Spark Fuse:
export SPARK_EMAIL="you@yourcompany.example"
export SPARK_PASSWORD="...your-spark-password..."

export SPARK_TOKEN=$(curl -sX POST "$SPARK_HOST/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$SPARK_EMAIL\",
    \"password\": \"$SPARK_PASSWORD\"
  }" | jq -r '.token')

echo "Got token? ${SPARK_TOKEN:0:20}..."
This token is your auth for both the Spark Fuse API and the ShareSync output server. Everything at Spark Cloud Studio runs off the same auth token — no second login dance. Tokens expire after a few minutes; re-run the snippet above when you get an HTTP 401 back.
Also export the API base URL once so the rest of these commands stay short:
export SPARK_HOST="https://api.prod.aapse1.sparkcloud.studio"
1.2 A workflow JSON
This is the same .json file ComfyUI exports when you click "Save (API Format)" in the web UI. The contents are an opaque graph of nodes; Spark Fuse doesn't parse it, it just hands the file to ComfyUI on the remote compute and lets Comfy do its thing.
If you don't have one yet, you can use this throwaway one for the smoke test in §3 — it's a tiny SDXL workflow that generates one 1024×1024 image in ~8 seconds and writes it to /output/:
{
  "prompt": {
    "1": {
      "class_type": "CheckpointLoaderSimple",
      "inputs": { "ckpt_name": "sd_xl_base_1.0.safetensors" }
    },
    "2": {
      "class_type": "CLIPTextEncode",
      "inputs": { "clip": ["1", 1], "text": "a cinematic concept sketch of a starship hangar, ink and watercolor" }
    },
    "3": {
      "class_type": "CLIPTextEncode",
      "inputs": { "clip": ["1", 1], "text": "blurry, low quality, watermark" }
    },
    "4": {
      "class_type": "EmptyLatentImage",
      "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }
    },
    "5": {
      "class_type": "KSampler",
      "inputs": {
        "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
        "latent_image": ["4", 0],
        "seed": 42, "steps": 20, "cfg": 7,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1
      }
    },
    "6": {
      "class_type": "VAEDecode",
      "inputs": { "samples": ["5", 0], "vae": ["1", 2] }
    },
    "7": {
      "class_type": "SaveImage",
      "inputs": { "images": ["6", 0], "filename_prefix": "smoke" }
    }
  }
}
Save this as workflow.json in a directory by itself (we'll upload the whole directory in §3.2).
1.3 A ComfyUI Docker image
You have two practical options.
Option A — public ComfyUI image, models downloaded at runtime. Simpler to start with; cold-start (first job after a long quiet period) is slower because Comfy pulls models on boot. We'll use yanwk/comfyui-boot:latest for the examples in this doc — it's a public image with ComfyUI pre-installed and a small entrypoint that auto-downloads missing checkpoints on first run. Cold-start: ~3–5 min.
Option B — your own image with models baked in. Faster cold-start (no model downloads), private, customizable with your own custom nodes. Recommended once you've validated the workflow and want to optimize for production. Push to Docker Hub or your preferred private container registry and reference it the same way. We'll write up a separate doc on custom-image bakes; for now, Option A is the fast path to getting things up and running.
A note on cold-start vs warm-start. First job in a quiet window = the platform allocates compute, pulls the image, downloads models. That's 3–8 min total for Option A. Every job within the next 10 min (or whatever you set idleHoldSeconds to) lands on the same warm compute with the image still cached and models still on disk — so back-to-back jobs run in seconds. The whole point of the "idle hold" pattern is to keep your concept-iteration loop hot.

2. Smoke test — does my account work?
Before we wire ComfyUI in, let's confirm your token, network, and the ShareSync output path are all healthy with a 5-second job that just prints "hello" and saves a file.
2.1 Submit
curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
    "command": ["bash", "-c", "nvidia-smi && echo hello-from-spark > /output/hello.txt && cat /output/hello.txt"],
    "instanceType": "g4dn.xlarge",
    "idleHoldSeconds": 0
  }' | jq '.'
g4dn.xlarge is the cheapest GPU SKU we offer at ~$0.50/hr with the highest availability. Perfect for smoke tests.
You'll get back a job ID, the output ShareSync URL, and a "queued" status. Hang onto these:
export JOB_ID="<jobId from response>"
export OUTPUT_URL="<output.shareSyncBaseUrl from response>"
2.2 Watch it run
Open a live log stream (this stays connected until the job finishes):
curl -N "$SPARK_HOST/api/compute/jobs/$JOB_ID/logs/stream" \
  -H "Authorization: Bearer $SPARK_TOKEN"
You'll see queue status frames first (you're #N in line, ETA M seconds), then the container's stdout (nvidia-smi output, then "hello-from-spark"), then the stream closes when the job ends.
2.3 Read the output back
curl -H "Authorization: Bearer $SPARK_TOKEN" \
  "$OUTPUT_URL/hello.txt"
# → hello-from-spark
If that printed "hello-from-spark", your account is fully wired. Move on to §3.
If anything failed:
HTTP 401 → your token expired. Re-run the auth snippet in §1.1.
HTTP 400 with instanceType ... is not eligible → call GET /api/compute/skus to see the current allow-list, pick another GPU SKU (a g6 or g7e will work).
Job lands in status='failed' with error_code='no_capacity' → the warm pool is exhausted for that SKU. Retry in a minute, or pick a different SKU.
Anything else → ping your Spark contact and include the jobId.

3. Running ComfyUI headless
Now the real thing: submit your workflow JSON to ComfyUI on Spark Fuse and pull the rendered images back.
3.1 The container shape
ComfyUI in "headless" mode is just ComfyUI running with --listen 0.0.0.0 and being fed workflows via its HTTP API instead of the web UI. Inside the Spark Fuse container, we:
Start ComfyUI in the background, listening on 127.0.0.1:8188.
Wait for it to come up (10–30 s the first time, models loading).
POST /input/workflow.json to http://127.0.0.1:8188/prompt.
Poll Comfy's history endpoint until our prompt is marked complete.
Copy ComfyUI's output/ directory into /output/ (the Spark
Fuse output mount).
Exit.
That entire orchestration is a short bash script. We pass it inline via the command field. You don't need to write a Dockerfile.
3.2 Upload your workflow
The simplest input path is the push workflow: Spark Fuse allocates a one-shot upload URL when you submit, and you PUT your input directory there as a tarball. The agent on the GPU downloads + extracts it to /input/ before running your command.
SUBMIT=$(curl -sX POST "$SPARK_HOST/api/compute/jobs" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "yanwk/comfyui-boot:latest",
    "instanceType": "g7e.2xlarge",
    "inputPushMode": "auto-prepare",
    "idleHoldSeconds": 600,
    "command": [
      "bash", "-c",
      "set -e; cd /root/ComfyUI && python3 main.py --listen 127.0.0.1 --port 8188 & SERVER_PID=$!; for i in $(seq 1 60); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break || sleep 2; done; PROMPT_RES=$(curl -sX POST -H '"'"'Content-Type: application/json'"'"' --data @/input/workflow.json http://127.0.0.1:8188/prompt); PROMPT_ID=$(echo \"$PROMPT_RES\" | jq -r '"'"'.prompt_id'"'"'); echo \"queued prompt $PROMPT_ID\"; while true; do H=$(curl -sf http://127.0.0.1:8188/history/$PROMPT_ID); if [ -n \"$H\" ] && [ \"$H\" != '"'"'{}'"'"' ]; then break; fi; sleep 2; done; cp -r /root/ComfyUI/output/* /output/ 2>/dev/null || true; echo done; kill $SERVER_PID 2>/dev/null || true"
    ]
  }')

echo "$SUBMIT" | jq '.'

export JOB_ID=$(echo "$SUBMIT"     | jq -r '.jobId')
export UPLOAD_URL=$(echo "$SUBMIT"  | jq -r '.input.uploadUrl')
export OUTPUT_URL=$(echo "$SUBMIT"  | jq -r '.output.shareSyncBaseUrl')
A few field-by-field notes on what just happened:
Field
What it does for you
image: "yanwk/comfyui-boot:latest"
Public ComfyUI image (Option A from §1.3). Spark Fuse pulls this on the GPU. A few minutes first time.
instanceType: "g7e.2xlarge"
A current Spark-recommended ComfyUI SKU (1 × NVIDIA RTX PRO 6000, 96 GB VRAM). Plenty for SDXL / FLUX-dev / AnimateDiff / pretty much anything ComfyUI throws at a single GPU. See §4 for picking.
inputPushMode: "auto-prepare"
Asks Spark to mint an uploadUrl for your /input/ tarball. You PUT your workflow.json there in the next step.
idleHoldSeconds: 600
After this job ends, keep the compute warm for 10 minutes so your next iteration's prompt change doesn't re-pay cold-start. Set to 0 if this is a one-shot.
command
The orchestration script (start Comfy → submit workflow → wait → copy outputs → exit). Worth reading once; you'll reuse it verbatim across most jobs.

Now upload workflow.json:
cd /path/to/the/folder/containing/workflow.json
tar czf - workflow.json | curl -sX PUT --data-binary @- \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  "$UPLOAD_URL"
echo "Workflow uploaded."
The agent waits up to 5 min for that upload. Once it lands, the job moves from queued → provisioning → running and your container takes off.
3.3 Watch + harvest
Same SSE stream as §2.2:
curl -N "$SPARK_HOST/api/compute/jobs/$JOB_ID/logs/stream" \
  -H "Authorization: Bearer $SPARK_TOKEN"
You'll see ComfyUI's stdout — model loads, sampler progress, "done". When the stream closes, your outputs are in ShareSync:
# List what landed.
curl -X PROPFIND -H "Depth: 1" \
  -H "Authorization: Bearer $SPARK_TOKEN" "$OUTPUT_URL" \
  | xmllint --xpath '//*[local-name()="href"]/text()' - 2>/dev/null
# → .../smoke_00001_.png

# Download.
curl -O -H "Authorization: Bearer $SPARK_TOKEN" \
  "$OUTPUT_URL/smoke_00001_.png"
That .png is the image Comfy just rendered. You're done with the first real ComfyUI job.

4. Picking the right compute
If you're rendering...
Try
Cost (estimate)
Why
SDXL / SD1.5 / SDXL-Lightning concept sketches, 1024²
g4dn.xlarge
~$0.50/hr
Cheapest GPU we offer. Fine for SDXL up to ~768²; 1024² works but slow.
SDXL-Lightning concept sketches, 1024², faster iteration
g6.xlarge (1 × NVIDIA L4, 24 GB VRAM)
~$1.20/hr
2-3× faster than g4dn on SDXL. Sweet spot for storyboard iteration.
Recommended default for ComfyUI — FLUX-dev / FLUX-schnell at 1024²-2048², SDXL-XL, AnimateDiff, multi-controlnet stacks, IPAdapter, video, big batches
g7e.2xlarge (1 × NVIDIA RTX PRO 6000, 96 GB VRAM)
~$2.89/hr
Best perf-per-dollar for FLUX-era work today, and 96 GB VRAM is enough headroom for nearly any single-GPU ComfyUI workflow.
Same GPU as g7e.2xlarge plus more vCPU + system RAM
g7e.4xlarge (1 × NVIDIA RTX PRO 6000, 96 GB VRAM)
~$3.78/hr
Pick this only when you're CPU-bound on the host side (heavy preprocessing, many parallel model loads) — the GPU itself is identical to g7e.2xlarge.

Rule of thumb: start with g7e.2xlarge for new ComfyUI jobs. It's the modern default for FLUX-era work, has the biggest healthy warm pool right now, and the 96 GB of VRAM on a single RTX PRO 6000 means you'll almost never bump into memory limits on real-world ComfyUI graphs.
If you want a real-time pre-flight cost quote before submitting (e.g. "this 90-second job will cost about $X"), use the estimate endpoint:
curl -sX POST "$SPARK_HOST/api/compute/jobs/estimate" \
  -H "Authorization: Bearer $SPARK_TOKEN" \
  -H 'content-type: application/json' \
  -d '{
    "instanceType": "g7e.2xlarge",
    "estimatedRuntimeSeconds": 90,
    "idleHoldSeconds": 600
  }' | jq '{rate, estimate}'
The response tells you costPerHourUsd, billedPerHourUsd, and a total in cents and USD if you passed runtime.
Full SKU list — GET /api/compute/skus. Capacity reality may not match the catalog at any given moment (a SKU we accept isn't a guarantee that a warm member is currently available); if you hit error_code='no_capacity', retry or pick a sibling SKU.

5. Your client-app integration: reading outputs as they appear
The Spark Fuse agent uploads everything in /output/ to ShareSync as the container produces files (not just at the end of the job), so your concept / storyboard / animatic app can pick up images as Comfy writes them. "Poll the output ShareSync folder" is the right pattern.
5.1 The output URL
You got output.shareSyncBaseUrl back from the submit response. That's a WebDAV directory URL. Standard PROPFIND to list, GET to download. Same Spark bearer token authenticates against both.
https://your-org.files.sparkcloud.studio/dav/spaces/<space-id>/Spark%20Fuse%20Jobs/<jobId>/
5.2 Node.js — poll for new files
import fetch from 'node-fetch';
import { XMLParser } from 'fast-xml-parser';

const sparkHost = process.env.SPARK_HOST;
const sparkToken = process.env.SPARK_TOKEN;
const jobId = process.env.JOB_ID;
const outputUrl = process.env.OUTPUT_URL; // from submit response

const seen = new Set();
const parser = new XMLParser();

async function pollOnce() {
  const xml = await fetch(outputUrl, {
    method: 'PROPFIND',
    headers: {
      Authorization: `Bearer ${sparkToken}`,
      Depth: '1',
    },
  }).then((r) => r.text());

  const tree = parser.parse(xml);
  // PROPFIND multistatus responses: D:multistatus > D:response[] > D:href
  const responses = [].concat(tree['D:multistatus']?.['D:response'] ?? []);
  for (const r of responses) {
    const href = r['D:href'];
    if (!href || href.endsWith('/')) continue; // skip the directory entry itself
    if (seen.has(href)) continue;
    seen.add(href);

    // New file landed. Download it.
    const fileUrl = `${new URL(outputUrl).origin}${href}`;
    const buf = await fetch(fileUrl, {
      headers: { Authorization: `Bearer ${sparkToken}` },
    }).then((r) => r.arrayBuffer());

    onNewImage(href, Buffer.from(buf));
  }
}

async function jobDone() {
  const job = await fetch(`${sparkHost}/api/compute/jobs/${jobId}`, {
    headers: { Authorization: `Bearer ${sparkToken}` },
  }).then((r) => r.json());
  return ['succeeded', 'failed', 'cancelled', 'spot-interrupted'].includes(job.status);
}

// Poll every 2 s until the job ends + one more pass to catch the final batch.
(async () => {
  while (!(await jobDone())) {
    await pollOnce();
    await new Promise((res) => setTimeout(res, 2000));
  }
  await pollOnce(); // final sweep
})();

function onNewImage(href, buf) {
  // Your app's "new image arrived" hook — push into the UI, write to
  // local disk, kick off a downstream node, etc.
  console.log(`new output: ${href} (${buf.length} bytes)`);
}
5.3 Python — same thing, shorter
import os
import time
import requests
import xml.etree.ElementTree as ET

SPARK_HOST   = os.environ["SPARK_HOST"]
SPARK_TOKEN  = os.environ["SPARK_TOKEN"]
JOB_ID       = os.environ["JOB_ID"]
OUTPUT_URL   = os.environ["OUTPUT_URL"]

NS = {"D": "DAV:"}
seen = set()

def poll_once():
    r = requests.request(
        "PROPFIND", OUTPUT_URL,
        headers={"Authorization": f"Bearer {SPARK_TOKEN}", "Depth": "1"},
    )
    root = ET.fromstring(r.text)
    for resp in root.findall("D:response", NS):
        href = resp.findtext("D:href", default="", namespaces=NS)
        if not href or href.endswith("/"):
            continue
        if href in seen:
            continue
        seen.add(href)
        # New file. Download.
        file_url = f"{OUTPUT_URL.split('//')[0]}//{OUTPUT_URL.split('/')[2]}{href}"
        data = requests.get(file_url,
            headers={"Authorization": f"Bearer {SPARK_TOKEN}"}).content
        on_new_image(href, data)

def job_done():
    r = requests.get(f"{SPARK_HOST}/api/compute/jobs/{JOB_ID}",
        headers={"Authorization": f"Bearer {SPARK_TOKEN}"}).json()
    return r["status"] in ("succeeded", "failed", "cancelled", "spot-interrupted")

def on_new_image(href, data):
    print(f"new output: {href} ({len(data)} bytes)")

while not job_done():
    poll_once()
    time.sleep(2)
poll_once()  # final sweep
5.4 Why polling, not push?
Spark Fuse can fire webhooks at terminal job events (job.succeeded, job.failed, etc.) but not per-file. The per-file granularity needed for "new image landed" is best served by ShareSync polling — 2 s ticks are cheap, WebDAV PROPFIND is fast, and your app needs to read the file anyway. Fine-grained per-file event webhooks may land later; the polling pattern is what we recommend today.

6. Idle-hold + the iteration loop
Your concept iteration loop will look like this:
App submits prompt 1 → 30 s render → output appears → reviewed.
App submits prompt 2 (same workflow, tweaked seed) → wants
zero cold-start.
Achieve this with idleHoldSeconds. When you submit prompt 1, set idleHoldSeconds: 600 (10 min). After the container exits, the compute stays running with the image cached + ComfyUI ready to run for 10 minutes. Submit prompt 2 within that window with the same instanceType, and the platform routes it to that same warm compute with no cold-start — your second prompt starts rendering within ~1 s.
The catch: you're paying for the 10-min hold (the compute is running). For a tight iteration loop where you're going to submit again every 30–60 s anyway, this is the right call. For a one-shot batch, set idleHoldSeconds: 0 so the compute spins down the moment your container exits.
A sensible default for an interactive concept session is idleHoldSeconds: 300 (5 min) — long enough to cover "look at the result, think, tweak, resubmit", short enough that you're not paying for inactivity if you wander off.
Idle-hold is InstantCompute-only. SmartCompute jobs don't honor idleHoldSeconds and don't reuse compute across submits. If you're iterating quickly, stay in InstantCompute (the default).

7. ComfyUI patterns you'll actually use
7.1 "Same workflow, different prompt"
In the workflow JSON you uploaded, the prompt text lives in a CLIPTextEncode node's inputs.text. To change just the prompt across runs, you have two options:
Easiest: re-upload the workflow JSON with the new prompt baked in. Generate a new workflow.json on the client side with the new prompt string, submit a new job that points at it. Costs you one auto-prepare upload per iteration (small text file, instant).
More flexible: use ComfyUI's --input-directory and a wrapper script. Pass the prompt + seed + other tweakables as env vars or small files in /input/, have a Python wrapper in your image read them, mutate the workflow graph, then POST to Comfy. We'll write a dedicated doc for the wrapper-script pattern; for now, the simpler "re-upload" path is fine for storyboard / concept iteration.
7.2 Multi-shot batches
If you want to render N variants of the same workflow in one job (cheaper than N submissions because you pay one cold-start, not N): edit your workflow JSON so the EmptyLatentImage node has batch_size: 4 (or whatever N), or add a Repeat Latent Batch node. Comfy will run the entire pipeline N times and save N images to /output/. Your client app sees N new files appear via the polling loop in §5.
For animation passes (e.g. AnimateDiff 16-frame loops), the same mechanic applies — Comfy writes one PNG per frame to output/, the agent streams each one to ShareSync as it's produced, your app gets them in order.
7.3 Custom nodes
If your workflow needs ComfyUI custom nodes that aren't in the yanwk/comfyui-boot image, your options in order of effort:
pip install it at runtime via startScriptB64 — works for
pure-Python custom nodes that have an installable package. Slow (runs every cold-start) but no Dockerfile.
**Clone the custom node repo into /root/ComfyUI/custom_nodes/
in your command** — works for any custom node. Same "slow on cold-start, fast on warm-start" tradeoff.
Bake a custom image — derive from yanwk/comfyui-boot, add
your custom nodes + your model checkpoints, push to Docker Hub or your preferred private container registry. Faster cold-starts, private, version-locked. Recommended once your workflow stabilizes.
We'll write up the custom-image pattern in a follow-on doc. For first exploration with stock ComfyUI nodes, you don't need to touch any of this.

8. Common gotchas
Symptom
What's happening
Fix
Job stays queued longer than 30 s
Warm pool is saturated; the platform is launching fresh compute for you. Cold-start is ~3–5 min.
Just wait. If it's > 10 min, cancel + retry with a different SKU (GET /api/compute/skus for alternates).
status='failed', error_code='no_capacity'
All warm members of that SKU are busy AND the underlying infrastructure has no capacity for a fresh allocation.
Retry in 1–2 min, or pick a sibling SKU (e.g. g6.xlarge if g7e.2xlarge is unavailable).
status='failed', error_code='image_pull_failed'
Typo in image field, or registry needs auth.
Confirm the image is public and the tag exists. docker pull yanwk/comfyui-boot:latest from your laptop as a sanity check.
status='failed', error_code='input_download_failed'
You submitted with inputPushMode: 'auto-prepare' but didn't PUT to the uploadUrl within 5 min.
Submit again, upload immediately after the submit response comes back.
Container exits 0 but no files in /output/
ComfyUI saved to its own output/ directory but your script didn't copy to /output/.
The cp -r /root/ComfyUI/output/* /output/ line in the §3.2 command does this — confirm it ran (check the SSE log).
Cold-start way longer than 5 min
Likely first-ever pull of the image to a brand-new warm member. Subsequent jobs of the same SKU shape will be much faster.
Be patient on the first job; verify with a second back-to-back job.
Comfy hangs on a node, never returns
A workflow problem (bad model reference, missing custom node, etc.), not a Spark problem.
Pull the log archive from log_archive_share_sync_path — Comfy's stderr is in there.

When in doubt, check the error_message field on GET /api/compute/jobs/:jobId — it's English prose, usually pointing right at the cause. And the log archive at log_archive_share_sync_path has the full container stdout/stderr, re-readable indefinitely after the job ends.

9. Cost in practice
Concrete numbers for a typical headless-ComfyUI iteration session, assuming g7e.2xlarge at ~$2.89/hr:
Activity
Time
Cost
Cold-start (first job after a quiet period)
3-5 min
~$0.14-$0.24
Run a SDXL 1024² concept render
8-15 s
~$0.006-$0.012
Run a FLUX-dev 1024² render
20-40 s
~$0.016-$0.032
Run a 16-frame AnimateDiff loop
60-120 s
~$0.048-$0.096
Idle hold between rapid-fire iterations (10 min default)
10 min
~$0.48
An hour of iterative concept work (cold-start + 30 renders + held idle between)
60 min
~$2.89

Compare to running the same workflow on a Spark workstation for an hour: you pay for 60 minutes of compute whether the GPU is idle or not. Headless on Spark Fuse pays only for the seconds the container is running plus whatever idle-hold window you set.
For pure batch work (no idle hold, fire-and-forget):
100 storyboard frames at 30 s each =  ~$2.14 total
For interactive concept work (rapid-fire iteration, 5-min idle hold):
10 prompts in a 30-min session = ~$1.30 total (mostly the idle hold; the renders themselves are pennies).

10. Where to go from here
You've now:
Got a Spark bearer token.
Submitted a smoke-test compute job and read its output.
Run a real ComfyUI workflow headless on a Spark GPU.
Read outputs back via WebDAV polling from your client app.
Next steps that aren't urgent for first use but worth knowing:
Cancel a job: POST /api/compute/jobs/:jobId/cancel. Idempotent. Skips the idle hold and stops the compute immediately. See spark-compute-api.md §7.
List your jobs: GET /api/compute/jobs. Returns the full history for your org. Useful for a "recent renders" UI in your app. See spark-compute-api.md §6.
SmartCompute (mode: "smart"): the other compute mode from §0.1. Submitting with mode: "smart" gives you typically ~60% cheaper compute at the cost of the platform being able to preempt your job mid-run. Good for long batch work (animations, look-dev sweeps) where the rare interruption can be retried automatically; not recommended for time-sensitive concept iteration. See spark-compute-api.md §13.
Bake your own ComfyUI image with your custom nodes + favorite checkpoints. Doc coming separately; ping us when you're ready.
Webhooks for "job finished" notifications instead of polling. Useful for fire-and-forget batch jobs. See spark-compute-api.md §14 (when published).
If you hit anything that this doc didn't cover, ping your Spark contact — we want this to be the doc that gets you from zero to shipping in under an hour, so any gap is something we want to fix.


