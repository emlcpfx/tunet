"""
comfy_launch.py — run a ComfyUI workflow as a batch job on Spark Fuse.

A generic runner with a "preset" layer on top:

  * GENERIC:  give it any API-format workflow (ComfyUI → Save (API Format)),
              the input media it references, and optional per-node overrides.
  * PRESET:   a named bundle (presets/<name>.preset.json) that supplies the
              image / GPU / workflow + maps friendly flags (--prompt, --lora,
              --strength, ...) onto specific workflow nodes. Ships with a
              `cleanplate_ltx` template for LTX-2.3 + Obscura Remova v2v.

It mirrors spark_launch.py's submit→pack→upload→tail flow, but the job command
runs comfy_run.py (the headless ComfyUI driver) inside the container instead of
train.py.

Usage:
    # generic
    python comfy_spark/comfy_launch.py \
        --workflow my_workflow_api.json \
        --input clip.mp4 \
        --image <registry>/comfy-ltx23:latest --gpu rtxpro6000 \
        --set 6.inputs.text="remove the curtains"

    # preset (cleanplate)
    python comfy_spark/comfy_launch.py --preset cleanplate_ltx \
        clip.mp4 --prompt "remove the window curtains" --strength 1.3

    # housekeeping
    python comfy_spark/comfy_launch.py --inspect my_workflow_api.json
    python comfy_spark/comfy_launch.py --list-jobs
    python comfy_spark/comfy_launch.py --logs <JOB_ID>
    python comfy_spark/comfy_launch.py --cancel <JOB_ID>

Auth: SPARK_EMAIL / SPARK_PASSWORD (or spark_email / spark_pass) in .env, same
as spark_launch.py.
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# ── Spark Compute v1 ──────────────────────────────────────────────────────────

SPARK_API = "https://api.prod.aapse1.sparkcloud.studio"

# Mandatory billing tag — Spark bills jobs carrying `cpfx_tunet` under the Clean
# Plate FX cost + revenue-share deal; anything else is standard rate. ALWAYS
# sent. `cpfx_comfy` is added on top so ComfyUI jobs are filterable.
BILLING_TAG = "cpfx_tunet"
GROUP_TAG = "cpfx_comfy"

# A ComfyUI image is required (the runpod/pytorch default has no ComfyUI). There
# is no safe default — submit fails if neither --image nor a preset supplies one.
DEFAULT_IDLE = 0

# GPU shortcut → Spark SKU (mirrors spark_launch.py). LTX-2.3 22B wants the big
# cards; t4/a10 will OOM on the base model.
GPU_TYPES = {
    "t4":           "g4dn.xlarge",
    "a10":          "g5.xlarge",
    "l4":           "g6.2xlarge",
    "l40s":         "g6e.8xlarge",
    "l40sx4":       "g6e.12xlarge",
    "rtxpro6000":   "g7e.2xlarge",
    "rtxpro6000x8": "g7e.48xlarge",
}

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfy_run.py")

# ── Auth (self-contained; same shape as spark_launch.py) ──────────────────────

_token = None
_token_expires_at = 0


def _creds():
    email = os.environ.get("SPARK_EMAIL") or os.environ.get("spark_email")
    pw = os.environ.get("SPARK_PASSWORD") or os.environ.get("spark_pass")
    if not email or not pw:
        sys.exit("ERROR: set SPARK_EMAIL / SPARK_PASSWORD (or spark_email / spark_pass) in .env")
    return email, pw


def _jwt_expiry(token):
    try:
        import base64
        payload = json.loads(base64.b64decode(token.split(".")[1] + "==").decode())
        return payload.get("exp", 0) * 1000
    except Exception:
        return 0


def get_token():
    global _token, _token_expires_at
    if _token and _token_expires_at > time.time() * 1000 + 60_000:
        return _token
    email, pw = _creds()
    r = requests.post(f"{SPARK_API}/auth/login", json={"email": email, "password": pw}, timeout=15)
    if not r.ok:
        sys.exit(f"ERROR: Spark auth failed (HTTP {r.status_code}): {r.text[:200]}")
    tok = r.json().get("token") or r.json().get("access_token")
    if not tok:
        sys.exit("ERROR: Spark auth: no token in response")
    _token, _token_expires_at = tok, _jwt_expiry(tok)
    return _token


def spark(method, path, body=None, **kw):
    tok = get_token()
    headers = {"Authorization": f"Bearer {tok}"}
    headers.update(kw.pop("headers", {}))
    if body is not None and "data" not in kw:
        headers.setdefault("Content-Type", "application/json")
        kw["json"] = body
    r = requests.request(method, f"{SPARK_API}{path}", headers=headers,
                         timeout=kw.pop("timeout", 30), **kw)
    if not r.ok:
        sys.exit(f"ERROR: Spark API {method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
    return r


# ── Job lifecycle ─────────────────────────────────────────────────────────────

def list_skus():
    return spark("GET", "/api/compute/skus").json().get("skus", [])


def list_jobs(tag=GROUP_TAG):
    path = f"/api/compute/jobs?tag={tag}" if tag else "/api/compute/jobs"
    return spark("GET", path).json().get("jobs", [])


def cancel_job(job_id):
    return spark("POST", f"/api/compute/jobs/{job_id}/cancel").json()


def submit_job(name, instance_type, image, command, env_vars, mode, idle, max_retries, extra_tags):
    tags = []
    for t in [BILLING_TAG, GROUP_TAG, *(extra_tags or [])]:
        if t and t not in tags:
            tags.append(t)
    body = {
        "name": name,
        "instanceType": instance_type,
        "image": image,
        "command": command,
        "tags": tags,
        "inputPushMode": "auto-prepare",
        "env": env_vars or {},
        "mode": mode,
    }
    if mode == "instant":
        body["idleHoldSeconds"] = idle
    if mode == "smart" and max_retries is not None:
        body["maxRetriesOnInterrupt"] = max_retries
    return spark("POST", "/api/compute/jobs", body).json()


def upload_tarball(upload_url, tar_path):
    size_mb = os.path.getsize(tar_path) / 1024 / 1024
    print(f"[upload] PUT {size_mb:.1f} MB → uploadUrl")
    tok = get_token()
    with open(tar_path, "rb") as f:
        r = requests.put(upload_url, data=f, timeout=1200,
                         headers={"Authorization": f"Bearer {tok}",
                                  "Content-Type": "application/octet-stream"})
    if not r.ok:
        sys.exit(f"ERROR: upload failed HTTP {r.status_code}: {r.text[:200]}")
    print("[upload] Done.")


def stream_logs(job_id):
    tok = get_token()
    print(f"\n[logs] Streaming {job_id} (Ctrl+C to detach; job keeps running)\n")
    try:
        with requests.get(f"{SPARK_API}/api/compute/jobs/{job_id}/logs/stream",
                          headers={"Authorization": f"Bearer {tok}",
                                   "Accept": "text/event-stream"},
                          stream=True, timeout=None) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    print(line[6:], flush=True)
                elif not line.startswith(":"):
                    print(line, flush=True)
    except KeyboardInterrupt:
        print("\n[logs] Detached.")


# ── Workflow patching ─────────────────────────────────────────────────────────

def _coerce(v):
    """Parse a --set value as JSON (numbers/bools/null/arrays) else leave a string."""
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def apply_set(workflow, node_id, dotted_path, value):
    """Set workflow[node_id][a][b]... = value. node_id is an API-format key (str)."""
    node_id = str(node_id)
    if node_id not in workflow:
        sys.exit(f"ERROR: node id {node_id!r} not in workflow. Run --inspect to list node ids.")
    cur = workflow[node_id]
    parts = dotted_path.split(".")
    for p in parts[:-1]:
        if p not in cur:
            sys.exit(f"ERROR: path {dotted_path!r} not found on node {node_id} ({'.'.join(parts)})")
        cur = cur[p]
    cur[parts[-1]] = value


def inspect(workflow_path):
    with open(workflow_path) as f:
        wf = json.load(f)
    print(f"{workflow_path}: {len(wf)} nodes\n")
    for nid in sorted(wf, key=lambda x: (len(x), x)):
        node = wf[nid]
        ct = node.get("class_type", "?")
        ins = node.get("inputs", {})
        # show only literal (non-link) inputs — those are what you'd patch
        lits = {k: v for k, v in ins.items() if not isinstance(v, list)}
        lit_str = ", ".join(f"{k}={v!r}" for k, v in lits.items())
        print(f"  [{nid:>3}] {ct:<28} {lit_str}")


# ── Preset layer ──────────────────────────────────────────────────────────────

def load_preset(name):
    path = os.path.join(PRESETS_DIR, f"{name}.preset.json")
    if not os.path.isfile(path):
        avail = [f[:-12] for f in os.listdir(PRESETS_DIR) if f.endswith(".preset.json")] \
            if os.path.isdir(PRESETS_DIR) else []
        sys.exit(f"ERROR: preset {name!r} not found. Available: {', '.join(avail) or '(none)'}")
    with open(path) as f:
        return json.load(f)


def resolve_preset_workflow(preset):
    wf = preset.get("workflow")
    if not wf:
        sys.exit("ERROR: preset has no 'workflow' key")
    wf_path = wf if os.path.isabs(wf) else os.path.join(PRESETS_DIR, wf)
    if not os.path.isfile(wf_path):
        sys.exit(
            f"ERROR: preset workflow not found: {wf_path}\n"
            f"       This preset ships as a TEMPLATE. Export your validated graph from\n"
            f"       ComfyUI via 'Save (API Format)', drop it there, then fill in the\n"
            f"       node ids in the .preset.json 'params' map (use --inspect to find them)."
        )
    return wf_path


def check_preset_finalized(preset):
    bad = []
    for name, spec in preset.get("params", {}).items():
        if str(spec.get("node", "")).startswith("REPLACE"):
            bad.append(name)
    if bad:
        sys.exit(
            f"ERROR: preset param mapping is still a template (node id REPLACE_* for: "
            f"{', '.join(bad)}).\n       Edit the .preset.json with real node ids "
            f"(run --inspect on your exported workflow)."
        )


# ── Tarball builder ───────────────────────────────────────────────────────────

def build_tar(workflow_obj, input_files, patches=None):
    """Pack /input/workflow.json + /input/comfy_run.py (+ patches.json) + media."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()
    with tarfile.open(tmp.name, "w:gz") as tf:
        def add_json(obj, arc):
            jt = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(obj, jt)
            jt.close()
            tf.add(jt.name, arcname=arc)
            os.unlink(jt.name)
        add_json(workflow_obj, "workflow.json")
        if patches:
            add_json(patches, "patches.json")
        tf.add(RUNNER, arcname="comfy_run.py")
        # media — referenced by bare basename inside the workflow
        for p in input_files:
            if not os.path.isfile(p):
                sys.exit(f"ERROR: input file not found: {p}")
            mb = os.path.getsize(p) / 1024 / 1024
            print(f"[pack] {os.path.basename(p)} ({mb:.1f} MB)")
            tf.add(p, arcname=os.path.basename(p))
    print(f"[pack] → {os.path.getsize(tmp.name) / 1024 / 1024:.1f} MB")
    return tmp.name


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run a ComfyUI workflow on Spark Fuse")
    ap.add_argument("video", nargs="?", help="(preset mode) primary input media file")
    ap.add_argument("--preset", help="named preset under presets/ (e.g. cleanplate_ltx)")
    ap.add_argument("--workflow", help="(generic) path to an API-format workflow JSON")
    ap.add_argument("--input", action="append", default=[],
                    help="media file to upload (repeatable). Referenced by basename in the workflow")
    ap.add_argument("--set", action="append", default=[], metavar="NODE.path=value",
                    help="override a workflow value, e.g. 6.inputs.text='remove the curtains' (repeatable)")
    # preset-friendly flags
    ap.add_argument("--prompt", help="(preset) positive prompt")
    ap.add_argument("--negative", help="(preset) negative prompt")
    ap.add_argument("--lora", help="(preset) direct .safetensors URL for the LoRA (overrides preset default)")
    ap.add_argument("--lora-name", help="(preset) filename to save the LoRA as")
    ap.add_argument("--strength", type=float, help="(preset) LoRA strength")
    ap.add_argument("--fps", type=int, help="(preset) output frame rate")
    # compute
    ap.add_argument("--image", help="ComfyUI Docker image (required unless preset supplies one)")
    ap.add_argument("--gpu", choices=list(GPU_TYPES), help="GPU shortcut")
    ap.add_argument("--instance-type", help="raw Spark SKU (overrides --gpu)")
    ap.add_argument("--mode", choices=["instant", "smart"], help="compute mode (default instant, or preset)")
    ap.add_argument("--max-retries", type=int, help="(smart) re-launches on preemption [0-5]")
    ap.add_argument("--idle-hold", type=int, default=DEFAULT_IDLE, help="(instant) warm-hold secs after exit")
    ap.add_argument("--name", help="job name")
    ap.add_argument("--tag", action="append", default=[], help="extra grouping tag (repeatable)")
    ap.add_argument("--ready-timeout", type=int, default=300, help="secs to wait for ComfyUI startup")
    ap.add_argument("--convert-only", action="store_true",
                    help="convert a UI graph → API format on Spark, write it to ShareSync, and exit (no render)")
    ap.add_argument("--no-tail", action="store_true", help="don't stream logs after submit")
    ap.add_argument("--dry-run", action="store_true", help="patch + print the workflow and planned submit; don't submit")
    # housekeeping
    ap.add_argument("--inspect", metavar="WORKFLOW", help="print node ids/types of a workflow and exit")
    ap.add_argument("--list-skus", action="store_true")
    ap.add_argument("--list-jobs", action="store_true")
    ap.add_argument("--logs", metavar="JOB_ID")
    ap.add_argument("--cancel", metavar="JOB_ID")
    args = ap.parse_args()

    # ── housekeeping sub-commands ─────────────────────────────────────────────
    if args.inspect:
        return inspect(args.inspect)
    if args.list_skus:
        for s in list_skus():
            print(f"  {s['instanceType']:20s}  {s['gpuCount']}× {s['gpuType']:28s}  {s['gpuMemoryGb']}GB")
        return
    if args.list_jobs:
        jobs = list_jobs()
        print(f"{len(jobs)} job(s) tagged {GROUP_TAG}:")
        for j in jobs:
            print(f"  {j.get('id','?'):40s}  {j.get('status','?'):12s}  {j.get('instance_type_name','?')}")
        return
    if args.cancel:
        print(json.dumps(cancel_job(args.cancel), indent=2))
        return
    if args.logs:
        return stream_logs(args.logs)

    # ── resolve config from preset (if any) ───────────────────────────────────
    preset = load_preset(args.preset) if args.preset else {}
    image = args.image or preset.get("image")
    gpu = args.gpu or preset.get("gpu", "rtxpro6000")
    mode = args.mode or preset.get("mode", "instant")
    instance_type = args.instance_type or GPU_TYPES[gpu]

    # ── load + patch the workflow ─────────────────────────────────────────────
    input_files = list(args.input)
    if args.preset:
        check_preset_finalized(preset)
        wf_path = resolve_preset_workflow(preset)
    else:
        if not args.workflow:
            sys.exit("ERROR: pass --workflow <api.json> (generic) or --preset <name>")
        wf_path = args.workflow
    with open(wf_path) as f:
        workflow = json.load(f)

    lora_url = args.lora or preset.get("lora_default")
    lora_name = args.lora_name or preset.get("lora_name", "lora.safetensors")
    is_ui = isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list)

    # Collect patch ops (node, path, value) from the preset map + --set.
    patch_ops = []
    if args.preset:
        if not args.video and not args.convert_only:
            sys.exit("ERROR: preset mode needs a primary input file, e.g. "
                     "`... --preset cleanplate_ltx clip.mp4` (or --convert-only)")
        if args.video:
            input_files.insert(0, args.video)
        flag_values = {
            "prompt": args.prompt, "negative": args.negative,
            "strength": args.strength, "fps": args.fps, "lora": lora_name,
            "video": os.path.basename(args.video) if args.video else None,
        }
        for pname, spec in preset.get("params", {}).items():
            val = flag_values.get(pname)
            if val is None:
                val = spec.get("default")
            if val is None:
                continue
            patch_ops.append({"node": str(spec["node"]), "path": spec["path"], "value": val})

    for s in args.set:  # generic overrides, applied last so they win
        if "=" not in s:
            sys.exit(f"ERROR: --set needs NODE.path=value, got {s!r}")
        lhs, rhs = s.split("=", 1)
        nid, _, path = lhs.partition(".")
        if not path:
            sys.exit(f"ERROR: --set needs a node path, got {lhs!r} (e.g. 6.inputs.text=...)")
        patch_ops.append({"node": nid, "path": path, "value": _coerce(rhs)})

    # API-format workflows are patched client-side now (so bad node ids/paths
    # fail immediately). UI/graph workflows are converted to API on Spark, so
    # their patches — which target the post-conversion graph — ride along in
    # patches.json and are applied there.
    patches_to_pack = None
    if is_ui:
        patches_to_pack = patch_ops
    else:
        for op in patch_ops:
            apply_set(workflow, op["node"], op["path"], op["value"])

    if not args.dry_run:
        if not image:
            sys.exit("ERROR: no --image and preset supplies none. ComfyUI image is required.")
        if str(image).startswith("REPLACE"):
            sys.exit("ERROR: preset 'image' is still a placeholder. Set it to your built "
                     "ComfyUI image in the .preset.json, or pass --image.")

    job_name = args.name or (f"comfy-{args.preset}" if args.preset else "comfy-job")

    env_vars = {
        "COMFY_WORKFLOW": "/input/workflow.json",
        "COMFY_READY_TIMEOUT": str(args.ready_timeout),
    }
    if patches_to_pack:
        env_vars["COMFY_PATCHES"] = "/input/patches.json"
    if args.convert_only:
        env_vars["COMFY_CONVERT_ONLY"] = "1"
    if lora_url:
        env_vars["COMFY_LORA_URL"] = lora_url
        env_vars["COMFY_LORA_NAME"] = lora_name
    if preset.get("extra_args"):
        env_vars["COMFY_EXTRA_ARGS"] = preset["extra_args"]
    # No-build path: tell comfy_run.py where ComfyUI lives in the base image and
    # what to clone/download onto the node at startup.
    if preset.get("comfy_home"):
        env_vars["COMFY_HOME"] = preset["comfy_home"]
    if preset.get("node_packs"):
        env_vars["COMFY_FETCH_NODES"] = json.dumps(preset["node_packs"])
    if preset.get("models"):
        env_vars["COMFY_FETCH_MODELS"] = json.dumps(preset["models"])

    # ── dry run: show what would happen ───────────────────────────────────────
    print(f"\n[comfy] {'DRY RUN — ' if args.dry_run else ''}submit plan")
    print(f"        name      : {job_name}")
    print(f"        image     : {image or '(none — required!)'}")
    print(f"        instance  : {instance_type} ({gpu})   mode={mode}")
    print(f"        tags      : {BILLING_TAG}, {GROUP_TAG}{''.join(', ' + t for t in args.tag)}")
    print(f"        inputs    : {[os.path.basename(p) for p in input_files] or '(none)'}")
    print(f"        format    : {'UI graph → converts on Spark' if is_ui else 'API (patched locally)'}")
    if args.convert_only:
        print(f"        convert   : emit converted api.json to ShareSync, no render")
    print(f"        env       : {json.dumps(env_vars)}")
    if patch_ops:
        print("        patches   :")
        for op in patch_ops:
            print(f"          {op['node']}.{op['path']} = {op['value']!r}")
    if args.dry_run:
        if is_ui:
            print(f"\n[comfy] {len(workflow['nodes'])}-node UI graph; converts to API on Spark, "
                  f"then the patches above are applied.\n        Use --convert-only to get the "
                  f"converted workflow_api.json back (in ShareSync) without rendering.")
        else:
            print("\n[comfy] patched workflow (API format):")
            print(json.dumps(workflow, indent=2))
        return

    # ── pack, submit, upload, tail ────────────────────────────────────────────
    # bash -c mirrors the proven quickstart invocation and dodges PATH/entrypoint
    # quirks across public ComfyUI base images.
    command = ["bash", "-c", "python3 /input/comfy_run.py"]
    tar_path = build_tar(workflow, input_files, patches_to_pack)
    try:
        resp = submit_job(job_name, instance_type, image, command, env_vars,
                          mode, args.idle_hold, args.max_retries, args.tag)
        job_id = resp.get("jobId") or resp.get("id")
        upload_url = (resp.get("input") or {}).get("uploadUrl")
        if not job_id or not upload_url:
            sys.exit(f"ERROR: submit response missing jobId/uploadUrl: {resp}")
        upload_tarball(upload_url, tar_path)
        print(f"[comfy] Job submitted: {job_id}")
        if args.no_tail:
            print(f"\nWatch:  python comfy_spark/comfy_launch.py --logs {job_id}")
            print(f"Cancel: python comfy_spark/comfy_launch.py --cancel {job_id}")
        else:
            stream_logs(job_id)
    finally:
        try:
            os.unlink(tar_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
