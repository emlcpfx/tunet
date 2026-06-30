"""
comfy_launch.py — run a ComfyUI workflow as a batch job on Spark Fuse.

A generic runner with a "preset" layer on top:

  * GENERIC:  give it any API-format workflow (ComfyUI → Save (API Format)),
              the input media it references, and optional per-node overrides.
  * PRESET:   a named bundle (presets/<name>.preset.json) that supplies the
              image / GPU / workflow + maps friendly flags (--prompt, --lora,
              --strength, ...) onto specific workflow nodes. Ships with a
              `ltx_Obscura_Remova` template for LTX-2.3 + Obscura Remova v2v.

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
    python comfy_spark/comfy_launch.py --preset ltx_Obscura_Remova \
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
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, unquote, quote

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import comfy_resolve  # local module (same dir); EZ-Comfy dependency resolver

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
LORAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfy_run.py")
DEFAULT_CATALOG = "ltx2.loras.json"
OUTPUTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "outputs.json")


def load_outputs():
    """The output-format registry (outputs/outputs.json). Returns {} if absent."""
    if not os.path.isfile(OUTPUTS_FILE):
        return {}
    with open(OUTPUTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def print_outputs(reg):
    fmts = reg.get("formats", {})
    print(f"{len(fmts)} output format(s) (default '{reg.get('_default', 'mp4')}'):")
    for name, spec in fmts.items():
        print(f"  {name:14s} [{spec.get('kind','?'):8s}] {spec.get('label','')}")


SEQ_EXTS = {".exr": "exr", ".dpx": "dpx", ".png": "img", ".jpg": "img",
            ".jpeg": "img", ".tif": "img", ".tiff": "img"}


def detect_sequence(folder):
    """Find a numbered frame sequence in `folder` and return its #### pattern, frame
    range, kind, and file list. Picks the largest consistent (prefix, ext, pad) group."""
    import re
    if not os.path.isdir(folder):
        sys.exit(f"ERROR: --input-sequence {folder!r} is not a folder")
    groups = {}
    for fn in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in SEQ_EXTS:
            continue
        m = re.search(r"(\d+)$", stem)
        if not m:
            continue
        key = (stem[:m.start()], ext.lower(), len(m.group(1)))
        groups.setdefault(key, []).append((int(m.group(1)), fn))
    if not groups:
        sys.exit(f"ERROR: no numbered frames (name####.ext) found in {folder} "
                 f"(looked for {sorted(SEQ_EXTS)})")
    (prefix, ext, pad), frames = max(groups.items(), key=lambda kv: len(kv[1]))
    frames.sort()
    nums = [n for n, _ in frames]
    return {
        "kind": SEQ_EXTS[ext],
        "pattern": f"/input/seq/{prefix}{'#' * pad}{ext}",
        "start": nums[0], "end": nums[-1], "step": 1,
        # raw components so a batch can re-home the sequence under /input/seq/<tag>/
        "prefix": prefix, "pad": pad, "ext": ext,
        "files": [os.path.join(folder, fn) for _, fn in frames],
    }


def prefix_targets(preset):
    """Normalize preset.output_prefix into [{node, path, template}]. Each `nodes`
    entry is either a bare node id (uses the top-level path/template) or a
    {node, path?, template?} object — the object form lets a second saver (e.g. an
    HDR EXR output_dir) get its own field + per-item template so a batch doesn't
    overwrite itself."""
    op = preset.get("output_prefix") or {}
    dpath = op.get("path", "inputs.filename_prefix")
    dtmpl = op.get("template", "{stem}_{preset}")
    out = []
    for e in op.get("nodes", []):
        if isinstance(e, dict):
            out.append({"node": str(e["node"]), "path": e.get("path", dpath),
                        "template": e.get("template", dtmpl)})
        else:
            out.append({"node": str(e), "path": dpath, "template": dtmpl})
    return out


def build_batch_manifest(batch_paths, preset, preset_name, colorspace):
    """Turn `--batch` PATHs (video files and/or image-sequence folders) into a
    COMFY_BATCH manifest the on-node runner loops over, plus the media to pack.

    The manifest is preset-agnostic — it carries the wiring the runner needs:
    where a video item's filename goes (the preset's primary input param), which
    node a sequence item swaps (the preset's `input_anchor`, e.g. a
    GetVideoComponents bridge), and where each item's output prefix is written
    (the preset's `output_prefix`). Returns
    (manifest, video_files, seq_groups, extra_node_packs)."""
    params = preset.get("params", {})
    prim = params.get("video") or params.get("image")
    if not prim:
        sys.exit("ERROR: --batch needs a preset whose primary input is a 'video' or 'image' "
                 "param (a video-to-video / upscale preset). This preset declares neither.")
    manifest = {
        "preset": preset_name or "comfy",
        "input_node": str(prim["node"]), "input_path": prim["path"],
        "input_anchor": str(preset["input_anchor"]["node"]) if preset.get("input_anchor") else None,
        "prefix_targets": prefix_targets(preset),
        "items": [],
    }
    video_files, seq_groups, extra_packs = [], [], []
    used = set()

    def uniq(stem):
        base = "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)[:60] or "item"
        s, i = base, 2
        while s in used:
            s, i = f"{base}_{i}", i + 1
        used.add(s)
        return s

    for p in batch_paths:
        if os.path.isfile(p):
            name = os.path.basename(p)
            manifest["items"].append({"kind": "video", "stem": uniq(os.path.splitext(name)[0]), "name": name})
            video_files.append(p)
        elif os.path.isdir(p):
            seq = detect_sequence(p)
            if seq["kind"] not in ("exr", "img"):
                sys.exit(f"ERROR: --batch folder {p!r} holds a {seq['kind']!r} sequence; only EXR and "
                         f"PNG/JPG/TIFF folders are supported today (DPX is the next add — see EZ_COMFY_TODO).")
            tag = uniq(os.path.basename(os.path.normpath(p)))
            cs = colorspace or ("linear" if seq["kind"] == "exr" else "as-is")
            if seq["kind"] == "exr":
                sd = {"kind": "exr",
                      "pattern": f"/input/seq/{tag}/{seq['prefix']}{'#' * seq['pad']}{seq['ext']}",
                      "start": seq["start"], "end": seq["end"], "step": seq["step"], "colorspace": cs}
                pack = "https://github.com/Conor-Collins/ComfyUI-CoCoTools_IO"
            else:
                sd = {"kind": "img", "dir": f"/input/seq/{tag}", "colorspace": cs}
                pack = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
            manifest["items"].append({"kind": seq["kind"], "stem": tag, "seq": sd})
            seq_groups.append((tag, seq["files"]))
            if pack not in extra_packs:
                extra_packs.append(pack)
        else:
            sys.exit(f"ERROR: --batch entry not found: {p}")
    return manifest, video_files, seq_groups, extra_packs


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
    # NOTE: the live API serves auth at /auth/login (returns HTTP 201) — NOT the
    # /api/auth/login that Spark Fuse API v1.18 §1.1 documents (that 404s). Doc
    # bug, verified 2026-05-21. The success/resp handling below DOES follow §1.1:
    # a failed login still returns 2xx with success=false / token=null, so branch
    # on `success` and surface `resp` rather than trusting the status code alone.
    r = None
    for _attempt in range(4):                      # retry transient network blips
        try:
            r = requests.post(f"{SPARK_API}/auth/login",
                              json={"email": email, "password": pw}, timeout=30)
            break
        except requests.exceptions.RequestException as e:
            if _attempt == 3:
                sys.exit(f"ERROR: can't reach Spark auth at {SPARK_API} after 4 tries "
                         f"({e.__class__.__name__}). Network/VPN issue or API down — try again.")
            time.sleep(2 * (_attempt + 1))
    if not r.ok:
        sys.exit(f"ERROR: Spark auth failed (HTTP {r.status_code}): {r.text[:200]}")
    data = r.json()
    if not data.get("success", True):
        sys.exit(f"ERROR: Spark auth failed: {data.get('resp', 'login unsuccessful')}")
    tok = data.get("token") or data.get("access_token")
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


def get_job(job_id):
    return spark("GET", f"/api/compute/jobs/{job_id}").json()


# Terminal job states (Spark Fuse API v1.18 §11.1). `smartcompute-interrupted`
# is the current name for a preempted smart-mode job; the old `spot-interrupted`
# spelling is gone. `completed` isn't in the spec but is kept as a harmless alias.
TERMINAL_STATUSES = {"succeeded", "completed", "failed", "cancelled",
                     "smartcompute-interrupted"}


def wait_for_terminal(job_id, poll=5):
    """Block until the job reaches a terminal state; return that status."""
    while True:
        status = (get_job(job_id) or {}).get("status", "")
        if status in TERMINAL_STATUSES:
            return status
        time.sleep(poll)


# ── ShareSync output download (WebDAV) ────────────────────────────────────────
# The Spark bearer token authenticates against the files.* host too (single
# sign-on), so the same getToken() works for PROPFIND/GET on output URLs.

_DAV = "{DAV:}"


def webdav_list(url):
    """PROPFIND Depth:1 -> [(name, full_url, is_dir)], excluding the dir itself."""
    tok = get_token()
    dir_url = url if url.endswith("/") else url + "/"
    r = requests.request(
        "PROPFIND", dir_url,
        headers={"Authorization": f"Bearer {tok}", "Depth": "1",
                 "Content-Type": "application/xml"},
        data='<?xml version="1.0" encoding="utf-8"?>'
             '<propfind xmlns="DAV:"><prop><resourcetype/></prop></propfind>',
        timeout=60,
    )
    if r.status_code == 404:
        return []
    if not r.ok:
        raise RuntimeError(f"PROPFIND {dir_url} -> HTTP {r.status_code}")
    root = ET.fromstring(r.text)
    base_path = urlparse(dir_url).path.rstrip("/")
    entries = []
    for resp in root.findall(f"{_DAV}response"):
        href = resp.findtext(f"{_DAV}href")
        if not href:
            continue
        full = urljoin(dir_url, href)
        path = urlparse(full).path.rstrip("/")
        if path == base_path:
            continue  # the directory itself
        is_dir = resp.find(f".//{_DAV}collection") is not None
        entries.append((unquote(path.split("/")[-1]), full, is_dir))
    return entries


def _space_base(any_dav_url):
    """Extract the ShareSync space root (…/dav/spaces/<space>) from any DAV URL,
    so we can rebuild an output URL from a job's output_share_sync_path."""
    u = urlparse(any_dav_url)
    parts = u.path.split("/")
    if len(parts) >= 4 and parts[1] == "dav" and parts[2] == "spaces":
        return f"{u.scheme}://{u.netloc}/" + "/".join(parts[1:4])
    return None


def _encode_path(p):
    return "/".join(quote(seg) for seg in p.strip("/").split("/") if seg)


def resolve_output_url(resp, job_id, upload_url):
    """Full WebDAV URL for a job's output folder. Prefer the *job detail's*
    shareSyncBaseUrl — it's the form that actually PROPFINDs (literal `$` in the
    space id). The submit-response value can encode the `$` as %24 and resolve to
    an empty/wrong path, which silently downloads 0 files. Fall back to the submit
    response, then to rebuilding from output_share_sync_path + the space root."""
    out_url = ((get_job(job_id) or {}).get("output") or {}).get("shareSyncBaseUrl")
    if out_url:
        return out_url
    out_url = (resp.get("output") or {}).get("shareSyncBaseUrl")
    if out_url:
        return out_url
    opath = (get_job(job_id) or {}).get("output_share_sync_path")
    base = _space_base(upload_url or "")
    if opath and base:
        return f"{base}/{_encode_path(opath)}/"
    return None


def download_outputs(base_url, dest_dir, _rel=""):
    """Recursively pull every file under base_url into dest_dir. Returns count."""
    os.makedirs(os.path.join(dest_dir, _rel), exist_ok=True)
    n = 0
    for name, href, is_dir in webdav_list(base_url):
        if is_dir:
            n += download_outputs(href, dest_dir, os.path.join(_rel, name))
            continue
        tok = get_token()
        r = requests.get(href, headers={"Authorization": f"Bearer {tok}"}, timeout=1200)
        if not r.ok:
            print(f"[download] WARN {name}: HTTP {r.status_code}")
            continue
        local = os.path.join(dest_dir, _rel, name)
        with open(local, "wb") as f:
            f.write(r.content)
        rel = os.path.join(_rel, name) if _rel else name
        print(f"[download] {rel} ({len(r.content) / 1024 / 1024:.1f} MB)")
        n += 1
    return n


def submit_job(name, instance_type, image, command, env_vars, mode, idle, max_retries, extra_tags,
               assets_path=None, assets_space=None):
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
    # Read-only model library, lazily mounted at /assets and cached across jobs.
    if assets_path:
        body["assetsShareSyncPath"] = assets_path
        if assets_space:
            body["assetsShareSyncSpaceName"] = assets_space
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
    with open(workflow_path, encoding="utf-8") as f:
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


# ── EZ-Comfy: resolve a workflow's dependencies (see comfy_resolve.py) ─────────

def resolve_cmd(workflow_path, out_path=None, fetch=True):
    res = comfy_resolve.resolve_workflow(comfy_resolve.load_graph(workflow_path), fetch=fetch)
    print(f"{workflow_path}:\n")
    print(comfy_resolve.format_report(res))
    if out_path:
        draft = comfy_resolve.to_draft_preset(res, os.path.basename(workflow_path))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=2, ensure_ascii=False)
        print(f"\nwrote draft preset -> {out_path}")
        if res["unresolved"]:
            print(f"  fill the {len(res['unresolved'])} REPLACE_ model URL(s), then --convert-only to validate.")


def resolve_selftest(fetch=False):
    """Resolve every shipped preset's workflow and diff against its hand-written
    node_packs / models. The model diff is the hard correctness signal (a
    declared weight the resolver misses is a regression); node-pack drift is a
    warning, since hand lists go stale relative to the graph."""
    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
    norm = comfy_resolve._norm_repo
    regressions = 0
    for fn in sorted(os.listdir(preset_dir)):
        if not fn.endswith(".preset.json"):
            continue
        preset = json.load(open(os.path.join(preset_dir, fn), encoding="utf-8"))
        wf = preset.get("workflow")
        if not wf or not os.path.isfile(os.path.join(preset_dir, wf)):
            continue
        res = comfy_resolve.resolve_workflow(
            comfy_resolve.load_graph(os.path.join(preset_dir, wf)), fetch=fetch)

        want_packs = {norm(p) for p in (preset.get("node_packs") or [])}
        got_packs  = set(res["node_packs"])
        want_models = {m["dest"] for m in (preset.get("models") or [])}
        got_models  = {m["dest"] for m in res["models"]} | {u["dest"] for u in res["unresolved"]}
        # A "missed" model = declared by the preset but not derived from the graph.
        # Excuse the legit multi-file MODEL-DIRECTORY case: some loaders (e.g. the
        # 19B LTXVGemmaCLIPModelLoader) read a whole HF folder, but the graph only
        # names ONE shard, so the resolver can only ever see that one file (and it
        # may even classify its dir differently). Support files (config/tokenizer/
        # index .json, *.model) and extra shards ("-of-") are therefore STRUCTURALLY
        # un-derivable — never count them as drift. The regression guard still fires
        # for an over-declared SINGLE weight file (the common drift case).
        # Also excuse an ALTERNATE weight that lives in the SAME dir as a
        # resolver-derived model — e.g. a task-adapter set (ltx_restore ships 4 IC-LoRA
        # adapters in loras/ltx2.3-train/ but the graph names only the 1-2 active ones;
        # the others are switched in via a `task` param). Keyed on the FULL dir, not the
        # category, so a stray weight in a top-level shared dir still flags.
        _WEIGHT_EXT = (".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".onnx", ".gguf")
        _got_dirs = {os.path.dirname(d) for d in got_models}
        def _model_dir_aux(dest):
            base = os.path.basename(dest).lower()
            if dest.replace("\\", "/").startswith("custom_nodes/"):
                return True                               # pack-internal weight (path hardcoded by the node pack, e.g. ComfyUI-LatentSyncWrapper/checkpoints/) — never in the graph
            if (not base.endswith(_WEIGHT_EXT)) or ("-of-" in base):
                return True                               # support file or sharded weight
            return os.path.dirname(dest) in _got_dirs     # alternate weight beside a resolved one
        missed_models = {d for d in (want_models - got_models) if not _model_dir_aux(d)}

        print(f"\n=== {fn}  (workflow {wf}) ===")
        print(f"  node packs: want {len(want_packs)}, got {len(got_packs)}, "
              f"match {len(want_packs & got_packs)}")
        for p in sorted(want_packs - got_packs):
            print(f"    - in preset, NOT resolved: {p}")
        for p in sorted(got_packs - want_packs):
            print(f"    + resolved, not in preset: {p}  [{res['node_pack_sources'].get(p,'?')}]")
        print(f"  models: want {len(want_models)}, matched {len(want_models & got_models)}, "
              f"resolver-extra {len(got_models - want_models)}")
        for d in sorted(missed_models):
            print(f"    ! MISSED declared model: {d}")
        if res["unknown_nodes"]:
            print(f"  unknown nodes: {', '.join(res['unknown_nodes'])}")
        if res["ambiguous_nodes"]:
            print(f"  ambiguous nodes: {', '.join(res['ambiguous_nodes'])}")
        regressions += len(missed_models)

    print(f"\n{'PASS' if regressions == 0 else 'FAIL'}: {regressions} missed declared model(s).")
    if regressions:
        sys.exit(1)


# ── Preset layer ──────────────────────────────────────────────────────────────

def load_preset(name):
    path = os.path.join(PRESETS_DIR, f"{name}.preset.json")
    if not os.path.isfile(path):
        avail = [f[:-12] for f in os.listdir(PRESETS_DIR) if f.endswith(".preset.json")] \
            if os.path.isdir(PRESETS_DIR) else []
        sys.exit(f"ERROR: preset {name!r} not found. Available: {', '.join(avail) or '(none)'}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_presets():
    """(name, preset_dict) for every shipped preset, ordered by optional 'order' then name."""
    if not os.path.isdir(PRESETS_DIR):
        return []
    items = []
    for f in sorted(os.listdir(PRESETS_DIR)):
        if not f.endswith(".preset.json"):
            continue
        try:
            with open(os.path.join(PRESETS_DIR, f), encoding="utf-8") as fh:
                items.append((f[: -len(".preset.json")], json.load(fh)))
        except (OSError, ValueError):
            pass
    items.sort(key=lambda kv: (kv[1].get("order", 99), kv[0]))
    return items


def docs_links(preset):
    """Normalize a preset's `docs` field to a list of (label, url). Accepts a bare
    URL string, a list of URL strings, or a list of {label,url} objects."""
    docs = preset.get("docs")
    if not docs:
        return []
    if isinstance(docs, str):
        return [("docs", docs)]
    out = []
    for d in docs:
        if isinstance(d, str):
            out.append(("docs", d))
        elif isinstance(d, dict) and d.get("url"):
            out.append((d.get("label", "docs"), d["url"]))
    return out


def print_presets(tag_filter=""):
    """Plain-language catalog of the shipped presets — what each does, what it
    takes, the knobs that matter, and a link to the canonical docs. Optional
    `tag_filter` keeps only presets with a tag containing that text (case-insensitive)."""
    tf = (tag_filter or "").strip().lower()
    shown = 0
    for name, p in iter_presets():
        tags = [str(t) for t in p.get("tags", [])]
        if tf and not any(tf in t.lower() for t in tags):
            continue
        shown += 1
        ui = p.get("ui", {})
        about = p.get("about") or {}
        print(f"\n● {name}")
        print(f"    {ui.get('title') or name}")
        if tags:
            print(f"    tags : {', '.join(tags)}")
        if about.get("what"):
            print(f"    What : {about['what']}")
        if about.get("inputs"):
            print(f"    Takes: {about['inputs']}")
        if about.get("key_knobs"):
            print(f"    Knobs: {about['key_knobs']}")
        for label, url in docs_links(p):
            print(f"    Docs : {url}  ({label})")
        pg = p.get("prompt_guide") or {}
        if pg.get("url"):
            print(f"    Prompt: {pg['url']}  (prompting guide)")
        if not about:                       # un-migrated preset: at least show the dev blurb
            print(f"    {p.get('description', '')[:200]}")
    if tf:
        print(f"\n{shown} preset(s) tagged ~{tag_filter!r}.")
    else:
        print(f"\n{shown} preset(s). Run one with --preset <name>; filter this list with "
              f"--list-presets <tag> (e.g. face, removal, ltx2.3).")


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


def unfinalized_models(preset):
    """Model entries whose URL is still a REPLACE_ placeholder (template preset)."""
    return [m.get("dest", m.get("url", "?")) for m in preset.get("models", [])
            if "REPLACE" in str(m.get("url", ""))]


# ── LoRA catalog (Tier-1, stackable, drop-in) ─────────────────────────────────

def load_catalog(name_or_path):
    """Load a LoRA catalog by bare name (under loras/) or explicit path."""
    path = name_or_path if os.path.isabs(name_or_path) or os.path.sep in name_or_path \
        else os.path.join(LORAS_DIR, name_or_path)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f).get("loras", {})


def _parse_lora_arg(spec):
    """'name', 'name:1.2', 'https://h/x.safetensors', 'https://h/x.safetensors:0.9'
    -> (ref, strength_or_None). Splits a trailing :<float> off; URLs keep their
    scheme colon because the right side of the split must parse as a number."""
    ref, _, tail = spec.rpartition(":")
    if ref and tail:
        try:
            return ref, float(tail)
        except ValueError:
            pass
    return spec, None


def _is_url(ref):
    return ref.startswith("http://") or ref.startswith("https://")


def resolve_loras(specs, catalog, preset_base):
    """Turn --lora specs into [{name,url,file,strength,trigger,base,kind}].
    A spec is a catalog name or a direct .safetensors URL, each with optional
    :<strength>. Warns on base-model mismatch and a too-hot combined stack."""
    resolved = []
    for spec in specs:
        ref, strength = _parse_lora_arg(spec)
        if _is_url(ref):
            file = unquote(urlparse(ref).path.split("/")[-1]) or "lora.safetensors"
            entry = {"name": file, "url": ref, "file": file, "trigger": "",
                     "base": None, "kind": "effect"}
        else:
            entry = catalog.get(ref)
            if not entry:
                avail = ", ".join(sorted(catalog)) or "(none)"
                sys.exit(f"ERROR: LoRA {ref!r} not in catalog. Known: {avail}\n"
                         f"       (or pass a direct .safetensors URL to --lora)")
            entry = {"name": ref, **entry}
        s = strength if strength is not None else entry.get("strength", 1.0)
        resolved.append({
            "name": entry["name"], "url": entry["url"], "file": entry["file"],
            "strength": float(s), "trigger": entry.get("trigger") or "",
            "base": entry.get("base"), "kind": entry.get("kind", "effect"),
        })
        if preset_base and entry.get("base") and entry["base"] != preset_base:
            print(f"[comfy] WARN: LoRA {entry['name']!r} targets base {entry['base']!r} "
                  f"but this preset's model is {preset_base!r} — it may not apply cleanly.")
    total = sum(l["strength"] for l in resolved)
    if total > 2.0:
        print(f"[comfy] WARN: combined LoRA strength is {total:.1f} (LTX docs suggest "
              f"keeping a stack under ~2.0); outputs may be over-cooked.")
    return resolved


def print_catalog(catalog):
    if not catalog:
        print("(catalog empty — pass a direct .safetensors URL to --lora instead)")
        return
    print(f"{len(catalog)} LoRA(s) in catalog:\n")
    for name in sorted(catalog):
        e = catalog[name]
        trig = f"  trigger={e['trigger']!r}" if e.get("trigger") else ""
        print(f"  {name:<14} [{e.get('kind','effect')}/{e.get('task','any')}] "
              f"base={e.get('base','?')}  str~{e.get('strength', 1.0)}{trig}")
        if e.get("note"):
            print(f"                 {e['note']}")
        if e.get("prompt_example"):
            print(f"                 e.g. prompt: \"{e['prompt_example']}\"")


# ── Tarball builder ───────────────────────────────────────────────────────────

def build_tar(workflow_obj, input_files, patches=None, seq_files=None, seq_groups=None):
    """Pack /input/workflow.json + /input/comfy_run.py (+ patches.json) + media.
    `seq_files` is a flat single-sequence (--input-sequence) → /input/seq/.
    `seq_groups` is a list of (subdir, files) for a BATCH of sequence folders →
    /input/seq/<subdir>/ each, so multiple sequences don't collide by basename."""
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
        # frame sequence (--input-sequence) -> /input/seq/, referenced by the swapped loader
        seq_mb = 0.0
        for p in (seq_files or []):
            seq_mb += os.path.getsize(p) / 1024 / 1024
            tf.add(p, arcname=f"seq/{os.path.basename(p)}")
        if seq_files:
            print(f"[pack] {len(seq_files)} sequence frame(s) ({seq_mb:.1f} MB) → /input/seq/")
        # batch sequence folders -> /input/seq/<subdir>/ (one subdir per folder)
        for subdir, files in (seq_groups or []):
            grp_mb = 0.0
            for p in files:
                if not os.path.isfile(p):
                    sys.exit(f"ERROR: sequence frame not found: {p}")
                grp_mb += os.path.getsize(p) / 1024 / 1024
                tf.add(p, arcname=f"seq/{subdir}/{os.path.basename(p)}")
            print(f"[pack] {len(files)} frame(s) ({grp_mb:.1f} MB) → /input/seq/{subdir}/")
    print(f"[pack] → {os.path.getsize(tmp.name) / 1024 / 1024:.1f} MB")
    return tmp.name


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run a ComfyUI workflow on Spark Fuse")
    ap.add_argument("video", nargs="?", help="(preset mode) primary input media file")
    ap.add_argument("--input-sequence", metavar="DIR",
                    help="(v2v presets) feed a folder of frames (EXR sequence) as the input plate "
                         "instead of a clip: swaps the input loader for LoadExrSequence and converts "
                         "scene-linear EXR -> sRGB for the model. Replaces the positional input.")
    ap.add_argument("--batch", action="append", default=[], metavar="PATH",
                    help="BATCH: a video file OR an image-sequence folder (EXR / PNG / JPG / TIFF). "
                         "Repeatable — render many inputs in ONE job, paying the model load ONCE "
                         "(the node stays warm across renders). Each input's output is named after it. "
                         "Use instead of the positional input. Preset-driven (v2v / upscale presets).")
    ap.add_argument("--preset", help="named preset under presets/ (e.g. ltx_Obscura_Remova)")
    ap.add_argument("--workflow", help="(generic) path to an API-format workflow JSON")
    ap.add_argument("--input", action="append", default=[],
                    help="media file to upload (repeatable). Referenced by basename in the workflow")
    ap.add_argument("--set", action="append", default=[], metavar="NODE.path=value",
                    help="override a workflow value, e.g. 6.inputs.text='remove the curtains' (repeatable)")
    # preset-friendly flags
    ap.add_argument("--prompt", help="(preset) positive prompt")
    ap.add_argument("--negative", help="(preset) negative prompt")
    ap.add_argument("--mask", help="(preset) mask file: a static image (.png/.jpg) OR a "
                    "per-frame mask video (.mp4/.mov/...). A video auto-selects the preset's "
                    "mask-video workflow. White = region to inpaint/remove")
    ap.add_argument("--face", help="(preset) face/identity reference IMAGE for face-swap "
                    "presets (e.g. ltx_faceswap). The body/motion clip is the positional arg; "
                    "this is the identity to swap in")
    ap.add_argument("--lora", action="append", default=[], metavar="NAME[:STR]",
                    help="stack a Tier-1 LoRA: a catalog name or a direct .safetensors URL, "
                         "with optional :<strength> (e.g. --lora transition:1.2). Repeatable.")
    ap.add_argument("--no-triggers", action="store_true",
                    help="don't auto-append catalog LoRA trigger words to the prompt")
    ap.add_argument("--catalog", help="LoRA catalog file (name under loras/ or path; "
                    "default per-preset or ltx2.loras.json)")
    ap.add_argument("--lora-url", help="(preset) direct .safetensors URL for the built-in LoRA (overrides preset default)")
    ap.add_argument("--lora-name", help="(preset) filename to save the built-in LoRA as")
    ap.add_argument("--strength", type=float, help="(preset) built-in LoRA strength")
    ap.add_argument("--fps", type=int, help="(preset) output frame rate")
    # compute
    ap.add_argument("--image", help="ComfyUI Docker image (required unless preset supplies one)")
    ap.add_argument("--gpu", choices=list(GPU_TYPES), help="GPU shortcut")
    ap.add_argument("--instance-type", help="raw Spark SKU (overrides --gpu)")
    ap.add_argument("--mode", choices=["instant", "smart"], help="compute mode (default instant, or preset)")
    ap.add_argument("--max-retries", type=int, help="(smart) re-launches on preemption [0-5]")
    ap.add_argument("--idle-hold", type=int, default=DEFAULT_IDLE, help="(instant) warm-hold secs after exit")
    ap.add_argument("--assets-path", default=os.environ.get("COMFY_ASSETS_PATH"),
                    help="ShareSync path to a read-only model library mounted at /assets and cached "
                         "on the node across jobs. Weights staged there are symlinked instead of "
                         "downloaded from HF/CivitAI. Defaults to $COMFY_ASSETS_PATH.")
    ap.add_argument("--assets-space",
                    help="ShareSync Project the --assets-path lives in (omit = your Personal space)")
    ap.add_argument("--name", help="job name")
    ap.add_argument("--tag", action="append", default=[], help="extra grouping tag (repeatable)")
    ap.add_argument("--ready-timeout", type=int, default=300, help="secs to wait for ComfyUI startup")
    ap.add_argument("--convert-only", action="store_true",
                    help="convert a UI graph → API format on Spark, write it to ShareSync, and exit (no render)")
    ap.add_argument("--no-tail", action="store_true", help="don't stream logs after submit")
    ap.add_argument("--download", metavar="DIR",
                    help="after the job finishes, pull its ShareSync outputs into DIR")
    # output format (high-bit / EXR / ProRes) — splices a parallel saver on the node
    ap.add_argument("--output", metavar="FORMAT",
                    help="output format (see --list-outputs): mp4 (default preview), exr32, exr16, "
                         "png16, tiff16, prores_hq, prores_4444, ... Adds a high-bit deliverable "
                         "alongside the mp4 preview.")
    ap.add_argument("--output-fps", type=float, default=24.0,
                    help="(video --output like prores) frame rate for the encoded file [24]")
    ap.add_argument("--colorspace", choices=["linear", "as-is"],
                    help="override colour handling on --output (default: linear for EXR, as-is otherwise)")
    ap.add_argument("--list-outputs", action="store_true", help="print the output-format registry and exit")
    ap.add_argument("--dry-run", action="store_true", help="patch + print the workflow and planned submit; don't submit")
    # housekeeping
    ap.add_argument("--inspect", metavar="WORKFLOW", help="print node ids/types of a workflow and exit")
    ap.add_argument("--resolve", metavar="WORKFLOW",
                    help="derive node packs + models for a workflow (EZ-Comfy) and exit")
    ap.add_argument("--resolve-out", metavar="FILE",
                    help="(with --resolve) write a draft .preset.json")
    ap.add_argument("--no-fetch", action="store_true",
                    help="(with --resolve) skip the ComfyUI-Manager DB fetch; use embedded + seed map only")
    ap.add_argument("--resolve-selftest", action="store_true",
                    help="resolve every shipped preset's workflow and diff vs its hand-written lists")
    ap.add_argument("--list-presets", nargs="?", const="", metavar="TAG",
                    help="list shipped presets in plain language (what each does, what it takes, "
                         "key knobs, docs link). Pass a TAG to filter (e.g. --list-presets face)")
    ap.add_argument("--list-skus", action="store_true")
    ap.add_argument("--list-loras", action="store_true", help="print the LoRA catalog and exit")
    ap.add_argument("--list-jobs", action="store_true")
    ap.add_argument("--logs", metavar="JOB_ID")
    ap.add_argument("--cancel", metavar="JOB_ID")
    args = ap.parse_args()

    # ── housekeeping sub-commands ─────────────────────────────────────────────
    if args.list_presets is not None:        # const="" when bare, so `is not None`
        return print_presets(args.list_presets)
    if args.inspect:
        return inspect(args.inspect)
    if args.resolve:
        return resolve_cmd(args.resolve, args.resolve_out, fetch=not args.no_fetch)
    if args.resolve_selftest:
        return resolve_selftest(fetch=not args.no_fetch)
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
    if args.list_loras:
        pre = load_preset(args.preset) if args.preset else {}
        cat_name = args.catalog or pre.get("lora_catalog") or DEFAULT_CATALOG
        print_catalog(load_catalog(cat_name))
        return
    if args.list_outputs:
        print_outputs(load_outputs())
        return

    # ── resolve config from preset (if any) ───────────────────────────────────
    preset = load_preset(args.preset) if args.preset else {}

    # A mask can be a static image or a per-frame mask video. A video extension
    # switches to the preset's mask-video workflow and remaps the mask param onto
    # that graph's video-loader input (mask_video_param).
    VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
    mask_is_video = bool(args.mask) and os.path.splitext(args.mask)[1].lower() in VIDEO_EXTS
    if args.preset and mask_is_video and preset.get("workflow_mask_video"):
        preset = {**preset, "workflow": preset["workflow_mask_video"]}
        if preset.get("mask_video_param"):
            preset = {**preset, "params": {**preset.get("params", {}),
                                           "mask": preset["mask_video_param"]}}

    image = args.image or preset.get("image")
    gpu = args.gpu or preset.get("gpu", "rtxpro6000")
    mode = args.mode or preset.get("mode", "instant")
    instance_type = args.instance_type or GPU_TYPES[gpu]

    # ── batch: many inputs (video files + image-sequence folders) in one job ──
    # Renders every input through one warm node (the model loads once), each
    # output named after its input. Preset-driven; replaces the positional input.
    batch_manifest = batch_video_files = batch_seq_groups = batch_packs = None
    if args.batch:
        if not args.preset:
            sys.exit("ERROR: --batch is preset-driven; pass --preset <name> (a v2v / upscale preset).")
        if args.video or args.input_sequence:
            sys.exit("ERROR: --batch replaces the positional input and --input-sequence; pass one or the other.")
        batch_manifest, batch_video_files, batch_seq_groups, batch_packs = \
            build_batch_manifest(args.batch, preset, args.preset, args.colorspace)

    # ── load + patch the workflow ─────────────────────────────────────────────
    input_files = list(args.input)
    if args.preset:
        check_preset_finalized(preset)
        wf_path = resolve_preset_workflow(preset)
    else:
        if not args.workflow:
            sys.exit("ERROR: pass --workflow <api.json> (generic) or --preset <name>")
        wf_path = args.workflow
    with open(wf_path, encoding="utf-8") as f:
        workflow = json.load(f)

    lora_url = args.lora_url or preset.get("lora_default")
    lora_name = args.lora_name or preset.get("lora_name", "lora.safetensors")
    is_ui = isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list)

    # ── resolve the stackable Tier-1 LoRA stack (catalog or raw URLs) ─────────
    # Each --lora adds a LoraLoaderModelOnly to a chain spliced in on Spark at the
    # preset's lora_chain anchor. Catalog trigger words are folded into the prompt.
    cat_name = args.catalog or preset.get("lora_catalog") or DEFAULT_CATALOG
    catalog = load_catalog(cat_name)
    lora_stack = resolve_loras(args.lora, catalog, preset.get("base")) if args.lora else []
    if lora_stack and not preset.get("lora_chain"):
        sys.exit("ERROR: --lora stacking needs the preset to declare a 'lora_chain' "
                 "anchor (the node whose MODEL output the stack extends). This preset has none.")
    triggers = [l["trigger"] for l in lora_stack if l["trigger"]]
    effective_prompt = args.prompt
    if triggers and not args.no_triggers:
        base_prompt = args.prompt if args.prompt is not None \
            else (preset.get("params", {}).get("prompt") or {}).get("default")
        suffix = ", ".join(triggers)
        effective_prompt = f"{base_prompt.rstrip().rstrip(',')}, {suffix}" if base_prompt else suffix
        print(f"[comfy] auto-appended LoRA trigger(s) to prompt: {suffix}  (--no-triggers to disable)")

    # Collect patch ops (node, path, value) from the preset map + --set.
    patch_ops = []
    if args.preset:
        if preset.get("requires_input", True) and not args.video \
                and not args.input_sequence and not args.batch and not args.convert_only:
            sys.exit("ERROR: preset mode needs a primary input file, e.g. "
                     "`... --preset ltx_Obscura_Remova clip.mp4` (or --input-sequence DIR, --batch PATH, or --convert-only)")
        # Batch inputs are packed below and re-pointed per item on the node — the
        # per-input `video`/output-prefix patches are skipped (args.video is None).
        if batch_video_files:
            input_files.extend(batch_video_files)
        if args.video:
            input_files.insert(0, args.video)
        if args.mask:
            input_files.append(args.mask)
        if args.face:
            input_files.append(args.face)
        # Static inputs the preset bundles (e.g. a placeholder for an optional/
        # bypassed loader so the graph runs without a user-supplied second image).
        # Packed by basename, like any input; the workflow references them so.
        for sf in (preset.get("static_inputs") or []):
            sp = os.path.join(PRESETS_DIR, os.path.basename(sf))
            if os.path.isfile(sp):
                input_files.append(sp)
            else:
                print(f"[warn] static_input {sf!r} not found at {sp}")
        # Presets that declare a `mask` param (e.g. VACE inpainting) need one to
        # render — fail early with the static-vs-video hint rather than at execution.
        if "mask" in preset.get("params", {}) and not args.mask and not args.convert_only:
            sys.exit("ERROR: this preset needs a mask: pass --mask mask.png (one static "
                     "region) or --mask mask.mp4 (per-frame mask video). White = inpaint area.")
        # Face-swap presets declare a `face` param: the identity reference image.
        if "face" in preset.get("params", {}) and not args.face and not args.convert_only:
            sys.exit("ERROR: this preset needs a face reference image: pass --face face.png "
                     "(the identity to swap in). The body/motion clip is the positional arg.")
        flag_values = {
            "prompt": effective_prompt, "negative": args.negative,
            "strength": args.strength, "fps": args.fps, "lora": lora_name,
            "video": os.path.basename(args.video) if args.video else None,
            # i2v presets name the primary input param "image" (it's a start image,
            # not a clip); the positional arg feeds it just like a "video" param.
            "image": os.path.basename(args.video) if args.video else None,
            "mask": os.path.basename(args.mask) if args.mask else None,
            "face": os.path.basename(args.face) if args.face else None,
        }
        for pname, spec in preset.get("params", {}).items():
            val = flag_values.get(pname)
            if val is None:
                val = spec.get("default")
            if val is None:
                continue
            patch_ops.append({"node": str(spec["node"]), "path": spec["path"], "value": val})

        # Output filename: name renders <inputstem>_<preset> so ComfyUI's own
        # counter yields e.g. myclip_ltx_Obscura_Remova_00001.mp4 instead of the
        # graph's baked "LTX2.3" prefix. A second saver target (e.g. an HDR EXR
        # output_dir) gets its own template. Only the {stem} is sanitized, so
        # slashes in a template (an output_dir) are kept. Overridable via --set
        # (applied after, so it wins). Skipped on convert-only (no input to name).
        if preset.get("output_prefix") and args.video:
            stem = "".join(c if (c.isalnum() or c in "._-") else "_"
                           for c in os.path.splitext(os.path.basename(args.video))[0])[:120] or "output"
            model_tag = str(preset.get("base", "")).split("-")[0]
            for t in prefix_targets(preset):
                val = (t["template"].replace("{stem}", stem)
                       .replace("{preset}", args.preset or "comfy").replace("{model}", model_tag))
                patch_ops.append({"node": t["node"], "path": t["path"], "value": val})
                print(f"[comfy] output name → {t['node']}.{t['path']} = {val!r}")

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
        # Convert-only doesn't fetch weights, so placeholder URLs are fine there
        # (that's exactly the cheap path to validate a template's converted graph).
        unset = unfinalized_models(preset)
        if unset and not args.convert_only:
            sys.exit(
                "ERROR: preset has placeholder model URLs (still a template) for:\n        "
                + "\n        ".join(unset)
                + "\n       Fill in the real .safetensors download URLs in the .preset.json "
                  "'models' list, then run --convert-only once to validate the converted graph.")

    job_name = args.name or (f"comfy-{args.preset}" if args.preset else "comfy-job")

    env_vars = {
        "COMFY_WORKFLOW": "/input/workflow.json",
        "COMFY_READY_TIMEOUT": str(args.ready_timeout),
        # The Spark agent doesn't sync /output to ShareSync — so comfy_run uploads
        # its own outputs via WebDAV. It finds its own job by this run-id marker
        # and authenticates with this (long-lived) token.
        "COMFY_RUN_ID": uuid.uuid4().hex,
        "COMFY_UPLOAD_TOKEN": get_token(),
    }
    # Forward a HuggingFace token (if set locally) so gated weights download on the
    # node — e.g. Lightricks/LTX-2.3 returns 401 otherwise.
    _hf = os.environ.get("COMFY_HF_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if _hf:
        env_vars["COMFY_HF_TOKEN"] = _hf
    if patches_to_pack:
        env_vars["COMFY_PATCHES"] = "/input/patches.json"
    if args.convert_only:
        env_vars["COMFY_CONVERT_ONLY"] = "1"
    if lora_url:
        env_vars["COMFY_LORA_URL"] = lora_url
        env_vars["COMFY_LORA_NAME"] = lora_name
    if lora_stack:
        env_vars["COMFY_LORAS"] = json.dumps(
            [{"url": l["url"], "file": l["file"], "strength": l["strength"]} for l in lora_stack])
        env_vars["COMFY_LORA_CHAIN"] = json.dumps(preset["lora_chain"])
    if preset.get("extra_args"):
        env_vars["COMFY_EXTRA_ARGS"] = preset["extra_args"]
    # No-build path: tell comfy_run.py where ComfyUI lives in the base image and
    # what to clone/download onto the node at startup.
    if preset.get("comfy_home"):
        env_vars["COMFY_HOME"] = preset["comfy_home"]
    if preset.get("comfy_bundle"):
        env_vars["COMFY_BUNDLE"] = preset["comfy_bundle"]
    node_packs = list(preset.get("node_packs") or [])
    if preset.get("models"):
        env_vars["COMFY_FETCH_MODELS"] = json.dumps(preset["models"])

    # ── batch manifest: many inputs, one warm node (run_batch on the node) ──
    if batch_manifest:
        env_vars["COMFY_BATCH"] = json.dumps(batch_manifest)
        for pk in (batch_packs or []):                 # CoCoTools (EXR) / VHS (PNG folders)
            if pk not in node_packs:
                node_packs.append(pk)

    # ── input image-sequence: pack frames + swap the loader on the node ──
    seq = None
    if args.input_sequence:
        seq = detect_sequence(args.input_sequence)
        if seq["kind"] != "exr":
            sys.exit(f"ERROR: --input-sequence currently supports EXR sequences; found a {seq['kind']!r} "
                     f"sequence. DPX / PNG-folder ingest is the next add (see EZ_COMFY_TODO).")
        env_vars["COMFY_INPUT_SEQ"] = json.dumps({
            "kind": seq["kind"], "pattern": seq["pattern"], "start": seq["start"],
            "end": seq["end"], "step": seq["step"],
            "colorspace": args.colorspace or ("linear" if seq["kind"] == "exr" else "as-is")})
        if preset.get("input_anchor"):                 # explicit loader id wins; else auto-detect
            env_vars["COMFY_INPUT_SEQ_ANCHOR"] = json.dumps(str(preset["input_anchor"]["node"]))
        seq_pack = {"exr": "https://github.com/Conor-Collins/ComfyUI-CoCoTools_IO"}.get(seq["kind"])
        if seq_pack and seq_pack not in node_packs:
            node_packs.append(seq_pack)

    # ── output format: splice a high-bit saver onto the frames (default mp4 = no-op) ──
    out_fmt = args.output
    if out_fmt and out_fmt != (load_outputs().get("_default") or "mp4"):
        reg = load_outputs()
        spec = (reg.get("formats") or {}).get(out_fmt)
        if not spec:
            sys.exit(f"ERROR: unknown --output {out_fmt!r}. See --list-outputs.")
        if spec.get("kind") != "builtin":
            env_vars["COMFY_OUTPUT_SPEC"] = json.dumps(spec)
            env_vars["COMFY_OUTPUT_FPS"] = str(args.output_fps)
            env_vars["COMFY_OUTPUT_PREFIX"] = job_name
            env_vars["COMFY_OUTPUT_COLORSPACE"] = args.colorspace or spec.get("default_colorspace", "as-is")
            if preset.get("output_anchor"):           # explicit anchor wins; else auto-detect on node
                a = preset["output_anchor"]
                env_vars["COMFY_OUTPUT_ANCHOR"] = json.dumps([str(a["node"]), int(a.get("slot", 0))])
            if spec.get("node_pack") and spec["node_pack"] not in node_packs:
                node_packs.append(spec["node_pack"])   # clone the saver's pack on the node

    if node_packs:
        env_vars["COMFY_FETCH_NODES"] = json.dumps(node_packs)
    # Optional per-preset pip pins/extras, installed AFTER node requirements.txt so
    # they win — for nodes whose deps resolve to a broken combo on the base image
    # (e.g. seedvr2's diffusers/transformers flash_attn import break).
    if preset.get("pip_extra"):
        env_vars["COMFY_PIP_EXTRA"] = json.dumps(preset["pip_extra"])

    # ── dry run: show what would happen ───────────────────────────────────────
    print(f"\n[comfy] {'DRY RUN — ' if args.dry_run else ''}submit plan")
    print(f"        name      : {job_name}")
    print(f"        image     : {image or '(none — required!)'}")
    print(f"        instance  : {instance_type} ({gpu})   mode={mode}")
    if args.assets_path:
        print(f"        assets    : {args.assets_path}{' [' + args.assets_space + ']' if args.assets_space else ''}"
              f"  → /assets (read-only, cached on node; staged weights symlinked not downloaded)")
    print(f"        tags      : {BILLING_TAG}, {GROUP_TAG}{''.join(', ' + t for t in args.tag)}")
    print(f"        inputs    : {[os.path.basename(p) for p in input_files] or '(none)'}")
    if seq:
        print(f"        in-seq    : {len(seq['files'])} {seq['kind']} frame(s) {seq['start']}-{seq['end']} "
              f"→ {seq['pattern']} [{json.loads(env_vars['COMFY_INPUT_SEQ'])['colorspace']}]")
    if batch_manifest:
        items = batch_manifest["items"]
        print(f"        batch     : {len(items)} input(s) in ONE job (model loads once) —")
        for it in items:
            if it["kind"] == "video":
                print(f"          • {it['name']}  → output prefix {it['stem']}_{args.preset}")
            else:
                seqd = it["seq"]
                loc = seqd.get("pattern") or seqd.get("dir")
                print(f"          • {it['stem']}/  [{it['kind']} seq → {loc}]  → output prefix {it['stem']}_{args.preset}")
    print(f"        format    : {'UI graph → converts on Spark' if is_ui else 'API (patched locally)'}")
    if env_vars.get("COMFY_OUTPUT_SPEC"):
        _os = json.loads(env_vars["COMFY_OUTPUT_SPEC"])
        print(f"        output    : {args.output} — {_os.get('label','')} "
              f"[{env_vars.get('COMFY_OUTPUT_COLORSPACE')}]"
              f"{', fps ' + env_vars['COMFY_OUTPUT_FPS'] if _os.get('kind')=='video' else ''}"
              f" + the preset's mp4 preview")
    elif args.output and args.output != "mp4":
        print(f"        output    : {args.output}")
    for _label, _url in docs_links(preset):
        print(f"        docs      : {_url}  ({_label})")
    if lora_stack:
        anchor = preset["lora_chain"].get("anchor")
        print(f"        lora stack: spliced on node {anchor} (MODEL) —")
        for l in lora_stack:
            print(f"          + {l['name']} @ {l['strength']}  ({l['file']})")
    if args.convert_only:
        print(f"        convert   : emit converted api.json to ShareSync, no render")
    # Redact secrets from the printed plan — these env vars still go to the job
    # (the node needs them), but they shouldn't sit in local stdout/logs.
    _safe_env = {k: ("<redacted>" if any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
                     else v) for k, v in env_vars.items()}
    print(f"        env       : {json.dumps(_safe_env)}")
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
    # Run under the image's real interpreter (preset "python", e.g. python3.13 on
    # yanwk — that's the one with pip + torch + ComfyUI deps; bare python3 has
    # none). bash -c also dodges PATH/entrypoint quirks across base images.
    python_bin = preset.get("python", "python3")
    command = ["bash", "-c", f"{python_bin} /input/comfy_run.py"]
    tar_path = build_tar(workflow, input_files, patches_to_pack,
                         seq_files=seq["files"] if seq else None,
                         seq_groups=batch_seq_groups)
    try:
        resp = submit_job(job_name, instance_type, image, command, env_vars,
                          mode, args.idle_hold, args.max_retries, args.tag,
                          assets_path=args.assets_path, assets_space=args.assets_space)
        job_id = resp.get("jobId") or resp.get("id")
        upload_url = (resp.get("input") or {}).get("uploadUrl")
        if not job_id or not upload_url:
            sys.exit(f"ERROR: submit response missing jobId/uploadUrl: {resp}")
        out_url = (resp.get("output") or {}).get("shareSyncBaseUrl")
        upload_tarball(upload_url, tar_path)
        print(f"[comfy] Job submitted: {job_id}")
        if out_url:
            print(f"[comfy] Output folder (ShareSync): {out_url}")
        # Stream the live log. With --download, run the tail in a DAEMON THREAD
        # and gate the download on job-status polling below — the SSE tail can
        # hang (the stream doesn't always close when the job ends), which would
        # otherwise block the download from ever running.
        if args.download and not args.no_tail:
            threading.Thread(target=stream_logs, args=(job_id,), daemon=True).start()
        elif not args.no_tail:
            stream_logs(job_id)            # plain blocking tail (no download to gate)
        elif not args.download:
            print(f"\nWatch:  python comfy_spark/comfy_launch.py --logs {job_id}")
            print(f"Cancel: python comfy_spark/comfy_launch.py --cancel {job_id}")
        if args.download:
            # Per-job subfolder so renders don't overwrite each other (every job
            # writes LTX2.3_00001.*). Name: <MMDDYY_MMSS>_<input basename>.
            stamp = time.strftime("%m%d%y_%M%S")
            base_in = os.path.splitext(os.path.basename(input_files[0]))[0] if input_files else job_name
            safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base_in)[:40] or "job"
            dest = os.path.join(args.download, f"{stamp}_{safe}")
            os.makedirs(dest, exist_ok=True)
            # Always confirm the job is really done before harvesting (a tail
            # may have been detached with Ctrl+C, or --no-tail was used).
            print("[download] waiting for the job to finish...")
            status = wait_for_terminal(job_id)
            dl_url = resolve_output_url(resp, job_id, upload_url)
            print(f"[download] job {status}; pulling outputs -> {os.path.abspath(dest)}")
            if not dl_url:
                print("[download] could not resolve the output URL.")
                print(f"[download] submit 'output' block was: {resp.get('output')}")
                print("[download] grab the file from the ShareSync web UI instead.")
            else:
                print(f"[download] source: {dl_url}")
                count = download_outputs(dl_url, dest)
                print(f"[download] {count} file(s) -> {os.path.abspath(dest)}")
    finally:
        try:
            os.unlink(tar_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
