"""
comfy_run.py — headless ComfyUI batch runner that executes INSIDE the
Spark Fuse container.

Packed into the input tarball at /input/comfy_run.py by comfy_launch.py and
invoked as the job command:  ["python", "/input/comfy_run.py"].

Pure stdlib (urllib/json/subprocess) so it runs on any ComfyUI image.

Flow:
    1. (optional) download a LoRA into <COMFY_HOME>/models/loras
    2. launch ComfyUI (--listen 127.0.0.1), relaying its logs into ours
    3. wait for /system_stats (server ready)
    4. load the workflow. If it's a UI/graph export (has a "nodes" list),
       convert it to API format using the server's own /object_info schema,
       and write the result to /output for reuse.
    5. apply any /input/patches.json edits (node/path/value) to the API graph
    6. POST the API graph to /prompt and poll /history to completion
       (skipped if COMFY_CONVERT_ONLY=1 — emit the converted graph and stop)
    7. exit 0 on success, non-zero on failure

Inputs are mounted at /input (ComfyUI --input-directory). Outputs go to /output
(ComfyUI --output-directory), streamed to ShareSync by the Spark agent.

Environment (set by comfy_launch.py):
    COMFY_HOME          ComfyUI install dir              (default /ComfyUI)
    COMFY_WORKFLOW      workflow JSON, UI or API format  (default /input/workflow.json)
    COMFY_PATCHES       patch ops JSON                   (default /input/patches.json)
    COMFY_PORT          port ComfyUI binds               (default 8188)
    COMFY_INPUT_DIR     ComfyUI input dir                (default /input)
    COMFY_OUTPUT_DIR    ComfyUI output dir               (default /output)
    COMFY_LORA_URL      optional direct .safetensors URL to fetch
    COMFY_LORA_NAME     filename to save the LoRA as     (default lora.safetensors)
    COMFY_EXTRA_ARGS    extra args appended to main.py
    COMFY_READY_TIMEOUT seconds to wait for startup      (default 300)
    COMFY_CONVERT_ONLY  "1" = convert + emit api.json, don't render

  No-build path (assemble the environment on the node, public base image):
    COMFY_BUNDLE        dir holding a pristine ComfyUI (with main.py) to seed
                        COMFY_HOME from if it's empty (e.g. the yanwk image's
                        /default-comfyui-bundle/ComfyUI). Run comfy_run.py under
                        the image's real interpreter (python3.13 for yanwk) so
                        pip + torch + ComfyUI deps resolve — see comfy_launch.py.
    COMFY_FETCH_NODES   JSON list of git repo URLs to clone into custom_nodes
                        (+ pip install requirements.txt). Always fetched.
    COMFY_FETCH_MODELS  JSON list of {"url","dest"} (dest relative to COMFY_HOME,
                        e.g. "models/vae/x.safetensors"). Skipped on CONVERT_ONLY
                        — /object_info only needs the node classes, not weights.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
# INPUT_TYPES whose first element marks a settable widget rather than a wired slot.
WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
# control_after_generate appends one of these keyword values to widgets_values.
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def log(msg):
    print(f"[comfy_run] {msg}", flush=True)


def hold_for_sync():
    """Files written to /output just before the container exits can miss the
    agent's ShareSync upload — a fast-exiting convert-only job loses the freshly
    written workflow_api.json this way. Hold briefly so the sync catches up."""
    secs = int(env("COMFY_EXIT_HOLD", "15"))
    if secs > 0:
        log(f"holding {secs}s so /output finishes syncing to ShareSync...")
        time.sleep(secs)


def http_json(url, method="GET", payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def download(url, dest, label="file"):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log(f"downloading {label} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "comfy_run/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    log(f"  {label} done ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")


def ensure_comfyui(comfy_home, bundle):
    """Some public images ship ComfyUI in a pristine bundle dir and copy it into
    place via their entrypoint (which our command override bypasses). If
    COMFY_HOME has no main.py but a bundle does, seed it (matches the image's own
    `cp --archive --update=none`, so it won't clobber already-cloned nodes)."""
    if os.path.isfile(os.path.join(comfy_home, "main.py")):
        return
    if bundle and os.path.isfile(os.path.join(bundle, "main.py")):
        log(f"seeding ComfyUI from bundle: {bundle} -> {comfy_home}")
        os.makedirs(comfy_home, exist_ok=True)
        subprocess.run(["cp", "--archive", "--update=none", f"{bundle}/.", f"{comfy_home}/"])
    if not os.path.isfile(os.path.join(comfy_home, "main.py")):
        log(f"WARNING: no ComfyUI main.py at {comfy_home} and no usable COMFY_BUNDLE")


def fetch_nodes(comfy_home, repos):
    """Clone custom-node repos into custom_nodes and pip-install their reqs.
    Idempotent: a repo already present is left as-is (warm-start reuse)."""
    if not repos:
        return
    cn = os.path.join(comfy_home, "custom_nodes")
    os.makedirs(cn, exist_ok=True)
    for url in repos:
        name = url.rstrip("/").split("/")[-1]
        dest = os.path.join(cn, name)
        if os.path.isdir(dest):
            log(f"node pack present: {name}")
        else:
            log(f"cloning node pack: {name}")
            if subprocess.run(["git", "clone", "--depth", "1", url, dest]).returncode != 0:
                log(f"WARNING: git clone failed for {name} — workflow may not load")
                continue
        req = os.path.join(dest, "requirements.txt")
        if os.path.isfile(req):
            log(f"pip install -r {name}/requirements.txt")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req])


def fetch_models(comfy_home, models):
    """Download each {url,dest} into COMFY_HOME if not already present."""
    for m in models:
        dest = os.path.join(comfy_home, m["dest"])
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            log(f"model present: {m['dest']}")
            continue
        download(m["url"], dest, os.path.basename(dest))


def relay(stream, prefix):
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    stream.close()


def wait_ready(port, timeout, proc):
    deadline = time.time() + timeout
    url = f"http://{HOST}:{port}/system_stats"
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"ERROR: ComfyUI exited during startup (code {proc.returncode}).")
            return False
        try:
            http_json(url, timeout=5)
            log("ComfyUI is up.")
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    log(f"ERROR: ComfyUI did not become ready within {timeout}s.")
    return False


# ── UI/graph → API-format conversion ─────────────────────────────────────────

def _is_widget(definition):
    """An INPUT_TYPES entry is a settable widget if its type is a scalar
    (INT/FLOAT/STRING/BOOLEAN/COMBO) or a literal list of combo options."""
    if not isinstance(definition, (list, tuple)) or not definition:
        return False
    t = definition[0]
    return isinstance(t, list) or t in WIDGET_SCALARS


def ui_to_api(graph, object_info):
    """Convert a ComfyUI UI/graph export to API (/prompt) format using the live
    node schema. Mirrors the frontend's graphToPrompt for the cases this tool
    targets: active nodes only, scalar/combo widgets, named or positional
    widgets_values, control_after_generate offsets. Bypassed (mode 4) and muted
    (mode 2) nodes are dropped — if a kept node depended on one, /prompt fails
    loudly rather than rendering something wrong."""
    links = {L[0]: (L[1], L[2]) for L in graph.get("links", [])}
    api = {}
    skipped = []
    for node in graph["nodes"]:
        if node.get("mode", 0) in (2, 4):
            skipped.append((node["id"], node["type"]))
            continue
        ct = node["type"]
        info = object_info.get(ct)
        if not info:
            # UI-only nodes (notes, reroutes the schema doesn't know) → drop
            continue
        nid = str(node["id"])
        inputs = {}
        connected = set()
        # 1. wired inputs
        for slot in node.get("inputs", []) or []:
            link_id = slot.get("link")
            name = slot.get("name")
            if link_id is not None and link_id in links:
                origin_node, origin_slot = links[link_id]
                inputs[name] = [str(origin_node), origin_slot]
                connected.add(name)
        # 2. widget values
        ordered = []
        spec = info.get("input", {}) or {}
        for group in ("required", "optional"):
            for name, definition in (spec.get(group) or {}).items():
                ordered.append((name, definition))
        wv = node.get("widgets_values")
        if isinstance(wv, dict):
            # newer frontend: named widget values (e.g. VHS_* nodes)
            valid = {n for n, _ in ordered}
            for name, val in wv.items():
                if name in valid and name not in connected:
                    inputs[name] = val
        elif isinstance(wv, list):
            # Positional widget mapping. Two graphToPrompt subtleties:
            #  (a) a widget converted to an input socket usually leaves its stale
            #      value in widgets_values; if the array carries a value for EVERY
            #      widget (incl. converted ones) we must consume the connected
            #      ones so the free widgets line up. We detect this by length.
            #  (b) seed/INT widgets with control_after_generate append a keyword
            #      ("fixed"/"randomize"/…) that is not an API input — skip it.
            widget_inputs = [(n, d) for n, d in ordered if _is_widget(d)]
            n_all = len(widget_inputs)
            n_free = sum(1 for n, _ in widget_inputs if n not in connected)
            stale_present = len(wv) >= n_all > n_free
            idx = 0
            for name, definition in widget_inputs:
                if name in connected:
                    if stale_present and idx < len(wv):
                        idx += 1  # consume the converted widget's stale value
                    continue
                if idx >= len(wv):
                    break
                inputs[name] = wv[idx]
                idx += 1
                if idx < len(wv) and isinstance(wv[idx], str) and wv[idx] in CONTROL_VALUES:
                    idx += 1  # control_after_generate companion
        api[nid] = {"class_type": ct, "inputs": inputs}
    if skipped:
        log(f"Skipped {len(skipped)} muted/bypassed node(s): {skipped}")
    # Prune to nodes reachable from output nodes — graphToPrompt drops dead-ends
    # (loggers, previews, anything not feeding a save/output node). Including them
    # only invites validation errors on nodes that wouldn't even execute.
    output_ids = [nid for nid, n in api.items()
                  if (object_info.get(n["class_type"]) or {}).get("output_node")]
    if output_ids:
        keep, stack = set(), list(output_ids)
        while stack:
            nid = stack.pop()
            if nid in keep or nid not in api:
                continue
            keep.add(nid)
            for v in api[nid]["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    stack.append(v[0])
        dropped = [nid for nid in api if nid not in keep]
        if dropped:
            log(f"Pruned {len(dropped)} node(s) not feeding an output: {dropped}")
        api = {nid: n for nid, n in api.items() if nid in keep}
    log(f"Converted UI graph → API format ({len(api)} nodes).")
    return api


def is_ui_graph(obj):
    return isinstance(obj, dict) and isinstance(obj.get("nodes"), list)


# ── patching (server-side, post-conversion) ──────────────────────────────────

def apply_patch(workflow, node_id, dotted_path, value):
    node_id = str(node_id)
    if node_id not in workflow:
        log(f"WARNING: patch target node {node_id} not in graph — skipping {dotted_path}")
        return
    cur = workflow[node_id]
    parts = dotted_path.split(".")
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# ── prompt lifecycle ──────────────────────────────────────────────────────────

# Model-name COMBO inputs whose "value not in list" errors are expected on
# convert-only (the weights aren't downloaded), so they're filtered from validation.
MODEL_INPUTS = {"unet_name", "vae_name", "clip_name", "clip_name1", "clip_name2",
                "lora_name", "ckpt_name", "model_name", "style_model_name",
                "control_net_name", "gguf_name"}


def validate_graph(port, workflow):
    """Submit to /prompt purely to surface conversion errors, then bail. Models
    aren't present on convert-only, so the expected 'value not in list' errors
    for model-name inputs are filtered; anything else is a real conversion bug."""
    try:
        resp = http_json(f"http://{HOST}:{port}/prompt", method="POST",
                         payload={"prompt": workflow, "client_id": "validate"})
    except urllib.error.HTTPError as e:
        try:
            resp = json.loads(e.read().decode())
        except Exception:  # noqa: BLE001
            resp = {}
    except Exception as e:  # noqa: BLE001
        log(f"validation: could not reach /prompt ({e})")
        return
    real = {}
    for nid, info in (resp.get("node_errors") or {}).items():
        kept = [er for er in info.get("errors", [])
                if not (er.get("type") == "value_not_in_list"
                        and (er.get("extra_info") or {}).get("input_name") in MODEL_INPUTS)]
        if kept:
            real[nid] = {"class_type": info.get("class_type"), "errors": kept}
    if real:
        log("VALIDATION FAILED — the converted graph still has problems:")
        log(json.dumps(real, indent=1)[:3000])
    else:
        log("VALIDATION OK — /prompt accepted the graph (model files aside).")


def queue_prompt(port, workflow, client_id):
    resp = http_json(f"http://{HOST}:{port}/prompt", method="POST",
                     payload={"prompt": workflow, "client_id": client_id})
    if "error" in resp or resp.get("node_errors"):
        log("ERROR: ComfyUI rejected the workflow:")
        log(json.dumps(resp, indent=2)[:2000])
        return None
    return resp.get("prompt_id")


def wait_for_completion(port, prompt_id, poll=3.0):
    url = f"http://{HOST}:{port}/history/{prompt_id}"
    while True:
        try:
            hist = http_json(url, timeout=15)
        except (urllib.error.URLError, OSError):
            time.sleep(poll)
            continue
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                _report_outputs(entry.get("outputs", {}))
                return True
            if status.get("status_str") == "error":
                log("ERROR: workflow execution failed:")
                for m in status.get("messages", []):
                    log(f"  {m}")
                return False
        time.sleep(poll)


def _report_outputs(outputs):
    n = 0
    for node_id, out in outputs.items():
        for key in ("images", "gifs", "videos", "files"):
            for item in out.get(key, []):
                fn = item.get("filename") if isinstance(item, dict) else item
                sub = item.get("subfolder", "") if isinstance(item, dict) else ""
                log(f"  output[{node_id}] {'/'.join(p for p in (sub, fn) if p)}")
                n += 1
    log(f"{n} output file(s) in {env('COMFY_OUTPUT_DIR', '/output')}")


def main():
    comfy_home = env("COMFY_HOME", "/ComfyUI")
    workflow_path = env("COMFY_WORKFLOW", "/input/workflow.json")
    patches_path = env("COMFY_PATCHES", "/input/patches.json")
    port = int(env("COMFY_PORT", "8188"))
    input_dir = env("COMFY_INPUT_DIR", "/input")
    output_dir = env("COMFY_OUTPUT_DIR", "/output")
    ready_timeout = int(env("COMFY_READY_TIMEOUT", "300"))
    convert_only = env("COMFY_CONVERT_ONLY") == "1"

    if not os.path.isfile(workflow_path):
        sys.exit(f"ERROR: workflow not found at {workflow_path}")
    with open(workflow_path) as f:
        workflow = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Assemble the environment (no-build path) ------------------------------
    # Seed ComfyUI from the image's bundle if needed, then clone node packs.
    # Node packs are always needed (ComfyUI must register the classes, and
    # /object_info drives UI->API conversion). Weights + LoRA are only needed to
    # actually render, so skip them on convert-only to keep that step fast.
    try:
        ensure_comfyui(comfy_home, env("COMFY_BUNDLE"))
        fetch_nodes(comfy_home, json.loads(env("COMFY_FETCH_NODES", "[]")))
        if not convert_only:
            fetch_models(comfy_home, json.loads(env("COMFY_FETCH_MODELS", "[]")))
            lora_url = env("COMFY_LORA_URL")
            if lora_url:
                download(lora_url, os.path.join(comfy_home, "models", "loras",
                                                env("COMFY_LORA_NAME", "lora.safetensors")), "LoRA")
    except Exception as e:  # noqa: BLE001 — fail the job loudly
        sys.exit(f"ERROR: environment assembly failed: {e}")

    # 2. Launch ComfyUI --------------------------------------------------------
    extra = env("COMFY_EXTRA_ARGS", "")
    cmd = [sys.executable, "main.py", "--listen", HOST, "--port", str(port),
           "--input-directory", input_dir, "--output-directory", output_dir,
           "--disable-auto-launch"] + (extra.split() if extra else [])
    log(f"Starting ComfyUI: {' '.join(cmd)} (cwd={comfy_home})")
    proc = subprocess.Popen(cmd, cwd=comfy_home, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=relay, args=(proc.stdout, "  | "), daemon=True).start()

    exit_code = 1
    try:
        if not wait_ready(port, ready_timeout, proc):
            return 1

        # 4. Convert UI → API if needed ---------------------------------------
        if is_ui_graph(workflow):
            log("Workflow is UI/graph format — converting via /object_info.")
            object_info = http_json(f"http://{HOST}:{port}/object_info", timeout=120)
            workflow = ui_to_api(workflow, object_info)
            with open(os.path.join(output_dir, "workflow_api.json"), "w") as f:
                json.dump(workflow, f, indent=2)
            log("Wrote converted workflow_api.json to output (pull it to reuse).")

        # 5. Apply patches -----------------------------------------------------
        if os.path.isfile(patches_path):
            with open(patches_path) as f:
                patches = json.load(f)
            for op in patches:
                apply_patch(workflow, op["node"], op["path"], op["value"])
            log(f"Applied {len(patches)} patch(es).")

        if convert_only:
            # Belt-and-suspenders: also emit the graph into the job log (which
            # uploads reliably even when the /output sync drops the file),
            # base64-chunked between sentinels for safe recovery from the log.
            blob = base64.b64encode(json.dumps(workflow).encode()).decode()
            print("===WF_API_B64_BEGIN===", flush=True)
            for i in range(0, len(blob), 120):
                print(blob[i:i + 120], flush=True)
            print("===WF_API_B64_END===", flush=True)
            log("COMFY_CONVERT_ONLY set — emitted graph (to /output and log), skipping render.")
            validate_graph(port, workflow)
            hold_for_sync()
            return 0

        # 6. Queue + wait ------------------------------------------------------
        client_id = f"spark-{int(time.time())}"
        log(f"Queuing workflow (client_id={client_id})")
        prompt_id = queue_prompt(port, workflow, client_id)
        if not prompt_id:
            return 1
        log(f"Queued prompt_id={prompt_id} — running...")
        ok = wait_for_completion(port, prompt_id)
        exit_code = 0 if ok else 1
        log("Done." if ok else "Failed.")
        hold_for_sync()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
