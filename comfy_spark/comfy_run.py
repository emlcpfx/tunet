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
    COMFY_LORA_URL      optional direct .safetensors URL to fetch (preset built-in)
    COMFY_LORA_NAME     filename to save the LoRA as     (default lora.safetensors)
    COMFY_LORAS         JSON list of stackable Tier-1 LoRAs: [{"url","file","strength"}].
                        Each is downloaded into models/loras and chained on Spark.
    COMFY_LORA_CHAIN    JSON {"anchor": <node id>, "slot": <out idx, default 0>} —
                        the MODEL output the LoRA stack is spliced onto.
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
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
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
    req = urllib.request.Request(url, headers={"User-Agent": "comfy_run/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        total_mb = total / 1048576
        log(f"downloading {label} ({total_mb:.0f} MB) -> {dest}" if total
            else f"downloading {label} -> {dest}")
        done, start, last = 0, time.time(), time.time()
        while True:
            buf = r.read(1048576)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            now = time.time()
            if now - last >= 3:
                spd = done / (now - start) / 1048576 if now > start else 0
                if total and spd > 0:
                    eta = (total - done) / 1048576 / spd
                    log(f"  {label} {done / 1048576:.0f}/{total_mb:.0f} MB "
                        f"({spd:.0f} MB/s, ~{eta:.0f}s left)")
                else:
                    log(f"  {label} {done / 1048576:.0f} MB ({spd:.0f} MB/s)")
                last = now
    log(f"  {label} done ({os.path.getsize(dest) / 1048576:.1f} MB)")


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
                        if idx < len(wv) and isinstance(wv[idx], str) and wv[idx] in CONTROL_VALUES:
                            idx += 1  # ...and its control_after_generate keyword
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


# ── LoRA chain injection (Tier-1 stacking) ───────────────────────────────────

def inject_lora_chain(workflow, loras, chain):
    """Splice a series of LoraLoaderModelOnly nodes onto a node's MODEL output.

    `chain` = {"anchor": <node id>, "slot": <output index, default 0>}. We build
    anchor -> lora1 -> lora2 -> ... -> loraN, then repoint every OTHER consumer of
    (anchor, slot) to loraN's output. That fan-out handling is why a node feeding
    two guiders (the generate graph) and one feeding a single guider (cleanplate)
    both work: all downstream consumers end up reading through the full stack."""
    if not loras:
        return workflow
    anchor = str(chain.get("anchor"))
    slot = int(chain.get("slot", 0))
    if anchor not in workflow:
        log(f"WARNING: lora_chain anchor {anchor} not in graph — skipping LoRA stack.")
        return workflow

    src = [anchor, slot]
    new_ids = []
    for i, lora in enumerate(loras):
        nid = f"lorastk_{i}"
        workflow[nid] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(src),
                "lora_name": lora["file"],
                "strength_model": float(lora.get("strength", 1.0)),
            },
        }
        new_ids.append(nid)
        src = [nid, 0]
    tail = src  # [last new id, 0]

    # Repoint existing consumers of (anchor, slot) to the chain tail. Skip the
    # nodes we just added (the head must keep reading from the real anchor).
    rewired = 0
    for nid, node in workflow.items():
        if nid in new_ids:
            continue
        for k, v in (node.get("inputs") or {}).items():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == anchor and int(v[1]) == slot:
                node["inputs"][k] = list(tail)
                rewired += 1
    log(f"Injected {len(loras)}-LoRA stack on node {anchor} (MODEL) -> rewired {rewired} consumer(s): "
        + ", ".join(f"{l['file']}@{l.get('strength', 1.0)}" for l in loras))
    return workflow


# ── output-format rewrite (high-bit EXR / ProRes etc.) ───────────────────────

# Terminal nodes whose `images` input carries the final RGB frames tensor — the
# anchor we branch a high-bit saver off. Order = preference when several exist.
OUTPUT_TERMINALS = ["CreateVideo", "VHS_VideoCombine", "SaveImage", "SaveAnimatedWEBP",
                    "SaveAnimatedPNG", "SaveWEBM", "PreviewImage"]


def find_output_anchor(workflow):
    """The [node, slot] feeding the `images` input of a terminal save/video node —
    i.e. the final frames tensor. Lets --output work on any preset without config."""
    for cls in OUTPUT_TERMINALS:
        for nid, node in workflow.items():
            if node.get("class_type") == cls:
                imgs = (node.get("inputs") or {}).get("images")
                if isinstance(imgs, list) and len(imgs) == 2:
                    return [str(imgs[0]), int(imgs[1])]
    return None


def rewrite_output(workflow, spec, anchor, prefix, fps, colorspace):
    """Splice the chosen format's node chain onto the frames anchor as a PARALLEL
    branch — the preset's own saver is left intact, so you still get the mp4
    preview plus the chosen high-bit deliverable. chain[0].in reads the anchor;
    each subsequent node reads the previous node's output."""
    if not spec or not spec.get("chain"):
        return workflow
    anchor = anchor or find_output_anchor(workflow)
    if not anchor:
        log("WARNING: --output set but no frames anchor found (no terminal "
            "CreateVideo/SaveImage with an 'images' input) — keeping the default output only.")
        return workflow
    linear = (colorspace or spec.get("default_colorspace", "as-is")) == "linear"
    src, added = list(anchor), []
    for i, step in enumerate(spec["chain"]):
        if step.get("only_if_linear") and not linear:
            continue
        nid = f"out_{i}"
        inputs = {}
        for k, v in (step.get("inputs") or {}).items():
            inputs[k] = prefix if v == "$PREFIX" else (float(fps) if v == "$FPS" else v)
        inputs[step["in"]] = list(src)
        workflow[nid] = {"class_type": step["class_type"], "inputs": inputs}
        added.append(nid)
        src = [nid, 0]
    log(f"Output rewrite: +{' -> '.join(added)} ({spec.get('label', '?')}) "
        f"off frames {anchor}{' [scene-linear]' if linear else ''}")
    return workflow


# ── input image-sequence ingestion (EXR/… plates in) ─────────────────────────

# Loaders that emit an IMAGE batch on output slot 0 — swappable in place for a
# sequence loader (the IMAGE consumers keep their [id, 0] links). A VIDEO-object
# loader (core LoadVideo) is NOT here: its slot 0 is a VIDEO, not IMAGE.
IMAGE_BATCH_LOADERS = ["VHS_LoadVideo", "VHS_LoadImagesPath", "VHS_LoadImages", "LoadImage"]


def build_seq_loader(seq):
    """The replacement loader node for a packed frame sequence. EXR only for now
    (CoCoTools LoadExrSequence, IMAGE on slot 0). Returns None for kinds we don't
    have a loader for yet (png/dpx folders) so the caller leaves the input alone."""
    if seq.get("kind") == "exr":
        return {"class_type": "LoadExrSequence",
                "inputs": {"sequence_path": seq["pattern"],
                           "start_frame": int(seq["start"]), "end_frame": int(seq["end"]),
                           "frame_step": int(seq.get("step", 1)), "normalize": False}}
    return None


def find_input_loader(workflow):
    for cls in IMAGE_BATCH_LOADERS:
        for nid, node in workflow.items():
            if node.get("class_type") == cls:
                return nid
    return None


def rewrite_input(workflow, seq, anchor):
    """Swap the workflow's primary IMAGE-batch loader for a sequence loader (in
    place, so [loader, 0] IMAGE consumers stay wired), encode scene-linear EXR ->
    display sRGB for the model, and drop the old loader's now-dangling non-IMAGE
    outputs (audio / fps / frame_count — a frame sequence has none)."""
    loader_def = build_seq_loader(seq)
    if not loader_def:
        log(f"WARNING: --input-sequence kind {seq.get('kind')!r} has no loader yet "
            f"(EXR only) — leaving the preset's own input.")
        return workflow
    lid = str(anchor) if anchor else find_input_loader(workflow)
    if not lid or lid not in workflow:
        log("WARNING: --input-sequence set but no swappable IMAGE loader "
            "(VHS_LoadVideo/VHS_LoadImages/LoadImage) found — leaving the input.")
        return workflow
    old = workflow[lid].get("class_type")
    workflow[lid] = loader_def                       # in-place: [lid,0] stays IMAGE
    log(f"Input rewrite: node {lid} {old} -> LoadExrSequence "
        f"({seq['pattern']} frames {seq['start']}-{seq['end']})")

    if (seq.get("colorspace") or "linear") == "linear":   # EXR is scene-linear
        cs = "inseq_cs"
        for nid, node in workflow.items():           # send IMAGE consumers through the encode
            if nid == cs:
                continue
            for k, v in (node.get("inputs") or {}).items():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == lid and int(v[1]) == 0:
                    node["inputs"][k] = [cs, 0]
        workflow[cs] = {"class_type": "ColorspaceNode",
                        "inputs": {"images": [lid, 0],
                                   "from_colorspace": "sRGB Linear", "to_colorspace": "sRGB"}}
        log(f"  + ColorspaceNode {cs} (scene-linear EXR -> display sRGB for the model)")

    dropped = 0                                      # old loader's audio/fps/etc. are gone
    for nid, node in workflow.items():
        if nid in (lid, "inseq_cs"):
            continue
        for k, v in list((node.get("inputs") or {}).items()):
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == lid and int(v[1]) >= 1:
                del node["inputs"][k]
                dropped += 1
    if dropped:
        log(f"  dropped {dropped} link(s) off the old loader's non-IMAGE outputs (no audio/fps in a sequence)")
    return workflow


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
    def _expected(er):
        # Errors that are artifacts of convert-only (no weights, no input clip),
        # not conversion bugs: model COMBO files absent, and the input video/image
        # files absent.
        t = er.get("type")
        name = (er.get("extra_info") or {}).get("input_name")
        if t == "value_not_in_list" and name in MODEL_INPUTS:
            return True
        if t == "custom_validation_failed" and name in {"video", "audio", "image"}:
            return True
        return False

    real = {}
    for nid, info in (resp.get("node_errors") or {}).items():
        kept = [er for er in info.get("errors", []) if not _expected(er)]
        if kept:
            real[nid] = {"class_type": info.get("class_type"), "errors": kept}
    if real:
        log("VALIDATION FAILED — the converted graph still has problems:")
        log(json.dumps(real, indent=1)[:3000])
    else:
        log("VALIDATION OK — /prompt accepted the graph (model files aside).")


def ws_progress(port, client_id, workflow):
    """Best-effort: stream ComfyUI's WebSocket progress events to the log with a
    rough ETA. Minimal pure-stdlib WS client; any failure is swallowed (progress
    is a nicety, not required for the render). Runs in a daemon thread until the
    server closes the socket at teardown."""
    try:
        sock = socket.create_connection((HOST, port), timeout=10)
    except OSError:
        return
    try:
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall(
            (f"GET /ws?clientId={client_id} HTTP/1.1\r\n"
             f"Host: {HOST}:{port}\r\n"
             "Upgrade: websocket\r\nConnection: Upgrade\r\n"
             f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        rbuf = b""
        while b"\r\n\r\n" not in rbuf:                      # consume 101 handshake
            chunk = sock.recv(4096)
            if not chunk:
                return
            rbuf += chunk
        rbuf = rbuf.split(b"\r\n\r\n", 1)[1]

        def recvn(n):
            nonlocal rbuf
            while len(rbuf) < n:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError
                rbuf += chunk
            out, rbuf = rbuf[:n], rbuf[n:]
            return out

        cur, base, last_log = None, {}, 0.0
        while True:
            h = recvn(2)
            opcode, ln = h[0] & 0x0F, h[1] & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", recvn(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", recvn(8))[0]
            if h[1] & 0x80:                                  # masked (unexpected)
                recvn(4)
            payload = recvn(ln) if ln else b""
            if opcode == 0x8:                                # close
                return
            if opcode != 0x1:                                # only text JSON
                continue
            try:
                msg = json.loads(payload.decode("utf-8", "ignore"))
            except ValueError:
                continue
            mtype, d = msg.get("type"), (msg.get("data") or {})
            if mtype == "executing" and d.get("node") is not None:
                cur, base = str(d["node"]), {}
                name = (workflow.get(cur) or {}).get("class_type", cur)
                log(f"running node {cur} ({name})")
            elif mtype == "progress":
                v, m = d.get("value"), d.get("max")
                now = time.time()
                if isinstance(v, int) and isinstance(m, int) and m > 0:
                    base.setdefault(m, (now, v))             # baseline per phase
                    if now - last_log >= 2:
                        t0, v0 = base[m]
                        per = (now - t0) / (v - v0) if v > v0 and now > t0 else 0
                        eta = f" (~{(m - v) * per:.0f}s left)" if per > 0 else ""
                        name = (workflow.get(cur) or {}).get("class_type", "") if cur else ""
                        log(f"  {name + ': ' if name else ''}step {v}/{m}{eta}")
                        last_log = now
    except (OSError, ConnectionError, ValueError):
        return
    finally:
        try:
            sock.close()
        except OSError:
            pass


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


# ── self-upload outputs to ShareSync (the Spark agent doesn't sync /output) ───

def _api_get(url, token):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _webdav(method, url, token, data=None):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=900)


_tok = {"access": "", "exp": 0.0, "refresh": ""}


def _bearer():
    """A currently-valid Spark bearer for the self-upload. COMFY_UPLOAD_TOKEN is
    baked at submit and expires in minutes, but a render can run far longer — so
    when refresh material is present (TUNET_REFRESH_TOKEN/URL) we mint a fresh
    token via tunet-web's /api/spark/refresh proxy, cached until ~1 min before
    expiry, tracking the rotated refresh token. Falls back to COMFY_UPLOAD_TOKEN
    when no refresh material is set (legacy jobs). Never raises."""
    refresh_url = env("TUNET_REFRESH_URL")
    if not _tok["refresh"]:
        _tok["refresh"] = env("TUNET_REFRESH_TOKEN") or ""
    static = env("COMFY_UPLOAD_TOKEN") or ""
    if not (refresh_url and _tok["refresh"]):
        return static
    now = time.time()
    if _tok["access"] and now < _tok["exp"] - 60:
        return _tok["access"]
    try:
        body = json.dumps({"refreshToken": _tok["refresh"]}).encode()
        req = urllib.request.Request(refresh_url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        if data.get("accessToken"):
            _tok["access"] = data["accessToken"]
            _tok["exp"] = now + int(data.get("expiresIn") or 300)
            if data.get("refreshToken"):
                _tok["refresh"] = data["refreshToken"]
            return _tok["access"]
    except Exception as e:  # noqa: BLE001
        log(f"token refresh failed: {e}")
    return _tok["access"] or static


def upload_outputs(output_dir):
    """The Spark agent (v1.177) doesn't ship /output to ShareSync — only its log
    archive. So upload our own outputs there via WebDAV PUT, into the job's own
    output folder (the same place --download and the web UI read). We find that
    folder by looking up our own job via the COMFY_RUN_ID marker. Best-effort:
    logs failures, never raises."""
    run_id = env("COMFY_RUN_ID")
    token = _bearer()
    api = env("COMFY_SPARK_API", "https://api.prod.aapse1.sparkcloud.studio")
    if not token or not run_id:
        return
    base = None
    try:
        jobs = _api_get(f"{api}/api/compute/jobs?tag=cpfx_comfy", token).get("jobs", [])
        for j in jobs:
            jid = j.get("id")
            if not jid:
                continue
            if (j.get("env") or {}).get("COMFY_RUN_ID") != run_id \
                    and j.get("status") not in ("running", "provisioning"):
                continue
            d = _api_get(f"{api}/api/compute/jobs/{jid}", token)
            if (d.get("env") or {}).get("COMFY_RUN_ID") == run_id:
                base = (d.get("output") or {}).get("shareSyncBaseUrl")
                break
    except Exception as e:  # noqa: BLE001
        log(f"self-upload: couldn't resolve output URL ({e})")
        return
    if not base:
        log("self-upload: own job/output URL not found; skipping")
        return
    base = base.rstrip("/")
    try:                                         # agent creates the folder only at
        _webdav("MKCOL", base + "/", token)      # job end, so make it ourselves now
    except Exception:  # noqa: BLE001
        pass
    n = 0
    for root, _, files in os.walk(output_dir):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, output_dir).replace(os.sep, "/")
            segs = [urllib.parse.quote(s) for s in rel.split("/")]
            acc = base                            # MKCOL any intermediate dirs
            for seg in segs[:-1]:
                acc += "/" + seg
                try:
                    _webdav("MKCOL", acc + "/", token)
                except Exception:  # noqa: BLE001
                    pass
            try:
                with open(p, "rb") as f:
                    # refresh per-file: a big render upload can outlast one token
                    _webdav("PUT", base + "/" + "/".join(segs), _bearer(), data=f.read())
                log(f"  uploaded {rel} ({os.path.getsize(p) / 1048576:.1f} MB)")
                n += 1
            except Exception as e:  # noqa: BLE001
                log(f"  upload FAILED {rel}: {e}")
    log(f"self-uploaded {n} output file(s) to ShareSync")


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
            for lora in json.loads(env("COMFY_LORAS", "[]")):
                dest = os.path.join(comfy_home, "models", "loras", lora["file"])
                if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                    log(f"LoRA present: {lora['file']}")
                    continue
                download(lora["url"], dest, f"LoRA {lora['file']}")
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

        # 5a2. Input image-sequence ingestion (swap the loader for a frame seq) --
        in_seq = json.loads(env("COMFY_INPUT_SEQ", "null"))
        if in_seq:
            workflow = rewrite_input(workflow, in_seq, json.loads(env("COMFY_INPUT_SEQ_ANCHOR", "null")))

        # 5b. Splice the stackable Tier-1 LoRA chain onto the model path -------
        lora_stack = json.loads(env("COMFY_LORAS", "[]"))
        lora_chain = json.loads(env("COMFY_LORA_CHAIN", "null"))
        if lora_stack and lora_chain:
            workflow = inject_lora_chain(workflow, lora_stack, lora_chain)
            # Re-emit so the saved/reusable api graph reflects the spliced stack.
            with open(os.path.join(output_dir, "workflow_api.json"), "w") as f:
                json.dump(workflow, f, indent=2)

        # 5c. Output-format rewrite (high-bit EXR / ProRes; default mp4 = no-op) -
        out_spec = json.loads(env("COMFY_OUTPUT_SPEC", "null"))
        if out_spec:
            workflow = rewrite_output(
                workflow, out_spec, json.loads(env("COMFY_OUTPUT_ANCHOR", "null")),
                env("COMFY_OUTPUT_PREFIX", "render"), env("COMFY_OUTPUT_FPS", "24"),
                env("COMFY_OUTPUT_COLORSPACE"))
            with open(os.path.join(output_dir, "workflow_api.json"), "w") as f:
                json.dump(workflow, f, indent=2)

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
            upload_outputs(output_dir)
            hold_for_sync()
            return 0

        # 6. Queue + wait ------------------------------------------------------
        client_id = f"spark-{int(time.time())}"
        log(f"Queuing workflow (client_id={client_id})")
        prompt_id = queue_prompt(port, workflow, client_id)
        if not prompt_id:
            return 1
        log(f"Queued prompt_id={prompt_id} — running...")
        threading.Thread(target=ws_progress, args=(port, client_id, workflow),
                         daemon=True).start()
        ok = wait_for_completion(port, prompt_id)
        exit_code = 0 if ok else 1
        log("Done." if ok else "Failed.")
        upload_outputs(output_dir)
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
