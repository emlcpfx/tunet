"""
lora_train_run.py — runs INSIDE the Spark container to train an LTX-2.3 LoRA.

The training analog of comfy_run.py. Pure stdlib so it runs on any CUDA-13 base
image without extra installs. Driven entirely by env vars set by lora_train.py:

    TRAIN_RECIPE        /input/recipe.json  (resolved recipe: image/gpu/lora/opt/…)
    TRAIN_DATASET_JSON  /input/dataset.json (list of {media_path, caption})
    TRAIN_DATASET_DIR   /input/dataset      (the images + optional .txt sidecars)
    TRAIN_TRIGGER       distinctive trigger word, auto-prepended to every caption
    TRAIN_OUTPUT_DIR    /output             (final LoRA + validation samples land here)
    TRAIN_UPLOAD_TOKEN  Spark bearer token  (self-upload /output to ShareSync)
    TRAIN_RUN_ID        marker to find our own job for the output URL
    TRAIN_GROUP_TAG     job tag used to look ourselves up (default cpfx_train)
    TRAIN_SPARK_API     Spark API base (default prod)

Flow: clone LTX-2 → uv sync the ltx-trainer env → fetch base weights → caption
gaps → process_dataset.py (cache latents) → write YAML → train.py → collect the
trained .safetensors into /output → self-upload to ShareSync.

The Spark agent doesn't ship /output to ShareSync for these jobs (same as comfy
jobs), so we PUT our own outputs there over WebDAV — see upload_outputs().
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


# ── tiny helpers (same shape as comfy_run.py) ─────────────────────────────────

def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def log(msg):
    print(f"[train_run] {msg}", flush=True)


def run(cmd, cwd=None, check=True, env_extra=None):
    """Run a shell command, streaming its output. Returns the exit code."""
    log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    e = dict(os.environ)
    if env_extra:
        e.update(env_extra)
    r = subprocess.run(cmd, cwd=cwd, env=e, shell=isinstance(cmd, str))
    if check and r.returncode != 0:
        sys.exit(f"[train_run] FATAL: command failed (exit {r.returncode}): {cmd}")
    return r.returncode


def download(url, dest, label="file"):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "lora_train_run/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
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
                log(f"  {label} {done / 1048576:.0f}"
                    + (f"/{total_mb:.0f}" if total else "") + f" MB ({spd:.0f} MB/s)")
                last = now
    log(f"  {label} done ({os.path.getsize(dest) / 1048576:.1f} MB)")


def hold_for_sync():
    secs = int(env("TRAIN_EXIT_HOLD", "15"))
    if secs > 0:
        log(f"holding {secs}s so /output finishes syncing...")
        time.sleep(secs)


# ── base-weight fetch ─────────────────────────────────────────────────────────

def hf_snapshot(uv, trainer_dir, repo, dest, token):
    """Download a whole HF model repo into `dest` (the trainer's --text-encoder-path
    wants a model DIRECTORY). Run inside the trainer's uv env, which has
    huggingface_hub; fall back to a system-python install if that fails."""
    os.makedirs(dest, exist_ok=True)
    code = ("from huggingface_hub import snapshot_download; "
            f"snapshot_download({repo!r}, local_dir={dest!r}, "
            f"token=({token!r} or None), ignore_patterns=['*.gguf'])")
    if run([uv, "run", "python", "-c", code], cwd=trainer_dir, check=False) != 0:
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
        run([sys.executable, "-c", code], check=False)


def fetch_models(models, root, uv, trainer_dir):
    """Fetch each weight under `root` if not already present. Two forms:
      - {"url": <direct .safetensors>, "dest": <file path>}  → single-file download
      - {"hf_repo": <repo id>,         "dest": <dir path>}   → snapshot the whole repo
    Returns {dest_rel: absolute_path}."""
    token = env("TRAIN_HF_TOKEN")
    paths = {}
    for m in models:
        dest = os.path.join(root, m["dest"])
        if m.get("hf_repo"):
            if os.path.isdir(dest) and os.listdir(dest):
                log(f"model dir present: {m['dest']}")
            else:
                log(f"snapshot-downloading {m['hf_repo']} -> {m['dest']}")
                hf_snapshot(uv, trainer_dir, m["hf_repo"], dest, token)
        else:
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                log(f"weight present: {m['dest']}")
            else:
                download(m["url"], dest, os.path.basename(dest))
        paths[m["dest"]] = dest
    return paths


# ── captioning (best-effort, runs in the trainer's own uv env) ────────────────

CAPTION_SNIPPET = r'''
import sys, json, os
imgs = json.load(open(sys.argv[1]))
out = {}
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large").to(dev)
    from PIL import Image
    for p in imgs:
        try:
            im = Image.open(p).convert("RGB")
            inp = proc(im, return_tensors="pt").to(dev)
            ids = model.generate(**inp, max_new_tokens=40)
            out[p] = proc.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            out[p] = ""
except Exception as e:
    sys.stderr.write(f"captioner unavailable: {e}\n")
json.dump(out, open(sys.argv[2], "w"))
'''


def auto_caption(uv, trainer_dir, image_paths):
    """Caption images using BLIP inside the trainer's uv env. Returns {path:cap}.
    Never raises — on any failure returns {} and we fall back to a default."""
    if not image_paths:
        return {}
    snip = "/tmp/_caption.py"
    inp = "/tmp/_caption_in.json"
    outp = "/tmp/_caption_out.json"
    with open(snip, "w") as f:
        f.write(CAPTION_SNIPPET)
    with open(inp, "w") as f:
        json.dump(image_paths, f)
    log(f"auto-captioning {len(image_paths)} image(s) with BLIP (best-effort)...")
    rc = run([uv, "run", "python", snip, inp, outp], cwd=trainer_dir, check=False)
    if rc != 0 or not os.path.isfile(outp):
        log("auto-caption step failed; falling back to default captions")
        return {}
    try:
        caps = json.load(open(outp))
        return {k: v for k, v in caps.items() if v}
    except Exception:  # noqa: BLE001
        return {}


# ── ShareSync self-upload (same mechanism as comfy_run.upload_outputs) ────────

def _api_get(url, token):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _webdav(method, url, token, data=None):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=900)


def upload_outputs(output_dir):
    token, run_id = env("TRAIN_UPLOAD_TOKEN"), env("TRAIN_RUN_ID")
    api = env("TRAIN_SPARK_API", "https://api.prod.aapse1.sparkcloud.studio")
    tag = env("TRAIN_GROUP_TAG", "cpfx_train")
    if not token or not run_id:
        return
    base = None
    try:
        jobs = _api_get(f"{api}/api/compute/jobs?tag={tag}", token).get("jobs", [])
        for j in jobs:
            jid = j.get("id")
            if not jid:
                continue
            if (j.get("env") or {}).get("TRAIN_RUN_ID") != run_id \
                    and j.get("status") not in ("running", "provisioning"):
                continue
            d = _api_get(f"{api}/api/compute/jobs/{jid}", token)
            if (d.get("env") or {}).get("TRAIN_RUN_ID") == run_id:
                base = (d.get("output") or {}).get("shareSyncBaseUrl")
                break
    except Exception as e:  # noqa: BLE001
        log(f"self-upload: couldn't resolve output URL ({e})")
        return
    if not base:
        log("self-upload: own job/output URL not found; skipping")
        return
    base = base.rstrip("/")
    try:
        _webdav("MKCOL", base + "/", token)
    except Exception:  # noqa: BLE001
        pass
    n = 0
    for root, _, files in os.walk(output_dir):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, output_dir).replace(os.sep, "/")
            segs = [urllib.parse.quote(s) for s in rel.split("/")]
            acc = base
            for seg in segs[:-1]:
                acc += "/" + seg
                try:
                    _webdav("MKCOL", acc + "/", token)
                except Exception:  # noqa: BLE001
                    pass
            try:
                with open(p, "rb") as f:
                    _webdav("PUT", base + "/" + "/".join(segs), token, data=f.read())
                log(f"  uploaded {rel} ({os.path.getsize(p) / 1048576:.1f} MB)")
                n += 1
            except Exception as e:  # noqa: BLE001
                log(f"  upload FAILED {rel}: {e}")
    log(f"self-uploaded {n} output file(s) to ShareSync")


# ── YAML config (write by hand — no PyYAML dependency in the base image) ──────

def write_config(path, recipe, ckpt_path, enc_path, data_root, output_dir, trigger):
    lo = recipe.get("lora", {})
    op = recipe.get("optimization", {})
    ck = recipe.get("checkpoints", {})
    va = recipe.get("validation", {})
    low_vram = bool(recipe.get("low_vram"))
    w, h = (recipe.get("resolution", "960x544").split("x") + ["544"])[:2]

    def ylist(items, indent):
        if not items:
            return " []"
        pad = " " * indent
        return "\n" + "\n".join(f'{pad}- "{str(s).replace(chr(34), chr(39))}"' for s in items)

    prompts = va.get("prompts") or ["a portrait, soft natural light"]
    lines = [
        "# Generated by lora_train_run.py from the recipe — image-only LTX-2.3 LoRA.",
        f'# trigger word (auto-prepended to every caption at preprocess): "{trigger}"',
        "model:",
        f'  model_path: "{ckpt_path}"',
        f'  text_encoder_path: "{enc_path}"',
        '  training_mode: "lora"',
        "  load_checkpoint: null",
        "lora:",
        f"  rank: {lo.get('rank', 32)}",
        f"  alpha: {lo.get('alpha', lo.get('rank', 32))}",
        f"  dropout: {lo.get('dropout', 0.0)}",
        "  target_modules: [ \"to_k\", \"to_q\", \"to_v\", \"to_out.0\" ]",
        "training_strategy:",
        '  name: "text_to_video"',
        "  first_frame_conditioning_p: 0.5",
        "  with_audio: false",
        "optimization:",
        f"  learning_rate: {op.get('learning_rate', 1e-4)}",
        f"  steps: {op.get('steps', 2000)}",
        f"  batch_size: {op.get('batch_size', 1)}",
        f"  gradient_accumulation_steps: {op.get('gradient_accumulation_steps', 1)}",
        "  max_grad_norm: 1.0",
        '  optimizer_type: "adamw"',
        '  scheduler_type: "linear"',
        "  scheduler_params: {}",
        "  enable_gradient_checkpointing: true",
        "acceleration:",
        '  mixed_precision_mode: "bf16"',
        f'  quantization: {"int8-quanto" if low_vram else "null"}',
        f"  load_text_encoder_in_8bit: {str(low_vram).lower()}",
        "  offload_optimizer_during_validation: false",
        "data:",
        f'  preprocessed_data_root: "{data_root}"',
        "  num_dataloader_workers: 2",
        "validation:",
        "  prompts:" + ylist(prompts, 4),
        '  negative_prompt: "worst quality, blurry, distorted, jittery"',
        "  images: null",
        f"  video_dims: [ {int(w)}, {int(h)}, 1 ]",
        "  frame_rate: 25.0",
        "  seed: 42",
        f"  inference_steps: {va.get('inference_steps', 30)}",
        f"  interval: {va.get('interval', 250)}",
        "  guidance_scale: 4.0",
        "  generate_audio: false",
        "  skip_initial_validation: true",
        "checkpoints:",
        f"  interval: {ck.get('interval', 250)}",
        f"  keep_last_n: {ck.get('keep_last_n', -1)}",
        '  precision: "bfloat16"',
        "flow_matching:",
        '  timestep_sampling_mode: "shifted_logit_normal"',
        "  timestep_sampling_params: {}",
        "hub: { push_to_hub: false, hub_model_id: null }",
        "wandb: { enabled: false }",
        "seed: 42",
        f'output_dir: "{output_dir}"',
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"wrote training config -> {path}")


# ── find the trained LoRA after training ──────────────────────────────────────

def collect_lora(output_dir, dest_dir, trigger):
    """The trainer writes checkpoints + a final LoRA under output_dir. Copy the
    newest .safetensors to dest_dir under a friendly name so --download grabs it."""
    found = []
    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn.endswith(".safetensors"):
                found.append(os.path.join(root, fn))
    if not found:
        log(f"WARNING: no .safetensors produced under {output_dir}")
        return None
    found.sort(key=lambda p: os.path.getmtime(p))
    final = found[-1]
    os.makedirs(dest_dir, exist_ok=True)
    name = f"{trigger or 'lora'}_ltx2.3.safetensors"
    out = os.path.join(dest_dir, name)
    if os.path.abspath(final) != os.path.abspath(out):
        shutil.copy2(final, out)
    log(f"trained LoRA: {out} ({os.path.getsize(out) / 1048576:.1f} MB) "
        f"[from {os.path.relpath(final, output_dir)}]")
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def find_uv():
    """Ensure `uv` is available; install it into the system python if missing."""
    for cand in ("uv", os.path.expanduser("~/.local/bin/uv"), "/root/.local/bin/uv"):
        if shutil.which(cand) or os.path.isfile(cand):
            return cand
    log("installing uv...")
    run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)
    return shutil.which("uv") or "uv"


def main():
    recipe = json.load(open(env("TRAIN_RECIPE", "/input/recipe.json")))
    dataset_json = env("TRAIN_DATASET_JSON", "/input/dataset.json")
    dataset_dir = env("TRAIN_DATASET_DIR", "/input/dataset")
    trigger = env("TRAIN_TRIGGER", "")
    output_dir = env("TRAIN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)

    work = "/opt/ltx2"
    repo_url = recipe.get("repo", "https://github.com/Lightricks/LTX-2")
    trainer_dir = os.path.join(work, recipe.get("trainer_subdir", "packages/ltx-trainer"))
    models_root = "/models"

    # 1) clone the trainer monorepo (idempotent for warm-node reuse)
    if not os.path.isdir(os.path.join(work, ".git")):
        run(["git", "clone", "--depth", "1", repo_url, work])
    else:
        log("LTX-2 repo present (warm node)")

    # 2) build the trainer env with uv
    uv = find_uv()
    run([uv, "sync"], cwd=work, check=False) or run([uv, "sync"], cwd=trainer_dir, check=False)

    # 3) base weights onto the node (single-file checkpoint + gemma model dir)
    paths = fetch_models(recipe.get("models", []), models_root, uv, trainer_dir)
    model_entries = recipe.get("models", [])
    ckpt_path = paths.get(model_entries[0]["dest"]) if model_entries else ""
    enc_path = paths.get(model_entries[1]["dest"]) if len(model_entries) > 1 else ""

    # 4) build the dataset.json the trainer reads: captions from .txt sidecars,
    #    auto-caption the rest (best-effort), default for anything still empty.
    entries = json.load(open(dataset_json))
    need_caption = []
    for e in entries:
        media = e["media_path"]
        if not os.path.isabs(media):
            media = os.path.join(dataset_dir, os.path.basename(media))
            e["media_path"] = media
        if not e.get("caption"):
            need_caption.append(media)
    caps = auto_caption(uv, trainer_dir, need_caption) if recipe.get("auto_caption", True) else {}
    default_cap = env("TRAIN_DEFAULT_CAPTION", "a photo")  # --lora-trigger is prepended by the trainer
    for e in entries:
        if not e.get("caption"):
            e["caption"] = caps.get(e["media_path"], default_cap)
    resolved_ds = "/tmp/dataset_resolved.json"
    json.dump(entries, open(resolved_ds, "w"), indent=2)
    log(f"dataset: {len(entries)} item(s), {len(need_caption)} auto-captioned, "
        f"{len(need_caption) - len(caps)} fell back to default")

    # 5) preprocess (cache latents + text embeddings). Images => frames bucket = 1.
    res = recipe.get("resolution", "960x544")
    buckets = f"{res}x1"
    precomp = "/tmp/precomputed"
    pre_cmd = [uv, "run", "python", "scripts/process_dataset.py", resolved_ds,
               "--resolution-buckets", buckets,
               "--model-path", ckpt_path,
               "--text-encoder-path", enc_path,
               "--output-dir", precomp]
    if trigger:
        pre_cmd += ["--lora-trigger", trigger]
    run(pre_cmd, cwd=trainer_dir)

    # 6) write the YAML config from the recipe
    cfg = "/tmp/train_config.yaml"
    train_out = os.path.join(output_dir, "lora")
    write_config(cfg, recipe, ckpt_path, enc_path, precomp, train_out, trigger)

    # 7) train
    run([uv, "run", "python", "scripts/train.py", cfg], cwd=trainer_dir)

    # 8) collect the trained LoRA into /output and self-upload to ShareSync
    collect_lora(train_out, output_dir, trigger)
    shutil.copy2(cfg, os.path.join(output_dir, "train_config.yaml"))
    upload_outputs(output_dir)
    hold_for_sync()
    log("done.")


if __name__ == "__main__":
    main()
