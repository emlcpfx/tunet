"""
lora_train.py — train your own LTX-2.3 LoRA on Spark Fuse, then stack it in comfy_spark.

The training sibling of comfy_launch.py. You point it at a folder of images and a
trigger word; it packs them with a recipe, submits a Spark job that runs the
official Lightricks ltx-trainer (lora_train_run.py inside the container), streams
the log, and downloads the trained .safetensors. Drop that file's ShareSync URL
into `comfy_launch.py --lora <url>` and you're generating with your own look.

It reuses comfy_launch.py's proven spine — auth, the Spark API helper, tarball
upload, log streaming, and ShareSync download — so billing, credentials, and
output handling are identical. The only differences here are the job payload
(a training run, not a render) and a distinct `cpfx_train` group tag.

Quick start:
    # 1) fill the two REPLACE_ base-weight URLs in trainers/ltx2_style.train.json
    # 2) validate for free (prints dataset.json + training plan; spends nothing):
    python comfy_spark/lora_train.py --recipe ltx2_style --dataset ./mylook \
        --trigger raxstyle --dry-run
    # 3) train:
    python comfy_spark/lora_train.py --recipe ltx2_style --dataset ./mylook \
        --trigger raxstyle --idle-hold 300 --download ./loras_out

Auth + billing: same as comfy_launch / spark_launch (SPARK_EMAIL / SPARK_PASSWORD
in the repo .env; every job carries the mandatory cpfx_tunet billing tag).
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

# Reuse the whole Spark spine from the renderer's launcher.
import comfy_launch as cl

TRAINERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainers")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_train_run.py")
GROUP_TAG = "cpfx_train"                 # filterable; cpfx_tunet (billing) is always added too
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


# ── recipes ───────────────────────────────────────────────────────────────────

def load_recipe(name):
    path = os.path.join(TRAINERS_DIR, f"{name}.train.json")
    if not os.path.isfile(path):
        avail = [f[:-11] for f in os.listdir(TRAINERS_DIR) if f.endswith(".train.json")] \
            if os.path.isdir(TRAINERS_DIR) else []
        sys.exit(f"ERROR: recipe {name!r} not found. Available: {', '.join(avail) or '(none)'}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_recipes():
    if not os.path.isdir(TRAINERS_DIR):
        print("(no trainers/ dir)")
        return
    for f in sorted(os.listdir(TRAINERS_DIR)):
        if f.endswith(".train.json"):
            r = json.load(open(os.path.join(TRAINERS_DIR, f), encoding="utf-8"))
            print(f"  {f[:-11]:16s} [{r.get('kind','?')}]  {r.get('description','')[:90]}")


# ── dataset ─────────────────────────────────────────────────────────────────

def collect_dataset(folder, kind):
    """Walk the dataset folder; return ([{media_path, caption}], [local file paths]).
    A caption comes from a same-named .txt sidecar if present, else "" (the node
    captions it). media_path is the bare basename (the tarball flattens into
    /input/dataset)."""
    if not os.path.isdir(folder):
        sys.exit(f"ERROR: --dataset {folder!r} is not a folder")
    exts = IMAGE_EXTS if kind == "image" else (IMAGE_EXTS | VIDEO_EXTS)
    entries, files, names = [], [], set()
    for fn in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in exts:
            continue
        p = os.path.join(folder, fn)
        if os.path.basename(p) in names:
            sys.exit(f"ERROR: duplicate media basename {fn!r} in dataset")
        names.add(os.path.basename(p))
        cap = ""
        txt = os.path.join(folder, stem + ".txt")
        if os.path.isfile(txt):
            cap = open(txt, encoding="utf-8").read().strip()
            files.append(txt)
        entries.append({"media_path": os.path.basename(p), "caption": cap})
        files.append(p)
    if not entries:
        sys.exit(f"ERROR: no {kind} files found in {folder} "
                 f"(looked for {sorted(exts)})")
    return entries, files


# ── submit (own payload; cpfx_train group tag) ────────────────────────────────

def submit_train_job(name, instance_type, image, command, env_vars, mode, idle, max_retries, extra_tags):
    tags = []
    for t in [cl.BILLING_TAG, GROUP_TAG, *(extra_tags or [])]:
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
    return cl.spark("POST", "/api/compute/jobs", body).json()


def build_tar(recipe, dataset_entries, dataset_files):
    """Pack /input/{lora_train_run.py, recipe.json, dataset.json} + /input/dataset/*."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()
    with tarfile.open(tmp.name, "w:gz") as tf:
        def add_json(obj, arc):
            jt = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(obj, jt, indent=2)
            jt.close()
            tf.add(jt.name, arcname=arc)
            os.unlink(jt.name)
        tf.add(RUNNER, arcname="lora_train_run.py")
        add_json(recipe, "recipe.json")
        add_json(dataset_entries, "dataset.json")
        total = 0
        for p in dataset_files:
            mb = os.path.getsize(p) / 1024 / 1024
            total += mb
            tf.add(p, arcname=f"dataset/{os.path.basename(p)}")
        print(f"[pack] {len(dataset_files)} dataset file(s), {total:.1f} MB")
    print(f"[pack] → {os.path.getsize(tmp.name) / 1024 / 1024:.1f} MB")
    return tmp.name


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train an LTX-2.3 LoRA on Spark Fuse")
    ap.add_argument("--recipe", help="named recipe under trainers/ (e.g. ltx2_style)")
    ap.add_argument("--dataset", help="folder of training images (+ optional same-named .txt captions)")
    ap.add_argument("--trigger", help="distinctive trigger word that activates your LoRA at inference "
                                       "(e.g. 'raxstyle'); auto-prepended to every caption")
    ap.add_argument("--describe", help="fallback caption for un-captioned images when auto-caption is off")
    # recipe overrides
    ap.add_argument("--rank", type=int, help="LoRA rank (override recipe; 32 default)")
    ap.add_argument("--steps", type=int, help="training steps (override recipe)")
    ap.add_argument("--low-vram", action="store_true", help="INT8/low-VRAM config for 48GB cards")
    ap.add_argument("--no-auto-caption", action="store_true", help="don't VLM-caption images that lack a .txt")
    ap.add_argument("--hf-token", help="Hugging Face token, only needed if a recipe model is a GATED hf_repo "
                                       "(the default gemma mirror is ungated). Falls back to $HF_TOKEN.")
    # compute
    ap.add_argument("--image", help="CUDA-13 Docker image (override recipe)")
    ap.add_argument("--gpu", choices=list(cl.GPU_TYPES), help="GPU shortcut (override recipe)")
    ap.add_argument("--instance-type", help="raw Spark SKU (overrides --gpu)")
    ap.add_argument("--mode", choices=["instant", "smart"], default="instant")
    ap.add_argument("--max-retries", type=int, help="(smart) re-launches on preemption [0-5]")
    ap.add_argument("--idle-hold", type=int, default=0, help="(instant) warm-hold secs after exit")
    ap.add_argument("--name", help="job name")
    ap.add_argument("--tag", action="append", default=[], help="extra grouping tag (repeatable)")
    ap.add_argument("--download", metavar="DIR", help="after the job finishes, pull the trained LoRA into DIR")
    ap.add_argument("--no-tail", action="store_true", help="don't stream logs after submit")
    ap.add_argument("--dry-run", action="store_true", help="print the dataset + training plan; submit nothing")
    # housekeeping
    ap.add_argument("--list-recipes", action="store_true")
    ap.add_argument("--list-skus", action="store_true")
    ap.add_argument("--list-jobs", action="store_true", help="list cpfx_train jobs")
    ap.add_argument("--logs", metavar="JOB_ID")
    ap.add_argument("--cancel", metavar="JOB_ID")
    args = ap.parse_args()

    if args.list_recipes:
        return list_recipes()
    if args.list_skus:
        for s in cl.list_skus():
            print(f"  {s['instanceType']:20s}  {s['gpuCount']}× {s['gpuType']:28s}  {s['gpuMemoryGb']}GB")
        return
    if args.list_jobs:
        jobs = cl.list_jobs(tag=GROUP_TAG)
        print(f"{len(jobs)} job(s) tagged {GROUP_TAG}:")
        for j in jobs:
            print(f"  {j.get('id','?'):40s}  {j.get('status','?'):12s}  {j.get('instance_type_name','?')}")
        return
    if args.cancel:
        print(json.dumps(cl.cancel_job(args.cancel), indent=2))
        return
    if args.logs:
        return cl.stream_logs(args.logs)

    if not args.recipe or not args.dataset:
        sys.exit("ERROR: --recipe and --dataset are required (see --list-recipes)")
    if not args.trigger:
        sys.exit("ERROR: --trigger is required — one distinctive word that will activate your LoRA "
                 "(e.g. --trigger raxstyle). It's auto-added to every caption.")

    recipe = load_recipe(args.recipe)

    # apply CLI overrides onto the recipe
    if args.rank is not None:
        recipe.setdefault("lora", {})["rank"] = args.rank
        recipe["lora"]["alpha"] = args.rank   # keep the recipe's 1:1 rank=alpha convention
    if args.steps is not None:
        recipe.setdefault("optimization", {})["steps"] = args.steps
    if args.low_vram:
        recipe["low_vram"] = True
    if args.no_auto_caption:
        recipe["auto_caption"] = False

    kind = recipe.get("kind", "image")
    entries, files = collect_dataset(args.dataset, kind)
    n_uncap = sum(1 for e in entries if not e["caption"])
    if len(entries) < 10:
        print(f"[warn] only {len(entries)} sample(s) — style/identity LoRAs usually want 20-50+ "
              f"(faces 30-100). Training will run, but may underfit.")
    if len(entries) > 200:
        print(f"[warn] {len(entries)} samples is a lot for a LoRA — more isn't better; "
              f"consistency matters more than quantity.")

    image = args.image or recipe.get("image")
    if not image:
        sys.exit("ERROR: no --image and recipe supplies none.")
    gpu = args.gpu or recipe.get("gpu", "rtxpro6000")
    instance_type = args.instance_type or cl.GPU_TYPES.get(gpu, gpu)
    mode = args.mode or recipe.get("mode", "instant")

    # template guard (same mechanism as comfy_launch): refuse placeholder weights
    unset = cl.unfinalized_models(recipe)
    if unset and not args.dry_run:
        sys.exit("ERROR: recipe has placeholder base-weight URLs (still a template) for:\n        "
                 + "\n        ".join(unset)
                 + "\n       Fill the real URLs in trainers/%s.train.json, then re-run "
                   "(use --dry-run to validate first)." % args.recipe)

    job_name = args.name or f"train-{args.recipe}-{args.trigger}"
    lo, op = recipe.get("lora", {}), recipe.get("optimization", {})

    # ── plan ──────────────────────────────────────────────────────────────────
    print(f"\n[train] {'DRY RUN — ' if args.dry_run else ''}plan")
    print(f"        recipe    : {args.recipe} ({kind})")
    print(f"        name      : {job_name}")
    print(f"        image     : {image}")
    print(f"        instance  : {instance_type} ({gpu})   mode={mode}")
    print(f"        tags      : {cl.BILLING_TAG}, {GROUP_TAG}{''.join(', ' + t for t in args.tag)}")
    print(f"        dataset   : {len(entries)} {kind}(s) from {args.dataset} "
          f"({len(entries) - n_uncap} captioned, {n_uncap} {'auto-captioned' if recipe.get('auto_caption', True) else 'default-captioned'})")
    print(f"        trigger   : {args.trigger!r}")
    print(f"        lora      : rank {lo.get('rank', 32)}, alpha {lo.get('alpha', lo.get('rank', 32))}")
    print(f"        train     : {op.get('steps', 2000)} steps, lr {op.get('learning_rate', 1e-4)}, "
          f"res {recipe.get('resolution', '960x544')}x1{', low-vram' if recipe.get('low_vram') else ''}")
    if unset:
        print(f"        weights   : TEMPLATE — fill {len(unset)} REPLACE_ URL(s) before a real run")
    if args.dry_run:
        print("\n[train] dataset.json (first 5):")
        print(json.dumps(entries[:5], indent=2))
        print("\n[train] container will run, in order:")
        print("        git clone LTX-2 → uv sync → fetch weights → "
              f"{'auto-caption → ' if (n_uncap and recipe.get('auto_caption', True)) else ''}"
              "process_dataset.py → write YAML → train.py → upload LoRA to ShareSync")
        return

    env_vars = {
        "TRAIN_RECIPE": "/input/recipe.json",
        "TRAIN_DATASET_JSON": "/input/dataset.json",
        "TRAIN_DATASET_DIR": "/input/dataset",
        "TRAIN_TRIGGER": args.trigger,
        "TRAIN_OUTPUT_DIR": "/output",
        "TRAIN_GROUP_TAG": GROUP_TAG,
        "TRAIN_SPARK_API": cl.SPARK_API,
        "TRAIN_RUN_ID": uuid.uuid4().hex,
        "TRAIN_UPLOAD_TOKEN": cl.get_token(),
    }
    if args.describe:
        env_vars["TRAIN_DEFAULT_CAPTION"] = args.describe
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        env_vars["TRAIN_HF_TOKEN"] = hf_token

    python_bin = recipe.get("python", "python3")
    command = ["bash", "-c", f"{python_bin} /input/lora_train_run.py"]
    tar_path = build_tar(recipe, entries, files)
    try:
        resp = submit_train_job(job_name, instance_type, image, command, env_vars,
                                mode, args.idle_hold, args.max_retries, args.tag)
        job_id = resp.get("jobId") or resp.get("id")
        upload_url = (resp.get("input") or {}).get("uploadUrl")
        if not job_id or not upload_url:
            sys.exit(f"ERROR: submit response missing jobId/uploadUrl: {resp}")
        out_url = (resp.get("output") or {}).get("shareSyncBaseUrl")
        cl.upload_tarball(upload_url, tar_path)
        print(f"[train] Job submitted: {job_id}")
        if out_url:
            print(f"[train] Output folder (ShareSync): {out_url}")
        if args.download and not args.no_tail:
            threading.Thread(target=cl.stream_logs, args=(job_id,), daemon=True).start()
        elif not args.no_tail:
            cl.stream_logs(job_id)
        elif not args.download:
            print(f"\nWatch:  python comfy_spark/lora_train.py --logs {job_id}")
            print(f"Cancel: python comfy_spark/lora_train.py --cancel {job_id}")
        if args.download:
            stamp = time.strftime("%m%d%y_%M%S")
            safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in args.trigger)[:40] or "lora"
            dest = os.path.join(args.download, f"{stamp}_{safe}")
            os.makedirs(dest, exist_ok=True)
            print("[download] waiting for the job to finish...")
            status = cl.wait_for_terminal(job_id)
            dl_url = cl.resolve_output_url(resp, job_id, upload_url)
            print(f"[download] job {status}; pulling outputs -> {os.path.abspath(dest)}")
            if not dl_url:
                print("[download] could not resolve the output URL — grab it from the ShareSync web UI.")
            else:
                count = cl.download_outputs(dl_url, dest)
                print(f"[download] {count} file(s) -> {os.path.abspath(dest)}")
                print(f"\n[train] Stack it: python comfy_spark/comfy_launch.py --preset ltx2_generate "
                      f"shot.png --prompt '... {args.trigger} ...' --lora <ShareSync URL of the .safetensors>")
    finally:
        try:
            os.unlink(tar_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
