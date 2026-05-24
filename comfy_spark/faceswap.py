"""
faceswap.py — run a diffusion video FACE SWAP engine on Spark Fuse.

The face-swap sibling of comfy_launch.py / lora_train.py. You point it at a target
(body/motion) video and a face reference image; it packs them with a recipe,
submits a Spark job that clones the chosen engine's repo and runs its OWN
inference (faceswap_run.py inside the container), streams the log, and downloads
the swapped video from ShareSync.

First engine: VividFace (swappers/vividface.swap.json). It is an R&D QUALITY
YARDSTICK, not a license-clean delivery engine — see the recipe's _RD_ONLY note
and the NOT-FOR-DELIVERY banner this launcher prints. The clean delivery path is
the LTX/BFS preset (comfy_launch.py --preset ltx_faceswap) or training your own.

It reuses comfy_launch.py's proven spine — auth, the Spark API helper, tarball
upload, log streaming, and ShareSync download — so billing, credentials, and
output handling are identical. The differences here are the job payload (a swap
run) and a distinct cpfx_faceswap group tag.

Quick start:
    # 1) validate the plan for free (prints the container steps; spends nothing):
    python comfy_spark/faceswap.py --recipe vividface --examples --dry-run
    # 2) prove the container builds + the model runs on the engine's own samples:
    python comfy_spark/faceswap.py --recipe vividface --examples --idle-hold 300
    # 3) swap your own footage:
    python comfy_spark/faceswap.py --recipe vividface body.mp4 --face id.png \
        --idle-hold 300 --download ./swaps_out

Auth + billing: same as comfy_launch / lora_train (SPARK_EMAIL / SPARK_PASSWORD
in the repo .env; every job carries the mandatory cpfx_tunet billing tag).
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid

# Reuse the whole Spark spine from the renderer's launcher.
import comfy_launch as cl

SWAPPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swappers")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faceswap_run.py")
GROUP_TAG = "cpfx_faceswap"              # filterable; cpfx_tunet (billing) is always added too
COMFY_TAG = "cpfx_comfy"                 # makes tunet-web show the media OUTPUTS panel (isComfyJob)
                                         # so the swapped mp4 is viewable in job details, like comfy renders
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


# ── recipes ───────────────────────────────────────────────────────────────────

def load_recipe(name):
    path = os.path.join(SWAPPERS_DIR, f"{name}.swap.json")
    if not os.path.isfile(path):
        avail = [f[:-10] for f in os.listdir(SWAPPERS_DIR) if f.endswith(".swap.json")] \
            if os.path.isdir(SWAPPERS_DIR) else []
        sys.exit(f"ERROR: recipe {name!r} not found. Available: {', '.join(avail) or '(none)'}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_recipes():
    if not os.path.isdir(SWAPPERS_DIR):
        print("(no swappers/ dir)")
        return
    for f in sorted(os.listdir(SWAPPERS_DIR)):
        if f.endswith(".swap.json"):
            r = json.load(open(os.path.join(SWAPPERS_DIR, f), encoding="utf-8"))
            rd = "  [R&D — not for delivery]" if r.get("rd_only") else ""
            print(f"  {f[:-10]:16s} [{r.get('engine','?')}]  {r.get('description','')[:80]}{rd}")


# ── inputs ────────────────────────────────────────────────────────────────────

def check_input(path, kinds, label):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {label} {path!r} is not a file")
    ext = os.path.splitext(path)[1].lower()
    if ext not in kinds:
        sys.exit(f"ERROR: {label} {path!r} is not a recognized {label} type ({sorted(kinds)})")
    return os.path.abspath(path)


def build_tar(recipe, video_path, face_path):
    """Pack /input/{faceswap_run.py, recipe.json} (+ the target video and face
    image by bare basename, unless --examples runs the engine's bundled samples)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()
    with tarfile.open(tmp.name, "w:gz") as tf:
        def add_json(obj, arc):
            jt = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(obj, jt, indent=2)
            jt.close()
            tf.add(jt.name, arcname=arc)
            os.unlink(jt.name)
        tf.add(RUNNER, arcname="faceswap_run.py")
        add_json(recipe, "recipe.json")
        total = 0
        for p in (video_path, face_path):
            if p:
                total += os.path.getsize(p) / 1024 / 1024
                tf.add(p, arcname=os.path.basename(p))
        if total:
            print(f"[pack] inputs {total:.1f} MB")
    print(f"[pack] → {os.path.getsize(tmp.name) / 1024 / 1024:.1f} MB")
    return tmp.name


# ── submit (own payload; cpfx_faceswap group tag) ─────────────────────────────

def submit_swap_job(name, instance_type, image, command, env_vars, mode, idle, max_retries, extra_tags):
    tags = []
    for t in [cl.BILLING_TAG, GROUP_TAG, COMFY_TAG, *(extra_tags or [])]:
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


def _chain_restore(dest, safe, idle_hold):
    """Tier-2: feed the just-downloaded swapped mp4 into the seedvr2 upscaler as a
    SECOND Spark job (comfy_launch.py --preset seedvr2). VividFace emits a 512
    face-centric clip; seedvr2 rebuilds detail + upscales it. Restored result lands
    in <dest>/restored/. Best-effort — a failure here doesn't undo the swap."""
    swaps = sorted(f for f in os.listdir(dest)
                   if f.lower().endswith((".mp4", ".mov")) and "preview" not in f.lower())
    if not swaps:
        print("[restore] no swapped video in the download dir; skipping seedvr2.")
        return
    swap_mp4 = os.path.join(dest, swaps[0])
    rdest = os.path.join(dest, "restored")
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"[restore] Tier-2: upscaling {swaps[0]} with seedvr2 (2nd Spark job) -> {rdest}")
    subprocess.run([sys.executable, os.path.join(here, "comfy_launch.py"),
                    "--preset", "seedvr2", swap_mp4,
                    "--name", f"seedvr2-{safe}-restore",
                    "--idle-hold", str(idle_hold),
                    "--download", rdest], cwd=here)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run a diffusion video face swap on Spark Fuse")
    ap.add_argument("video", nargs="?", help="target / body video (the motion + scene to keep)")
    ap.add_argument("--recipe", help="named engine recipe under swappers/ (e.g. vividface)")
    ap.add_argument("--face", help="face reference image (the identity to swap IN)")
    ap.add_argument("--examples", action="store_true",
                    help="ignore --video/--face and run the engine's OWN bundled samples "
                         "(they ship annotations) — the cheapest proof the container builds + model runs")
    # recipe overrides
    ap.add_argument("--detector", choices=["yunet", "insightface"],
                    help="per-frame face detector for a custom video (yunet=clean default; insightface=non-commercial)")
    ap.add_argument("--frame-cap", type=int, help="cap target frames processed (0=all)")
    ap.add_argument("--hf-token", help="Hugging Face token if a recipe weight repo is gated (falls back to $HF_TOKEN)")
    # compute
    ap.add_argument("--image", help="Docker image (override recipe)")
    ap.add_argument("--gpu", choices=list(cl.GPU_TYPES), help="GPU shortcut (override recipe)")
    ap.add_argument("--instance-type", help="raw Spark SKU (overrides --gpu)")
    ap.add_argument("--mode", choices=["instant", "smart"], default="instant")
    ap.add_argument("--max-retries", type=int, help="(smart) re-launches on preemption [0-5]")
    ap.add_argument("--idle-hold", type=int, default=0, help="(instant) warm-hold secs after exit")
    ap.add_argument("--name", help="job name")
    ap.add_argument("--tag", action="append", default=[], help="extra grouping tag (repeatable)")
    ap.add_argument("--download", metavar="DIR", help="after the job finishes, pull the swapped video into DIR")
    ap.add_argument("--restore", action="store_true",
                    help="Tier-2: after the swap downloads, upscale/restore it with the seedvr2 preset "
                         "(a 2nd Spark job on the swapped mp4). Requires --download.")
    ap.add_argument("--no-tail", action="store_true", help="don't stream logs after submit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + container steps; submit nothing")
    # housekeeping
    ap.add_argument("--list-recipes", action="store_true")
    ap.add_argument("--list-skus", action="store_true")
    ap.add_argument("--list-jobs", action="store_true", help="list cpfx_faceswap jobs")
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

    if not args.recipe:
        sys.exit("ERROR: --recipe is required (see --list-recipes)")
    if not args.examples and not (args.video and args.face):
        sys.exit("ERROR: give a target VIDEO and a --face image, or use --examples to run the engine's "
                 "own bundled samples (the recommended first run).")
    if args.restore and not args.download:
        sys.exit("ERROR: --restore needs --download DIR — the swap is pulled locally, then fed to a "
                 "seedvr2 upscale job.")

    recipe = load_recipe(args.recipe)

    # apply CLI overrides onto the recipe
    if args.detector:
        recipe["detector"] = args.detector
    if args.frame_cap is not None:
        recipe["frame_cap"] = args.frame_cap

    video_path = face_path = None
    if not args.examples:
        video_path = check_input(args.video, VIDEO_EXTS, "video")
        face_path = check_input(args.face, IMAGE_EXTS, "face image")

    image = args.image or recipe.get("image")
    if not image:
        sys.exit("ERROR: no --image and recipe supplies none.")
    gpu = args.gpu or recipe.get("gpu", "rtxpro6000")
    instance_type = args.instance_type or cl.GPU_TYPES.get(gpu, gpu)
    mode = args.mode or recipe.get("mode", "instant")

    # template guard (same mechanism as comfy_launch / lora_train): refuse placeholders
    unset = cl.unfinalized_models(recipe)
    if unset and not args.dry_run:
        sys.exit("ERROR: recipe has placeholder weight URLs (still a template) for:\n        "
                 + "\n        ".join(unset)
                 + f"\n       Fill the real URLs in swappers/{args.recipe}.swap.json, then re-run.")

    job_name = args.name or (f"swap-{args.recipe}-examples" if args.examples
                             else f"swap-{args.recipe}-{os.path.splitext(os.path.basename(args.video))[0]}")[:60]

    # ── plan ──────────────────────────────────────────────────────────────────
    print(f"\n[swap] {'DRY RUN — ' if args.dry_run else ''}plan")
    if recipe.get("rd_only"):
        print("        " + "!" * 64)
        print("        ! R&D ENGINE — NOT LICENSE-CLEAN FOR BILLED CLIENT SHOTS.        !")
        print("        ! Quality benchmark only. Deliver via ltx_faceswap or own-trained. !")
        print("        " + "!" * 64)
    print(f"        recipe    : {args.recipe} (engine={recipe.get('engine','?')})")
    print(f"        name      : {job_name}")
    print(f"        image     : {image}")
    print(f"        instance  : {instance_type} ({gpu})   mode={mode}")
    print(f"        tags      : {cl.BILLING_TAG}, {GROUP_TAG}, {COMFY_TAG}{''.join(', ' + t for t in args.tag)}")
    if args.examples:
        print(f"        inputs    : (engine's bundled examples — container/model proof)")
    else:
        print(f"        target    : {os.path.basename(args.video)}")
        print(f"        face      : {os.path.basename(args.face)}")
        print(f"        detector  : {recipe.get('detector','yunet')}"
              + ("  (NON-COMMERCIAL — research only)" if recipe.get('detector') == 'insightface' else "  (license-clean)"))
    if args.dry_run:
        print("\n[swap] container will run, in order:")
        print(f"        git clone {recipe.get('repo')} → apt {len(recipe.get('apt', []))} pkg(s) → "
              "pip install -r requirements.txt → build nvdiffrast → snapshot weights → "
              + ("infer.py examples" if args.examples
                 else f"detect faces ({recipe.get('detector','yunet')}) → infer.py")
              + " → upload outputs to ShareSync")
        print(f"        weights   : {', '.join(m.get('hf_repo') or m.get('url','?') for m in recipe.get('models', []))}")
        return

    env_vars = {
        # TUNET_PRESET gives the comfy job-detail layout (shown because of the
        # cpfx_comfy tag) a readable label instead of '—'.
        "TUNET_PRESET": f"{recipe.get('engine', args.recipe)} (face swap, R&D)",
        "SWAP_RECIPE": "/input/recipe.json",
        "SWAP_INPUT_DIR": "/input",
        "SWAP_OUTPUT_DIR": "/output",
        "SWAP_EXAMPLES": "1" if args.examples else "",
        "SWAP_VIDEO": "" if args.examples else os.path.basename(args.video),
        "SWAP_FACE": "" if args.examples else os.path.basename(args.face),
        "SWAP_DETECTOR": recipe.get("detector", "yunet"),
        "SWAP_FRAME_CAP": str(recipe.get("frame_cap", 0)),
        "SWAP_GROUP_TAG": GROUP_TAG,
        "SWAP_SPARK_API": cl.SPARK_API,
        "SWAP_RUN_ID": uuid.uuid4().hex,
        "SWAP_UPLOAD_TOKEN": cl.get_token(),
    }
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        env_vars["SWAP_HF_TOKEN"] = hf_token

    python_bin = recipe.get("python", "python")
    command = ["bash", "-c", f"{python_bin} /input/faceswap_run.py"]
    tar_path = build_tar(recipe, video_path, face_path)
    try:
        resp = submit_swap_job(job_name, instance_type, image, command, env_vars,
                               mode, args.idle_hold, args.max_retries, args.tag)
        job_id = resp.get("jobId") or resp.get("id")
        upload_url = (resp.get("input") or {}).get("uploadUrl")
        if not job_id or not upload_url:
            sys.exit(f"ERROR: submit response missing jobId/uploadUrl: {resp}")
        out_url = (resp.get("output") or {}).get("shareSyncBaseUrl")
        cl.upload_tarball(upload_url, tar_path)
        print(f"[swap] Job submitted: {job_id}")
        if out_url:
            print(f"[swap] Output folder (ShareSync): {out_url}")
        if args.download and not args.no_tail:
            threading.Thread(target=cl.stream_logs, args=(job_id,), daemon=True).start()
        elif not args.no_tail:
            cl.stream_logs(job_id)
        elif not args.download:
            print(f"\nWatch:  python comfy_spark/faceswap.py --logs {job_id}")
            print(f"Cancel: python comfy_spark/faceswap.py --cancel {job_id}")
        if args.download:
            stamp = time.strftime("%m%d%y_%M%S")
            base = "examples" if args.examples else os.path.splitext(os.path.basename(args.video))[0]
            safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)[:40] or "swap"
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
                if args.restore and status == "succeeded":
                    _chain_restore(dest, safe, args.idle_hold)
    finally:
        try:
            os.unlink(tar_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
