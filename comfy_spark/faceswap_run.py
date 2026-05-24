"""
faceswap_run.py — runs INSIDE the Spark container to perform a video face swap.

The face-swap analog of comfy_run.py / lora_train_run.py. Driven entirely by env
vars set by faceswap.py. Unlike the LoRA runner this one is NOT pure-stdlib by the
time it preprocesses (it imports cv2/numpy AFTER the env is built), but everything
before env-build uses only stdlib so it boots on the bare base image.

    SWAP_RECIPE      /input/recipe.json   (engine recipe: image/repo/apt/build/models/…)
    SWAP_INPUT_DIR   /input               (the uploaded target video + face image, by basename)
    SWAP_OUTPUT_DIR  /output              (swapped video lands here, then to ShareSync)
    SWAP_EXAMPLES    "1" => run the engine's OWN bundled samples (container/model proof)
    SWAP_VIDEO       target video basename (custom run)
    SWAP_FACE        face image basename   (custom run)
    SWAP_DETECTOR    "yunet" (clean) | "insightface" (non-commercial) — custom-run annotations
    SWAP_FRAME_CAP   max target frames (0 = all; engine caps at 80 anyway)
    SWAP_UPLOAD_TOKEN / SWAP_RUN_ID / SWAP_GROUP_TAG / SWAP_SPARK_API  (ShareSync self-upload)

VividFace flow (recipe engine=vividface):
  clone deepcs233/VividFace → apt GL/build deps → pip install -r requirements.txt
  → build nvdiffrast (cd Deep3DFaceRecon/nvdiffrast && pip install .)
  → snapshot ~12GB weights into weights/ → [examples] run infer.py examples
  OR [custom] detect per-frame faces → write <video>.txt (4 bbox + 10 kps [+ 3 pose])
  → build a data root (videos/ + faces/) → python infer.py <root>
  → collect outputs/ → /output → self-upload to ShareSync.

infer.py contract (verified from the repo): `python infer.py <root>` reads
<root>/videos/*.mp4 + <root>/faces/*, pairs by SORTED order, reads a sibling
<video>.txt (per-frame comma-ints: x1,y1,x2,y2 then five x,y keypoints in
InsightFace order; optional yaw,pitch,roll at [14:17]); writes
outputs/<ts>_<ckpt>/videos/<video>_<face>.mp4. 512x512, 40 steps, <=80 frames.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request


# ── tiny helpers (same shape as comfy_run.py / lora_train_run.py) ─────────────

def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def log(msg):
    print(f"[swap_run] {msg}", flush=True)


def run(cmd, cwd=None, check=True, env_extra=None):
    log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    e = dict(os.environ)
    if env_extra:
        e.update(env_extra)
    r = subprocess.run(cmd, cwd=cwd, env=e, shell=isinstance(cmd, str))
    if check and r.returncode != 0:
        sys.exit(f"[swap_run] FATAL: command failed (exit {r.returncode}): {cmd}")
    return r.returncode


def download(url, dest, label="file"):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "faceswap_run/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        total_mb = total / 1048576
        log(f"downloading {label} ({total_mb:.0f} MB) -> {dest}" if total
            else f"downloading {label} -> {dest}")
        while True:
            buf = r.read(1048576)
            if not buf:
                break
            f.write(buf)
    log(f"  {label} done ({os.path.getsize(dest) / 1048576:.1f} MB)")


def hold_for_sync():
    secs = int(env("SWAP_EXIT_HOLD", "15"))
    if secs > 0:
        log(f"holding {secs}s so /output finishes syncing...")
        time.sleep(secs)


# ── env build (the heavy part: GL/build deps + requirements + nvdiffrast) ─────

def pip(args, cwd=None, check=True):
    return run([sys.executable, "-m", "pip", "install", *args], cwd=cwd, check=check)


def apt_install(recipe):
    """Install OS packages — MUST run BEFORE the git clone, because the base conda
    image ships no git (and nvdiffrast/dlib need build-essential/cmake, opencv needs
    libGL, the transcode needs ffmpeg). Cheap no-op on a warm node."""
    apt = recipe.get("apt", [])
    if apt and shutil.which("apt-get"):
        run(["apt-get", "update", "-y"], check=False)
        run(["apt-get", "install", "-y", "--no-install-recommends", *apt], check=False)
    elif apt:
        log("WARNING: apt-get not found; assuming git + build tools + GL are already present")


def build_env(recipe, repo_dir):
    """pip the engine requirements + extras, then run the build_cmds (nvdiffrast
    from source). Runs AFTER the clone — requirements.txt and nvdiffrast live in the
    repo. The nvdiffrast build is gated by a sentinel so a warm node won't recompile."""
    reqs = os.path.join(repo_dir, "requirements.txt")
    if os.path.isfile(reqs):
        # torch is already in the base image; let pip resolve the rest. Don't fail
        # the whole job on one stubborn pin — surface it and continue (infer.py
        # will tell us what's actually missing).
        pip(["-r", reqs], check=False)
    for extra in recipe.get("pip_extra", []):
        pip([extra], check=False)
    if env("SWAP_DETECTOR", "yunet") == "insightface":
        pip(["insightface", "onnxruntime-gpu"], check=False)

    sentinel = os.path.join(repo_dir, ".nvdiffrast_built")
    if not os.path.isfile(sentinel):
        for cmd in recipe.get("build_cmds", []):
            run(cmd, cwd=repo_dir, check=False)   # shell string (has &&)
        open(sentinel, "w").close()
    else:
        log("build_cmds already done (warm node)")


# ── weights (hf snapshot / single-file, mirrors lora_train_run.fetch_models) ──

def hf_snapshot(repo, dest, token):
    os.makedirs(dest, exist_ok=True)
    code = ("from huggingface_hub import snapshot_download; "
            f"snapshot_download({repo!r}, local_dir={dest!r}, "
            f"token=({token!r} or None))")
    if run([sys.executable, "-c", code], check=False) != 0:
        pip(["-q", "huggingface_hub"], check=False)
        run([sys.executable, "-c", code], check=False)


def relocate_weights(recipe, repo_dir):
    """Some engines ship files in their HF weight repo that the code expects at a
    DIFFERENT path inside the cloned repo (VividFace reads the Basel model + the
    Deep3DFaceRecon checkpoint from Deep3DFaceRecon/BFM/, but they live in the
    weight snapshot's BFM/; and it reads the face-3D VAE from weights/face3dvae2
    while the repo ships face3dvae/). Two directive forms:
      {from, to}          recursively symlink every file under <repo>/<from> into
                          <repo>/<to>, preserving structure (copy as fallback).
      {symlink, target}   make <repo>/<symlink> a symlink to <repo>/<target> (a
                          directory alias)."""
    for rl in recipe.get("relocate", []):
        if rl.get("symlink"):
            link = os.path.join(repo_dir, rl["symlink"])
            target = os.path.join(repo_dir, rl["target"])
            if not os.path.exists(target):
                log(f"relocate: symlink target {rl['target']} missing, skipping")
                continue
            try:
                if os.path.lexists(link):
                    (os.remove if (os.path.islink(link) or os.path.isfile(link))
                     else lambda p: shutil.rmtree(p, ignore_errors=True))(link)
                os.symlink(target, link)
                log(f"relocate: symlink {rl['symlink']} -> {rl['target']}")
            except OSError as e:
                log(f"relocate: symlink failed ({e})")
            continue
        src, dst = os.path.join(repo_dir, rl["from"]), os.path.join(repo_dir, rl["to"])
        if not os.path.isdir(src):
            log(f"relocate: source {rl['from']} missing, skipping")
            continue
        n = 0
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            outdir = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(outdir, exist_ok=True)
            for name in files:
                s, d = os.path.join(root, name), os.path.join(outdir, name)
                try:
                    if os.path.lexists(d):
                        os.remove(d)
                    os.symlink(s, d)
                except OSError:
                    shutil.copy2(s, d)
                n += 1
        log(f"relocate: linked {n} file(s) {rl['from']} -> {rl['to']} (recursive)")


def fetch_models(models, root):
    token = env("SWAP_HF_TOKEN")
    for m in models:
        dest = os.path.join(root, m["dest"])
        if m.get("hf_repo"):
            # "present" = a specific marker file exists if `present_marker` is set
            # (a dir can be non-empty yet incomplete — e.g. VividFace ships a partial
            # stable-diffusion-v1-5/ with no model_index.json), else dir-non-empty.
            marker = m.get("present_marker")
            present = (os.path.isfile(os.path.join(dest, marker)) if marker
                       else (os.path.isdir(dest) and bool(os.listdir(dest))))
            if present:
                log(f"model dir present: {m['dest']}")
            else:
                log(f"snapshot-downloading {m['hf_repo']} -> {m['dest']}")
                hf_snapshot(m["hf_repo"], dest, token)
        else:
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                log(f"weight present: {m['dest']}")
            else:
                download(m["url"], dest, os.path.basename(dest))


# ── custom-input preprocessing: per-frame face annotations (CLEAN: YuNet) ─────

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def _estimate_pose(kps, w, h):
    """Rough head yaw/pitch/roll (degrees) from the 5 keypoints via solvePnP
    against a canonical 3D 5-point face. No extra weights — clean. Returns ints.
    VividFace example annos likely got pose from Deep3DFaceRecon/InsightFace; this
    is a license-clean approximation, logged as such. If infer.py reads only [:14]
    these three trailing values are simply ignored."""
    import math
    try:
        import numpy as np
        import cv2
        # canonical 3D model points (left_eye, right_eye, nose, left_mouth, right_mouth)
        model = np.ascontiguousarray(np.array(
            [[-30, -30, -30], [30, -30, -30], [0, 0, 0],
             [-25, 35, -30], [25, 35, -30]], dtype=np.float64).reshape(5, 1, 3))
        image_pts = np.ascontiguousarray(np.array(kps, dtype=np.float64).reshape(5, 1, 2))
        f = float(max(w, h))
        cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(model, image_pts, cam, np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return [0, 0, 0]
        rot, _ = cv2.Rodrigues(rvec)
        sy = (rot[0, 0] ** 2 + rot[1, 0] ** 2) ** 0.5
        yaw = math.degrees(math.atan2(-rot[2, 0], sy))
        pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
        roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        return [int(yaw), int(pitch), int(roll)]
    except Exception:   # pose is optional ([14:17]); never let it crash annotation
        return [0, 0, 0]


def detect_annotations(video_path, anno_path, detector, frame_cap):
    """Write a per-frame annotation matching VividFace's examples: 14 comma-ints —
    'x1,y1,x2,y2' bbox + five (x,y) keypoints (eyes & mouth corners image-left-first,
    nose middle), no pose. Frames with no detection reuse the last good box (keeps the
    line count == frame count)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if detector == "insightface":
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        det = None
    else:
        model = os.path.join("/tmp", "yunet.onnx")
        if not os.path.isfile(model):
            download(YUNET_URL, model, "yunet")
        det = cv2.FaceDetectorYN.create(model, "", (w, h), 0.6, 0.3, 5000)
        app = None

    lines, last = [], None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (frame_cap and idx >= frame_cap):
            break
        kps = bbox = None
        if det is not None:                                  # YuNet
            _, faces = det.detect(frame)
            if faces is not None and len(faces):
                f = max(faces, key=lambda r: r[2] * r[3])    # largest face
                x, y, bw, bh = f[:4]
                bbox = [int(x), int(y), int(x + bw), int(y + bh)]
                # YuNet kps order: r_eye,l_eye,nose,r_mouth,l_mouth → reorder to
                # InsightFace: l_eye,r_eye,nose,l_mouth,r_mouth
                re_, le_, no_, rm_, lm_ = f[4:6], f[6:8], f[8:10], f[10:12], f[12:14]
                kps = [le_[0], le_[1], re_[0], re_[1], no_[0], no_[1],
                       lm_[0], lm_[1], rm_[0], rm_[1]]
        else:                                                # InsightFace
            fcs = app.get(frame)
            if fcs:
                fc = max(fcs, key=lambda a: (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]))
                bbox = [int(v) for v in fc.bbox[:4]]
                kps = [v for xy in fc.kps for v in xy]       # already l_eye,r_eye,nose,l_mouth,r_mouth
        if bbox is None:
            if last is None:
                idx += 1
                lines.append(None)                            # placeholder; backfilled below
                continue
            bbox, kps = last
        kps = [int(v) for v in kps]
        # match VividFace's example annotations: eye pair & mouth pair image-left-first
        if kps[0] > kps[2]:
            kps[0:2], kps[2:4] = kps[2:4], kps[0:2]
        if kps[6] > kps[8]:
            kps[6:8], kps[8:10] = kps[8:10], kps[6:8]
        last = (bbox, kps)
        lines.append(",".join(map(str, bbox + kps)))   # 14 values (bbox + 5 kps); examples carry no pose
        idx += 1
    cap.release()
    # backfill any leading no-detection frames with the first good line
    first_good = next((l for l in lines if l), None)
    lines = [l if l else (first_good or "0,0,1,1," + ",".join(["0"] * 10)) for l in lines]
    with open(anno_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"annotations: {len(lines)} frame(s) -> {os.path.basename(anno_path)} "
        f"(detector={detector}, 14-val, kps image-left-first)")
    return len(lines)


def to_mp4(src, dst, frame_cap):
    """infer.py only globs *.mp4 — transcode/copy the target into an mp4 (ffmpeg
    is apt-installed). Optionally trim to frame_cap frames."""
    cmd = ["ffmpeg", "-y", "-i", src]
    if frame_cap:
        cmd += ["-frames:v", str(frame_cap)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", dst]
    run(cmd, check=True)


def square_face_clip(src_mp4, dst_mp4, frame_cap):
    """VividFace expects 512x512 FACE-CENTRIC clips (its examples are ~512 square with
    the face filling the frame), and its inference hard-reshapes to 512x512 — a full
    16:9 plate trips that ('shape [-1,1,512,512] invalid for size N'). Detect the face
    across the clip, compute ONE stable square crop (face + margin) that fits the
    frame, and ffmpeg crop+scale to 512x512. Returns (X, Y, S) of the crop used.
    NOTE: the swap output is therefore a 512 face-centric clip, not a full-plate
    composite — VividFace is not built to repaint a full plate (see recipe _RD_ONLY)."""
    import cv2
    cap = cv2.VideoCapture(src_mp4)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (frame_cap or 40)
    model = os.path.join("/tmp", "yunet.onnx")
    if not os.path.isfile(model):
        download(YUNET_URL, model, "yunet")
    det = cv2.FaceDetectorYN.create(model, "", (W, H), 0.6, 0.3, 5000)
    cx = cy = sz = cnt = 0.0
    step = max(1, n // 12)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or (frame_cap and i >= frame_cap):
            break
        if i % step == 0:
            _, faces = det.detect(fr)
            if faces is not None and len(faces):
                f = max(faces, key=lambda r: r[2] * r[3])
                x, y, bw, bh = f[:4]
                cx += x + bw / 2; cy += y + bh / 2; sz += max(bw, bh); cnt += 1
        i += 1
    cap.release()
    if cnt == 0:
        S = min(W, H); X = (W - S) // 2; Y = (H - S) // 2          # no face: center square
        log("square face clip: NO FACE detected — using a centered square crop")
    else:
        cx /= cnt; cy /= cnt; sz /= cnt
        S = int(min(W, H, sz * 1.8))                                # face + ~80% margin
        X = int(min(max(0, cx - S / 2), W - S))
        Y = int(min(max(0, cy - S / 2), H - S))
    run(["ffmpeg", "-y", "-i", src_mp4, "-vf", f"crop={S}:{S}:{X}:{Y},scale=512:512",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", dst_mp4], check=True)
    log(f"square face clip: crop {S}x{S}@({X},{Y}) of {W}x{H} -> 512x512 -> {os.path.basename(dst_mp4)}")
    return (X, Y, S)


def crop_face_image(src, dst):
    """VividFace's faces/ are tight face crops. Crop the reference image to a square,
    face-centred 512 so the identity isn't a tiny region of a full screenshot. Falls
    back to a centred square (then a raw copy) if no face is found."""
    import cv2
    img = cv2.imread(src)
    if img is None:
        shutil.copy2(src, dst)
        log(f"face crop: cv2 couldn't read {os.path.basename(src)} — copied raw")
        return
    h, w = img.shape[:2]
    model = os.path.join("/tmp", "yunet.onnx")
    if not os.path.isfile(model):
        download(YUNET_URL, model, "yunet")
    det = cv2.FaceDetectorYN.create(model, "", (w, h), 0.6, 0.3, 5000)
    _, faces = det.detect(img)
    if faces is not None and len(faces):
        f = max(faces, key=lambda r: r[2] * r[3])
        x, y, bw, bh = f[:4]
        cx, cy, sz = x + bw / 2, y + bh / 2, max(bw, bh)
        S = int(min(w, h, sz * 1.8))
        X = int(min(max(0, cx - S / 2), w - S))
        Y = int(min(max(0, cy - S / 2), h - S))
    else:
        S = min(w, h); X = (w - S) // 2; Y = (h - S) // 2
    cv2.imwrite(dst, cv2.resize(img[Y:Y + S, X:X + S], (512, 512)))
    log(f"face crop: {S}x{S}@({X},{Y}) of {w}x{h} -> 512x512 -> {os.path.basename(dst)}")


# ── ShareSync self-upload (same mechanism as lora_train_run.upload_outputs) ───

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
    token, run_id = env("SWAP_UPLOAD_TOKEN"), env("SWAP_RUN_ID")
    api = env("SWAP_SPARK_API", "https://api.prod.aapse1.sparkcloud.studio")
    tag = env("SWAP_GROUP_TAG", "cpfx_faceswap")
    if not token or not run_id:
        return
    base = None
    try:
        jobs = _api_get(f"{api}/api/compute/jobs?tag={tag}", token).get("jobs", [])
        for j in jobs:
            jid = j.get("id")
            if not jid:
                continue
            if (j.get("env") or {}).get("SWAP_RUN_ID") != run_id \
                    and j.get("status") not in ("running", "provisioning"):
                continue
            d = _api_get(f"{api}/api/compute/jobs/{jid}", token)
            if (d.get("env") or {}).get("SWAP_RUN_ID") == run_id:
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    recipe = json.load(open(env("SWAP_RECIPE", "/input/recipe.json")))
    in_dir = env("SWAP_INPUT_DIR", "/input")
    output_dir = env("SWAP_OUTPUT_DIR", "/output")
    examples = env("SWAP_EXAMPLES") == "1"
    detector = env("SWAP_DETECTOR", "yunet")
    frame_cap = int(env("SWAP_FRAME_CAP", "0"))
    os.makedirs(output_dir, exist_ok=True)

    repo_dir = recipe.get("repo_dir", "/opt/vividface")
    repo_url = recipe.get("repo", "https://github.com/deepcs233/VividFace")

    # 1) OS deps FIRST — the base conda image has no git to clone with
    apt_install(recipe)

    # 2) clone the engine repo (idempotent for warm-node reuse)
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        run(["git", "clone", "--depth", "1", repo_url, repo_dir])
    else:
        log("engine repo present (warm node)")

    # 3) build the environment (requirements + nvdiffrast)
    build_env(recipe, repo_dir)

    # 3) weights onto the node (~12GB VividFace snapshot + SD1.5 if needed).
    #    Model dests are relative to the REPO ROOT (e.g. "weights",
    #    "weights/stable-diffusion-v1-5"). Clean up a stale weights/weights/ left by
    #    an earlier path-nesting bug in case we landed on that warm node.
    stale = os.path.join(repo_dir, recipe.get("weights_subdir", "weights"),
                         recipe.get("weights_subdir", "weights"))
    if os.path.isdir(stale):
        log("cleaning stale nested weights/weights/ from a prior run")
        shutil.rmtree(stale, ignore_errors=True)
    fetch_models(recipe.get("models", []), repo_dir)
    relocate_weights(recipe, repo_dir)   # BFM -> Deep3DFaceRecon/BFM

    infer_script = recipe.get("infer_script", "infer.py")

    if examples:
        # cheapest proof: the engine's own bundled samples already ship videos/,
        # faces/ and .txt — no detector, no input prep.
        data_root = recipe.get("examples_arg", "examples")
        log(f"running bundled examples: {infer_script} {data_root}")
    else:
        # 4) build a custom data root: videos/<stem>.mp4 + faces/<face> + <stem>.txt
        data_root = "/tmp/swap_data"
        vdir, fdir = os.path.join(data_root, "videos"), os.path.join(data_root, "faces")
        os.makedirs(vdir, exist_ok=True)
        os.makedirs(fdir, exist_ok=True)
        src_video = os.path.join(in_dir, env("SWAP_VIDEO"))
        src_face = os.path.join(in_dir, env("SWAP_FACE"))
        stem = os.path.splitext(env("SWAP_VIDEO"))[0]
        mp4 = os.path.join(vdir, stem + ".mp4")
        raw = os.path.join("/tmp", "raw_" + os.path.basename(mp4))
        to_mp4(src_video, raw, frame_cap)                       # full-frame transcode
        square_face_clip(raw, mp4, frame_cap)                   # -> 512 face-centric (VividFace geometry)
        crop_face_image(src_face, os.path.join(fdir, os.path.splitext(os.path.basename(src_face))[0] + ".png"))
        detect_annotations(mp4, os.path.join(vdir, stem + ".txt"), detector, frame_cap)

    # 5) run the engine's own inference. Clear the repo's committed sample outputs
    #    first so the upload carries ONLY this run's render — but KEEP the outputs/
    #    dir itself (infer.py writes a timestamped subfolder into it without mkdir-ing
    #    the parent).
    out_dir_repo = os.path.join(repo_dir, "outputs")
    shutil.rmtree(out_dir_repo, ignore_errors=True)
    os.makedirs(out_dir_repo, exist_ok=True)
    run([sys.executable, infer_script, data_root], cwd=repo_dir)

    # 6) collect the swapped VIDEO(s) to the TOP LEVEL of /output and self-upload.
    #    The tunet-web outputs panel lists the ShareSync dir ONE LEVEL deep, so the
    #    deliverable mp4 must sit at the top — VividFace nests it under
    #    outputs/<ts>/videos/. We flatten it and SKIP the hundreds of per-frame jpgs
    #    + 3D-recon files (clutter + a very slow per-file WebDAV upload).
    out_src = os.path.join(repo_dir, "outputs")
    n = 0
    if os.path.isdir(out_src):
        for root, _, files in os.walk(out_src):
            for fn in files:
                if not fn.lower().endswith((".mp4", ".mov", ".webm")):
                    continue
                p = os.path.join(root, fn)
                d = os.path.join(output_dir, fn)
                i = 1
                while os.path.exists(d):                       # de-collide basenames
                    stem, ext = os.path.splitext(fn)
                    d = os.path.join(output_dir, f"{stem}_{i}{ext}"); i += 1
                shutil.copy2(p, d)
                n += 1
                log(f"  swapped video -> {os.path.basename(d)}")
    log(f"collected {n} swapped video(s) to {output_dir} (frames/recon left on the node)")
    if n == 0:
        log("WARNING: no swapped video produced — check the infer.py log above")
    upload_outputs(output_dir)
    hold_for_sync()
    log("done.")


if __name__ == "__main__":
    main()
