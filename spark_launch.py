"""
spark_launch.py — fully automated TuNet training on Spark Compute v1

Usage:
    python spark_launch.py --config config.yaml [options]

What it does:
    1. Authenticates against the Spark API (email/password → JWT)
    2. Submits a job via POST /api/compute/jobs
    3. Packs tunet/ source + data + config into a tar.gz
    4. PUTs the tarball to the uploadUrl returned by submit
    5. Streams logs via SSE GET /api/compute/jobs/:id/logs/stream
    6. (Outputs stream to ShareSync automatically — agent owns that.)

Environment variables (via .env):
    SPARK_EMAIL / spark_email
    SPARK_PASSWORD / spark_pass

Compare to runpod_launch.py: this is much simpler. There's no SSH, no
nohup, no monitor_api. The Spark agent runs your container in foreground
and uploads /output/ to ShareSync as files appear. Cancel = SIGTERM.

Path mapping in config (handled by rewrite_config_for_pod):
    /workspace/tunet         → /input/tunet
    /workspace/data/{src,dst} → /input/data/{src,dst}
    /workspace/output/<job>  → /output/<job>
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time

import requests
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass

# Re-use the packing + config-rewrite helpers from the RunPod launcher.
# The only difference is the remote prefix (`/input/...` vs `/workspace/...`),
# which we patch by overriding constants before calling.
import runpod_launch as _rl

# ── Spark Compute v1 endpoints ────────────────────────────────────────────────

SPARK_API = 'https://api.prod.aapse1.sparkcloud.studio'

# Defaults — match what Walt validated in the doc
DEFAULT_IMAGE   = 'runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204'
DEFAULT_SKU     = 'g6e.4xlarge'   # 1× L40S 48GB
DEFAULT_IDLE    = 0               # 0 = stop immediately when job exits

# Mandatory billing tag. Spark identifies jobs carrying this as official Clean
# Plate FX / TuNet work under the cost + revenue-share agreement (Spark Fuse API
# v1.17 §2.3). Submissions WITHOUT it are billed at STANDARD rates, so it's
# force-merged onto every job we submit (mirrors tunet-web/src/lib/spark.ts).
TUNET_JOB_TAG   = 'cpfx_tunet'

# Spark container paths (replaces the /workspace/* layout RunPod used)
INPUT_REMOTE    = '/input'
OUTPUT_REMOTE   = '/output'
TUNET_REMOTE    = f'{INPUT_REMOTE}/tunet'
DATA_REMOTE     = f'{INPUT_REMOTE}/data'
CONFIG_REMOTE   = f'{INPUT_REMOTE}/config.yaml'

# GPU shortcuts → Spark instanceType (24 SKUs available, see /api/compute/skus)
GPU_TYPES = {
    't4':         'g4dn.xlarge',     # 1× T4 16GB
    'a10':        'g5.xlarge',       # 1× A10 24GB  (cheapest single-GPU)
    'a10x4':      'g5.24xlarge',     # 4× A10 24GB
    'l4':         'g6.2xlarge',      # 1× L4 24GB
    'l40s':       'g6e.4xlarge',     # 1× L40S 48GB
    'l40sx4':     'g6e.12xlarge',    # 4× L40S 48GB
    'rtxpro6000': 'g7e.xlarge',      # 1× RTX PRO 6000 96GB
    'rtxpro6000x8': 'g7e.48xlarge',  # 8× RTX PRO 6000 96GB
}


# ── Auth — lazy JWT with refresh ─────────────────────────────────────────────

_token = None
_token_expires_at = 0


def _creds():
    email = os.environ.get('SPARK_EMAIL') or os.environ.get('spark_email')
    pw    = os.environ.get('SPARK_PASSWORD') or os.environ.get('spark_pass')
    if not email or not pw:
        sys.exit('ERROR: Set SPARK_EMAIL / SPARK_PASSWORD (or spark_email / spark_pass) in .env')
    return email, pw


def _jwt_expiry(token):
    try:
        import base64
        payload = json.loads(base64.b64decode(token.split('.')[1] + '==').decode())
        return payload.get('exp', 0) * 1000
    except Exception:
        return 0


def get_token():
    global _token, _token_expires_at
    if _token and _token_expires_at > time.time() * 1000 + 60_000:
        return _token
    email, pw = _creds()
    r = requests.post(f'{SPARK_API}/auth/login',
                      json={'email': email, 'password': pw}, timeout=15)
    if not r.ok:
        sys.exit(f'ERROR: Spark auth failed (HTTP {r.status_code}): {r.text[:200]}')
    data = r.json()
    tok = data.get('token') or data.get('access_token')
    if not tok:
        sys.exit('ERROR: Spark auth: no token in response')
    _token = tok
    _token_expires_at = _jwt_expiry(tok)
    return _token


def spark(method, path, body=None, **kw):
    tok = get_token()
    headers = {'Authorization': f'Bearer {tok}'}
    headers.update(kw.pop('headers', {}))
    if body is not None and 'data' not in kw:
        headers.setdefault('Content-Type', 'application/json')
        kw['json'] = body
    r = requests.request(method, f'{SPARK_API}{path}', headers=headers, timeout=kw.pop('timeout', 30), **kw)
    if not r.ok:
        sys.exit(f'ERROR: Spark API {method} {path} -> HTTP {r.status_code}: {r.text[:300]}')
    return r


# ── Job lifecycle ─────────────────────────────────────────────────────────────

def list_skus():
    return spark('GET', '/api/compute/skus').json().get('skus', [])


def list_jobs():
    return spark('GET', '/api/compute/jobs').json().get('jobs', [])


def get_job(job_id):
    return spark('GET', f'/api/compute/jobs/{job_id}').json()


def cancel_job(job_id):
    return spark('POST', f'/api/compute/jobs/{job_id}/cancel').json()


def submit_job(name, instance_type, image, command, idle_hold_seconds=DEFAULT_IDLE):
    """
    Submit a new compute job in auto-prepare mode (we PUT the tarball after).
    Walt's doc: response includes input.uploadUrl + input.exampleCurl.
    """
    body = {
        'name':              name,
        'instanceType':      instance_type,
        'image':             image,
        'command':           command,                # list[str] — argv to run inside container
        'tags':              [TUNET_JOB_TAG],         # mandatory billing tag (see above)
        'inputPushMode':     'auto-prepare',          # we'll upload a tarball
        'idleHoldSeconds':   idle_hold_seconds,
    }
    return spark('POST', '/api/compute/jobs', body).json()


def upload_tarball(upload_url, tar_path):
    """PUT the packed tar.gz to the uploadUrl returned by submit."""
    size_mb = os.path.getsize(tar_path) / 1024 / 1024
    print(f'[upload] PUT {size_mb:.1f} MB → uploadUrl')
    tok = get_token()
    with open(tar_path, 'rb') as f:
        r = requests.put(upload_url,
                         data=f,
                         headers={'Authorization': f'Bearer {tok}',
                                  'Content-Type': 'application/octet-stream'},
                         timeout=600)
    if not r.ok:
        sys.exit(f'ERROR: upload failed HTTP {r.status_code}: {r.text[:200]}')
    print(f'[upload] Done.')


def stream_logs(job_id):
    """SSE log stream — replays from start, then live-tails."""
    tok = get_token()
    print(f'\n[logs] Streaming {job_id} (Ctrl+C to detach, job keeps running)\n')
    try:
        with requests.get(
            f'{SPARK_API}/api/compute/jobs/{job_id}/logs/stream',
            headers={'Authorization': f'Bearer {tok}',
                     'Accept': 'text/event-stream'},
            stream=True, timeout=None,
        ) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # SSE lines come as "data: ..." — strip the prefix
                if line.startswith('data: '):
                    print(line[6:], flush=True)
                elif line.startswith(':'):
                    continue   # heartbeat
                else:
                    print(line, flush=True)
    except KeyboardInterrupt:
        print('\n[logs] Detached.')


# ── Tarball builder (input/ layout) ───────────────────────────────────────────

def build_input_tar(tunet_root, config_path, data_uploads, resume_pth=None):
    """
    Pack a tar.gz that the agent will extract into /input/. Layout:
        /input/tunet/...        (source)
        /input/data/src/...     (training pairs)
        /input/data/dst/...
        /input/uploads/<file>   (finetune .pth, etc.)
        /input/config.yaml
        /input/spark_start.sh
        /input/output/<job>/<file>  (resume checkpoints, training.log)
    """
    import tarfile

    tmp = tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False)
    tmp.close()
    n_files = 0

    SOURCE_EXCLUDES = ['__pycache__', '*.pyc', '.git', 'node_modules',
                       'Spark', '_archive', '*.pth', '*.onnx',
                       'tunet_session.yaml', 'tunet-web', 'docs']

    def _add_dir(tf, src, arcprefix, exclude=None):
        nonlocal n_files
        import fnmatch
        exclude = exclude or []
        src = src.rstrip('/\\')

        def _is_excluded(rel_path):
            parts = rel_path.replace('\\', '/').split('/')
            for pattern in exclude:
                for part in parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True
            return False

        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs
                       if not _is_excluded(os.path.relpath(os.path.join(root, d), src))]
            for fname in files:
                abs_p = os.path.join(root, fname)
                rel_p = os.path.relpath(abs_p, src).replace('\\', '/')
                if _is_excluded(rel_p):
                    continue
                tf.add(abs_p, arcname=f'{arcprefix}/{rel_p}')
                n_files += 1

    with tarfile.open(tmp.name, 'w:gz') as tf:
        # 1. tunet source → tunet/
        print(f'[pack] tunet/ ← {tunet_root}')
        _add_dir(tf, tunet_root, 'tunet', exclude=SOURCE_EXCLUDES)

        # 2. data uploads — keyed by remote path under /input
        for local_path, remote_path in data_uploads.items():
            arcname = remote_path.lstrip('/')
            if arcname.startswith('input/'):
                arcname = arcname[len('input/'):]
            elif arcname.startswith('output/'):
                # resume checkpoints / log → /input/output/<job>/...
                # The Spark agent only mounts /input/ from our tarball; train.py
                # will copy them into /output/ on first run via rewrite below.
                arcname = arcname  # keep as-is, see __rewrite at bottom
            if os.path.isdir(local_path):
                print(f'[pack] {arcname}/ ← {local_path}')
                _add_dir(tf, local_path, arcname)
            elif os.path.isfile(local_path):
                size_mb = os.path.getsize(local_path) / 1024 / 1024
                print(f'[pack] {arcname} ← {os.path.basename(local_path)} ({size_mb:.0f} MB)')
                tf.add(local_path, arcname=arcname)
                n_files += 1

        # 3. config.yaml
        cfg_arc = 'config.yaml'
        print(f'[pack] {cfg_arc}')
        tf.add(config_path, arcname=cfg_arc)
        n_files += 1

        # 4. spark_start.sh
        start_sh = os.path.join(tunet_root, 'spark_start.sh')
        if os.path.isfile(start_sh):
            print(f'[pack] spark_start.sh')
            tf.add(start_sh, arcname='spark_start.sh')
            n_files += 1

    size_mb = os.path.getsize(tmp.name) / 1024 / 1024
    print(f'[pack] {n_files} files → {size_mb:.1f} MB → {tmp.name}')
    return tmp.name


# ── Config rewriting for /input + /output paths ──────────────────────────────

def rewrite_config_for_spark(config, data_uploads, resume_pth=None):
    """
    Rewrite local paths to Spark agent paths. Mirrors rewrite_config_for_pod
    from runpod_launch.py but with /input/ and /output/ prefixes.

    data_uploads: dict populated here, mapping local_path → remote_path
                  (relative to /input or /output, used by build_input_tar).
    """
    cfg = copy.deepcopy(config)
    data_sect = cfg.get('data') or {}

    job_name = os.path.basename(
        data_sect.get('output_dir', 'job').rstrip('/\\')
    ) or 'job'
    remote_output = f'{OUTPUT_REMOTE}/{job_name}'
    local_output  = data_sect.get('output_dir', '')

    # 1. src_dir / dst_dir → /input/data/{src,dst}
    for key, sub in (('src_dir', 'src'), ('dst_dir', 'dst')):
        local_path = data_sect.get(key, '')
        if local_path and os.path.isdir(local_path):
            remote_path = f'{DATA_REMOTE}/{sub}'
            data_uploads[local_path] = remote_path
            cfg['data'][key] = remote_path
            print(f'[rewrite] {key:10s}: {local_path} → {remote_path}')
        elif local_path:
            print(f'[rewrite] {key:10s}: {local_path!r} not found locally — leaving as-is')

    # 2. finetune_from → /input/uploads/<basename>
    ft = (cfg.get('training') or {}).get('finetune_from')
    if ft and os.path.isfile(ft):
        remote = f'{INPUT_REMOTE}/uploads/{os.path.basename(ft)}'
        data_uploads[ft] = remote
        cfg['training']['finetune_from'] = remote
        print(f'[rewrite] finetune_from: {ft} → {remote}')

    # 3. resume checkpoint
    # Spark agent only mounts /input/ from our tarball, so we ship the .pth
    # under /input/output/<job>/ and have spark_start copy it into /output/<job>/
    # on first run. Same for training.log — see _rewrite_resume_helper below.
    ckpt = None
    if resume_pth and os.path.isfile(resume_pth):
        ckpt = resume_pth
        print(f'[rewrite] resume (override): {resume_pth}')
    elif local_output and os.path.isdir(local_output):
        ckpts = _rl.scan_checkpoints(local_output)
        latest = next((p for p, il in ckpts if il), None)
        if latest:
            ckpt = latest
            print(f'[rewrite] resume (latest): {latest}')
        elif ckpts:
            ckpt = ckpts[0][0]
            print(f'[rewrite] resume (newest): {ckpt}')
        else:
            print(f'[rewrite] no checkpoint in {local_output} — fresh training')
    else:
        print(f'[rewrite] no output_dir — fresh training')

    if ckpt:
        # Stage the checkpoint inside the tarball at output/<job>/<basename>
        # (relative path; build_input_tar arcname-strips the leading /).
        # spark_start.sh seeds /output/<job>/ from /input/output/<job>/ before
        # train.py runs, so the resume picks it up naturally.
        staged = f'output/{job_name}/{os.path.basename(ckpt)}'
        data_uploads[ckpt] = staged
        print(f'[rewrite] staged → /input/{staged}')

    # 4. existing training.log — same staging trick as the checkpoint
    if local_output:
        local_log = os.path.join(local_output, 'training.log')
        if os.path.isfile(local_log):
            staged = f'output/{job_name}/training.log'
            data_uploads[local_log] = staged
            size_mb = os.path.getsize(local_log) / 1024 / 1024
            print(f'[rewrite] training.log: {local_log} ({size_mb:.1f} MB) → /input/{staged}')

    cfg['data']['output_dir'] = remote_output
    return cfg, job_name, remote_output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Launch TuNet training on Spark Compute v1')
    ap.add_argument('--config',     default=None, help='Path to config.yaml (required for submit)')
    ap.add_argument('--gpu',        default='l40s', choices=list(GPU_TYPES.keys()),
                    help='GPU shortcut (or pass --instance_type for raw SKU)')
    ap.add_argument('--instance_type', default=None,
                    help='Raw Spark SKU (overrides --gpu). e.g. g6e.4xlarge')
    ap.add_argument('--image',      default=DEFAULT_IMAGE, help='Docker image')
    ap.add_argument('--name',       default=None, help='Job name (default: derived from config)')
    ap.add_argument('--idle_hold',  type=int, default=DEFAULT_IDLE,
                    help='Seconds to keep instance warm after exit (0 = stop immediately)')
    ap.add_argument('--no_tail',    action='store_true', help='Skip log streaming after submit')
    ap.add_argument('--list_skus',  action='store_true', help='Print available SKUs and exit')
    ap.add_argument('--list_jobs',  action='store_true', help='Print recent jobs and exit')
    ap.add_argument('--cancel',     metavar='JOB_ID', help='Cancel a job by ID')
    ap.add_argument('--logs',       metavar='JOB_ID', help='Stream logs for an existing job')
    args = ap.parse_args()

    # ── Sub-commands ─────────────────────────────────────────────────────────
    if args.list_skus:
        skus = list_skus()
        print(f'{len(skus)} SKUs available:')
        for s in skus:
            print(f'  {s["instanceType"]:20s}  {s["gpuCount"]}× {s["gpuType"]:30s}  {s["gpuMemoryGb"]}GB')
        return

    if args.list_jobs:
        jobs = list_jobs()
        print(f'{len(jobs)} recent jobs:')
        for j in jobs:
            jid    = j.get('id', '?')
            status = j.get('status', '?')
            inst   = j.get('instance_type_name') or j.get('instanceType') or '?'
            print(f'  {jid:40s}  {status:12s}  {inst}')
        return

    if args.cancel:
        print(json.dumps(cancel_job(args.cancel), indent=2))
        return

    if args.logs:
        stream_logs(args.logs)
        return

    # ── Submit ───────────────────────────────────────────────────────────────
    if not args.config:
        sys.exit('ERROR: --config required for submit (or use --list_skus / --list_jobs / --cancel / --logs)')
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_uploads = {}
    pod_config, job_name, remote_output = rewrite_config_for_spark(config, data_uploads)

    instance_type = args.instance_type or GPU_TYPES[args.gpu]
    job_label = args.name or f'tunet-{job_name}'

    print(f'\n[spark] Submitting: {job_label}')
    print(f'        Instance : {instance_type}')
    print(f'        Image    : {args.image}')
    print(f'        Idle hold: {args.idle_hold}s')
    print()

    # ── Pack the tarball (config first as a temp file) ───────────────────────
    tunet_root = os.path.dirname(os.path.abspath(args.config))
    if not os.path.exists(os.path.join(tunet_root, 'train.py')):
        # config wasn't in the tunet root — figure out the real one
        tunet_root = os.path.dirname(os.path.abspath(__file__))

    cfg_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(pod_config, cfg_tmp, default_flow_style=False, sort_keys=False)
    cfg_tmp.close()

    tar_path = build_input_tar(tunet_root, cfg_tmp.name, data_uploads)
    os.unlink(cfg_tmp.name)

    # ── Submit + upload + tail ───────────────────────────────────────────────
    try:
        # Command run inside the container by the agent.
        # spark_start.sh seeds /output/<job>/ from /input/output/<job>/ if a
        # resume checkpoint was staged, then runs train.py in foreground.
        command = ['bash', '/input/spark_start.sh', remote_output, CONFIG_REMOTE]

        resp = submit_job(job_label, instance_type, args.image, command, args.idle_hold)
        print(f'[spark] Response: {json.dumps(resp, indent=2)[:500]}')

        # Walt's doc shape: { jobId, input: { uploadUrl, exampleCurl }, ... }
        job_id = resp.get('jobId') or resp.get('id')
        upload_url = (resp.get('input') or {}).get('uploadUrl')
        if not job_id or not upload_url:
            sys.exit(f'ERROR: submit response missing jobId or uploadUrl: {resp}')

        upload_tarball(upload_url, tar_path)
        print(f'[spark] Job submitted: {job_id}')

        if not args.no_tail:
            stream_logs(job_id)
        else:
            print(f'\nTo watch logs:    python spark_launch.py --logs {job_id}')
            print(f'To cancel:        python spark_launch.py --cancel {job_id}')

    finally:
        try:
            os.unlink(tar_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()
