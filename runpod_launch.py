"""
runpod_launch.py — fully automated TuNet training on RunPod

Usage:
    python runpod_launch.py --config config.yaml [options]

What it does:
    1. Creates a RunPod pod (L40S by default) via GraphQL API
    2. Waits for SSH to become available
    3. Rsyncs the tunet repo to /workspace/tunet on the pod
    4. Uploads config.yaml to /workspace/config.yaml
    5. If config has finetune_from or a resume checkpoint, uploads
       the local .pth file and rewrites the path in the config
    6. Runs runpod_start.sh which installs deps, starts monitor API,
       and launches train.py
    7. Prints the Spark monitor URL and tails logs

Prerequisites:
    pip install requests pyyaml

Environment variables:
    RUNPOD_API_KEY   — your RunPod API key (or pass --api_key)
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import requests
import yaml

# Load .env from the project root (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass

# ── RunPod GraphQL API ────────────────────────────────────────────────────────

RUNPOD_API = 'https://api.runpod.io/graphql'

GPU_TYPES = {
    'rtxpro6000': 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
    'l40s':       'NVIDIA L40S',
    'a100':       'NVIDIA A100 80GB PCIe',
    'a100sxm':    'NVIDIA A100-SXM4-80GB',
    '4090':       'NVIDIA GeForce RTX 4090',
    'a40':        'NVIDIA A40',
    'l4':         'NVIDIA L4',
}

# Recommended PyTorch Docker image — has CUDA 12.4, Python 3.11, no GUI deps
DEFAULT_IMAGE = 'runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204'

MONITOR_PORT = 8080
TUNET_REMOTE = '/workspace/tunet'
CONFIG_REMOTE = '/workspace/config.yaml'
OUTPUT_REMOTE = '/workspace/output'


def gql(api_key, query, variables=None):
    resp = requests.post(
        RUNPOD_API,
        json={'query': query, 'variables': variables or {}},
        headers={'Authorization': f'Bearer {api_key}'},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"RunPod API HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    if 'errors' in data:
        raise RuntimeError(f"RunPod API error: {json.dumps(data['errors'], indent=2)}")
    return data['data']


def find_ssh_key():
    """Return the path to the RunPod SSH private key."""
    priv = os.path.expanduser('~/.ssh/runpod')
    if not os.path.isfile(priv):
        raise FileNotFoundError(f'RunPod SSH private key not found: {priv}')
    return priv


def create_pod(api_key, name, gpu_type, image, disk_gb, volume_gb):
    q = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        costPerHr
        machineId
      }
    }
    """
    inp = {
        'name':              name,
        'gpuTypeId':         gpu_type,
        'imageName':         image,
        'gpuCount':          1,
        'containerDiskInGb': disk_gb,
        'volumeInGb':        volume_gb,
        'volumeMountPath':   '/workspace',
        'startSsh':          True,
        'ports':             f'22/tcp,{MONITOR_PORT}/http',
        'env': [
            {'key': 'OPENCV_IO_ENABLE_OPENEXR', 'value': '1'},
            {'key': 'RUNPOD_API_KEY', 'value': api_key},
        ],
        'cloudType':         'SECURE',
    }
    data = gql(api_key, q, {'input': inp})
    return data['podFindAndDeployOnDemand']


def get_pod(api_key, pod_id):
    q = """
    query GetPod($input: PodFilter) {
      pod(input: $input) {
        id
        name
        desiredStatus
        costPerHr
        runtime {
          ports { ip isIpPublic privatePort publicPort type }
        }
      }
    }
    """
    data = gql(api_key, q, {'input': {'podId': pod_id}})
    return data['pod']


def terminate_pod(api_key, pod_id):
    q = """
    mutation TerminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    gql(api_key, q, {'input': {'podId': pod_id}})
    print(f'[runpod] Pod {pod_id} terminated.')


def get_ssh_details(pod):
    """Extract host/port from pod runtime ports."""
    if not pod.get('runtime') or not pod['runtime'].get('ports'):
        return None, None
    for p in pod['runtime']['ports']:
        if p.get('privatePort') == 22 and p.get('isIpPublic'):
            return p['ip'], p['publicPort']
    return None, None


def wait_for_pod(api_key, pod_id, timeout=600):
    """Poll until pod is RUNNING and has SSH port assigned."""
    print(f'[runpod] Waiting for pod {pod_id} to be ready...')
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        pod = get_pod(api_key, pod_id)
        desired = pod.get('desiredStatus', '')
        if desired != last_status:
            elapsed = int(timeout - (deadline - time.time()))
            print(f'[runpod] {desired}  ({elapsed}s elapsed)', flush=True)
            last_status = desired
        if desired == 'RUNNING':
            host, port = get_ssh_details(pod)
            if host and port:
                print(f'[runpod] Ready!')
                return pod, host, port
        elif desired in ('EXITED', 'DEAD', 'FAILED'):
            raise RuntimeError(f'Pod entered status {desired}')
        time.sleep(5)
    raise TimeoutError(f'Pod not ready after {timeout}s')


def wait_for_ssh(host, port, key_path, timeout=120):
    """Try SSH connection until it succeeds."""
    print(f'[ssh] Waiting for SSH at {host}:{port}', end='', flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ['ssh', '-n', '-T',
             '-o', 'StrictHostKeyChecking=no',
             '-o', 'BatchMode=yes',
             '-o', 'ConnectTimeout=5',
             '-i', key_path,
             '-p', str(port), f'root@{host}', 'echo ok'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(' connected!')
            return
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError('SSH never became available')


def ssh(host, port, key_path, cmd, check=True, capture=False, timeout=120):
    """Run a command on the pod via SSH."""
    result = subprocess.run(
        ['ssh', '-n', '-T',
         '-o', 'StrictHostKeyChecking=no',
         '-o', 'BatchMode=yes',
         '-o', 'ConnectTimeout=15',
         '-i', key_path,
         '-p', str(port), f'root@{host}', cmd],
        capture_output=capture, text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f'SSH command failed: {result.stderr.strip() or cmd}')
    return result


def rsync(src, host, port, key_path, dest, exclude=None):
    """Upload a local directory to the pod. Uses tar+scp (works on Windows, no unzip needed)."""
    import tarfile, fnmatch, tempfile

    src = src.rstrip('/\\')
    exclude = exclude or []

    def _is_excluded(rel_path):
        parts = rel_path.replace('\\', '/').split('/')
        for pattern in exclude:
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    tmp = tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False)
    tmp.close()
    try:
        print(f'[upload] Packing {src} ...', flush=True)
        n_files = 0
        with tarfile.open(tmp.name, 'w:gz') as tf:
            for root, dirs, files in os.walk(src):
                dirs[:] = [d for d in dirs
                           if not _is_excluded(os.path.relpath(os.path.join(root, d), src))]
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, src).replace('\\', '/')
                    if not _is_excluded(rel_path):
                        tf.add(abs_path, arcname=rel_path)
                        n_files += 1
        size_mb = os.path.getsize(tmp.name) / 1024 / 1024
        print(f'[upload] Packed {n_files} files → {size_mb:.1f} MB, uploading...')
        remote_tar = '/tmp/_tunet_upload.tar.gz'
        scp(tmp.name, host, port, key_path, remote_tar)
        print(f'[upload] Extracting on pod...')
        ssh(host, port, key_path, f'mkdir -p "{dest}" && tar -xzf {remote_tar} -C "{dest}" && rm {remote_tar}')
        print(f'[upload] Done → {dest}')
    finally:
        os.unlink(tmp.name)


def scp(local_path, host, port, key_path, remote_path):
    """Copy a single file to the pod."""
    result = subprocess.run(
        ['scp', '-o', 'StrictHostKeyChecking=no',
         '-o', 'BatchMode=yes',
         '-o', 'ConnectTimeout=15',
         '-i', key_path,
         '-P', str(port), local_path, f'root@{host}:{remote_path}'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f'scp failed: {result.stderr.strip()}')


# ── Config rewriting for remote paths ────────────────────────────────────────

DATA_REMOTE = '/workspace/data'


def scan_checkpoints(output_dir):
    """
    Return a list of .pth files in output_dir sorted newest-first.
    Flags the _latest file (if present) separately.
    Returns: list of (path, is_latest) tuples.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return []
    pths = []
    for fname in os.listdir(output_dir):
        if not fname.endswith('.pth'):
            continue
        fp = os.path.join(output_dir, fname)
        is_latest = fname.endswith('_tunet_latest.pth') or fname == 'tunet_latest.pth'
        pths.append((fp, is_latest, os.path.getmtime(fp)))
    pths.sort(key=lambda x: x[2], reverse=True)
    return [(fp, il) for fp, il, _ in pths]


def rewrite_config_for_pod(config, data_uploads, resume_pth=None):
    """
    Return a copy of config with local paths replaced by pod paths.

    data_uploads: dict populated here, mapping local_path -> remote_path.
    resume_pth:   optional explicit local .pth to resume from (overrides auto-detect).

    Covers:
      1. src_dir / dst_dir  — training image folders → /workspace/data/src|dst
      2. finetune_from      — explicit .pth path in config
      3. resume checkpoint  — resume_pth if given, else auto-detect *_tunet_latest.pth
                              in output_dir; if nothing found, trains fresh.
    """
    cfg = copy.deepcopy(config)
    data_sect = cfg.get('data') or {}

    # ── Derive job name + remote output_dir ──────────────────────────────────
    job_name = os.path.basename(
        data_sect.get('output_dir', 'job').rstrip('/\\')
    ) or 'job'
    remote_output = f'{OUTPUT_REMOTE}/{job_name}'
    local_output  = data_sect.get('output_dir', '')

    # ── Case 1: src_dir / dst_dir ────────────────────────────────────────────
    for key, remote_sub in (('src_dir', 'src'), ('dst_dir', 'dst')):
        local_path = data_sect.get(key, '')
        if local_path and os.path.isdir(local_path):
            remote_path = f'{DATA_REMOTE}/{remote_sub}'
            data_uploads[local_path] = remote_path
            cfg['data'][key] = remote_path
            print(f'[upload] {key:10s}: {local_path}')
            print(f'                      → {remote_path}')
        elif local_path:
            print(f'[upload] {key:10s}: {local_path!r} not found locally, leaving as-is')

    # ── Case 2: finetune_from in config ──────────────────────────────────────
    ft = (cfg.get('training') or {}).get('finetune_from')
    if ft and os.path.isfile(ft):
        remote = f'/workspace/uploads/{os.path.basename(ft)}'
        data_uploads[ft] = remote
        cfg['training']['finetune_from'] = remote
        print(f'[upload] finetune_from: {ft}')
        print(f'                      → {remote}')

    # ── Case 3: resume checkpoint ────────────────────────────────────────────
    # Priority: explicit resume_pth arg > auto-detect *_tunet_latest.pth > fresh start
    ckpt_to_upload = None
    if resume_pth and os.path.isfile(resume_pth):
        ckpt_to_upload = resume_pth
        print(f'[upload] resume (override): {resume_pth}')
    elif local_output and os.path.isdir(local_output):
        # Find the _latest file first, then fall back to newest .pth
        ckpts = scan_checkpoints(local_output)
        latest = next((p for p, il in ckpts if il), None)
        if latest:
            ckpt_to_upload = latest
            print(f'[upload] resume (latest): {latest}')
        elif ckpts:
            ckpt_to_upload = ckpts[0][0]
            print(f'[upload] resume (newest): {ckpt_to_upload}')
        else:
            print(f'[upload] no checkpoint found in {local_output} — training fresh')
    else:
        print(f'[upload] no output_dir found — training fresh')

    if ckpt_to_upload:
        remote_path = f'{remote_output}/{os.path.basename(ckpt_to_upload)}'
        data_uploads[ckpt_to_upload] = remote_path
        print(f'                      → {remote_path}')

    cfg['data']['output_dir'] = remote_output
    return cfg, job_name


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Launch TuNet training on RunPod')
    ap.add_argument('--config',    required=True, help='Path to config.yaml')
    ap.add_argument('--api_key',   default=os.environ.get('RUNPOD_API_KEY') or os.environ.get('runpodapi'), help='RunPod API key')
    ap.add_argument('--gpu',       default='rtxpro6000', choices=list(GPU_TYPES.keys()), help='GPU type')
    ap.add_argument('--image',     default=DEFAULT_IMAGE, help='Docker image')
    ap.add_argument('--disk',      type=int, default=50,  help='Container disk GB')
    ap.add_argument('--volume',    type=int, default=100, help='Persistent volume GB (for checkpoints)')
    ap.add_argument('--name',      default=None, help='Pod name (default: derived from config)')
    ap.add_argument('--data_dir',  default=None, help='Local data directory to upload to /workspace/data')
    ap.add_argument('--no_upload_code', action='store_true', help='Skip uploading tunet source (already on pod)')
    ap.add_argument('--terminate_on_finish', action='store_true', help='Terminate pod when training ends')
    ap.add_argument('--tail',      action='store_true', help='Stream training log after launch')
    ap.add_argument('--sync',      default=None, metavar='DIR', help='Continuously sync .pth/.onnx to this local dir')
    ap.add_argument('--sync_interval', type=int, default=120, help='Seconds between sync polls (default 120)')
    args = ap.parse_args()

    if not args.api_key:
        sys.exit('ERROR: Set RUNPOD_API_KEY env var or pass --api_key')

    # Load and validate config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_uploads = {}
    pod_config, job_name = rewrite_config_for_pod(config, data_uploads)

    pod_name = args.name or f'tunet-{job_name}'
    gpu_type = GPU_TYPES[args.gpu]

    # ── Load SSH key ─────────────────────────────────────────
    key_path = find_ssh_key()
    print(f'[ssh] Using key: {key_path}')

    print(f'\n[runpod] Creating pod: {pod_name}')
    print(f'         GPU    : {gpu_type}')
    print(f'         Image  : {args.image}')
    print(f'         Disk   : {args.disk}GB container + {args.volume}GB volume')
    print()

    # ── 1. Create pod ────────────────────────────────────────
    pod = create_pod(args.api_key, pod_name, gpu_type, args.image, args.disk, args.volume)
    pod_id = pod['id']
    print(f'[runpod] Pod created: {pod_id}  (${pod.get("costPerHr", "?")}/hr)')

    try:
        # ── 2. Wait for ready + SSH ──────────────────────────
        pod, host, port = wait_for_pod(args.api_key, pod_id)
        wait_for_ssh(host, port, key_path)

        # ── 3. Upload tunet source ───────────────────────────
        if not args.no_upload_code:
            tunet_root = os.path.dirname(os.path.abspath(__file__))
            print(f'[upload] Syncing tunet source → {TUNET_REMOTE}')
            ssh(host, port, key_path, f'mkdir -p {TUNET_REMOTE}')
            rsync(
                tunet_root + '/',
                host, port,
                key_path,
                TUNET_REMOTE,
                exclude=[
                    '__pycache__', '*.pyc', '.git', 'node_modules',
                    'Spark', '_archive', '*.pth', '*.onnx',
                    'tunet_session.yaml',
                ]
            )

        # ── 4. Upload data dirs + checkpoints ───────────────
        # data_uploads contains all local→remote mappings from rewrite_config_for_pod:
        #   src_dir → /workspace/data/src
        #   dst_dir → /workspace/data/dst
        #   finetune_from .pth → /workspace/uploads/...
        #   resume .pth → /workspace/output/<job>/...
        # Dirs are uploaded with rsync; individual files (.pth) with scp.
        if data_uploads:
            remote_dirs = {os.path.dirname(p) for p in data_uploads.values()}
            for d in remote_dirs:
                ssh(host, port, key_path, f'mkdir -p "{d}"')
            for local_path, remote_path in data_uploads.items():
                if os.path.isdir(local_path):
                    print(f'[upload] {local_path} → {remote_path}')
                    rsync(local_path + '/', host, port, key_path, remote_path)
                else:
                    size_mb = os.path.getsize(local_path) / 1024 / 1024
                    print(f'[upload] {os.path.basename(local_path)} ({size_mb:.0f} MB) → {remote_path}')
                    scp(local_path, host, port, key_path, remote_path)

        # ── 6. Write rewritten config to pod ────────────────
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
            yaml.dump(pod_config, tf, default_flow_style=False, sort_keys=False)
            tmp_cfg = tf.name
        print(f'[upload] Config → {CONFIG_REMOTE}')
        scp(tmp_cfg, host, port, key_path, CONFIG_REMOTE)
        os.unlink(tmp_cfg)

        # ── 7. Upload and run start script ──────────────────
        start_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runpod_start.sh')
        scp(start_sh, host, port, key_path, '/workspace/runpod_start.sh')
        ssh(host, port, key_path, 'chmod +x /workspace/runpod_start.sh')

        output_dir = f'{OUTPUT_REMOTE}/{job_name}'
        print(f'\n[launch] Starting training...')
        print(f'         Output dir : {output_dir}')
        monitor_url = f'https://{pod_id}-{MONITOR_PORT}.proxy.runpod.net'
        print(f'         Monitor URL: {monitor_url}')
        print(f'         Spark UI   : open Spark/frontend/job-detail.html?pod={monitor_url}')
        print()

        # Run start script in background via nohup so SSH disconnect doesn't kill it
        launch_cmd = (
            f'nohup bash /workspace/runpod_start.sh '
            f'"{output_dir}" "{CONFIG_REMOTE}" "{MONITOR_PORT}" '
            f'> /workspace/launch.log 2>&1 &'
        )
        ssh(host, port, key_path, launch_cmd)
        print('[launch] Training started. SSH connection can be closed safely.')
        print(f'\nTo SSH into the pod:')
        print(f'  ssh -i ~/.ssh/runpod -p {port} root@{host}')
        print(f'\nTo stream the training log:')
        print(f'  ssh -i ~/.ssh/runpod -p {port} root@{host} "tail -f {output_dir}/training.log"')
        print(f'\nTo stop training:')
        print(f'  python runpod_launch.py --stop {pod_id}')
        print(f'\nTo download results now:')
        print(f'  python runpod_launch.py --download {pod_id} --dest YOUR_PATH --job {job_name}')
        print(f'\nTo terminate pod:')
        print(f'  python runpod_launch.py --terminate {pod_id}')

        # ── 8. Start background sync ─────────────────────────
        if args.sync:
            sync_loop(host, port, key_path, output_dir, args.sync, args.sync_interval)

        # ── 9. Optionally tail the log ───────────────────────
        if args.tail:
            print('\n[tail] Streaming training log (Ctrl+C to detach, pod keeps running)...\n')
            time.sleep(8)  # give train.py a moment to start
            try:
                subprocess.run(
                    ['ssh', '-o', 'StrictHostKeyChecking=no',
                     '-i', key_path,
                     '-p', str(port), f'root@{host}',
                     f'tail -f {output_dir}/training.log']
                )
            except KeyboardInterrupt:
                print('\n[tail] Detached. Pod is still running.')

    except Exception as e:
        print(f'\nERROR: {e}')
        print(f'Pod {pod_id} is still running. Terminate with:')
        print(f'  python runpod_launch.py --terminate {pod_id} --api_key YOUR_KEY')
        sys.exit(1)


def sync_loop(host, port, key_path, remote_output, local_dest, interval):
    """
    Background thread: poll for new .pth/.onnx files on the pod every N seconds
    and scp them locally. Uses SSH ls + scp (no rsync needed, works on Windows).
    """
    import threading

    os.makedirs(local_dest, exist_ok=True)
    _seen = {}  # fname -> size, to detect new/changed files

    EXTENSIONS = ('.pth', '.onnx', '.jpg')

    def _poll():
        # List files with sizes in remote_output
        result = subprocess.run(
            ['ssh', '-o', 'StrictHostKeyChecking=no', '-i', key_path,
             '-p', str(port), f'root@{host}',
             f'find {remote_output} -maxdepth 2 -type f 2>/dev/null | xargs ls -l 2>/dev/null || true'],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 9:
                continue
            size  = parts[4]
            rpath = parts[8]
            fname = os.path.basename(rpath)
            if not any(fname.endswith(ext) for ext in EXTENSIONS):
                continue
            key = rpath
            if _seen.get(key) == size:
                continue
            _seen[key] = size
            local_path = os.path.join(local_dest, fname)
            try:
                subprocess.run(
                    ['scp', '-o', 'StrictHostKeyChecking=no', '-i', key_path,
                     '-P', str(port), f'root@{host}:{rpath}', local_path],
                    capture_output=True
                )
                mb = os.path.getsize(local_path) / 1024 / 1024
                print(f'[sync] ↓ {fname}  ({mb:.0f} MB)')
            except Exception as e:
                print(f'[sync] failed to download {fname}: {e}')

    def _run():
        print(f'[sync] Watching for new checkpoints → {local_dest}  (every {interval}s)')
        while True:
            time.sleep(interval)
            try:
                _poll()
            except Exception as e:
                print(f'[sync] error: {e}')

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ── Sub-commands: stop / terminate ───────────────────────────────────────────

def download(api_key, pod_id, local_dest, job_name=None):
    """Download checkpoints and exports from the pod."""
    key_path = find_ssh_key()
    pod = get_pod(api_key, pod_id)
    host, port = get_ssh_details(pod)
    if not host:
        sys.exit('ERROR: Pod has no SSH port — is it running?')

    remote_output = f'{OUTPUT_REMOTE}/{job_name}' if job_name else OUTPUT_REMOTE
    local_dest = os.path.abspath(local_dest)
    os.makedirs(local_dest, exist_ok=True)

    print(f'[download] {host}:{port}  {remote_output} → {local_dest}')
    print(f'[download] Fetching .pth, .onnx and log files...')

    EXTENSIONS = ('.pth', '.onnx', '.jpg', '.log')
    result = subprocess.run(
        ['ssh', '-o', 'StrictHostKeyChecking=no', '-i', key_path,
         '-p', str(port), f'root@{host}',
         f'find {remote_output} -maxdepth 2 -type f 2>/dev/null || true'],
        capture_output=True, text=True, check=True
    )
    files = [l.strip() for l in result.stdout.splitlines() if l.strip()
             and any(l.strip().endswith(ext) for ext in EXTENSIONS)]
    for rpath in files:
        fname = os.path.basename(rpath)
        local_path = os.path.join(local_dest, fname)
        print(f'  ↓ {fname}')
        subprocess.run(
            ['scp', '-o', 'StrictHostKeyChecking=no', '-i', key_path,
             '-P', str(port), f'root@{host}:{rpath}', local_path],
            check=True
        )
    print(f'[download] Done → {local_dest}  ({len(files)} files)')


def cli():
    """
    Also handles:
      python runpod_launch.py --terminate POD_ID
      python runpod_launch.py --stop POD_ID
      python runpod_launch.py --status POD_ID
      python runpod_launch.py --download POD_ID --dest PATH [--job JOB_NAME]
    """
    # Peek at argv before argparse so we can handle sub-commands cleanly
    if len(sys.argv) >= 3 and sys.argv[1] in ('--terminate', '--stop', '--status', '--download'):
        action  = sys.argv[1].lstrip('-')
        pod_id  = sys.argv[2]
        api_key = None
        dest    = None
        job     = None
        for i, a in enumerate(sys.argv):
            if a == '--api_key' and i + 1 < len(sys.argv):
                api_key = sys.argv[i + 1]
            if a == '--dest' and i + 1 < len(sys.argv):
                dest = sys.argv[i + 1]
            if a == '--job' and i + 1 < len(sys.argv):
                job = sys.argv[i + 1]
        if not api_key:
            api_key = os.environ.get('RUNPOD_API_KEY') or os.environ.get('runpodapi')
        if not api_key:
            sys.exit('ERROR: --api_key required')

        if action == 'terminate':
            terminate_pod(api_key, pod_id)
        elif action == 'status':
            pod = get_pod(api_key, pod_id)
            print(json.dumps(pod, indent=2))
        elif action == 'stop':
            import urllib.request
            url = f'https://{pod_id}-{MONITOR_PORT}.proxy.runpod.net/api/stop'
            req = urllib.request.Request(url, method='POST', data=b'')
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    print(json.loads(r.read()))
            except Exception as e:
                print(f'Could not reach monitor API: {e}')
                print('SSH in and: touch /workspace/output/<job>/.stop_training')
        elif action == 'download':
            if not dest:
                sys.exit('ERROR: --dest PATH required for --download')
            download(api_key, pod_id, dest, job_name=job)
        return True
    return False


if __name__ == '__main__':
    if not cli():
        main()
