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


def fetch_gpu_pricing(api_key):
    """Return list of {id, displayName, memoryInGb, securePrice, communityPrice} dicts."""
    q = '{ gpuTypes { id displayName memoryInGb securePrice communityPrice } }'
    data = gql(api_key, q)
    return data.get('gpuTypes') or []


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


def _pack_dir(src, exclude=None):
    """Pack a local directory into a temp .tar.gz. Returns the temp file path — caller must delete."""
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
    return tmp.name, n_files


def _upload_tar(tar_path, host, port, key_path, dest):
    """SCP a pre-packed tar.gz to the pod and extract it into dest."""
    remote_tar = '/tmp/_tunet_upload.tar.gz'
    scp(tar_path, host, port, key_path, remote_tar)
    ssh(host, port, key_path, f'mkdir -p "{dest}" && tar -xzf {remote_tar} -C "{dest}" && rm {remote_tar}')


def rsync(src, host, port, key_path, dest, exclude=None):
    """Upload a local directory to the pod. Uses tar+scp (works on Windows, no unzip needed)."""
    src = src.rstrip('/\\')
    print(f'[upload] Packing {src} ...', flush=True)
    tar_path, n_files = _pack_dir(src, exclude)
    try:
        size_mb = os.path.getsize(tar_path) / 1024 / 1024
        print(f'[upload] Packed {n_files} files → {size_mb:.1f} MB, uploading...')
        _upload_tar(tar_path, host, port, key_path, dest)
        print(f'[upload] Done → {dest}')
    finally:
        os.unlink(tar_path)


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

    # ── Case 4: existing training.log ────────────────────────────────────────
    # Upload so the monitor chart shows full history, not just the current session.
    if local_output:
        local_log = os.path.join(local_output, 'training.log')
        if os.path.isfile(local_log):
            remote_log = f'{remote_output}/training.log'
            data_uploads[local_log] = remote_log
            size_mb = os.path.getsize(local_log) / 1024 / 1024
            print(f'[upload] training.log   : {local_log}  ({size_mb:.1f} MB)')
            print(f'                      → {remote_log}')

    cfg['data']['output_dir'] = remote_output
    return cfg, job_name


# ── Benchmark chart ──────────────────────────────────────────────────────────

def _save_benchmark_chart(results, steps):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        import numpy as np
    except ImportError:
        print('[chart] matplotlib not installed — skipping chart (pip install matplotlib)')
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    costs  = [r['cost_per_hr']    for r in results]
    speeds = [r['images_per_hr']  for r in results]
    c1ks   = [r['cost_per_1k']    for r in results]
    names  = [r['gpu_name']       for r in results]

    # Shorten names: strip "NVIDIA " prefix and long suffixes
    def short(n):
        n = n.replace('NVIDIA ', '')
        n = n.replace(' Server Edition', '')
        n = n.replace(' PCIe', '')
        return n
    labels = [short(n) for n in names]

    # Colour by $/1k (green = cheap, red = expensive)
    norm = plt.Normalize(min(c1ks), max(c1ks))
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in c1ks]

    # Bubble size ~ images/hr
    max_spd = max(speeds)
    sizes = [350 + 1200 * (s / max_spd) for s in speeds]

    sc = ax.scatter(costs, speeds, s=sizes, c=c1ks, cmap='RdYlGn_r',
                    norm=norm, edgecolors='#334155', linewidths=1.2, zorder=3, alpha=0.92)

    # Colorbar
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('$/1k images  (green = cheaper)', color='#94a3b8', fontsize=10)
    cb.ax.yaxis.set_tick_params(color='#475569')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=9)
    cb.outline.set_edgecolor('#1e293b')

    # Labels on each bubble
    for x, y, lbl, c1k in zip(costs, speeds, labels, c1ks):
        ax.annotate(
            lbl, xy=(x, y),
            xytext=(0, 14), textcoords='offset points',
            ha='center', va='bottom',
            fontsize=9, color='#e2e8f0', fontweight='600',
            path_effects=[pe.withStroke(linewidth=2, foreground='#0f172a')],
        )
        ax.annotate(
            f'${c1k:.3f}/1k', xy=(x, y),
            xytext=(0, -18), textcoords='offset points',
            ha='center', va='top',
            fontsize=8, color='#64748b',
            path_effects=[pe.withStroke(linewidth=2, foreground='#0f172a')],
        )

    # "Value frontier" — pareto-optimal points (cheapest for given speed)
    sorted_by_cost = sorted(zip(costs, speeds, labels), key=lambda t: t[0])
    pareto = []
    best_spd = 0
    for c, s, l in sorted_by_cost:
        if s > best_spd:
            pareto.append((c, s))
            best_spd = s
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.step(px, py, where='post', color='#7c3aed', linewidth=1.2,
                linestyle='--', alpha=0.5, zorder=2, label='Value frontier')
        ax.legend(facecolor='#1e293b', edgecolor='#334155',
                  labelcolor='#94a3b8', fontsize=9)

    # Grid
    ax.grid(True, color='#1e293b', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Axes styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#f8fafc')

    ax.set_xlabel('Cost  ($/hr)', fontsize=11, labelpad=8)
    ax.set_ylabel('Throughput  (images / hr)', fontsize=11, labelpad=8)
    ax.set_title(f'GPU Speed vs Cost — {steps}-step benchmark', fontsize=13, pad=14)

    # Y axis: comma-formatted integers
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f'{int(v):,}'))

    plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_chart.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[chart] Saved → {out}')
    # Open in default viewer
    try:
        import subprocess as _sp
        _sp.Popen(['explorer', out])
    except Exception:
        pass


# ── Parallel GPU benchmark ───────────────────────────────────────────────────

_SOURCE_EXCLUDES = ['__pycache__', '*.pyc', '.git', 'node_modules',
                    'Spark', '_archive', '*.pth', '*.onnx', 'tunet_session.yaml']


def benchmark_all_gpus(config_path, api_key, steps=100, gpu_keys=None,
                       image=DEFAULT_IMAGE, disk_gb=50, volume_gb=20):
    """
    Spin up one pod per GPU type in parallel, run N training steps on each,
    collect T/Step + batch size, terminate all pods, then print a sorted table.
    """
    import threading, re as _re, copy as _copy, tempfile as _tmp

    if not gpu_keys:
        gpu_keys = list(GPU_TYPES.keys())

    key_path = find_ssh_key()

    # ── Build shared config (fresh training, max_steps=steps) ────────────────
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    data_uploads_all = {}
    pod_config_base, job_name = rewrite_config_for_pod(raw_config, data_uploads_all)

    # Benchmark always trains fresh — drop checkpoint & log uploads
    data_uploads_bench = {k: v for k, v in data_uploads_all.items()
                          if not k.endswith('.pth') and not k.endswith('.log')}

    pod_config_base.setdefault('training', {})['max_steps'] = steps
    # Remove any resume/finetune keys — benchmark always trains fresh
    pod_config_base['training'].pop('resume', None)
    pod_config_base['training'].pop('finetune_from', None)
    pod_config_base['training'].pop('resume_from', None)

    # ── Pre-pack source + data dirs once (shared across threads) ─────────────
    tunet_root = os.path.dirname(os.path.abspath(__file__))
    print('[benchmark_all] Pre-packing tunet source...')
    source_tar, _ = _pack_dir(tunet_root, exclude=_SOURCE_EXCLUDES)

    data_tars = {}          # local_dir -> tar_path
    data_files = {}         # local_file -> remote_path (scp directly)
    for lp, rp in data_uploads_bench.items():
        if os.path.isdir(lp):
            print(f'[benchmark_all] Pre-packing {os.path.basename(lp)}...')
            tar, _ = _pack_dir(lp)
            data_tars[lp] = tar
        else:
            data_files[lp] = rp

    with _tmp.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
        yaml.dump(pod_config_base, tf, default_flow_style=False, sort_keys=False)
        config_tmp = tf.name

    start_sh = os.path.join(tunet_root, 'runpod_start.sh')

    # Limit concurrent uploads to avoid saturating bandwidth
    upload_sem = threading.Semaphore(2)

    results = []
    results_lock = threading.Lock()

    # ── Per-GPU worker ────────────────────────────────────────────────────────
    def _bench_one(gpu_key):
        gpu_display = GPU_TYPES[gpu_key]
        pfx = f'[bench:{gpu_key:>12}]'
        pod_id = None

        def log(msg):
            print(f'{pfx} {msg}', flush=True)

        try:
            log(f'Creating pod ({gpu_display})...')
            pod = create_pod(api_key, f'tunet-bench-{gpu_key}', GPU_TYPES[gpu_key],
                             image, disk_gb, volume_gb)
            pod_id  = pod['id']
            cph     = pod.get('costPerHr') or 0.0
            log(f'Pod {pod_id}  (${cph:.2f}/hr)')

            pod_info, host, port = wait_for_pod(api_key, pod_id, timeout=600)
            wait_for_ssh(host, port, key_path)
            log('SSH ready — uploading...')

            with upload_sem:
                _upload_tar(source_tar, host, port, key_path, TUNET_REMOTE)
                log(f'Source → {TUNET_REMOTE}')

                remote_dirs = {os.path.dirname(rp) for rp in data_uploads_bench.values()}
                for d in remote_dirs:
                    ssh(host, port, key_path, f'mkdir -p "{d}"')

                for lp, tar in data_tars.items():
                    rp = data_uploads_bench[lp]
                    _upload_tar(tar, host, port, key_path, rp)
                    log(f'Data → {rp}')

                for lp, rp in data_files.items():
                    scp(lp, host, port, key_path, rp)
                    log(f'File → {rp}')

                scp(config_tmp, host, port, key_path, CONFIG_REMOTE)
                scp(start_sh,   host, port, key_path, '/workspace/runpod_start.sh')
                ssh(host, port, key_path, 'chmod +x /workspace/runpod_start.sh')

            output_dir = f'{OUTPUT_REMOTE}/{job_name}'
            ssh(host, port, key_path, f'mkdir -p {output_dir}')
            launch_cmd = (
                f'nohup bash /workspace/runpod_start.sh '
                f'"{output_dir}" "{CONFIG_REMOTE}" "{MONITOR_PORT}" '
                f'> /workspace/launch.log 2>&1 &'
            )
            ssh(host, port, key_path, launch_cmd)
            log(f'Training started — waiting for {steps} steps...')

            # ── Poll until max_steps reached ──────────────────────────────────
            log_path  = f'{output_dir}/training.log'
            deadline  = time.time() + 3600
            last_step = 0
            while time.time() < deadline:
                time.sleep(20)
                r = ssh(host, port, key_path,
                        f'test -f {log_path} && echo yes || echo no',
                        check=False, capture=True)
                if r.stdout.strip() != 'yes':
                    continue
                r2 = ssh(host, port, key_path,
                         f'grep -oP "Step\\[\\K[0-9]+" {log_path} | tail -1',
                         check=False, capture=True)
                step = int(r2.stdout.strip()) if r2.stdout.strip().isdigit() else 0
                if step != last_step:
                    log(f'Step {step}/{steps}')
                    last_step = step
                r3 = ssh(host, port, key_path,
                         f'grep -c "Reached max_steps" {log_path} 2>/dev/null',
                         check=False, capture=True)
                if r3.stdout.strip() not in ('', '0'):
                    log('Done!')
                    break
            else:
                log('WARNING: timed out waiting for benchmark.')

            # ── Parse T/Step from log ─────────────────────────────────────────
            r = ssh(host, port, key_path, f'cat {log_path}', check=False, capture=True)
            log_text = r.stdout

            step_times = [float(m) for m in _re.findall(r'T/Step:([\d.]+)s', log_text)]
            batch_size = 1
            bm = _re.search(r'Auto batch size resolved to (\d+)', log_text)
            if bm:
                batch_size = int(bm.group(1))

            gr = ssh(host, port, key_path,
                     'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1',
                     check=False, capture=True)
            gpu_name = gr.stdout.strip() or gpu_display

            if step_times:
                warmup   = max(5, len(step_times) // 5)
                measured = step_times[warmup:]
                avg      = sum(measured) / len(measured)
                imgs_hr  = int(batch_size * 3600 / avg)
                c1k      = cph * 3600 / imgs_hr * 1000 if imgs_hr else 0
                with results_lock:
                    results.append({
                        'gpu_key': gpu_key, 'gpu_name': gpu_name,
                        'batch_size': batch_size, 'avg_step_s': avg,
                        'images_per_hr': imgs_hr, 'cost_per_hr': cph, 'cost_per_1k': c1k,
                    })
                log(f'batch={batch_size}  T/Step={avg:.3f}s  {imgs_hr:,} img/hr  ${c1k:.4f}/1k')
            else:
                log('Could not parse T/Step values.')

        except Exception as e:
            log(f'ERROR: {e}')
        finally:
            if pod_id:
                try:
                    terminate_pod(api_key, pod_id)
                    log(f'Pod {pod_id} terminated.')
                except Exception as te:
                    log(f'Failed to terminate pod {pod_id}: {te}')

    # ── Launch all threads ────────────────────────────────────────────────────
    print(f'\n[benchmark_all] {len(gpu_keys)} GPUs × {steps} steps  (deps install ~5-10 min each)')
    print(f'[benchmark_all] GPUs: {", ".join(gpu_keys)}\n')

    threads = [threading.Thread(target=_bench_one, args=(gk,), daemon=True) for gk in gpu_keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ── Cleanup shared temp files ─────────────────────────────────────────────
    for p in [source_tar, config_tmp] + list(data_tars.values()):
        try:
            os.unlink(p)
        except Exception:
            pass

    # ── Print results table ───────────────────────────────────────────────────
    if not results:
        print('[benchmark_all] No results collected.')
        return

    results.sort(key=lambda x: -x['images_per_hr'])
    best = results[0]['images_per_hr']

    W = 72
    print(f'\n{"═"*W}')
    print(f'  BENCHMARK RESULTS — {steps} steps, sorted by throughput')
    print(f'{"═"*W}')
    print(f'  {"GPU":<36} {"Batch":>5} {"T/Step":>8} {"Img/hr":>9} {"$/hr":>6} {"$/1k img":>9} {"vs best":>8}')
    print(f'  {"─"*36} {"─"*5} {"─"*8} {"─"*9} {"─"*6} {"─"*9} {"─"*8}')
    for r in results:
        ratio = r['images_per_hr'] / best * 100
        print(f'  {r["gpu_name"]:<36} {r["batch_size"]:>5} {r["avg_step_s"]:>7.3f}s '
              f'{r["images_per_hr"]:>9,} {r["cost_per_hr"]:>6.2f} '
              f'{r["cost_per_1k"]:>8.4f}  {ratio:>6.0f}%')
    print(f'{"═"*W}\n')

    _save_benchmark_chart(results, steps)


# ── Benchmark helper ─────────────────────────────────────────────────────────

def _run_benchmark(host, port, key_path, output_dir, pod_id, api_key, n_steps, cost_per_hr=0.0):
    import re as _re

    log_path = f'{output_dir}/training.log'
    print(f'\n[benchmark] Waiting for {n_steps} steps to complete (deps install first, ~5-10 min)...')

    deadline = time.time() + 3600
    last_step = 0

    while time.time() < deadline:
        time.sleep(20)

        # Check if training log exists yet
        r = ssh(host, port, key_path, f'test -f {log_path} && echo yes || echo no',
                check=False, capture=True)
        if r.stdout.strip() != 'yes':
            print('[benchmark] Waiting for training to start...', flush=True)
            continue

        # Get latest step count
        r = ssh(host, port, key_path,
                f'grep -oP "Step\\[\\K[0-9]+" {log_path} | tail -1',
                check=False, capture=True)
        step = int(r.stdout.strip()) if r.stdout.strip().isdigit() else 0
        if step != last_step:
            print(f'[benchmark] Step {step}/{n_steps}...', flush=True)
            last_step = step

        # Check for completion
        r = ssh(host, port, key_path,
                f'grep -c "Reached max_steps" {log_path} 2>/dev/null',
                check=False, capture=True)
        if r.stdout.strip() not in ('', '0'):
            print('[benchmark] Training complete.')
            break
    else:
        print('[benchmark] WARNING: timed out waiting for benchmark to finish.')

    # ── Parse results ────────────────────────────────────────
    r = ssh(host, port, key_path, f'cat {log_path}', check=False, capture=True)
    log_text = r.stdout

    step_times = [float(m) for m in _re.findall(r'T/Step:([\d.]+)s', log_text)]

    # Parse batch size from auto-batch log line
    batch_size = 1
    bm = _re.search(r'Auto batch size resolved to (\d+)', log_text)
    if bm:
        batch_size = int(bm.group(1))
    else:
        bm2 = _re.search(r'batch_size["\s:=]+(\d+)', log_text)
        if bm2:
            batch_size = int(bm2.group(1))

    # Parse GPU name
    gpu_name = 'unknown'
    gm = ssh(host, port, key_path,
             'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1',
             check=False, capture=True)
    if gm.stdout.strip():
        gpu_name = gm.stdout.strip()

    # Skip first 20% as warmup
    if step_times:
        warmup = max(5, len(step_times) // 5)
        measured = step_times[warmup:]
        avg = sum(measured) / len(measured)
        images_per_hr = int(batch_size * 3600 / avg)

        print(f'\n{"─"*52}')
        print(f'  BENCHMARK RESULTS — {gpu_name}')
        print(f'{"─"*52}')
        print(f'  Steps measured : {len(measured)}  (skipped {warmup} warmup)')
        print(f'  Batch size     : {batch_size}')
        print(f'  T/Step avg     : {avg:.3f}s')
        print(f'  T/Step min     : {min(measured):.3f}s')
        print(f'  T/Step max     : {max(measured):.3f}s')
        print(f'  Images/hour    : {images_per_hr:,}')
        cost_per_1k = cost_per_hr * 3600 / images_per_hr * 1000 if images_per_hr else 0
        print(f'  Cost           : ${cost_per_hr:.2f}/hr  →  ${cost_per_1k:.4f} per 1k images')
        print(f'{"─"*52}\n')
    else:
        print('[benchmark] Could not parse T/Step values from log.')

    # ── Terminate pod ────────────────────────────────────────
    print(f'[benchmark] Terminating pod {pod_id}...')
    terminate_pod(api_key, pod_id)
    print('[benchmark] Done.')


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
    ap.add_argument('--benchmark', type=int, nargs='?', const=100, default=0,
                    metavar='STEPS',
                    help='Benchmark mode: run N steps (default 100), print T/Step stats, then terminate pod')
    args = ap.parse_args()

    if not args.api_key:
        sys.exit('ERROR: Set RUNPOD_API_KEY env var or pass --api_key')

    # Load and validate config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_uploads = {}
    pod_config, job_name = rewrite_config_for_pod(config, data_uploads)

    if args.benchmark:
        if 'training' not in pod_config:
            pod_config['training'] = {}
        pod_config['training']['max_steps'] = args.benchmark
        print(f'[benchmark] Mode: will run {args.benchmark} steps, report throughput, then terminate.')

    pod_name = args.name or (f'tunet-bench-{job_name}' if args.benchmark else f'tunet-{job_name}')
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

        if args.benchmark:
            _run_benchmark(host, port, key_path, output_dir, pod_id, args.api_key, args.benchmark,
                           cost_per_hr=pod.get('costPerHr') or 0.0)
        else:
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
      python runpod_launch.py --benchmark_all --config CONFIG [--steps N] [--gpus a100,l40s,...]
    """
    # ── benchmark_all sub-command ─────────────────────────────────────────────
    if '--benchmark_all' in sys.argv:
        config_path = None
        steps       = 100
        gpu_keys    = None
        api_key     = None
        image       = DEFAULT_IMAGE
        for i, a in enumerate(sys.argv):
            if a == '--config'  and i + 1 < len(sys.argv): config_path = sys.argv[i + 1]
            if a == '--steps'   and i + 1 < len(sys.argv): steps       = int(sys.argv[i + 1])
            if a == '--gpus'    and i + 1 < len(sys.argv): gpu_keys    = sys.argv[i + 1].split(',')
            if a == '--api_key' and i + 1 < len(sys.argv): api_key     = sys.argv[i + 1]
            if a == '--image'   and i + 1 < len(sys.argv): image       = sys.argv[i + 1]
        if not config_path:
            sys.exit('ERROR: --benchmark_all requires --config CONFIG_PATH')
        if not api_key:
            api_key = os.environ.get('RUNPOD_API_KEY') or os.environ.get('runpodapi')
        if not api_key:
            sys.exit('ERROR: Set RUNPOD_API_KEY or pass --api_key')
        if gpu_keys:
            bad = [g for g in gpu_keys if g not in GPU_TYPES]
            if bad:
                sys.exit(f'ERROR: Unknown GPU keys: {bad}. Valid: {list(GPU_TYPES.keys())}')
        benchmark_all_gpus(config_path, api_key, steps=steps, gpu_keys=gpu_keys, image=image)
        return True

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
