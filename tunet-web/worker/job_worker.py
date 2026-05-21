"""
job_worker.py — SSH-based fallback worker for Spark Flint

This worker is ONLY needed if RunPod's startScript API field is unavailable.
It polls the Supabase jobs table for pending jobs and provisions them via SSH
using the existing runpod_launch.py infrastructure.

Run this on any machine that has:
  - SSH access (the ~/.ssh/runpod key)
  - Network access to Supabase and RunPod
  - The tunet repo available locally (for rsync upload)

Usage:
    pip install supabase requests
    python worker/job_worker.py

Environment variables:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    RUNPOD_API_KEY
    TUNET_REPO_DIR  — local path to the tunet repo (default: parent of this file)
"""

import os
import sys
import time
import json
import subprocess
import tempfile
import threading
from pathlib import Path

try:
    from supabase import create_client, Client
except ImportError:
    sys.exit("pip install supabase")

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ── Config ────────────────────────────────────────────────────────────────────

SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
RUNPOD_API_KEY  = os.environ["RUNPOD_API_KEY"]
TUNET_REPO_DIR  = os.environ.get("TUNET_REPO_DIR") or str(Path(__file__).parent.parent)
SSH_KEY_PATH    = os.path.expanduser("~/.ssh/runpod")
POLL_INTERVAL   = 15  # seconds

supa: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── RunPod API ────────────────────────────────────────────────────────────────

RUNPOD_API = "https://api.runpod.io/graphql"

def runpod_gql(query, variables=None):
    resp = requests.post(
        RUNPOD_API,
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"]))
    return data["data"]

def create_pod(name, gpu_type_id, container_disk_gb, volume_gb):
    q = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) { id name costPerHr machineId }
    }
    """
    data = runpod_gql(q, {"input": {
        "name":              f"tunet-{name}",
        "gpuTypeId":         gpu_type_id,
        "imageName":         "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuCount":          1,
        "containerDiskInGb": container_disk_gb,
        "volumeInGb":        volume_gb,
        "volumeMountPath":   "/workspace",
        "startSsh":          True,
        "ports":             "22/tcp,8080/http",
        "cloudType":         "SECURE",
        "env":               [{"key": "RUNPOD_API_KEY", "value": RUNPOD_API_KEY}],
    }})
    return data["podFindAndDeployOnDemand"]

def get_pod(pod_id):
    q = """
    query GetPod($input: PodFilter) {
      pod(input: $input) {
        id desiredStatus costPerHr
        runtime { ports { ip isIpPublic privatePort publicPort type } }
        machine { gpuDisplayName }
      }
    }
    """
    return runpod_gql(q, {"input": {"podId": pod_id}})["pod"]

def wait_for_ssh(pod_id, timeout=600):
    print(f"  [wait] Waiting for SSH on pod {pod_id}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = get_pod(pod_id)
        if pod and pod.get("desiredStatus") == "RUNNING":
            for p in (pod.get("runtime") or {}).get("ports") or []:
                if p.get("privatePort") == 22 and p.get("isIpPublic"):
                    host, port = p["ip"], p["publicPort"]
                    # Check actual SSH connectivity
                    r = subprocess.run(
                        ["ssh", "-n", "-T",
                         "-o", "StrictHostKeyChecking=no",
                         "-o", "BatchMode=yes",
                         "-o", "ConnectTimeout=5",
                         "-i", SSH_KEY_PATH,
                         "-p", str(port), f"root@{host}", "echo ok"],
                        capture_output=True, text=True
                    )
                    if r.returncode == 0:
                        print(f"  [wait] SSH ready at {host}:{port}")
                        return pod, host, port
        print("  [wait] .", end="", flush=True)
        time.sleep(10)
    raise TimeoutError("Pod SSH never became available")

def ssh_run(host, port, cmd, check=True):
    return subprocess.run(
        ["ssh", "-n", "-T",
         "-o", "StrictHostKeyChecking=no",
         "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=15",
         "-i", SSH_KEY_PATH,
         "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True,
        timeout=300, check=check,
    )

# ── Supabase storage download ─────────────────────────────────────────────────

def download_file(storage_path: str, local_path: str):
    """Download a file from Supabase Storage to a local path."""
    res = supa.storage.from_("tunet-jobs").download(storage_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(res)

# ── Job processor ─────────────────────────────────────────────────────────────

def process_job(job: dict):
    job_id   = job["id"]
    name     = job["name"]
    gpu_type = job["gpu_type_id"]
    cdisk    = job["container_disk_gb"]
    vgb      = job["volume_gb"]

    print(f"\n[worker] Processing job {job_id} ({name})")

    try:
        # 1. Create pod
        print(f"  [pod] Creating pod on {gpu_type}...")
        pod_meta = create_pod(name, gpu_type, cdisk, vgb)
        pod_id   = pod_meta["id"]
        print(f"  [pod] Pod ID: {pod_id}")

        supa.table("jobs").update({
            "pod_id":     pod_id,
            "status":     "provisioning",
        }).eq("id", job_id).execute()

        # 2. Wait for SSH
        pod, host, port = wait_for_ssh(pod_id)

        supa.table("jobs").update({
            "runpod_cost_per_hr": pod.get("costPerHr"),
            "gpu_display_name":   pod.get("machine", {}).get("gpuDisplayName"),
        }).eq("id", job_id).execute()

        # 3. Upload tunet code
        print(f"  [upload] Uploading tunet repo...")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        import tarfile, fnmatch
        with tarfile.open(tmp_path, "w:gz") as tf:
            EXCLUDE = ["__pycache__", "*.pyc", ".git", "node_modules", ".env", "*.pth"]
            for root, dirs, files in os.walk(TUNET_REPO_DIR):
                dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in EXCLUDE)]
                for fname in files:
                    if any(fnmatch.fnmatch(fname, p) for p in EXCLUDE):
                        continue
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, TUNET_REPO_DIR).replace("\\", "/")
                    tf.add(abs_path, arcname=rel_path)

        # SCP the tarball
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=15", "-i", SSH_KEY_PATH,
             "-P", str(port), tmp_path, f"root@{host}:/tmp/tunet.tar.gz"],
            check=True, capture_output=True
        )
        os.unlink(tmp_path)
        ssh_run(host, port,
                "mkdir -p /workspace/tunet && tar -xzf /tmp/tunet.tar.gz -C /workspace/tunet && rm /tmp/tunet.tar.gz")
        print(f"  [upload] Code uploaded")

        # 4. Download config from Supabase, upload to pod
        if job.get("config_path"):
            print(f"  [upload] Uploading config...")
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                tmp_cfg = tmp.name
            download_file(job["config_path"], tmp_cfg)
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                 "-i", SSH_KEY_PATH, "-P", str(port), tmp_cfg, f"root@{host}:/workspace/config.yaml"],
                check=True, capture_output=True
            )
            os.unlink(tmp_cfg)

        # 5. Download and extract src/dst zips if provided
        for role, remote_dir in [("src_zip_path", "/workspace/data/src"), ("dst_zip_path", "/workspace/data/dst")]:
            if not job.get(role):
                continue
            print(f"  [upload] Uploading {role}...")
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_zip = tmp.name
            download_file(job[role], tmp_zip)
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                 "-i", SSH_KEY_PATH, "-P", str(port), tmp_zip, f"root@{host}:/tmp/data.zip"],
                check=True, capture_output=True
            )
            os.unlink(tmp_zip)
            ssh_run(host, port, f"mkdir -p {remote_dir} && unzip -q /tmp/data.zip -d {remote_dir} && rm /tmp/data.zip")

        # 6. Download checkpoint if provided
        if job.get("checkpoint_path"):
            print(f"  [upload] Uploading checkpoint...")
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp_pth = tmp.name
            download_file(job["checkpoint_path"], tmp_pth)
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                 "-i", SSH_KEY_PATH, "-P", str(port), tmp_pth, f"root@{host}:/workspace/uploads/checkpoint.pth"],
                check=True, capture_output=True
            )
            os.unlink(tmp_pth)

        # 7. Patch config for pod paths
        patch = """python3 - <<'PYEOF'
import yaml, os
with open('/workspace/config.yaml') as f:
    cfg = yaml.safe_load(f) or {}
if 'data' not in cfg: cfg['data'] = {}
cfg['data']['output_dir'] = '/workspace/output/job'
if os.path.isdir('/workspace/data/src'): cfg['data']['src_dir'] = '/workspace/data/src'
if os.path.isdir('/workspace/data/dst'): cfg['data']['dst_dir'] = '/workspace/data/dst'
if os.path.isfile('/workspace/uploads/checkpoint.pth'):
    cfg.setdefault('training', {})['finetune_from'] = '/workspace/uploads/checkpoint.pth'
with open('/workspace/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Config patched')
PYEOF"""
        ssh_run(host, port, "pip install -q pyyaml && " + patch)

        # 8. Launch training in background (nohup)
        print(f"  [train] Launching training...")
        ssh_run(host, port,
                "nohup bash /workspace/tunet/runpod_start.sh "
                "/workspace/output/job /workspace/config.yaml 8080 "
                "> /workspace/bootstrap.log 2>&1 &")

        # 9. Update DB to running
        supa.table("jobs").update({
            "status":               "running",
            "started_at":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "billing_last_tick_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }).eq("id", job_id).execute()

        print(f"  [done] Job {job_id} running on pod {pod_id}")

    except Exception as e:
        print(f"  [error] Job {job_id} failed: {e}", file=sys.stderr)
        supa.table("jobs").update({
            "status": "failed",
        }).eq("id", job_id).execute()

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f"[worker] Spark Flint job worker starting...")
    print(f"[worker] Polling every {POLL_INTERVAL}s for pending jobs")
    print(f"[worker] Supabase: {SUPABASE_URL}")
    print(f"[worker] SSH key:  {SSH_KEY_PATH}")
    print()

    active_jobs: set[str] = set()

    while True:
        try:
            result = supa.table("jobs") \
                .select("*") \
                .eq("status", "pending") \
                .execute()

            for job in result.data or []:
                if job["id"] in active_jobs:
                    continue
                active_jobs.add(job["id"])
                t = threading.Thread(
                    target=lambda j=job: (process_job(j), active_jobs.discard(j["id"])),
                    daemon=True,
                )
                t.start()

        except Exception as e:
            print(f"[worker] Poll error: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
