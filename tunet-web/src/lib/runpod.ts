import type { RunPodPod } from '@/types'

const RUNPOD_API = 'https://api.runpod.io/graphql'
const MONITOR_PORT = 8080

function apiKey(): string {
  const key = process.env.RUNPOD_API_KEY
  if (!key) throw new Error('RUNPOD_API_KEY not set')
  return key
}

async function gql<T = unknown>(query: string, variables?: Record<string, unknown>): Promise<T> {
  const res = await fetch(RUNPOD_API, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey()}`,
    },
    body: JSON.stringify({ query, variables: variables ?? {} }),
    next: { revalidate: 0 },
  })
  if (!res.ok) throw new Error(`RunPod API HTTP ${res.status}`)
  const json = await res.json()
  if (json.errors) throw new Error(JSON.stringify(json.errors))
  return json.data as T
}

// ── Queries ───────────────────────────────────────────────────────────────────

export async function listPods(): Promise<RunPodPod[]> {
  const data = await gql<{ myself: { pods: RunPodPod[] } }>(`
    query {
      myself {
        pods {
          id name desiredStatus costPerHr gpuCount imageName
          uptimeSeconds lastStartedAt
          runtime {
            ports { ip isIpPublic privatePort publicPort type }
            gpus  { id gpuUtilPercent }
          }
          machine { gpuDisplayName }
        }
      }
    }
  `)
  return data.myself.pods
}

export async function getPod(podId: string): Promise<RunPodPod | null> {
  const data = await gql<{ pod: RunPodPod | null }>(`
    query GetPod($input: PodFilter) {
      pod(input: $input) {
        id name desiredStatus costPerHr gpuCount imageName
        uptimeSeconds lastStartedAt
        runtime {
          ports { ip isIpPublic privatePort publicPort type }
          gpus  { id gpuUtilPercent }
        }
        machine { gpuDisplayName }
      }
    }
  `, { input: { podId } })
  return data.pod
}

// ── Mutations ─────────────────────────────────────────────────────────────────

export interface CreatePodInput {
  name: string
  gpuTypeId: string
  startScript: string     // base64-encoded bash
  containerDiskGb: number
  volumeGb: number
  env?: Record<string, string>
}

export interface CreatedPod {
  id: string
  name: string
  costPerHr: number
  machineId: string
}

const DEFAULT_IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'

export async function createPod(input: CreatePodInput): Promise<CreatedPod> {
  const envArray = Object.entries(input.env ?? {}).map(([key, value]) => ({ key, value }))

  const data = await gql<{ podFindAndDeployOnDemand: CreatedPod }>(`
    mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) {
        id name costPerHr machineId
      }
    }
  `, {
    input: {
      name:              input.name,
      gpuTypeId:         input.gpuTypeId,
      imageName:         DEFAULT_IMAGE,
      gpuCount:          1,
      containerDiskInGb: input.containerDiskGb,
      volumeInGb:        input.volumeGb,
      volumeMountPath:   '/workspace',
      startSsh:          false,
      ports:             `${MONITOR_PORT}/http`,
      startScript:       input.startScript,
      cloudType:         'SECURE',
      env:               envArray,
    },
  })
  return data.podFindAndDeployOnDemand
}

export async function stopPod(podId: string): Promise<void> {
  await gql(`
    mutation StopPod($input: PodStopInput!) {
      podStop(input: $input) { id desiredStatus }
    }
  `, { input: { podId } })
}

export async function terminatePod(podId: string): Promise<void> {
  await gql(`
    mutation TerminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
  `, { input: { podId } })
}

// ── Monitor proxy ─────────────────────────────────────────────────────────────

export function monitorBaseUrl(podId: string): string {
  return `https://${podId}-${MONITOR_PORT}.proxy.runpod.net`
}

export async function proxyMonitor<T = unknown>(
  podId: string,
  path: string,
  init?: RequestInit,
): Promise<{ data: T | null; error: string | null }> {
  const url = `${monitorBaseUrl(podId)}${path}`
  try {
    const res = await fetch(url, {
      ...init,
      signal: AbortSignal.timeout(8000),
      next: { revalidate: 0 },
    })
    if (!res.ok) return { data: null, error: `Monitor HTTP ${res.status}` }
    const data = await res.json() as T
    return { data, error: null }
  } catch (e) {
    return { data: null, error: e instanceof Error ? e.message : 'Monitor unreachable' }
  }
}

// ── Bootstrap script builder ──────────────────────────────────────────────────

export interface BootstrapConfig {
  codeUrl: string
  configUrl: string
  srcZipUrl?: string
  dstZipUrl?: string
  checkpointUrl?: string
  runpodApiKey: string
}

export function buildBootstrapScript(cfg: BootstrapConfig): string {
  const script = `#!/bin/bash
set -e
echo "=== Spark Flint Bootstrap ==="

# Install bootstrap deps
pip install -q pyyaml

# Download tunet code
echo "[bootstrap] Downloading tunet code..."
mkdir -p /workspace/tunet
wget -q -O /tmp/tunet.tar.gz "${cfg.codeUrl}"
tar -xzf /tmp/tunet.tar.gz -C /workspace/tunet --strip-components=1
rm /tmp/tunet.tar.gz
echo "[bootstrap] Code extracted"

# Download config
echo "[bootstrap] Downloading config..."
wget -q -O /workspace/config.yaml "${cfg.configUrl}"

${cfg.srcZipUrl ? `# Download src training data
echo "[bootstrap] Downloading src data..."
mkdir -p /workspace/data/src
wget -q -O /tmp/src.zip "${cfg.srcZipUrl}"
unzip -q /tmp/src.zip -d /workspace/data/src
rm /tmp/src.zip` : '# No src data provided (paths must be in config)'}

${cfg.dstZipUrl ? `# Download dst training data
echo "[bootstrap] Downloading dst data..."
mkdir -p /workspace/data/dst
wget -q -O /tmp/dst.zip "${cfg.dstZipUrl}"
unzip -q /tmp/dst.zip -d /workspace/data/dst
rm /tmp/dst.zip` : '# No dst data provided (paths must be in config)'}

${cfg.checkpointUrl ? `# Download checkpoint
echo "[bootstrap] Downloading checkpoint..."
mkdir -p /workspace/uploads
wget -q -O /workspace/uploads/checkpoint.pth "${cfg.checkpointUrl}"
CHECKPOINT_PATH="/workspace/uploads/checkpoint.pth"` : 'CHECKPOINT_PATH=""'}

# Patch config paths
python3 - <<'PYEOF'
import yaml, os

with open('/workspace/config.yaml') as f:
    cfg = yaml.safe_load(f) or {}

if 'data' not in cfg:
    cfg['data'] = {}

cfg['data']['output_dir'] = '/workspace/output/job'

if os.path.isdir('/workspace/data/src'):
    cfg['data']['src_dir'] = '/workspace/data/src'
if os.path.isdir('/workspace/data/dst'):
    cfg['data']['dst_dir'] = '/workspace/data/dst'

ckpt = os.environ.get('CHECKPOINT_PATH', '').strip()
if ckpt and os.path.isfile(ckpt):
    cfg.setdefault('training', {})['finetune_from'] = ckpt

with open('/workspace/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print('[bootstrap] Config patched:')
for k in ('src_dir','dst_dir','output_dir'):
    print(f"  {k}: {cfg['data'].get(k,'—')}")
PYEOF

export CHECKPOINT_PATH="$CHECKPOINT_PATH"
export RUNPOD_API_KEY="${cfg.runpodApiKey}"

OUTPUT_DIR=$(python3 -c "
import yaml
with open('/workspace/config.yaml') as f:
    c = yaml.safe_load(f)
print(c.get('data',{}).get('output_dir','/workspace/output/job'))
")

echo "[bootstrap] Launching training, output_dir=$OUTPUT_DIR"
bash /workspace/tunet/runpod_start.sh "$OUTPUT_DIR" /workspace/config.yaml 8080
`
  return Buffer.from(script).toString('base64')
}
