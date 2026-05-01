/**
 * Server-side tarball packer for Spark Compute v1 jobs.
 *
 * Walt's spec (auto-prepare mode): pack tunet/ source + config.yaml +
 * spark_start.sh into a tar.gz, PUT to the uploadUrl returned by submit.
 * The agent extracts to /input/ inside the container.
 *
 * Layout inside the tarball:
 *     tunet/...                — full tunet source (with realistic excludes)
 *     config.yaml              — synthesized for this job
 *     spark_start.sh           — bootstrap script that runs train.py
 *     [output/<job>/...]       — staged resume checkpoints (when present)
 *
 * Source tree is read from TUNET_REPO_ROOT (env) or auto-detected by walking
 * up from cwd looking for `train.py` + `spark_start.sh`.
 *
 * Size budget (verified empirically): ~800 KB compressed for the full repo
 * minus tests/data/checkpoints. Re-uploaded on every submit; not optimized.
 */

import 'server-only'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'
import * as tar from 'tar'
import * as zlib from 'node:zlib'
import { stringify as yamlStringify } from 'yaml'

// ── Tunet repo location ──────────────────────────────────────────────────────

const REPO_ENV = 'TUNET_REPO_ROOT'

/** Find the tunet repo root by env var or by walking up from this module. */
export function findTunetRepoRoot(): string {
  const envPath = process.env[REPO_ENV]
  if (envPath && fs.existsSync(path.join(envPath, 'train.py'))) {
    return envPath
  }
  // tunet-web sits inside tunet/; walk up two levels from this file
  // (src/lib/spark-packer.ts → src/lib/ → src/ → tunet-web/ → tunet/)
  let dir = path.resolve(process.cwd())
  for (let i = 0; i < 6; i++) {
    if (
      fs.existsSync(path.join(dir, 'train.py')) &&
      fs.existsSync(path.join(dir, 'spark_start.sh'))
    ) {
      return dir
    }
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  // Last resort: assume tunet-web/ is at $REPO/tunet-web
  return path.resolve(process.cwd(), '..')
}

// ── Excludes ─────────────────────────────────────────────────────────────────

// Mirrors the SOURCE_EXCLUDES list used by spark_launch.py / runpod_launch.py.
// Glob-style: a path part matches if any segment of the relative path matches
// any pattern. This intentionally uses simple substring/extension rules rather
// than full fnmatch — tar supports filter functions.
const EXCLUDE_PATTERNS = [
  // Build/cache directories
  '__pycache__', '.git', '.venv', 'node_modules', '.pytest_cache',
  '_archive', '_inference_cache', '_internal',
  // Sub-projects we don't ship
  'Spark', 'tunet-web', 'docs',
  // Local data / outputs (could be huge)
  'output', 'finetuned_outputs', 'data', 'src', 'dst',
  'inputs', 'outputs', 'paintout-test-A', 'paintout-test-B',
  'finetune-test', 'finetune-test-1024',
  // Sessions & local state
  'tunet_session.yaml', 'spark_dashboard_settings.json',
  '_spark_panel.json',
]

const EXCLUDE_EXTS = new Set([
  '.pth', '.onnx', '.pyc', '.log', '.tar.gz', '.zip',
  '.exr', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
])

const EXCLUDE_FILES = new Set([
  'benchmark_chart.png',
])

function isExcluded(relPath: string): boolean {
  const norm = relPath.replace(/\\/g, '/')
  const parts = norm.split('/')
  for (const part of parts) {
    if (EXCLUDE_PATTERNS.includes(part)) return true
  }
  const base = parts[parts.length - 1]
  if (EXCLUDE_FILES.has(base)) return true
  const ext = path.extname(base).toLowerCase()
  if (EXCLUDE_EXTS.has(ext)) return true
  return false
}

// ── Pack ─────────────────────────────────────────────────────────────────────

export interface PackInput {
  /** Absolute path to tunet repo root. Defaults to findTunetRepoRoot(). */
  tunetRoot?: string
  /** Synthesized config object (will be YAML-dumped at the tarball root). */
  config:    Record<string, unknown>
  /** Optional: name of the spark_start.sh inside the tarball. Defaults to 'spark_start.sh'. */
  startScriptName?: string
  /**
   * Optional: stage directory containing user-uploaded data folders.
   * If present, its subdirs (src/, dst/, val_src/, val_dst/, mask/) are
   * copied into the tarball as `data/src`, `data/dst`, etc. The tarball
   * extracts to `/input/`, so the agent will see them at `/input/data/...`.
   */
  stageDir?: string
}

export interface PackResult {
  /** Compressed tar.gz buffer ready to PUT to the Spark uploadUrl. */
  buffer:        Buffer
  /** Compressed size (bytes). */
  compressedSize: number
  /** Number of files in the tarball. */
  fileCount:     number
}

/**
 * Pack the tunet source, config, and start script into a tar.gz buffer.
 *
 * Strategy: write the staged files (tunet/, config.yaml, spark_start.sh)
 * into a temp directory, then call `tar.create` on it. tar handles the
 * gzip level and streams to a Buffer concat. On Windows this is fast
 * because we're working off SSD and the tunet repo is <4MB raw.
 */
export async function packInputTarball(input: PackInput): Promise<PackResult> {
  const tunetRoot = input.tunetRoot ?? findTunetRepoRoot()
  if (!fs.existsSync(path.join(tunetRoot, 'train.py'))) {
    throw new Error(`tunet repo not found at ${tunetRoot} (no train.py). Set ${REPO_ENV} env var.`)
  }
  const startScriptSrc = path.join(tunetRoot, 'spark_start.sh')
  if (!fs.existsSync(startScriptSrc)) {
    throw new Error(`spark_start.sh not found at ${startScriptSrc}`)
  }

  // Stage everything into a temp directory so tar.create can pack it as a
  // single tree. This is ~800 KB raw — copy cost is negligible.
  const stamp   = Date.now().toString(36)
  const stageDir = path.join(os.tmpdir(), `tunet-spark-pack-${stamp}-${process.pid}`)
  await fs.promises.mkdir(stageDir, { recursive: true })

  let fileCount = 0

  try {
    // 1. Mirror tunet source under stageDir/tunet/
    const tunetStage = path.join(stageDir, 'tunet')
    await fs.promises.mkdir(tunetStage, { recursive: true })
    fileCount += await mirrorDir(tunetRoot, tunetStage, isExcluded)

    // 2. Write config.yaml at the root
    const configPath = path.join(stageDir, 'config.yaml')
    await fs.promises.writeFile(
      configPath,
      yamlStringify(input.config, { indent: 2 }),
      'utf8',
    )
    fileCount += 1

    // 3. Copy spark_start.sh at the root
    const startDest = path.join(stageDir, input.startScriptName ?? 'spark_start.sh')
    await fs.promises.copyFile(startScriptSrc, startDest)
    fileCount += 1

    // 3b. Bundle staged data if provided. Copy each role dir to data/<role>/
    if (input.stageDir && fs.existsSync(input.stageDir)) {
      const dataRoot = path.join(stageDir, 'data')
      await fs.promises.mkdir(dataRoot, { recursive: true })
      const roleDirs = await fs.promises.readdir(input.stageDir, { withFileTypes: true })
      for (const r of roleDirs) {
        if (!r.isDirectory()) continue
        const srcRole = path.join(input.stageDir, r.name)
        const dstRole = path.join(dataRoot, r.name)
        await fs.promises.mkdir(dstRole, { recursive: true })
        // No exclude logic for user data — copy as-is
        const copied = await mirrorDir(srcRole, dstRole, () => false)
        fileCount += copied
      }
    }

    // 4. tar -czf - .
    // tar.create accepts a string of patterns relative to `cwd`. Listing
    // them explicitly keeps the archive tidy (no leading "./").
    const entries = await fs.promises.readdir(stageDir)

    const chunks: Buffer[] = []
    await new Promise<void>((resolve, reject) => {
      const stream = tar.create(
        {
          gzip: { level: 6 },
          cwd:  stageDir,
          // Don't follow symlinks; preserve mtimes for cache friendliness
          follow: false,
          portable: true,
        },
        entries,
      )
      stream.on('data', (chunk: Buffer) => chunks.push(chunk))
      stream.on('end', () => resolve())
      stream.on('error', (err: unknown) => reject(err))
    })

    const buffer = Buffer.concat(chunks)
    return { buffer, compressedSize: buffer.length, fileCount }
  } finally {
    // Cleanup
    await fs.promises.rm(stageDir, { recursive: true, force: true }).catch(() => {})
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Recursively mirror `src` to `dst`, skipping anything that returns true
 * from `excluded`. Returns number of files copied.
 *
 * Uses copyFile (not symlinks) so the tarball is self-contained even if
 * stageDir gets archived via a follow-up step that doesn't dereference.
 */
async function mirrorDir(
  src:      string,
  dst:      string,
  excluded: (relPath: string) => boolean,
  rel:      string = '',
): Promise<number> {
  let count = 0
  const entries = await fs.promises.readdir(src, { withFileTypes: true })
  for (const e of entries) {
    const childRel = rel ? `${rel}/${e.name}` : e.name
    if (excluded(childRel)) continue

    const srcPath = path.join(src, e.name)
    const dstPath = path.join(dst, e.name)

    if (e.isDirectory()) {
      await fs.promises.mkdir(dstPath, { recursive: true })
      count += await mirrorDir(srcPath, dstPath, excluded, childRel)
    } else if (e.isFile()) {
      await fs.promises.copyFile(srcPath, dstPath)
      count += 1
    }
    // skip symlinks, sockets, etc.
  }
  return count
}

/** Diagnostic: pack and return size info without actually shipping anything. */
export async function dryRunPackSize(config: Record<string, unknown>): Promise<{
  fileCount: number
  compressedKB: number
  rawKB: number
}> {
  const result = await packInputTarball({ config })
  // Estimate raw size by inflating
  const raw = zlib.gunzipSync(result.buffer)
  return {
    fileCount:    result.fileCount,
    compressedKB: Math.round(result.buffer.length / 1024),
    rawKB:        Math.round(raw.length / 1024),
  }
}
