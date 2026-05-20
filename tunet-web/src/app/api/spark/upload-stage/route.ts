/**
 * /api/spark/upload-stage
 *
 *   POST  — receive a single batch of files for a stage and write them to disk
 *           Form fields:
 *             stageId    (string)  — opaque ID returned from the first POST; pass
 *                                     to subsequent batches and finally to /training-jobs
 *             role       (string)  — 'src' | 'dst' | 'val_src' | 'val_dst' | 'mask'
 *             files      (file[])  — one or more files. webkitRelativePath is
 *                                     ignored; only the basename is preserved.
 *
 *   GET ?stageId=...  — return the current state of a stage (file count + size by role)
 *
 *   DELETE ?stageId=...  — cleanup
 *
 * Why batched? Browser fetch+FormData has practical limits around 1-2 GB
 * per request. Splitting by role and chunking large folders into several
 * POSTs lets the UI show progress and recover from individual failures.
 *
 * Stages live under <os.tmpdir>/tunet-stages/<stageId>/{src,dst,val_src,val_dst,mask}/.
 * They're cleaned up after submit (success or failure) and on a daily cron
 * (TODO; for dev, just rely on tmpdir's own eviction).
 */

import { NextResponse } from 'next/server'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'
import * as crypto from 'node:crypto'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
// Allow large upload bodies. The default 4 MB cap is too small even for a
// modest batch of EXRs.
export const maxDuration = 300

// 'checkpoint' is for local-resume uploads: the user trained off-Spark
// (on their workstation) and wants to continue training on Spark using
// the .pth they have on disk. Single file, not batched. training-jobs
// reads it from <stageDir>/checkpoint/<filename> and threads it into the
// extraFiles bundle just like a Spark-source resume would.
const ALLOWED_ROLES = new Set(['src', 'dst', 'val_src', 'val_dst', 'mask', 'checkpoint'])

function stageDir(stageId: string): string {
  return path.join(os.tmpdir(), 'tunet-stages', stageId)
}

function newStageId(): string {
  return crypto.randomBytes(12).toString('base64url')
}

// ── POST ────────────────────────────────────────────────────────────────────

export async function POST(req: Request) {
  // Large single-file uploads (e.g. ~1 GB .pth checkpoints) come in as a raw
  // octet-stream body, NOT multipart. We stream those straight to disk so we
  // never hold the whole file in memory — req.formData() would buffer the
  // entire ~1 GB, which OOM-kills the service on a small (1 GB RAM) host.
  const contentType = req.headers.get('content-type') ?? ''
  if (!contentType.includes('multipart/form-data')) {
    return handleRawUpload(req)
  }

  let form: FormData
  try {
    form = await req.formData()
  } catch (e) {
    return NextResponse.json(
      { error: 'Expected multipart/form-data', detail: e instanceof Error ? e.message : '' },
      { status: 400 },
    )
  }

  let stageId = String(form.get('stageId') ?? '')
  if (!stageId) stageId = newStageId()

  const role = String(form.get('role') ?? '')
  if (!ALLOWED_ROLES.has(role)) {
    return NextResponse.json({ error: `role must be one of: ${[...ALLOWED_ROLES].join(', ')}` }, { status: 400 })
  }

  const files = form.getAll('files').filter(f => f instanceof File) as File[]
  if (files.length === 0) {
    return NextResponse.json({ error: 'no files in request' }, { status: 400 })
  }

  const targetDir = path.join(stageDir(stageId), role)
  await fs.promises.mkdir(targetDir, { recursive: true })

  // Write each file. We use the basename only to avoid path traversal — the
  // browser sends webkitRelativePath like 'project/src/frame.png' but we just
  // want 'frame.png' under the role dir.
  let totalBytes = 0
  for (const file of files) {
    const safeName = path.basename(file.name)
    if (!safeName || safeName.startsWith('.')) continue
    const buf = Buffer.from(await file.arrayBuffer())
    await fs.promises.writeFile(path.join(targetDir, safeName), buf)
    totalBytes += buf.length
  }

  // Compute per-role totals to return as state
  const summary = await summarizeStage(stageId)

  return NextResponse.json({
    stageId,
    received: { role, files: files.length, bytes: totalBytes },
    stage:    summary,
  }, { status: 201 })
}

// ── Raw streamed upload (single file, no multipart) ──────────────────────────
//
// Used for large files (~1 GB .pth checkpoints). The client sends the file as
// the raw request body with metadata in headers:
//   x-role      — same role values as the multipart path
//   x-filename  — URL-encoded basename (we take basename only, no traversal)
//   x-stage-id  — optional; omit on the first upload to mint a new stage
// We pipe req.body → disk so peak memory stays at the stream buffer size, not
// the file size.
async function handleRawUpload(req: Request): Promise<Response> {
  let stageId = req.headers.get('x-stage-id') ?? ''
  if (!stageId) stageId = newStageId()

  const role = req.headers.get('x-role') ?? ''
  if (!ALLOWED_ROLES.has(role)) {
    return NextResponse.json({ error: `role must be one of: ${[...ALLOWED_ROLES].join(', ')}` }, { status: 400 })
  }

  const rawName  = req.headers.get('x-filename') ?? ''
  const safeName = path.basename(decodeURIComponent(rawName))
  if (!safeName || safeName.startsWith('.')) {
    return NextResponse.json({ error: 'x-filename header missing or invalid' }, { status: 400 })
  }
  if (!req.body) {
    return NextResponse.json({ error: 'empty request body' }, { status: 400 })
  }

  const targetDir = path.join(stageDir(stageId), role)
  await fs.promises.mkdir(targetDir, { recursive: true })
  const destPath = path.join(targetDir, safeName)

  try {
    // Readable.fromWeb adapts the web ReadableStream (req.body) to a Node
    // stream; pipeline handles backpressure + closes the file handle.
    await pipeline(Readable.fromWeb(req.body as Parameters<typeof Readable.fromWeb>[0]), fs.createWriteStream(destPath))
  } catch (e) {
    await fs.promises.rm(destPath, { force: true }).catch(() => {})
    return NextResponse.json({ error: 'upload stream failed', detail: e instanceof Error ? e.message : '' }, { status: 500 })
  }

  const bytes   = (await fs.promises.stat(destPath)).size
  const summary = await summarizeStage(stageId)
  return NextResponse.json({
    stageId,
    received: { role, files: 1, bytes },
    stage:    summary,
  }, { status: 201 })
}

// ── GET ─────────────────────────────────────────────────────────────────────

export async function GET(req: Request) {
  const stageId = new URL(req.url).searchParams.get('stageId')
  if (!stageId) return NextResponse.json({ error: 'stageId query param required' }, { status: 400 })
  if (!fs.existsSync(stageDir(stageId))) {
    return NextResponse.json({ error: 'stage not found' }, { status: 404 })
  }
  const summary = await summarizeStage(stageId)
  return NextResponse.json({ stageId, stage: summary })
}

// ── DELETE ──────────────────────────────────────────────────────────────────

export async function DELETE(req: Request) {
  const stageId = new URL(req.url).searchParams.get('stageId')
  if (!stageId) return NextResponse.json({ error: 'stageId query param required' }, { status: 400 })

  const dir = stageDir(stageId)
  if (fs.existsSync(dir)) {
    await fs.promises.rm(dir, { recursive: true, force: true })
  }
  return NextResponse.json({ stageId, deleted: true })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

async function summarizeStage(stageId: string): Promise<Record<string, { files: number; bytes: number }>> {
  const root = stageDir(stageId)
  const out: Record<string, { files: number; bytes: number }> = {}
  if (!fs.existsSync(root)) return out

  for (const role of ALLOWED_ROLES) {
    const roleDir = path.join(root, role)
    if (!fs.existsSync(roleDir)) continue
    const entries = await fs.promises.readdir(roleDir, { withFileTypes: true })
    let files = 0, bytes = 0
    for (const e of entries) {
      if (!e.isFile()) continue
      const st = await fs.promises.stat(path.join(roleDir, e.name))
      files += 1
      bytes += st.size
    }
    if (files > 0) out[role] = { files, bytes }
  }
  return out
}
