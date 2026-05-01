/**
 * /api/spark/jobs
 *   GET  — list all jobs for the authenticated Spark account
 *   POST — submit a new job
 *
 * Server-only. Bearer token never leaves the server.
 *
 * POST body shape:
 *   {
 *     name:          string,
 *     gpu:           keyof GPU_TYPES   (e.g. "l40s")
 *     image?:        string            (defaults to DEFAULT_IMAGE)
 *     command:       string[]          (argv inside the container)
 *     idleHoldSeconds?: number
 *     env?:          Record<string,string>
 *   }
 *
 * Response: the full Spark submit response (jobId, uploadUrl, ShareSync paths).
 * The client either:
 *   (a) uploads the input tarball directly to uploadUrl using their own
 *       Spark credentials (they don't have any — placeholder for future), or
 *   (b) posts the tarball back to /api/spark/jobs/[id]/upload and we proxy it.
 *
 * For now we expose uploadUrl in the response and let server-side helpers
 * proxy the upload (see /api/spark/jobs/[id]/upload).
 */

import { NextResponse } from 'next/server'
import { listJobs, submitJob, GPU_TYPES, DEFAULT_IMAGE, type GpuKey } from '@/lib/spark'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const jobs = await listJobs()
    return NextResponse.json({ jobs })
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Failed to list jobs' },
      { status: 500 },
    )
  }
}

export async function POST(req: Request) {
  let body: {
    name?: string
    gpu?: string
    instanceType?: string
    image?: string
    command?: string[]
    idleHoldSeconds?: number
    env?: Record<string, string>
  }
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 })
  }

  if (!body.name) {
    return NextResponse.json({ error: 'name is required' }, { status: 400 })
  }
  if (!body.command || !Array.isArray(body.command) || body.command.length === 0) {
    return NextResponse.json({ error: 'command (string[]) is required' }, { status: 400 })
  }

  // Resolve instance type — accept either gpu shortcut or raw SKU
  let instanceType: string | null = null
  if (body.instanceType) {
    instanceType = body.instanceType
  } else if (body.gpu && body.gpu in GPU_TYPES) {
    instanceType = GPU_TYPES[body.gpu as GpuKey].sku
  } else {
    return NextResponse.json(
      { error: `Specify gpu (one of: ${Object.keys(GPU_TYPES).join(', ')}) or instanceType` },
      { status: 400 },
    )
  }

  try {
    const resp = await submitJob({
      name:            body.name,
      instanceType,
      image:           body.image ?? DEFAULT_IMAGE,
      command:         body.command,
      idleHoldSeconds: body.idleHoldSeconds ?? 0,
      env:             body.env ?? {},
    })
    return NextResponse.json(resp, { status: 201 })
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Submit failed' },
      { status: 500 },
    )
  }
}
