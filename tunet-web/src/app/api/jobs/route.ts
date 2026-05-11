import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { createServiceClient, getBootstrapSignedUrl, getCodeBundleUrl } from '@/lib/supabase'
import { createPod, buildBootstrapScript } from '@/lib/runpod'
import { requiredCreditsToLaunch } from '@/lib/pricing'
import type { JobCreatePayload } from '@/types'

export async function GET() {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const svc = createServiceClient()
  const { data, error } = await svc
    .from('jobs')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })
  return NextResponse.json(data)
}

export async function POST(req: Request) {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const body = await req.json() as JobCreatePayload & { id?: string }
  const { name, gpu_type_id, config_path, src_zip_path, dst_zip_path, checkpoint_path,
          container_disk_gb, volume_gb } = body

  if (!name || !gpu_type_id || !config_path) {
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
  }

  const svc = createServiceClient()

  // Fetch GPU pricing
  const { data: gpu } = await svc
    .from('gpu_pricing')
    .select('*')
    .eq('gpu_type_id', gpu_type_id)
    .single()

  if (!gpu) return NextResponse.json({ error: 'Unknown GPU type' }, { status: 400 })

  // Check user balance
  const { data: user } = await svc
    .from('users')
    .select('credit_balance_cents')
    .eq('id', userId)
    .single()

  const required = requiredCreditsToLaunch(gpu.platform_cost_per_hr as number)
  if (!user || user.credit_balance_cents < required) {
    return NextResponse.json({
      error: `Insufficient credits. Need at least $${(required / 100).toFixed(2)} to launch.`,
    }, { status: 402 })
  }

  // Use provided ID or generate a new one
  const jobId = body.id ?? crypto.randomUUID()

  // Create job record (status: pending)
  const { error: insertErr } = await svc.from('jobs').insert({
    id:                   jobId,
    user_id:              userId,
    name,
    status:               'pending',
    gpu_type_id,
    platform_cost_per_hr: gpu.platform_cost_per_hr,
    config_path,
    src_zip_path:         src_zip_path ?? null,
    dst_zip_path:         dst_zip_path ?? null,
    checkpoint_path:      checkpoint_path ?? null,
    container_disk_gb:    container_disk_gb ?? 50,
    volume_gb:            volume_gb ?? 100,
  })

  if (insertErr) return NextResponse.json({ error: insertErr.message }, { status: 500 })

  // Generate signed download URLs for the bootstrap script (24h TTL)
  const [codeUrl, configUrl, srcUrl, dstUrl, checkpointUrl] = await Promise.all([
    getCodeBundleUrl(),
    getBootstrapSignedUrl(config_path),
    src_zip_path  ? getBootstrapSignedUrl(src_zip_path)  : Promise.resolve(null),
    dst_zip_path  ? getBootstrapSignedUrl(dst_zip_path)  : Promise.resolve(null),
    checkpoint_path ? getBootstrapSignedUrl(checkpoint_path) : Promise.resolve(null),
  ])

  if (!codeUrl) {
    await svc.from('jobs').update({ status: 'failed' }).eq('id', jobId)
    return NextResponse.json({ error: 'TuNet code bundle not found. Admin must upload it first.' }, { status: 500 })
  }
  if (!configUrl) {
    await svc.from('jobs').update({ status: 'failed' }).eq('id', jobId)
    return NextResponse.json({ error: 'Config file not accessible' }, { status: 500 })
  }

  // Build and base64-encode the bootstrap script
  const startScript = buildBootstrapScript({
    codeUrl,
    configUrl,
    srcZipUrl:      srcUrl  ?? undefined,
    dstZipUrl:      dstUrl  ?? undefined,
    checkpointUrl:  checkpointUrl ?? undefined,
    runpodApiKey:   process.env.RUNPOD_API_KEY!,
  })

  // Update status to provisioning
  await svc.from('jobs').update({ status: 'provisioning' }).eq('id', jobId)

  // Launch RunPod pod
  try {
    const pod = await createPod({
      name:            `tunet-${name}`,
      gpuTypeId:       gpu_type_id,
      startScript,
      containerDiskGb: container_disk_gb ?? 50,
      volumeGb:        volume_gb ?? 100,
    })

    await svc.from('jobs').update({
      pod_id:             pod.id,
      runpod_cost_per_hr: pod.costPerHr,
      gpu_display_name:   gpu.display_name,
    }).eq('id', jobId)

    return NextResponse.json({ id: jobId, pod_id: pod.id }, { status: 201 })
  } catch (err) {
    await svc.from('jobs').update({ status: 'failed' }).eq('id', jobId)
    return NextResponse.json({
      error: err instanceof Error ? err.message : 'Pod creation failed',
    }, { status: 500 })
  }
}
