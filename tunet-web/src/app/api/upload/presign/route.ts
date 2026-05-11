import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { createServiceClient, STORAGE_BUCKET } from '@/lib/supabase'

const ALLOWED_ROLES = ['config', 'src', 'dst', 'checkpoint'] as const
type Role = typeof ALLOWED_ROLES[number]

const ROLE_EXTENSIONS: Record<Role, string[]> = {
  config:     ['.yaml', '.yml'],
  src:        ['.zip'],
  dst:        ['.zip'],
  checkpoint: ['.pth'],
}

const ROLE_FILENAMES: Record<Role, string> = {
  config:     'config.yaml',
  src:        'src.zip',
  dst:        'dst.zip',
  checkpoint: 'checkpoint.pth',
}

export async function POST(req: Request) {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { fileName, jobId, role } = await req.json() as {
    fileName: string
    fileType: string
    jobId: string
    role: Role
  }

  if (!ALLOWED_ROLES.includes(role)) {
    return NextResponse.json({ error: 'Invalid role' }, { status: 400 })
  }

  // Validate file extension
  const ext = fileName.toLowerCase().slice(fileName.lastIndexOf('.'))
  if (!ROLE_EXTENSIONS[role].includes(ext)) {
    return NextResponse.json({
      error: `Invalid file type for ${role}. Expected: ${ROLE_EXTENSIONS[role].join(', ')}`,
    }, { status: 400 })
  }

  // Scoped path: userId/jobId/role-filename
  const storagePath = `${userId}/${jobId}/${ROLE_FILENAMES[role]}`

  const svc = createServiceClient()
  const { data, error } = await svc.storage
    .from(STORAGE_BUCKET)
    .createSignedUploadUrl(storagePath)

  if (error || !data) {
    return NextResponse.json({ error: error?.message ?? 'Could not create upload URL' }, { status: 500 })
  }

  return NextResponse.json({
    signedUrl: data.signedUrl,
    token:     data.token,
    path:      storagePath,
  })
}
