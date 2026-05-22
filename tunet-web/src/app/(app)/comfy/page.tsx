// /comfy — EZ-Comfy submission form. `?clone=<jobId>` pre-fills the form from a
// prior job's stored settings (read server-side here and passed as a prop so the
// client form needn't reach for useSearchParams / a Suspense boundary).

import { ComfyForm } from '@/components/comfy/comfy-form'

export default async function ComfyPage({
  searchParams,
}: {
  searchParams: Promise<{ clone?: string }>
}) {
  const { clone } = await searchParams
  return <ComfyForm jobsBase="/jobs" cloneFromId={clone ?? null} />
}
