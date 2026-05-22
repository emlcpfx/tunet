// /comfy — EZ-Comfy inside the demo skin. Same <ComfyForm> as the real
// /comfy page; only the job hand-off link differs (demo job detail).

import { ComfyForm } from '@/components/comfy/comfy-form'

export default function DemoComfyPage() {
  return <ComfyForm jobsBase="/jobs" />
}
