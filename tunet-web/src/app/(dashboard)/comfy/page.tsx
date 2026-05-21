// EZ-Comfy — web port of the tkinter comfy_ui.py ComfyUI-on-Spark workflow.
// The form lives in <ComfyForm> (shared with /demo/comfy); here it links job
// hand-off to the real-app job detail page.

import { ComfyForm } from '@/components/comfy/comfy-form'

export default function ComfyPage() {
  return <ComfyForm jobsBase="/jobs" />
}
