/**
 * GET /api/comfy/presets — list the comfy_spark presets (read from the repo on
 * the VPS) with their `ui` metadata, params, gpu, and LoRA catalog. The EZ-Comfy
 * page renders its form entirely from this, exactly like comfy_ui.py does off
 * the *.preset.json files.
 */
import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { loadComfyPresets, loadLoraCatalog } from '@/lib/comfy'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function GET() {
  const session = await auth()
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  const presets = loadComfyPresets().map(p => ({
    ...p,
    loraCatalog: p.lora_chain ? loadLoraCatalog(p) : undefined,
  }))
  return NextResponse.json({ presets })
}
