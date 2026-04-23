'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Card, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input, FormRow, Select } from '@/components/ui/input'
import { TierBadge } from '@/components/ui/badge'
import { FileUpload } from '@/components/dashboard/file-upload'
import { formatCredits } from '@/types'
import type { DbGpuPricing } from '@/types'

const MOCK_GPUS: DbGpuPricing[] = [
  { gpu_type_id: 'spark_168', display_name: 'NVIDIA A10 24GB',   short_key: 'a10',     vram_gb: 24, platform_cost_per_hr: 1.19, runpod_cost_per_hr: 0.79, is_available: true, tier: 'standard',    sort_order: 1 },
  { gpu_type_id: 'spark_179', display_name: 'NVIDIA L4 24GB',    short_key: 'l4',      vram_gb: 24, platform_cost_per_hr: 1.79, runpod_cost_per_hr: 1.14, is_available: true, tier: 'standard',    sort_order: 2 },
  { gpu_type_id: 'spark_181', display_name: 'NVIDIA L40S 48GB',  short_key: 'l40s',    vram_gb: 48, platform_cost_per_hr: 4.50, runpod_cost_per_hr: 2.99, is_available: true, tier: 'recommended', sort_order: 3 },
  { gpu_type_id: 'spark_204', display_name: 'NVIDIA L40S Pro',   short_key: 'l40s-pro',vram_gb: 48, platform_cost_per_hr: 5.25, runpod_cost_per_hr: 3.49, is_available: true, tier: 'pro',         sort_order: 4 },
]

const MOCK_BALANCE = 5250

export default function DemoNewJobPage() {
  const router = useRouter()
  const [selectedGpu, setSelectedGpu] = useState<DbGpuPricing>(MOCK_GPUS[2])
  const [name, setName] = useState('')
  const [containerDisk, setContainerDisk] = useState(50)
  const [volumeGb, setVolumeGb] = useState(100)
  const [configFile, setConfigFile] = useState<File | null>(null)
  const [srcFile, setSrcFile] = useState<File | null>(null)
  const [dstFile, setDstFile] = useState<File | null>(null)
  const [checkpointFile, setCheckpointFile] = useState<File | null>(null)

  const canAfford = MOCK_BALANCE >= Math.ceil(selectedGpu.platform_cost_per_hr * 100)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    // Demo: just navigate back to jobs list
    router.push('/demo/jobs')
  }

  return (
    <div className="space-y-6 animate-slide-in max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-[#111827]">New Training Job</h1>
        <p className="text-sm text-[#6b7280] mt-1">Configure and launch a GPU training run</p>
      </div>

      <div className="px-3 py-2 bg-[#FFFBEB] border border-[#FDE68A] rounded-lg text-xs text-[#D97706]">
        Demo mode — form is interactive but job launch is disabled.
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        <Card>
          <CardTitle className="mb-4">GPU</CardTitle>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {MOCK_GPUS.map(gpu => (
              <button
                key={gpu.gpu_type_id}
                type="button"
                onClick={() => setSelectedGpu(gpu)}
                className={`text-left p-4 rounded-xl border-2 transition-all duration-150 ${
                  selectedGpu.gpu_type_id === gpu.gpu_type_id
                    ? 'border-[#ae69f4] bg-[#F7F4FC]'
                    : 'border-[#e5e7eb] hover:border-[#D1D5DB]'
                }`}
              >
                <div className="flex items-start justify-between mb-1">
                  <span className="font-semibold text-sm text-[#111827]">{gpu.display_name}</span>
                  <TierBadge tier={gpu.tier} />
                </div>
                <p className="text-xs text-[#6b7280]">{gpu.vram_gb} GB VRAM</p>
                <p className="text-sm font-semibold text-[#ae69f4] mt-2">
                  ${gpu.platform_cost_per_hr.toFixed(2)}<span className="text-xs font-normal text-[#9ca3af]">/hr</span>
                </p>
                {selectedGpu.gpu_type_id === gpu.gpu_type_id && (
                  <div className="flex gap-3 mt-2 text-xs text-[#6b7280]">
                    <span>1h: ${gpu.platform_cost_per_hr.toFixed(2)}</span>
                    <span>4h: ${(gpu.platform_cost_per_hr * 4).toFixed(2)}</span>
                    <span>8h: ${(gpu.platform_cost_per_hr * 8).toFixed(2)}</span>
                  </div>
                )}
              </button>
            ))}
          </div>
        </Card>

        <Card>
          <CardTitle className="mb-4">Job Configuration</CardTitle>
          <div className="space-y-4">
            <FormRow label="Job Name" hint="Leave blank to auto-generate">
              <Input
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder={`${selectedGpu.short_key}-job`}
              />
            </FormRow>
            <FormRow label="Config YAML" required hint="Your tunet config.yaml">
              <FileUpload accept=".yaml,.yml" label="YAML config file" maxMb={1} onFile={setConfigFile} currentName={configFile?.name} />
            </FormRow>
            <div className="grid grid-cols-2 gap-4">
              <FormRow label="Src Images" hint="ZIP of source images">
                <FileUpload accept=".zip" label="src.zip" maxMb={10000} onFile={setSrcFile} currentName={srcFile?.name} />
              </FormRow>
              <FormRow label="Dst Images" hint="ZIP of destination images">
                <FileUpload accept=".zip" label="dst.zip" maxMb={10000} onFile={setDstFile} currentName={dstFile?.name} />
              </FormRow>
            </div>
            <FormRow label="Resume Checkpoint" hint=".pth file to resume from (optional)">
              <FileUpload accept=".pth" label=".pth checkpoint file" maxMb={10000} onFile={setCheckpointFile} currentName={checkpointFile?.name} />
            </FormRow>
          </div>
        </Card>

        <Card>
          <CardTitle className="mb-4">Storage</CardTitle>
          <div className="grid grid-cols-2 gap-4">
            <FormRow label="Container Disk" hint="OS + dependencies">
              <Select value={containerDisk} onChange={e => setContainerDisk(Number(e.target.value))}>
                {[20, 50, 100, 200].map(v => <option key={v} value={v}>{v} GB</option>)}
              </Select>
            </FormRow>
            <FormRow label="Persistent Volume" hint="For checkpoints and outputs">
              <Select value={volumeGb} onChange={e => setVolumeGb(Number(e.target.value))}>
                {[50, 100, 200, 500].map(v => <option key={v} value={v}>{v} GB</option>)}
              </Select>
            </FormRow>
          </div>
        </Card>

        <div className="flex items-center justify-between pt-2">
          <div className="text-sm text-[#6b7280]">
            Balance: <span className="font-semibold text-[#111827]">{formatCredits(MOCK_BALANCE)}</span>
          </div>
          <div className="flex gap-3">
            <Button type="button" variant="ghost" onClick={() => router.back()}>Cancel</Button>
            <Button type="submit" disabled={!canAfford}>Launch Job</Button>
          </div>
        </div>
      </form>
    </div>
  )
}
