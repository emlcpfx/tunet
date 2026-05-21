// EZ-Comfy — a web front-end for the comfy_spark ComfyUI-on-Spark workflow
// (a port of the tkinter comfy_ui.py). The form is preset-driven, exactly like
// the desktop tool: it reads comfy_spark/presets/*.preset.json and renders the
// knobs each preset declares in its `ui` metadata. Submit + live logs + output
// download are wired in a follow-up (see /api/comfy/*).

export default function ComfyPage() {
  return (
    <div>
      <div className="flex items-center gap-2.5 mb-2">
        <h1 className="text-xl font-bold text-[#111827]">EZ-Comfy</h1>
        <span className="text-xs font-semibold text-[#7E3AF2] bg-[#faf5ff] border border-[#e9d5ff] rounded px-2 py-0.5">
          Beta
        </span>
      </div>
      <p className="text-sm text-[#6b7280] mb-6 max-w-2xl">
        Run ComfyUI workflows on Spark Fuse — the same engine as the desktop tool,
        in your browser. Pick a workflow preset, drop in your clip, tune the knobs,
        and submit. Outputs land in ShareSync and download here when the job finishes.
      </p>
      <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 card-shadow text-sm text-[#6b7280]">
        Setting up the workflow form. Hang tight.
      </div>
    </div>
  )
}
