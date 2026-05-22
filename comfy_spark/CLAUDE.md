# comfy_spark — instructions for Claude

This is the ComfyUI-on-Spark layer (CLI `comfy_launch.py`, desktop GUI
`comfy_ui.py`, on-node runner `comfy_run.py`, and `presets/*.preset.json`
bundles). Read `README.md` for the full tour. The rules below are **requirements**
when changing this folder.

## Adding a new workflow / preset — REQUIRED metadata

Every `presets/*.preset.json` we ship MUST carry these three keys, written for a
**non-technical user**. The CLI (`--list-presets`) and the desktop GUI ("About
this workflow" panel) both read them, so they are the front door to the workflow,
not internal notes. Keep ComfyUI / AI / ML jargon out — say "outlines" not
"canny", "clip" not "latent", "add-on" not "LoRA" where you can.

1. **`docs`** — a link (or links) to the **canonical** documentation for the
   model / workflow (the model card, the official example-workflow page, etc.).
   This is what a user clicks to learn more. Shape:
   ```json
   "docs": [
     { "label": "Obscura Remova model card", "url": "https://huggingface.co/WepeNerd/Obscura_Remova" },
     { "label": "LTX-2 docs", "url": "https://docs.ltx.video" }
   ]
   ```
   (A bare URL string or a list of URL strings is also accepted, but prefer the
   labelled form.)

2. **`about`** — a plain-language description, three short fields:
   ```json
   "about": {
     "what":      "What it does, in one or two everyday sentences.",
     "inputs":    "What the user gives it (a clip? an image? a mask?).",
     "key_knobs": "The 2-3 settings that matter most, and a sane starting value."
   }
   ```
   Litmus test: someone who has never opened ComfyUI should understand what the
   workflow is for and what to feed it from these three lines alone.

3. **`tags`** — a flat list of lowercase, hyphenated tags so workflows are easy to
   sift (`--list-presets <tag>` filters on a substring match). Pick from the
   shared vocabulary below; add new tags sparingly and reuse existing ones.

   | facet | tags |
   |-------|------|
   | output | `video` (add `image` / `audio` if/when we ship those) |
   | model | `ltx2.3`, `wan2.1`, `wan2.2`, … (match the preset's `base` family token) |
   | task | `text-to-video`, `image-to-video`, `video-to-video`, `generation`, `inpainting`, `object-removal`, `cleanplate`, `face-swap`, `hdr`, `color-grade`, `structural-control` |
   | structure cues | `canny`, `depth`, `pose` |
   | inputs / needs | `mask-required`, `no-mask`, `two-inputs`, `lora-stacking`, `ic-lora` |
   | readiness | `ready-to-run` (all weights pinned) **vs** `template` (user must fill `REPLACE_*` URLs) |

The existing six presets are the reference — match their tone and depth.

## Other standing rules

- **CLI ↔ GUI parity.** Every `comfy_launch.py` CLI feature must also be reachable
  in `comfy_ui.py`. Per-preset `params` flow to the GUI automatically *if* they
  carry a `ui` block; a new **global** flag needs an explicit control wired into
  the GUI. (Same rule the README states.)
- **Per-knob `ui.tooltip`s** should stay human and cite their source where a value
  is non-obvious (e.g. `(src: LTX-2.3 docs)`), like the shipped presets do.
- **Validate cheaply first.** After editing a preset, run
  `python comfy_launch.py --preset <name> --convert-only` (no GPU, no model pull)
  before any paid render, and `--resolve-selftest` if you touched `node_packs` /
  `models`.
- **Billing tag** `cpfx_tunet` is mandatory on every job (enforced in
  `comfy_launch.py`); `cpfx_comfy` is added so ComfyUI jobs are filterable.
- **Output formats** are a global choice (`--output`, registry `outputs/outputs.json`),
  not per-preset: `comfy_run.rewrite_output` splices the chosen saver as a parallel
  branch onto the **auto-detected** frames anchor, so a new preset gets high-bit
  EXR/ProRes output for free. Only add an explicit `output_anchor` to a preset if
  its terminal node isn't a `CreateVideo`/`SaveImage`/`VHS_VideoCombine` (i.e.
  auto-detect can't find the final IMAGE). Add a new format by editing the registry
  (save-node chain + `node_pack`), not code. mp4 stays the default preview.
- **Input sequences** (`--input-sequence DIR`, v2v presets): `comfy_run.rewrite_input`
  swaps the preset's IMAGE-batch loader (auto-detected `VHS_LoadVideo`/`LoadImage`,
  or a preset's `input_anchor: {node}`) in place for CoCoTools `LoadExrSequence` and
  inserts a scene-linear→sRGB `ColorspaceNode`. EXR only today; new kinds = a
  `build_seq_loader` entry. Core `LoadVideo` (VIDEO-object) presets aren't swappable.
