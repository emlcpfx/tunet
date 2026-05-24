# VOID on Spark — end-to-end plan

Getting **Netflix VOID** (video object + interaction removal) running through the
comfy_spark ingest pipeline, with an **interactive in-browser masking** front end
on the website. Written against the WORKFLOW_INGEST pipeline and the existing
two-input/mask plumbing.

> **Status:** design only — no code yet. This doc is the hand-off / review artifact.

---

## Decisions locked (2026-05-22)

| Decision | Choice | Implication |
|----------|--------|-------------|
| Quadmask fidelity | **True VLM-reasoned quadmask** | Need SAM + a Gemini VLM reasoner to find "affected regions" (the `127` band). Adds a `GEMINI_API_KEY` + per-clip cost. This is what makes output look like *VOID*, not ordinary inpaint. |
| Interactive masking | **Browser decoder + GPU encoder** | ~20 MB SAM decoder runs in-browser (onnxruntime-web) for instant clicks; the heavy encoder + video propagation + VLM reasoner run on a GPU (NOT the 1 GB VPS). Biggest build, best UX. |
| First step | **Write this plan** | Build order TBD after review. Recommended start: Phase 0. |

Open sub-decision: **which GPU hosts the masking encoder/propagation/reasoner** —
the user's **local ComfyUI** (free, low-latency, must be reachable) vs a short
cloud GPU job. Recommended: local ComfyUI, with a cloud fallback. VOID *removal*
(Stage B) always runs on Spark.

---

## What VOID actually is (and why it's two stages)

VOID removes a target object **and the physical interactions it caused** (things
that fall/shift once it's gone) — built on CogVideoX-Fun-V1.5-5b-InP, fine-tuned
for inpainting. It is **natively in ComfyUI core** now (template
`utility_void_video_inpainting.json`, weights `Comfy-Org/void-model`), with a
third-party node pack `ComfyUI_RH_Void` as an alternative.

It does **not** take a binary mask. It takes a **quadmask** + a text prompt of the
clean scene. The quadmask is the crux of the whole project (see spec below).

This forces a clean split:

| Stage | Work | Cost profile | Home |
|-------|------|--------------|------|
| **A. Masking** | clicks → per-frame object mask → **quadmask** (incl. VLM affected-region reasoning) | interactive, light-to-medium GPU | **local ComfyUI** (browser does clicks) |
| **B. Removal** | VOID inference (CogVideoX-Fun 5B) | batch, **24–40 GB VRAM**, minutes | **Spark** via comfy_spark |

The **1 GB VPS runs no ML** — pure relay: upload staging, job submit, ShareSync
download, static hosting. Matches its current role.

```
   Browser (artist machine)          1 GB VPS (relay only)        GPUs
   ────────────────────────         ─────────────────────       ──────
   ① scrub clip, click object ──►    proxy points + clip  ──►  Local ComfyUI
      onnx decoder = instant                                    SAM encode + propagate
      preview overlay          ◄──   per-frame masks      ◄──   (your GPU)
   ② confirm object + prompt                                     │
                              ──►    relay to reasoner   ──►    VLM reasoner (SAM+Gemini)
   ③ quadmask video ready     ◄──    quadmask            ◄──    → quadmask.mp4
   ④ clip+quadmask+prompt     ──►    comfy_spark submit  ──►   Spark (L40S 48GB / RTX PRO)
                                                                 VOID render
   ⑤ download clean plate     ◄──    ShareSync           ◄──    comfy_run self-upload
```

---

## The quadmask spec (the crux)

A per-frame **greyscale MP4**, same length + resolution as the source, four values:

| Value | Meaning | Source |
|-------|---------|--------|
| `0`   | primary object to remove | **interactive SAM selection** |
| `63`  | overlap of primary + affected | derived |
| `127` | **affected region** (objects that fall/shift after removal) | **Gemini VLM reasoning** |
| `255` | background to **keep** | everything else |

**Inverted convention:** white (255) = keep, black (0) = remove.

Interactive SAM gives only the `0` region. The `127` "affected" reasoning is what
needs the VLM — that's why "true quadmask" implies a Gemini dependency. Netflix's
`VLM-MASK-REASONER/` does exactly this (SAM2/SAM3 + Gemini); the `ComfyUI_RH_Void`
**Mask Reasoner** node wraps the same idea (`api_key`, `analysis_model:
gemini-3-flash-preview`).

---

## Phase 0 — prove VOID renders on Spark (no UI)

Lowest-risk first move; directly the WORKFLOW_INGEST cheap-gate loop. Uses a
**hand-made quadmask** so we test the *model*, not the masking.

1. **Get the template.** Fetch `utility_void_video_inpainting.json` from the
   Comfy-Org `workflow_templates` repo. (Alternatively the `ComfyUI_RH_Void`
   `example/void_example.json`.)
2. **Resolve deps.**
   `python comfy_launch.py --resolve utility_void_video_inpainting.json --resolve-out void.draft.preset.json`
   → fill any `REPLACE_*` model URLs:
   - `Comfy-Org/void-model` → `void_pass1.safetensors` (~11 GB), `void_pass2.safetensors` (~11 GB)
   - base `CogVideoX-Fun-V1.5-5b-InP` (separate, several GB)
3. **Strip the in-graph mask-gen.** The template suggests SAM3 to make the
   initial mask. For Spark we feed a **pre-made quadmask** as an input instead —
   mute/drop the SAM/reasoner sub-cluster (WORKFLOW_INGEST §F: mute the dead
   *output* leaves; the prune drops their ancestors). VOID render then takes
   `clip + quadmask + prompt`.
4. **Cheap gate.** `python comfy_launch.py --preset void --convert-only --gpu t4`
   → iterate until `VALIDATION OK`. Watch for subgraph flatten (§G — should be
   automatic), combo drift (§B), DynamicCombo (§C), version pins (§D).
5. **One full render** on a big GPU (g6e L40S 48 GB, or g7e RTX PRO 6000) with a
   hand-painted quadmask + prompt. Confirm output + ShareSync upload.
6. **Capture** the converted `workflow_api.json` for reuse.

**⚠ Verify first:** the ComfyUI version baked into the comfy_spark Spark image
(no-build yanwk + py3.13 per project notes) must be recent enough to have **native
VOID nodes**. If not: either bump the image's ComfyUI, or use `ComfyUI_RH_Void`
(comfy_resolve can fetch it as a node pack) — note RH_Void looks **Pass-1 only**.

**Billing:** every job must carry the `cpfx_tunet` tag (enforced in
`comfy_launch.py`); `cpfx_comfy` is added for filtering.

### Phase 0 findings (2026-05-22)

Pulled the real template + resolved it. Several plan-shaping discoveries:

- **Template** = `utility_void_video_inpainting.json` (Comfy-Org/workflow_templates,
  117 KB). Top level is `LoadVideo → [subgraph] → 2× SaveVideo`; the work is a
  **subgraph "Video Inpaint (VOID)"** (37 nodes) with a **nested subgraph "Image
  Segmentation (SAM3)"** (3 nodes). It ships **both Pass 1 and Pass 2** (RAFT
  optical flow + `VOIDWarpedNoise` refinement).
- **Flatten-then-resolve is required.** `comfy_resolve` does NOT descend into
  subgraph defs, so a raw `--resolve` found 0 deps. Flattening first
  (`comfy_run.flatten_subgraphs`, 42 nodes) then resolving gives the true picture:
  **0 custom node packs** (all 27 classes are ComfyUI **core**) and **6 weights,
  all URLs known** (no `REPLACE_`): `void_pass1`/`void_pass2`/`cogvideox_vae`/`raft`
  (Comfy-Org/void-model), `t5xxl_fp16` (flux encoders), `sam3.1_multiplex_fp16`
  (Comfy-Org/sam3.1). → **`comfy_resolve` should flatten before resolving; file as
  an EZ-Comfy fix.**
- **The native template is TEXT-driven SAM3, NOT Gemini.** Masking is an in-graph
  `CLIPTextEncode("human") → SAM3_Detect → mask → VOIDInpaintConditioning`. There
  is **no VLM quadmask reasoner** in the native nodes — the Gemini/4-tone quadmask
  is only in Netflix's standalone repo / `ComfyUI_RH_Void`. **Implication: the
  "true VLM quadmask" decision does not apply to the native path; getting VOID
  working needs no Gemini.** (Still TBD: does `VOIDInpaintConditioning` take a
  plain binary MASK or a 4-tone quadmask? Verify once the node is available.)
- **SAM3 is native and already does what Stage A needs.** Local ComfyUI 0.20.1
  exposes `SAM3_Detect` (text conditioning + **interactive `positive_coords` /
  `negative_coords` JSON point prompts** + bboxes → MASK) and `SAM3_VideoTrack →
  SAM3_TrackToMask` (**native video propagation**). So the interactive masking UI
  can drive native SAM3 with clicked point JSON — **no kijai/SAM2 custom nodes
  needed.**
- **Version blocker.** Local ComfyUI **0.20.1 (RTX 3090, 24 GB)** has SAM3 but
  **NOT VOID** — VOID core nodes are newer than 0.20.1. Both local and the Spark
  image need a ComfyUI bump to a VOID-capable build. The 3090's 24 GB can run 480p
  VOID locally, so local validation (free, WORKFLOW_INGEST "iterate locally") is
  the cheapest first proof.

**Revised recommendation:** start with the native, text-driven template (one job,
no Gemini, no separate masking stage). Validate locally on the 3090 first, then
port to Spark. Add interactive SAM3 point-selection (Stage A) and — only if the
native removal quality needs it — the Gemini quadmask, as later upgrades.

---

## Phase 1 — the VOID preset (Stage B, batch removal)

Model the preset on `wan_vace_inpaint` — it already proves the clip+mask-video
pattern end to end.

**New files in `comfy_spark/presets/`:**
- `void_inpaint.preset.json`
- `void_inpaint.workflow.json` (converted from Phase 0)
- `void_inpaint.derive.py` *(only if a hand edit is needed — mute mask-gen, etc.)*

**Preset shape (mirrors `wan_vace_inpaint.preset.json`):**
```jsonc
{
  "tags": ["video","void","inpainting","object-removal","mask-required","ready-to-run"],
  "docs": [
    { "label": "VOID model card", "url": "https://huggingface.co/netflix/void-model" },
    { "label": "ComfyUI VOID announce", "url": "https://blog.comfy.org/p/new-open-source-models-now-in-comfyui" }
  ],
  "about": {
    "what":  "Removes an object from a clip — and the things it was affecting — leaving a clean plate.",
    "inputs":"A video, a 4-tone quadmask video marking what to remove vs keep, and a short line describing the scene after removal.",
    "key_knobs":"Resolution (672x384 default), steps (~30), frame count (~85)."
  },
  "ui": {
    "title": "VOID — video object removal",
    "primary_input":   { "label": "Input video",  "filetypes": [["Video","*.mp4 *.mov *.webm"]] },
    "secondary_input": { "param": "mask", "label": "Quadmask video",
                         "tooltip": "4-tone mask: black=remove, white=keep. Use the Masking tool to make one.",
                         "filetypes": [["Video","*.mp4 *.mov"]] }
  },
  "workflow": "void_inpaint.workflow.json",
  "params": {
    "prompt": { "node": "<id>", "path": "inputs.text" },
    "mask":   { "node": "<id>", "path": "inputs.image" }
  },
  "mask_video_param": { "node": "<id>", "path": "inputs.file" }
}
```
Because the quadmask is always a **video**, this is the `workflow_mask_video` /
`mask_video_param` path — the same auto-switch `wan_vace_inpaint` uses
(`comfy_launch.py:844-850`). The web form will show the second uploader purely
because `ui.secondary_input` is present (`comfySecondaryParam`, `comfy.ts:109`).

**Per CLAUDE.md:** ship `docs` + `about` + `tags` in plain language; add per-knob
`ui.tooltip`s; run `--convert-only` and `--resolve-selftest` before any paid render.

**GPU sizing:** default 480p (720×480) needs ~24 GB (g6e L40S works); 720p needs
more. Expose a resolution knob; default conservative.

---

## Phase 2 — interactive masking (Stage A): browser decoder + GPU encoder

Goal: artist clicks the object, gets an instant overlay, confirms, and a
**quadmask.mp4** comes back. All ML on a GPU; VPS only relays.

**Pieces:**
1. **Frame extraction** — ffmpeg on the GPU host (or precompute) → frames the
   browser can show + click on.
2. **In-browser decoder** — bundle the ~20 MB SAM mask **decoder** ONNX, run with
   **onnxruntime-web** (WASM, or WebGPU). Clicks → coords → instant mask preview.
   Single-image only; this is just the live preview.
3. **GPU encoder** — on the masking GPU, run the SAM image **encoder** per viewed
   frame; return the embedding (`.npy`/ORT) to the browser to drive the decoder.
4. **Confirm → propagate** — send confirmed points (frame idx, obj id,
   coords/labels) to the GPU; run SAM2 video predictor
   (`init_state → add_new_points_or_box → propagate_in_video`) → per-frame binary
   object mask. ComfyUI node: kijai `Sam2VideoSegmentation` /
   `Sam2VideoSegmentationAddPoints`.
5. **VLM reasoner → quadmask** — feed video + object mask + prompt to the
   reasoner (Netflix `VLM-MASK-REASONER`, or `ComfyUI_RH_Void` **Mask Reasoner**
   node) → assemble the 4-tone quadmask → encode `quadmask.mp4`.

**Recommended host:** run steps 3–5 as a **ComfyUI workflow on local ComfyUI**
(the user has it). `ComfyUI_RH_Void`'s **Point Editor → Mask Reasoner** nodes do
points→quadmask already; the browser decoder just makes the point-picking instant.
So Stage A can itself be a comfy_spark-style local workflow.

**SAM2 vs SAM3:**
- **Start SAM2** — battle-tested, mature ComfyUI nodes
  (kijai/ComfyUI-segment-anything-2), proven click→propagate video API, clean
  encoder/decoder ONNX split for the browser.
- **SAM3 = upgrade** — adds text/concept prompts ("select every X") + faster
  multi-object tracking. **But:** gated HF access + a custom **"SAM License"**
  (not Apache/MIT) → **legal review before commercial ship.** VOID's template
  *suggests* SAM3 for the initial mask; SAM2 is a fine substitute.

**Browser/video caveat:** in-browser SAM is single-frame only; **video
propagation must run on the GPU** (SAM2's stateful memory-attention is too heavy
for the browser). So the browser does live single-frame preview; the GPU does the
real cross-frame masklet.

---

## Phase 3 — website integration

The two-input upload path is **already built** — we mostly flip it on and add the
masking UI. Verified insertion points:

| Need | File:line | Today |
|------|-----------|-------|
| Show 2nd uploader | `comfy-form.tsx:52` `secondaryOf()` | fires on `ui.secondary_input` → VOID preset enables it for free |
| Primary clip upload | `comfy-form.tsx:229-233` | role `comfy_input` → `/tmp/tunet-stages/<id>/comfy_input/` |
| Mask upload | `comfy-form.tsx:235-242` | role `comfy_input2` → `…/comfy_input2/` |
| Submit handling | `api/comfy/submit/route.ts:93-103` | resolves `secondaryName`, packs into tarball |
| Param resolve | `comfy.ts:109` `comfySecondaryParam()` | returns `'mask'` for VOID |
| Output return | `comfy_run.py` self-upload → ShareSync | unchanged |

**New work on the website:**
- **Masking page/modal** — clip scrubber + click-to-select (onnxruntime-web
  decoder), "propagate" + "build quadmask" actions that call the masking GPU
  (local ComfyUI endpoint or a relay), preview the quadmask, then hand the
  resulting `quadmask.mp4` to the existing secondary-upload slot of the VOID
  preset.
- **Relay endpoints** on the VPS to proxy browser ↔ local ComfyUI (frames,
  embeddings, points, quadmask) without running ML. Handle CORS / reachability
  (local Comfy on the artist's LAN).
- **Prompt field** for the after-removal scene description (`params.prompt`).

---

## Secrets — the Gemini key

The VLM reasoner needs `GEMINI_API_KEY`. It is a **server-side** secret; treat it
exactly like `AUTH_KEYCLOAK_SECRET` / `SUPABASE_SERVICE_ROLE_KEY`.

- **Storage:** `tunet-web/.env.local` (dev) and `tunet-web/deploy/tunet-web.env`
  (prod → `/etc/tunet-web/env` via `deploy.ps1 -PushEnv`). Both are gitignored.
  Prod file ends up `0640 root:tunet` (others have no read); the deploy now also
  `chmod 600`s the transient `/tmp/tunet-web.env` during `-PushEnv` so it's not
  exposed on the shared box during the deploy window.
- **No `NEXT_PUBLIC_` prefix** — that would inline it into the browser bundle.
- **The Gemini call runs on the server, never the browser.** The reasoning step
  is plain HTTPS (no GPU), so the cleanest design is: GPU node does SAM → returns
  masks/segments → **the VPS server makes the Gemini call** and assembles the
  quadmask. Key lives in exactly one place; also a natural spot to meter cost +
  rate-limit.
- **⚠ Never put the key in the ComfyUI workflow JSON / preset / tarball.** The
  `ComfyUI_RH_Void` Mask Reasoner node takes `api_key` as a *node input* — do not
  fill that widget (it serializes into `workflow.json` and ships in the tarball).
  If the reasoner must run on a GPU node, pass the key as a **job env var**
  (Spark, like `COMFY_UPLOAD_TOKEN`) or the local ComfyUI **process env**, never a
  graph value. Don't log it; don't echo Gemini errors verbatim to the client.
- **Harden at Google's end:** restrict the key to the Generative Language API and
  set a billing budget/quota alert so a leak can't run up unbounded spend.

---

## Cross-cutting risks / things to verify

1. **Spark image ComfyUI version** has native VOID — else bump image or use
   `ComfyUI_RH_Void` (Pass-1 only). *(Phase 0 gate will reveal this.)*
2. **Local GPU VRAM** — can it run the SAM encoder + propagation + reasoner? (VOID
   *removal* is on Spark, so local only needs masking ~ SAM2 ≤ 16 GB.)
3. **Gemini key + cost** — required for the true quadmask reasoner; per-clip API
   spend. Store server-side; never ship to the browser.
4. **SAM3 license/gating** — legal review before commercial use; SAM2 avoids this.
5. **Weight pull size** — VOID ~22 GB + CogVideoX-Fun base; first Spark render is
   slow. Confirm caching across jobs.
6. **Pass 2 refinement** — only in the standalone CLI / not in RH_Void; decide if
   the native template exposes it. Quality-critical jobs may want it.
7. **CLI ↔ GUI parity** (CLAUDE.md) — any new `comfy_launch.py` flag must reach
   `comfy_ui.py`.

---

## Suggested build order

1. **Phase 0** — ingest template, pass `--convert-only`, one full render w/ hand
   mask. *Confirms VOID + finds the image-version answer.*
2. **Phase 1** — ship the `void_inpaint` preset (clip + quadmask video).
3. **Phase 3 (lite)** — enable the VOID preset on the website with **manual
   quadmask upload** (no masking UI yet). Now it's usable end to end by anyone who
   can make a quadmask.
4. **Phase 2** — build the interactive masking tool (browser decoder + GPU
   encoder + SAM2 propagate + VLM reasoner → quadmask).
5. **Phase 3 (full)** — wire the masking tool into the form; add relay endpoints.
6. **Upgrades** — SAM3 text prompts; VOID Pass 2; 720p tier.

---

## References

- VOID repo — https://github.com/netflix/void-model
- VOID weights — https://huggingface.co/netflix/void-model · https://huggingface.co/Comfy-Org/void-model
- ComfyUI VOID announce — https://blog.comfy.org/p/new-open-source-models-now-in-comfyui
- RH_Void node pack — https://github.com/HM-RunningHub/ComfyUI_RH_Void
- SAM 2 — https://github.com/facebookresearch/sam2 · kijai nodes: https://github.com/kijai/ComfyUI-segment-anything-2
- SAM 3 — https://github.com/facebookresearch/sam3 (gated; custom SAM License)
- In-browser SAM (decoder/onnxruntime-web) — https://github.com/lucasgelfond/webgpu-sam2
- comfy_spark ingest pipeline — `comfy_spark/WORKFLOW_INGEST.md`
- Reference preset (clip+mask-video) — `comfy_spark/presets/wan_vace_inpaint.preset.json`
