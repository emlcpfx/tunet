# EZ-Comfy — roadmap / to-do

**Goal:** let people bring their own ComfyUI workflow and have comfy_spark
auto-grab the custom nodes + model weights it needs, then run it on Spark —
first from the CLI, then in tunet-web.

**Working principle: build & test locally first.** The resolver is logic-heavy
and will need many passes against messy real-world workflows; the local Python
loop is seconds, a tunet-web deploy loop is minutes on a *shared* VPS. Crucially
we already have ground truth: the shipped presets
(`ltx2_generate`, `ltx_control`, `wan_vace_inpaint`, `ltx_Obscura_Remova`) have
hand-written `node_packs` + `models`, so the resolver can be regression-tested by
checking it reproduces them (`comfy_launch.py --resolve-selftest`).

**Format (decided):** the input is a **full UI/graph workflow export**, not API
format. API format is lossy — it drops `properties.cnr_id`/`aux_id` (node
provenance) and widget metadata. The resolver's output is the existing preset
schema (`node_packs` + `models` + gaps), so "resolve a workflow" == "generate a
draft preset," and presets already flow to both the CLI and tunet-web.

**Architecture (decided):** the resolver *brain* is one Python module
(`comfy_resolve.py`, stdlib-only like `comfy_run.py`) used by the CLI now and
called — never reimplemented — by tunet-web later. The provisioning engine
(`comfy_run.py`, on the Spark node) and the preset schema are already shared; the
client orchestration (`comfy_launch.py` / `comfy.ts`) is the duplicated layer, so
keep the hardest logic out of TS.

---

## Phase 0 — Decisions (blocking the web side)
- [ ] **Per-user preset storage**: Supabase (already in tunet-web) vs. each user's
      ShareSync space. Shapes the whole web side.
- [ ] **Security stance on BYO**: a BYO workflow = arbitrary node-pack `git clone`
      + `pip install` + arbitrary model URLs on the Spark node = remote code
      execution. Per-user-billed node contains it somewhat. Decide: allowlist
      known packs / warn-and-proceed / sandbox.
- [ ] **Model auth**: how gated HF / Civitai weights get a token to `download()`
      in `comfy_run.py` (env passthrough? per-user secret?).

## Phase 1 — Resolver brain (local Python) — DONE (validated offline + live)
- [x] Inspect real workflow shape: provenance in `properties.cnr_id`/`aux_id`/`ver`;
      `comfy-core` == built-in; many custom nodes carry **no** provenance →
      class→repo fallback is mandatory. Models live in `widgets_values`; no
      embedded download URLs in our presets → URLs come from catalog/user.
- [x] `comfy_resolve.py`: `workflow → { node_packs[], models[], unresolved[],
      unknown_nodes[], ambiguous_nodes[], notes[] }`. Stdlib + urllib only.
      Handles UI-graph **and** API format (degraded: API has no provenance).
- [x] **Nodes**: active nodes (mode 0) only; resolve via (1) embedded `aux_id`
      → `github.com/<aux_id>`, (2) curated seed map for the LTX/Wan ecosystem,
      (3) ComfyUI-Manager `extension-node-map.json` (cached) for the long tail —
      which also translates `cnr_id` nodes to github URLs and identifies core
      nodes (mapped to `comfyanonymous/ComfyUI`). Canonical-owner tiebreak picks
      the official repo over forks; genuine collisions flagged ambiguous;
      no-pack classes flagged unknown.
- [x] **Models**: model-extension widget values on active nodes; `dest` folder
      from loader type (checkpoints/loras/vae/text_encoders/…), subpaths
      preserved; basename-dedup with folder priority (one physical file ref'd by
      many loaders → downloaded once); URL via embedded `properties.models` →
      Manager `model-list.json` → else `unresolved`; preprocessor weights tagged
      auto-download so they're not falsely required.
- [x] CLI: `comfy_launch.py --resolve workflow.json [--resolve-out draft.preset.json]
      [--no-fetch]` — prints a report and writes a draft preset with `REPLACE_*`
      markers for gaps.
- [x] **Validation harness**: `comfy_launch.py --resolve-selftest [--no-fetch]`
      resolves every shipped preset's workflow and diffs vs its hand-written
      lists; exits non-zero if a *declared model* is missed. **Result: PASS —
      0 missed declared models across all 6 presets** (offline & live).

> **Phase 1 findings (carry into Phase 2/3):**
> - Model resolution is reliable: the hard correctness signal (every declared
>   weight found, with the right dest) is green on all presets.
> - The hand-written `node_packs` lists had **drifted** from their graphs. Two
>   were *missing a pack the graph needs* (runtime-failure bugs, now fixed):
>   `ltx_faceswap` lacked `RES4LYF` (the `bong_tangent` scheduler — see below);
>   `ltx_control` listed `kijai/DepthAnythingV2` but the graph uses
>   `yuvraj108c/Video-Depth-Anything` (swapped). Others list *extra/unused* packs
>   (harmless — just slower cold start; left as-is). `--resolve-selftest` now
>   shows no preset missing a needed pack (faceswap's `Ollama-Describer` is the
>   one intentional exception — vestigial, can't run on Spark).
> - **Resolver gained `WIDGET_VALUE_PACKS`** (enum→pack): some packs register only
>   a scheduler/sampler *value* into a core node's combo (no custom class in the
>   graph), so class-based resolution misses them. `bong_tangent`→RES4LYF is the
>   first entry; extend as new ones surface. Also added `UI_ONLY_NODES` (rgthree
>   muter/bypasser etc. need no pack).
> - Residual `unknown`/`ambiguous` are real edge cases (e.g. rgthree's UI-only
>   "Fast Groups Muter"); `--convert-only` on Spark (`/object_info`) stays the
>   authoritative final check.
> - **Open**: `cnr:<id>` fallback (registry id with no github in the map) isn't
>   cloneable by `comfy_run.fetch_nodes` — needs a cnr→github step or comfy-cli
>   install. Didn't trigger on our presets but will on arbitrary BYO workflows.

## Phase 2 — Close the loop, still CLI
- [ ] Chain `--resolve` → review/fill gaps → existing `--convert-only` (free,
      surfaces any missed node classes) → render. Prove a real BYO workflow runs
      end-to-end from the terminal.
- [ ] (Later) Snapshot/lockfile mode — pin git hashes for reproducible bundles
      instead of `clone --depth 1` of HEAD.
- [ ] Doc the known limit: UI→API converter is flat-graph only; subgraph
      workflows (VOID-style) need a flattening pass first — out of scope for v1.

## Phase 3 — tunet-web wrap (only after Phase 1–2 are solid)
- [ ] API route: upload workflow → invoke the **same** `comfy_resolve.py` (shell
      out to the deployed repo, or a tiny CPU Spark job that writes the manifest to
      ShareSync like convert-only) → return `{resolved, unresolved}`. Do **not**
      reimplement resolution in TS.
- [ ] Gap-fill UI: upload, show resolved nodes/models, paste URLs for unresolved,
      "Save as preset."
- [ ] Extend `loadComfyPresets()` to merge **user-scoped** presets with built-ins
      so a BYO workflow runs the exact same submit path.
- [ ] Parity check: `comfy_ui.py` (tkinter) can also consume a resolved preset.

## Cross-cutting
- [ ] **Generalize subgraph flattening**: `presets/wan22.derive.py` has a working
      single-subgraph flattener (boundary inputs = inner links from `origin_id
      -10`, outputs = `target_id -20`; promoted widget values already sit on the
      inner nodes, so the boundary widget links just drop). Promote it into
      `comfy_run.py`'s UI→API converter so subgraphed workflows (WAN 2.2 i2v,
      VOID, and BYO graphs) convert without a manual derive step. Today the
      converter is flat-only — the same gap the VOID roadmap and Phase 2 flag.
- [ ] Cost guard: resolution + convert-only must never spin a GPU; gate GPU spend
      behind "all models resolved."
- [ ] UI expectation-setting: "auto-resolve what we can, fill the rest" — model
      filenames are not addresses, so 100% hands-off isn't promised.

---

## Face-swap engines — VividFace sibling pipeline (VALIDATED on Spark 2026-05-23)

A **non-Comfy** face-swap engine added beside EZ-Comfy, mirroring the `lora_train`
trio (NOT a ComfyUI workflow → does not use the resolver / `comfy_run.py`):
`faceswap.py` (launcher, reuses `comfy_launch` spine, `cpfx_faceswap`+`cpfx_comfy`
tags), `faceswap_run.py` (in-container: clone → build → fetch weights → preprocess →
run the engine's own `infer.py` → flatten swapped mp4 to `/output` → self-upload),
`swappers/vividface.swap.json` (recipe). Built because the user's pasted face-swap
research named **VividFace** the Tier-1 quality yardstick.

- **Run it:** `python faceswap.py --recipe vividface body.mp4 --face id.png --frame-cap 40 --name <x> --idle-hold 300`
  (or `--examples` for the bundled-sample proof, or `--dry-run` to spend nothing).
- **R&D-ONLY — never a billed shot.** Academic weights + WebFace42M identity encoder
  + SD-1.5 (OpenRAIL) + InsightFace default detector. `rd_only:true` → launcher prints
  a NOT-FOR-DELIVERY banner. Deliverable face-swap stays `ltx_faceswap` (LTX-2 is
  commercial-OK under <$10M ARR) or training your own on rights-cleared data.
- **MUST be L40S (Ada sm_89), NOT Blackwell** — VividFace pins torch 2.4.1 (kernels
  ≤ sm_90); RTX PRO 6000 Blackwell (sm_120) fails "CUDA capability not compatible".
  Recipe defaults `gpu:l40s`. Base `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`.
- **Validated:** `--examples` renders 4 swaps; custom real-footage path
  (transcode → YuNet square 512 face-crop → 14-val annotations → infer) runs on a 2K
  plate. Hard-won fixes are in the `comfy-spark-vividface` memory / the file headers
  (git-before-clone, weights nesting, recursive BFM+epoch_20 relocate, face3dvae2
  alias, SD-1.5 `model_index.json` marker, square-crop geometry, outputs-dir keep).
- **Custom output is a 512 FACE-CENTRIC clip, not a full-plate composite** —
  VividFace's inference hard-reshapes to 512×512 and isn't built to repaint a full
  plate. Fine for quality evaluation; a full-plate deliverable would need a
  crop→swap→paste-back layer (or just use `ltx_faceswap`).
- **Web visibility:** jobs carry `cpfx_comfy` so tunet-web's job-detail shows the
  media outputs panel; the swapped mp4 is flattened to the ShareSync top level (the
  files API lists one level only). No web deploy needed (deployed app already detects
  `cpfx_comfy`).

### Still TODO on this engine
- [ ] **Judge the swap quality** on the named run's output (PF0013) — is VividFace's
      512 swap worth pursuing vs `ltx_faceswap`? (the actual go/no-go).
- [x] **Tier-2 restoration DONE**: `faceswap.py --restore --download DIR` runs the swap,
      then auto-fires a `seedvr2` upscale job on the swapped mp4 (→ `DIR/restored/`).
      Required a `pip_extra` hook (comfy_launch sets `COMFY_PIP_EXTRA`, comfy_run installs
      after node reqs) to pin `diffusers==0.35.0`/`transformers==4.55.4` and fix SeedVR2's
      `flash_attn` import break. Validated: PF0013 512 swap → 1080 upscale (115s, rtxpro6000).
- [ ] **Tier-3 license-clean path**: detector default is already YuNet (clean); note
      ByteDance **LVFace** (MIT) as a clean identity-embedding option; real delivery =
      train an equivalent on consented data.
- [ ] **CLI↔GUI↔web parity**: faceswap is CLI-only today (like `lora_train`). Expose in
      `comfy_ui.py` (tkinter) and a tunet-web submit path.
- [ ] **Rollout**: `-SkipWeb` repo deploy so prod can launch it from the deployed repo.

---

## Shipped-but-untested (verify before relying on it)

Work landed in the repo but **never run against live ComfyUI/Spark or a browser**.
Verified only structurally (graph integrity, `tsc --noEmit`). This is the "haven't
tested yet" backlog — clear these before treating the features as working.

### `seedvr2` preset + generic BATCH (`--batch` / web Batch toggle) — built, never run on Spark
(SeedVR2 diffusion video upscaler — numz node, IceClear/ByteDance model. Graph cleaned
from the upstream HD-video example by `presets/seedvr2.derive.py` (7 nodes, validates).
Generic batch engine: `comfy_run.run_batch` loops the converted graph over many inputs
on one warm node; `comfy_launch.py --batch` + web submit build an identical COMFY_BATCH
manifest. `ltx_hdr` + `ltx_Obscura_Remova` gained `output_prefix`/`input_anchor` so they
batch too. Resolver taught the SeedVR2 loaders → `models/SEEDVR2` (selftest PASS).)
- [x] `--resolve-selftest --no-fetch` PASS (0 missed models; seedvr2 model dests match).
- [x] CLI dry-runs: seedvr2 single + mixed batch (2 video + EXR + PNG folders); ltx_hdr
      single + batch (mp4 prefix + per-item EXR `output_dir`). `tsc --noEmit` clean.
- [ ] **`--convert-only` on Spark** for `seedvr2` — first proof the cleaned graph converts
      (SeedVR2VideoUpscaler/LoadDiTModel/LoadVAEModel via `/object_info`, V3 IO, all scalar
      widgets — no DynamicCombo, so the flat converter should handle it).
- [ ] **One real render**: `seedvr2 shot.mp4` on `rtxpro6000` (default model is now **7B-fp16
      ~15 GB**) — confirm the node pip-installs its reqs (diffusers/peft/rotary_embedding_torch/
      gguf), the 7B+VAE download to `models/SEEDVR2`, and the upscale + CreateVideo/SaveVideo emit
      an mp4. (3B / 7B-fp8 are selectable; they download on first use. Tune blocks_to_swap for l40s.)
- [ ] **Auto mp4 preview** (`comfy_run.generate_previews`): confirm ffmpeg on the image encodes
      a `*.preview.mp4` for an image-sequence output (`--output exr32`/png16) and a ProRes `.mov`
      (`--output prores_hq`), and that the web panel plays it. Verify the EXR linear→sRGB tonemap
      (`-apply_trc iec61966_2_1`, gamma fallback) looks right; if the build lacks the filter it
      degrades to the gamma path (and the preset's own mp4 still covers most v2v presets).
- [ ] **Batch render end-to-end**: confirm the model loads ONCE across items, the per-item
      output prefix lands distinct files, and a SEQUENCE item swaps `LoadExrSequence` /
      `VHS_LoadImagesPath` in at `input_anchor` (GetVideoComponents) + the orphaned
      `LoadVideo` is pruned. Watch for: VHS `VHS_LoadImagesPath` `directory` arg name; the
      `4n+1` batch_size validation; 7B picked in the form auto-downloading.
- [ ] **Web batch round-trip**: multi-video + image-sequence folder upload (comfy_input /
      comfy_seq roles) → submit `batch:true` → outputs panel groups per input.
- [ ] **HDR EXR batch**: the web files API only lists top-level outputs, so per-item EXR
      subfolders (`output/<stem>_hdr_exr/`) show via ShareSync/CLI, not the browser — the
      mp4 previews group per input in the browser. Confirm no EXR collisions across items.
- [ ] **Rollout**: needs a `-SkipWeb` repo deploy (preset/derive/`comfy_run.py`) AND a
      `-SkipRepo` web deploy — same split as the other EZ-Comfy features.

### `ltx_faceswap` preset — first Spark run FAILED, root-caused + fixed; re-run pending
(LTX-2.3 video face/head swap, Alissonerdx BFS V3 LoRA; body clip + `--face` image.
Graph derived by `presets/ltx_faceswap.derive.py`.)
- [x] **First run failed at prompt validation** — output 341 (`SaveVideo` "Result")
      rejected: `BasicScheduler 575: scheduler 'bong_tangent' not in [...]`.
      Root cause: `bong_tangent` is a **RES4LYF** scheduler, but RES4LYF was
      missing from `node_packs` (it contributes only an enum value, no node class,
      so nothing flagged it). **Fixed:** added `ClownsharkBatwing/RES4LYF` to the
      preset, and taught the resolver `WIDGET_VALUE_PACKS` (enum→pack) so it now
      detects this class of dependency. 341's render path depends on no muted node
      and none of the Ollama/switch subsystem (traced), so this should unblock it.
- [x] **Second run** got past `bong_tangent` (RES4LYF fix worked — output 341
      validated) but the whole prompt was still rejected by a `KeyError` in
      `validate_inputs`: the vestigial `ComfySwitchNode`s (493/516) referenced
      inner nodes the flat converter never emitted. An unhandled `KeyError`
      (unlike a graceful "Value not in list") rejects the ENTIRE prompt.
- [x] **Muted the vestigial Ollama path in `derive.py`** (regenerated workflow):
      the cluster did NOT auto-prune because it terminates in three `PreviewAny`
      OUTPUT nodes (426/507/572) — the converter keeps anything feeding an output.
      Added 426/507/572 + the two `ComfySwitchNode`s (493/516) + `OllamaVideoDescriber`
      (586) to `derive.py`'s MUTE list. Verified: only active output is 341, 341
      has zero muted ancestors, selftest still PASS.
- [ ] **RE-RUN** `--convert-only` then a real render — should now convert to a
      single-output graph and render end-to-end (first real proof).
- [ ] **Conversion of the flattened graph**: `derive.py` flattened ~68 KJNodes
      Set/Get virtual nodes + reconnected through 2 bypassed LoRA loaders. Confirm
      `comfy_run.py`'s UI→API converter actually emits a valid API graph from it.
      (First run converted to 40 nodes OK — so this is largely de-risked, but the
      `bong_tangent` failure was downstream of conversion.)
- [x] All **5** node packs clone + register — `alisson-anjos/ComfyUI-BFSNodes`
      (`ReservedRegionFrameComposer`), LTXVideo, KJNodes (note: `LTXVAddGuideMulti`
      is actually a KJNodes class, not LTXVideo), VHS, **RES4LYF** (schedulers).
- [ ] **Distilled-transformer substitution**: I pinned Kijai's
      `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled` in place of Alisson's
      unverifiable private `..._input_scaled_v3` build. Confirm the V3 head-swap
      LoRA gives good likeness on it; swap to the exact build if not.
- [ ] Head-swap LoRA found at the subfolder dest
      `models/loras/ltxv/ltx2/head_swap_v3_rank_adaptive_fro_098.safetensors`
      (subfoldered `lora_name` in `LoraLoaderModelOnly`).
- [ ] **Audio**: a body clip *with* audio drives the AV-latent path (lip motion);
      confirm a *silent* clip doesn't hard-fail `LTXVAudioVAEEncode`.
- [ ] Output is the single `SaveVideo` "Result" (node 341); the 2 comparison
      combiners (365/559) + dead upscaler loader (418) were muted — confirm intended.
- [ ] Uncensored path (re-enable node 424 + supply the abliterated-gemma CLIP LoRA
      weight) is unpinned and untried.

### EZ-Comfy two-input support (web) — built, never run end-to-end
(Generalized secondary input: `comfy_input2` stage role + `ui.secondary_input`
form picker + `comfySecondaryParam`/multi-input packer. Files: `upload-stage`
route+lib, `comfy.ts`, `comfy/submit/route.ts`, `comfy-form.tsx`.)
- [ ] Full browser→stage→submit→Spark round-trip for a 2-input preset
      (face image alongside the body clip).
- [ ] Both files land in `/input` by bare basename; patches map the secondary onto
      its node (`face`→269 for faceswap, `mask`→215 for VACE).
- [ ] 422 gate fires when a 2-input preset is submitted without its secondary file;
      **regression check** the 4 single-input presets still submit unaffected.
- [ ] **`wan_vace_inpaint` static mask now ships via the web** (was silently a
      no-op before this) — confirm it actually inpaints.
- [ ] Mask-**video** variant (`workflow_mask_video`, per-frame masks) is **not**
      wired on the web side — static image masks only. Out of scope here; track it.
- [ ] Only `tsc --noEmit` was run (clean). `next build` (full lint + typecheck) and
      `next lint` (not configured in-repo) were not.
- [ ] `comfy_ui.py` (tkinter) face picker added for parity but not run.

### LoRA training (`lora_train.py` / `lora_train_run.py` / `trainers/`) — built, never run
The train→generate loop sibling of comfy_spark (May 2026): point it at a folder of
images + a `--trigger`, it runs the official Lightricks **ltx-trainer** on Spark and
the trained `.safetensors` lands in ShareSync to feed `comfy_launch.py --lora <url>`.
Mirrors the comfy_launch/comfy_run/preset trio; reuses the same auth/billing
(`cpfx_tunet` + `cpfx_train`)/ShareSync spine. Dry-run validated; all URLs filled
(dev checkpoint + ungated `unsloth/gemma-3-12b-it` model dir, snapshot-downloaded).
- [ ] **First real run**: image-only `ltx2_style` on `rtxpro6000`. Confirm `uv sync`
      builds the trainer env on the `yanwk` cu130 image; if a CUDA-extension build
      fails (slim image, no nvcc), switch the recipe `image` to a CUDA-13 `-devel` base.
- [ ] **Pin the trainer version**: confirm `process_dataset.py` / `train.py` CLI +
      flags + YAML schema match the installed commit (built against HEAD docs; pin a
      git ref in the clone if upstream drifts).
- [ ] **Gemma encoder**: confirm `--text-encoder-path` accepts the model dir and
      `unsloth/gemma-3-12b-it` loads cleanly in the trainer.
- [ ] **Auto-caption**: BLIP sanity check on the node (transformers + Pillow in the
      uv env?); else pip-install in `CAPTION_SNIPPET` or use `--no-auto-caption`.
- [ ] **Round-trip**: trained LoRA loads + stacks via
      `comfy_launch.py --preset ltx2_generate --lora <ShareSync url>`.
- [ ] **GUI parity**: expose the trainer in `comfy_ui.py` (recipe already carries
      `ui` metadata; CLI-only today).

### `wan22_i2v` preset — built (subgraph flattened), never run
(Wan 2.2 14B image-to-video, 4-step lightx2v Lightning, two-expert MoE. Additive —
`wan_vace_inpaint` stays the masked-inpaint tool. The official template is
**subgraphed**; `presets/wan22.derive.py` flattens it offline into
`wan22_i2v.workflow.json` since the converter is flat-only. All URLs real,
Comfy-Org repackaged; param map + model coverage validated structurally.)
- [ ] `--convert-only` on Spark, then a real render — first end-to-end proof the
      flattened graph converts + the two-expert/Lightning split renders.
- [ ] Confirm `wan2.2_i2v_*_14B_fp8_scaled` fits the chosen GPU (rtxpro6000 96GB
      ample; l40s 48GB likely OK since experts load sequentially).
- [ ] (Maybe) WAN 2.2 t2v + Fun Control / Fun InP — no flat official template yet;
      they'd need flattening too (see the subgraph note in Cross-cutting).

### Output-format system (`--output`, `outputs/outputs.json`) — built, never run
(High-bit output as a choice instead of assumed 8-bit mp4. `comfy_run.rewrite_output`
splices the chosen saver as a **parallel branch** onto the auto-detected frames
anchor — EXR 16/32-bit scene-linear via CoCoTools `ColorspaceNode`→`SaverNode`,
or ProRes ladder via core `CreateVideo`→`SaveVideoHQ`. mp4 stays the default.
Validated **structurally against a real 40-node converted graph** (anchor found,
chains splice, all links resolve) + launcher dry-runs; GUI dropdown added.)
- [ ] **First Spark run** per format: `--output exr32` then `prores_hq` on
      `wan22_i2v --convert-only`, then a render. Confirm the spliced nodes register
      and write files.
- [ ] **CoCoTools_IO deps**: confirm its `requirements.txt` (OpenEXR/Imath/cv2/
      colour-science) pip-installs cleanly via `fetch_nodes` on the yanwk image.
- [ ] **SaveVideoHQ**: it uses the new `comfy_api` `io.ComfyNode` + needs ffmpeg
      with `prores_ks` — confirm the yanwk cu130 ComfyUI is recent enough and ffmpeg
      has ProRes. Confirm `CreateVideo`→`SaveVideoHQ` (VIDEO input) wiring renders.
- [ ] **Anchor auto-detect** edge cases: presets whose terminal isn't
      CreateVideo/SaveImage/VHS (add an explicit `output_anchor` to the preset then).
- [ ] **Colour**: verify `sRGB`→`sRGB Linear` in `ColorspaceNode` gives a correct
      scene-linear EXR (vs double-applying a transform). Spot-check in Nuke.

### Input image-sequence ingestion (`--input-sequence`) — EXR built, never run
The symmetric input side of the output system. `comfy_launch` auto-detects a
numbered sequence in the folder, packs it to `/input/seq/`, and `comfy_run.rewrite_input`
**swaps the preset's IMAGE-batch loader in place** for CoCoTools `LoadExrSequence`
(IMAGE on slot 0, so consumers keep their links), inserts a `ColorspaceNode`
(scene-linear EXR → display sRGB), and drops the old loader's audio/fps outputs.
Validated structurally on a synthetic VHS_LoadVideo graph (swap, colour insert,
consumer rewire, audio drop, links resolve) + launcher dry-run on `ltx_Obscura_Remova`.
- [ ] **First Spark run**: `--input-sequence ./plate_exr` on a v2v preset; confirm
      `LoadExrSequence` reads `/input/seq/shot_####.exr` and the swapped graph renders.
- [ ] **Colour round-trip**: linear→sRGB on ingest vs sRGB→linear on `--output exr32`
      — verify a plate survives EXR-in → EXR-out without a colour shift (check in Nuke).
- [ ] **HDR / >1.0 plates**: `normalize=False` keeps values; confirm the model path
      doesn't clip super-whites badly (or expose `normalize`).
- [ ] **DPX + PNG/TIFF folders**: only EXR has a loader wired (others error clearly).
      Add DPX (needs a DPX sequence loader — not in CoCoTools) and a folder image
      loader (VHS `LoadImagesPath`) keyed by `seq.kind` in `comfy_run.build_seq_loader`.
- [ ] **VIDEO-object loaders**: presets whose input is core `LoadVideo` (VIDEO out,
      e.g. `wan_vace_inpaint`) aren't swappable in place — would need an IMAGE→VIDEO
      bridge (`CreateVideo`). Currently auto-detect skips them; document/handle.

### Rollout (not done)
- [ ] Preset/derive files need a **`-SkipWeb` repo deploy** to the prod VPS —
      EZ-Comfy reads presets from the *deployed* repo, so `ltx_faceswap` won't
      appear in the dropdown until then.
- [ ] The web changes need a **web deploy** (`-SkipRepo`). (See the deploy-split
      note in the project memory.)

---

## LTX-2 expansion (May 2026) — gap-closing pass vs the official repos

Goal: close the gaps from the EZ-Comfy gap analysis (two-stage upscaler, lipdub,
camera control, detailer, restoration) + a few trending models. All recon done;
weight URLs verified ungated via the HF API. Scratch + downloaded workflows in
`_ingest_work/` (helpers `inspect_wf.py`, `trace.py`).

### Built + free-validated (resolve-selftest PASS) this pass
- **`ltx_two_stage`** — LTX-2.3 base + 2× spatial upscaler. **convert-only GREEN**;
  T2V render fired (rtxpro6000). The headline gap.
- **`ltx_lipdub`** — LipDub IC-LoRA, two-stage, audio-conditioned. **convert-only GREEN**.
- **`ltx19_generate`** + **`ltx19_detailer`** + **`loras/ltx2-19b.loras.json`** (7
  camera LoRAs).
  - **`ltx19_generate` convert BLOCKED (2 subgraph issues):** (1) `KeyError '5230'/'5231'`
    — bypassed camera-LoRA loaders feeding the subgraph; FIXED by `ltx19_generate.derive.py`
    (drop them, rewire subgraph model inputs -> checkpoint 5228). (2) After that, `Required
    input is missing: upscale_model` — FIXED by a GENERAL change to `flatten_subgraphs`:
    map subgraph boundary inputs by SLOT (not the litegraph-suffixed NAME) and rewire any
    boundary input that has a real source link (dropped the `_SUBGRAPH_DATA` type-allowlist
    that missed `LATENT_UPSCALE_MODEL`). Verified locally (all inner inputs wired) + on Spark.
    **`ltx19_generate` convert-only is now GREEN; render in flight.** No offline-flatten needed.
  - **`ltx19_detailer`** FLAT — **convert-only GREEN.**
  - 19B = SEPARATE base (`Lightricks/LTX-2`); gemma is a 16-file HF FOLDER
    (`Lightricks/gemma-3-12b-it-qat-q4_0-unquantized`); dest `models/text_encoders/...`
    CONFIRMED correct from `gemma_encoder.py` (it scans `get_filename_list("text_encoders")`).
- **`wan_vace_ref2video`** — Wan-VACE reference-to-video (3-input: video + `--mask` +
  `--face` reference). Derived from the flat `wan_vace_inpaint.workflow.json`. **convert-only GREEN** (1 combo self-heal).
- **`ltx_restore` (ICEdit-Insight) — convert-only GREEN.** Built with NO derive (dest-rename + prompt-link override). `task` param (node 5011) switches the 4 adapters.
- **`latentsync` — convert-only GREEN.** Native deps install on yanwk; weights in the wrapper's `custom_nodes/.../checkpoints/`; version-drift fix (inference_steps/lips_expression params).
- **ALL 7 NEW PRESETS CONVERT-GREEN.** Renders proven: `ltx_two_stage` ✅. Render in flight: `ltx19_generate`. Render-unverified (need real inputs): `ltx_lipdub`, `wan_vace_ref2video`, `ltx_restore`, `latentsync`, `ltx19_detailer`.
- **Launcher papercut found:** `--convert-only` does NOT exit after printing
  `VALIDATION OK` (keeps following the finished job) → wastes the run until killed.
  Fix: have convert-only exit on the job's terminal status. Validate via streamed
  output + `timeout`, NOT `| tail` (tail buffers until exit → zero visibility).

### TODO — ICEdit-Insight (joyfox, #1 trending) — restoration/HD/watermark/subtitle
Standalone `run_pipeline.py` (`github.com/Valiant-Cat/LTX2-ICEdit-Insight`) that ALSO
ships ComfyUI workflows in its `workflows/` folder. **Decision still open: ComfyUI
preset vs wrapping run_pipeline.py as a Spark job (like lora_train.py).** Having seen
the graph, the wrap may be cleaner. If doing the ComfyUI preset, use the **official**
version (`LTX-2.3-ICLORA编辑-官方模型版本.json`, in `_ingest_work/icedit_official.json`):
flat (41 nodes, 0 subgraphs, 1 bypass), runs the task adapters on the STANDARD 22B.
- [ ] Weights (all ungated, verified): checkpoint `Lightricks/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors`;
      distill `Lightricks/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors`;
      gemma — workflow wants `gemma_3_12B_it_fp8_e4m3fn` but only `gemma_3_12B_it_fp8_scaled`
      exists at `Comfy-Org/ltx-2` → must patch the in-graph widget to `_fp8_scaled`.
- [ ] Task adapters (joyfox/LTX2.3-ICEdit-Insight, ungated): `ltx2.3-video-restoration-general`,
      `ltx2.3-ic-video-upscale-general`, `ltx2.3-ic-watermark-remove-general`,
      `ltx2.3-ic-subtitles-remove-general`. Graph's widget names (`...-v2`) DON'T match
      → a derive must rewrite the active IC-LoRA loader (**node 5011**; 5133 is an
      inactive alternate) lora_name + expose a `task` param picking 1 of 4.
- [ ] Prompt indirection: prompt feeds via **CR Prompt Text** (Comfyroll) 5132 → CLIPTextEncode 2483;
      map `prompt` to the 5132 widget (confirm field name) or rewire 2483 to a plain box.
      Input video = VHS_LoadVideo **5099**. Negative 2612 (empty).
- [ ] 9 node packs (LTXVideo, VHS, KJNodes, essentials, rgthree, **ComfyUI_LayerStyle**
      [pick over Swwan for the ambiguous `LayerUtility: ImageScaleByAspectRatio V2`],
      Comfyroll, Easy-Use, masquerade). Many are utility/decoration — gate-iterate +
      mute what's not on the active path. Expect 1-2 convert-only heal cycles.

### TODO — LatentSync (ByteDance, audio-driven lip-sync) — SEPARATE ecosystem
Not LTX. ComfyUI node pack `ShmuelRonen/ComfyUI-LatentSyncWrapper` (node `LatentSync1.6`).
Needs a DIFFERENT image (mediapipe / face-alignment / decord / ffmpeg deps — the yanwk
LTX image won't have them; either find a comfy image with these or pip-install via
`fetch_nodes`' requirements). Weights ~8GB (latentsync_unet.pt, stable_syncnet.pt,
whisper tiny, SD VAE) — ByteDance repo is private; use the **chunyu-li/LatentSync** mirror.
Fits 24GB (could even run on the local 3090). Inputs: 25fps frontal-face video + audio.
- [ ] Confirm chunyu-li mirror file paths + the wrapper's expected model dirs.
- [ ] Decide the base image (deps); build preset (video + `--face`?/audio input). Distinct
      from LTX presets — its own `base`. Lower priority than the LTX line.
