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

## Shipped-but-untested (verify before relying on it)

Work landed in the repo but **never run against live ComfyUI/Spark or a browser**.
Verified only structurally (graph integrity, `tsc --noEmit`). This is the "haven't
tested yet" backlog — clear these before treating the features as working.

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

### Input image-sequence ingestion (EXR/DPX in) — NOT built yet (next step)
The symmetric input side of the output system, and the other half of the user's
"exr/dpx general image sequence support" ask. Design (mirror the output rewrite):
- [ ] `--input-sequence DIR` packs a frame folder into the tarball (`/input/seq/`).
- [ ] `comfy_run` rewrites the graph's primary input loader → a sequence loader
      (`LoadExrSequence` from CoCoTools_IO for EXR, or a folder image loader),
      rewiring its IMAGE output to wherever the old loader fed (the input analog of
      `find_output_anchor` / `output_anchor`).
- [ ] Colour on **ingest**: EXR plates are scene-linear → convert linear→sRGB for
      the model, then the output side converts back. Get the round-trip right.
- [ ] DPX in (10-bit log) via CoCoTools too.

### Rollout (not done)
- [ ] Preset/derive files need a **`-SkipWeb` repo deploy** to the prod VPS —
      EZ-Comfy reads presets from the *deployed* repo, so `ltx_faceswap` won't
      appear in the dropdown until then.
- [ ] The web changes need a **web deploy** (`-SkipRepo`). (See the deploy-split
      note in the project memory.)
