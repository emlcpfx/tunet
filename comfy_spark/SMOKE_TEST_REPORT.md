# comfy_spark — smoke-test results (agentic run, 2026-05-22)

Goal: get every shipped preset running start→finish on Spark, or document why it
can't. I launched/diagnosed/fixed/re-ran the Spark jobs autonomously. The
self-healing ingest patterns are written up in `WORKFLOW_INGEST.md`.

## Scoreboard

| preset | result | proof / blocker |
|--------|--------|-----------------|
| **ltx_Obscura_Remova** | ✅ WORKS | rendered mp4, GPU 99% |
| **wan_vace_inpaint** | ✅ WORKS | rendered mp4 (video + mask) |
| **wan22_i2v** | ✅ WORKS | rendered mp4 (after input-wiring fix) |
| **ltx_faceswap** | ✅ WORKS | **real render of your PF0013 plate + screenshot** + smoke render |
| **ltx2_generate** | ✅ WORKS | rendered mp4 (GPU 100%) — DynamicCombo fix confirmed |
| **ltx_control** | ✅ WORKS | rendered mp4 (512×512, len 49) — DynamicCombo + IC-LoRA path |
| **ltx_hdr** | ✅ WORKS | **41 float16 EXR frames + mp4** — full HDR pass |

**🎉 7 of 7 render end-to-end.** Both converter gaps that blocked the LTX presets —
**DynamicCombo** *and* **subgraphs** — are SOLVED in the converter (below). ltx_hdr's
issues were all environmental (gated-weight access, then OpenCV EXR), now fixed.

Outputs: `comfy_spark/_dl/r_obscura/`, `_dl/r_wan22b/`, `_dl/r_wanvace/`,
`_dl/faceswap_REAL/` (+ `_dl/faceswap_smoke5/`).

## Pipeline hardening shipped (helps every preset)
In `comfy_run.py` / `comfy_launch.py`, all verified live:
1. **convert-only is a real pass/fail gate** (exits non-zero on a true validation
   failure) — run with `--gpu t4`: no 40 GB pull, no sampling. Catches build bugs
   for pennies before a paid render.
2. **Combo self-heal** (`_heal_combos`): repairs widget values that no longer match
   the live node schema (option rename/reorder); reads both object_info combo
   formats (v1 `[[...],{}]` and v3 `["COMBO",{options}]`); skips empty (model)
   dropdowns so it never drops a valid model input. Only fires on invalid combos.
3. **Silent-input audio guard** + **writable input staging** (the Spark `/input`
   mount is read-only) — verified by the real faceswap on your silent plate.
4. **Registry-version node pinning** (`node_packs` entry `{name,zip:cdn.comfy.org/…}`).
5. **HF-token passthrough** for gated weights (`HF_TOKEN`→`COMFY_HF_TOKEN`→bearer
   header) — confirmed: it pulled the gated 44 GB `Lightricks/LTX-2.3` checkpoint.
6. **Resolver** gains enum→pack (`bong_tangent`→RES4LYF) and DynamicCombo notes.
7. **Security:** secrets redacted from the printed plan + scrubbed from 27 local
   logs; `.env`/`_dl` gitignored. (See the security discussion: keep `HF_TOKEN`
   fine-grained read-only — it lives in the Spark job env during a run.)
8. **Converter: DynamicCombo serialization** (`_map_widgets_dynamic_combo`) and
   **subgraph flattening** (`flatten_subgraphs`) — the two flat-only gaps, both
   verified on a local ComfyUI (convert→validate→execute) before any Spark render.
9. **`OPENCV_IO_ENABLE_OPENEXR=1`** set for the ComfyUI process so HDR / high-bit
   savers can write EXR (ltx_hdr; also unblocks the EXR output format generally).
10. **Fail-fast gated-weight precheck** (`precheck_model_access`): a 1-byte Range
    request per model URL before any big pull — a gated 401/403 aborts the job with
    a clear message instead of after a 40+ GB download of the others.
11. **Preset `static_inputs`**: a preset can bundle static files (packed by
    basename). `ltx_control` uses it to ship a placeholder for its bypassed init
    image (node 2004), so it runs with just the control video — no `--input`/`--set`
    workaround. Its width/height defaults are now ÷64 (576) for the IC-LoRA's even
    latent (was 544 → odd latent → guide error).

## faceswap — DONE (6 bug-layers fixed)
Real render of your `PF0013-03_Clip-3_REC709_2K.mov` + screenshot succeeded
(GPU 100%, 8 s output). Layers fixed, in order:
1. `bong_tangent` scheduler → added **RES4LYF** + resolver enum→pack.
2. `KeyError` → muted the vestigial Ollama/`ComfySwitchNode`/`PreviewAny` cluster
   in `ltx_faceswap.derive.py`.
3. silent plate + read-only `/input` → audio guard + writable staging.
4. `LTXVAddGuideMulti` **DynamicCombo** (`num_guides`) → swapped to
   `LTXVAddGuidesFromBatch` (plain `images`+`strength`) in the derive script.
5. UNET re-synced to the exact `…_input_scaled_v3` build the V3 graph specifies.
> Note: your downloaded clip used the *old* substituted UNET (rendered before the
> correction). Re-render for best likeness on the now-correct `input_scaled_v3`.

## DynamicCombo — SOLVED in the converter
All three LTX presets use `ResizeImageMaskNode` (ComfyUI **core**) whose
`resize_type` is a **DynamicCombo** (a grouped value: a mode like
"scale to multiple" + its dimension sub-input). The flat converter didn't build
it, so execution failed `missing 1 required positional argument: 'resize_type'`.

**Fix implemented** (`_map_widgets_dynamic_combo` in `comfy_run.py`): detect a
`COMFY_DYNAMICCOMBO_V3` input from object_info and emit the format core expects —
`inputs[id] = <selected option key>` + the selected option's nested inputs as
`inputs[id.<nested>]` (e.g. `resize_type="scale to multiple"`,
`resize_type.multiple=32`, `scale_method="lanczos"`). Also fixed `_heal_combos` to
skip DynamicCombo (its `options` are dicts, not string values — healing corrupted
the selection). **Closes the whole class** (the trio *and* faceswap-style nodes).

**Verified the fast way — locally, free, in seconds** (the key process win): the
user's local ComfyUI gave a convert → `validate_workflow` → **`enqueue` (execute)**
loop on core `ResizeImageMaskNode` — confirmed in three ~10s loops before any paid
render. **ltx2_generate then rendered end-to-end on Spark (GPU 100%).**

**Residual, non-converter items:**
- **ltx_control**: the resize + IC-LoRA conditioning all executed; the only failure
  was my shrink size 512×288 → odd latent (288/32=9), and the IC-LoRA guide needs
  latent ÷2 (output dims ÷64). Re-running at 512×512 to confirm. **Finding:** the
  preset's default height **544** is *not* ÷64 (544/32=17, odd) — likely needs to
  be a multiple of 64 for the IC-LoRA path. And `ltx_control` has a *second* input
  (`LoadImage` node 2004) the preset doesn't map — wants a `--face`-style
  secondary-image param (worked around here via `--input` + `--set`).
- **ltx_hdr**: converted + downloaded fine, then **HTTP 403** on the gated HDR
  IC-LoRA (`Lightricks/LTX-2.3-22b-IC-LoRA-HDR`). The HF token reaches it but lacks
  access → **accept that repo's license** on HF for the token's account.
