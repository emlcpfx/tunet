# Workflow ingest & self-heal pipeline

How comfy_spark takes an arbitrary ComfyUI workflow and gets it running on Spark —
auto-resolving dependencies and **auto-healing** the common ways a graph fails to
build because the installed node packs have drifted from the version the graph
was authored against. Written from the bugs actually hit bringing up the shipped
presets (see `SMOKE_TEST_REPORT.md` for the run log).

## The pipeline (cheap → expensive)

```
workflow.json
   │
   ├─1 RESOLVE deps        comfy_resolve.py  → node_packs + models (+ unresolved)
   │                       (provenance, seed map, ComfyUI-Manager, enum→pack)
   ├─2 PACK + SUBMIT       comfy_launch.py   → tiny tarball, Spark job
   │
   on the Spark node (comfy_run.py):
   ├─3 FETCH               clone node packs (or pinned registry zips) + weights
   ├─4 STAGE INPUTS        copy /input (read-only) → writable dir; silent-audio guard
   ├─5 CONVERT UI→API      ui_to_api(): positional widgets + SELF-HEAL combos
   ├─6 GATE (convert-only) POST /prompt for validation only — no sampling
   └─7 RENDER              queue + sample + save + self-upload
```

**Cost rule:** stages 1–6 are cheap. `--convert-only --gpu t4` runs 1–6 with **no
40 GB weight pull and no GPU sampling** — it catches almost every build failure
(missing pack, wiring, version drift) for pennies. Only run a full render once the
gate is green. This single discipline is what made an agentic fix-loop affordable.

## Self-heal mechanisms (each: symptom → fix → where)

### A. Missing node pack — resolve from the graph
- **Symptom:** a node class isn't installed; `/object_info` lacks it.
- **Heal:** `comfy_resolve.py` derives packs from node `properties` (`aux_id`→github,
  `cnr_id`→registry), a curated seed map, then ComfyUI-Manager's
  `extension-node-map.json`. Plus **enum→pack** (`WIDGET_VALUE_PACKS`): some packs
  contribute only a scheduler/sampler *value* (e.g. `bong_tangent`→RES4LYF) with no
  node class — caught by scanning widget values.
- **Gotcha:** a pack can be needed for a widget enum even when none of its node
  classes appear. Always run the gate after editing `node_packs`.

### B. Combo option drift — self-heal against the live schema  ✅ (fixes ltx_control/2_generate/hdr)
- **Symptom:** `Value not in list: <input>: 'X' not in [...]`. A saved widget value
  no longer matches the installed node's combo options (pack renamed/reordered
  options, or the value belonged to a different input).
- **Heal:** `_heal_combos()` in `comfy_run.py` — after positional widget mapping,
  for every combo input whose value isn't a valid option, swap in a value from the
  node's own widget pool that *is* valid; else the combo default; else drop it
  (node default). **Only fires on invalid combos**, so healthy graphs are untouched.
- **Key details:**
  - Reads BOTH object_info combo formats: v1 `[[opt,...], {meta}]` and v3 IO
    `["COMBO", {"options":[...], "default":...}]`.
  - **Skips empty option lists** — an empty combo = a model dropdown with no
    weights downloaded (convert-only); its value is fine at render time, so never
    "heal" (drop) a model input.
- **Real example:** `ResizeImageMaskNode.scale_method = "scale to multiple"`
  (an old resize-mode value) → healed to `"lanczos"` (the valid interpolation that
  was sitting elsewhere in the widget list).

### C. DynamicCombo inputs — node substitution  ✅ (fixes faceswap)
- **Symptom:** `execute() got an unexpected keyword argument 'X'`, or a dotted
  input name like `num_guides.image_1` / `resize_type.multiple` in the graph. The
  node uses ComfyUI's **DynamicCombo** (a grouped dynamic input the frontend
  assembles into one structured value). The flat UI→API converter can't synthesize
  it, so neither the dotted name nor a de-namespaced name is accepted.
- **Heal — converter support (now implemented).** `ui_to_api` detects a
  `COMFY_DYNAMICCOMBO_V3` input from object_info and serializes the grouped value
  ComfyUI core expects (`_map_widgets_dynamic_combo` in `comfy_run.py`):
  - `inputs[id] = <selected option key>` (e.g. `resize_type = "scale to multiple"`)
  - the SELECTED option's nested inputs as `inputs[id.<nested>]` (the `parent.child`
    prefix), from widgets or links (e.g. `resize_type.multiple = 32`).
  It walks `widgets_values` in `input_order`, expanding the DynamicCombo into its
  selection + the chosen option's nested widgets; a wired nested input just consumes
  its stale value so following widgets stay aligned. Verified end-to-end (convert →
  validate → **execute**) on core `ResizeImageMaskNode`. This closes the whole class
  — both the LTX trio's `resize_type` and faceswap-style `num_guides` (the faceswap
  node-swap is no longer strictly required, though it's left in place).
  - **`_heal_combos` must skip DynamicCombo**: its `options` are `[{key,inputs}…]`
    dicts, not string options — only treat *string*-option lists as combos, else
    the heal corrupts the selection.
- **Alternative (still valid):** when a graph is simpler to edit than the converter,
  swap the DynamicCombo node for a plain-input equivalent in a derive script
  (faceswap: `LTXVAddGuideMulti` → `LTXVAddGuidesFromBatch`).
- **Gotcha:** convert-only (validation) does NOT exercise a DynamicCombo at execute
  time — a graph can `VALIDATION OK` yet fail with `missing required positional
  argument` at render. Confirm DynamicCombo nodes by **executing** them.

> **Iterate the converter LOCALLY, not on Spark.** Converter/format work needs only
> a running ComfyUI for `/object_info` + `/prompt` validation + executing the node —
> and core nodes like `ResizeImageMaskNode` are CPU ops needing no big weights/GPU.
> A local ComfyUI (or the ComfyUI MCP) turns a ~15-min, paid Spark cycle into a
> ~10-second free loop. Reserve Spark for the final full-model render. The
> DynamicCombo fix above was nailed in 3 local loops before any Spark render.

### D. Version drift you must pin — registry version pinning
- **Symptom:** a node's *behavior/signature* changed (not just a combo option), and
  no self-heal recovers it — e.g. `AudioVAE.__init__() takes 2 args but 3 given`
  (an old KJNodes `VAELoaderKJ` against a newer core).
- **Heal:** pin the pack to the version the graph was authored with. A `node_packs`
  entry may be `{ "name": "<dir>", "zip": "https://cdn.comfy.org/<owner>/<id>/<ver>/node.zip" }`
  (Comfy Registry CDN) instead of a HEAD git URL. Get `<ver>` from the graph's
  `extra.node_versions` or a node's `properties.ver`.
- **Caveat:** pinning a pack OLD can break it against a NEWER ComfyUI core (and
  vice-versa). True reproducibility needs a *full snapshot* (core + all packs).
  Pin the minimum that fixes the issue; prefer self-heal (B) where it applies.

### E. Read-only `/input` + silent clips — stage & guard
- **Symptom:** preprocessing input fails (`Read-only file system`); AV-aware graphs
  fail audio extraction on a silent clip (`VHS failed to extract audio, exit 234`).
- **Heal:** `comfy_run.py` copies `/input` (read-only Spark mount) to a writable
  staging dir and points ComfyUI there; `ensure_input_audio()` adds a silent stereo
  track to any audio-less input video (video stream copied — fast/lossless).

### F. Vestigial / dead-end nodes — mute, don't fight
- **Symptom:** outputs depend on muted nodes, or a sub-cluster needs a service that
  doesn't exist on Spark (e.g. an Ollama server), throwing a hard `KeyError` that
  rejects the whole prompt.
- **Heal:** mute the dead leaf outputs in the graph's derive script; the converter's
  reachability prune then drops their exclusive ancestors. (faceswap: muted the
  Ollama/`ComfySwitchNode`/`PreviewAny` cluster.) Note the prune keeps anything
  feeding an output, so you must mute the *output* nodes, not just the interior.

### G. Subgraphs — auto-flattened (now supported)
- `ui_to_api` calls `flatten_subgraphs()` first: it inlines ComfyUI subgraph
  `definitions` into a flat graph. Litegraph encodes the boundary with virtual ids
  (inner links from `origin_id -10` = subgraph inputs, links to `target_id -20` =
  outputs); promoted-widget boundary inputs keep their value on the inner node
  (link dropped), real data wires (IMAGE/LATENT/…) are rewired across the boundary;
  colliding inner ids are renumbered and each node's `inputs[].link` re-synced (the
  converter traces edges via `inputs[].link`, so stale ids would prune the graph).
- Handles **any number** of subgraph definitions/instances and **nesting** (it
  loops until no instance node remains). Generalised from `wan22.derive.py`;
  verified locally to reproduce that proven flattening exactly (17 nodes/22 links,
  identical node-type multiset) and to convert+validate the raw WAN 2.2 subgraph
  template clean. So new subgraphed workflows (WAN 2.2 i2v, VOID, BYO) convert
  without a hand-written derive script.
- The standing per-preset `*.derive.py` scripts still work and are kept, but are no
  longer required just to flatten a subgraph.

## Ingesting a NEW workflow — checklist
1. `comfy_launch.py --resolve wf.json --resolve-out draft.preset.json` — get
   node_packs + models; fill any `REPLACE_` model URLs.
2. `--convert-only --gpu t4` — the cheap gate. Read the log:
   - `Self-healed N combo value(s)` → drift auto-fixed (B); review the choices.
   - `VALIDATION FAILED` with a non-empty combo list → real bug; check for
     DynamicCombo (C), a signature mismatch needing a pin (D), or a missing pack (A).
   - `unexpected keyword argument` / dotted input → DynamicCombo (C).
   - `KeyError` in validate_inputs → a kept node references a dropped one (F).
3. Iterate cheaply on the gate until `VALIDATION OK`.
4. One full render to catch execution-time issues (audio E, OOM, sampler).
5. Capture the converted `workflow_api.json` (written every run) to reuse.

## What's automatic vs. needs a human
- **Automatic:** pack resolution (A), combo drift (B), input staging + silent audio
  (E), reachability prune (F).
- **Per-graph workaround (scripted in a derive.py):** DynamicCombo node swap (C),
  subgraph flattening (G), muting dead clusters (F).
- **Needs a decision:** version pins (D) and unresolved model URLs (registry/HF
  gated weights). The pipeline surfaces these clearly rather than guessing.
