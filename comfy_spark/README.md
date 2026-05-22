# comfy_spark — run ComfyUI workflows on Spark Fuse

A command-line tool to submit a [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
workflow as a batch job on **Spark Fuse**, the same way [`spark_launch.py`](../spark_launch.py)
submits TuNet training: auth → pack a tarball → `POST /api/compute/jobs` →
`PUT` the upload → stream logs. Outputs land in ShareSync.

It has two layers:

- **Generic runner** — feed it any API-format workflow (ComfyUI → *Save (API
  Format)*), the media it references, and optional per-node overrides.
- **Presets** — named bundles under [`presets/`](presets/) that supply the
  image/GPU/workflow and map friendly flags (`--prompt`, `--strength`,
  `--mask`, …) onto specific nodes. Ships with:
  - **`ltx_Obscura_Remova`** — LTX-2.3 + the [Obscura Remova](https://huggingface.co/WepeNerd/Obscura_Remova)
    LoRA (video-to-video occlusion / foreground removal, no mask).
  - **`ltx2_generate`** — LTX-2.3 22B image/text-to-video **generation**, the
    natural home for stacking effect/style LoRAs. Ships as a *template* (fill in
    two weight URLs once — see below).
  - **`ltx_hdr`** / **`ltx_control`** — Tier-2 **IC-LoRA** presets (HDR grade;
    canny/depth/pose structural control). Templates — see *IC-LoRA presets*.
  - **`wan_vace_inpaint`** — Wan2.1 VACE 14B video inpainting / object removal,
    guided by a **mask** (static image *or* per-frame mask video). See below.
  - **`ltx_faceswap`** — LTX-2.3 22B video **face / head swap** ([Alissonerdx's
    "BFS" V3](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video)
    head-swap LoRA). A body/motion clip **+ a face reference image** (`--face`).
    All weights pinned — runs as-is. See below.
- **LoRA stacking** — a curated [`loras/`](loras/) catalog of drop-in LTX-2 LoRAs
  plus `--lora name[:strength]` (repeatable, or pass a raw `.safetensors` URL).
  Triggers are auto-added to the prompt. See *Stacking LoRAs* below.
- **Desktop UI** — [`comfy_ui.py`](comfy_ui.py), a tiny tkinter front-end that
  renders only the knobs each preset declares (with `[?]` tooltips), a file
  picker, and a LoRA-stack picker. See *Desktop UI* below.

## Files

| File | Runs where | What it does |
|------|-----------|--------------|
| `comfy_launch.py` | your machine | the CLI: pack + submit + tail |
| `comfy_ui.py` | your machine | optional tkinter desktop front-end over the CLI |
| `comfy_run.py` | inside the container | starts ComfyUI headless, converts UI→API if needed, splices the LoRA stack, queues the workflow, polls to completion |
| `lora_train.py` | your machine | **train** your own LTX-2.3 LoRA: pack a dataset + recipe, submit, tail, download the result. See *Training your own LoRA* |
| `lora_train_run.py` | inside the container | runs the official Lightricks ltx-trainer (clone → uv sync → caption → preprocess → train → self-upload the LoRA) |
| `Dockerfile` | build step | ComfyUI + LTX node-pack image (a starting point) |
| `presets/*.preset.json` | — | preset bundles (image/GPU/workflow + flag→node maps) |
| `trainers/*.train.json` | — | training recipes (rank/steps/lr + base weights + GPU) for `lora_train.py` |
| `loras/ltx2.loras.json` | — | curated catalog of stackable LTX-2.3 LoRAs |

## Auth

Same as `spark_launch.py` — put `SPARK_EMAIL` / `SPARK_PASSWORD` in the repo
`.env`.

## Billing tags

Every job is tagged **`cpfx_tunet`** (mandatory — this is what Spark bills under
the Clean Plate FX cost + revenue-share deal; without it you pay standard rates)
**plus `cpfx_comfy`** so ComfyUI jobs are filterable. `--list-jobs` filters on
`cpfx_comfy`. Add more with `--tag`.

## Quick start (generic)

```bash
python comfy_spark/comfy_launch.py \
    --workflow my_workflow_api.json \
    --input clip.mp4 \
    --image <registry>/comfy-ltx23:latest --gpu rtxpro6000 \
    --set 6.inputs.text="remove the curtains" \
    --set 14.inputs.strength_model=1.3
```

`--input` files are referenced inside the workflow by **bare basename** (the
container mounts them at `/input`, which ComfyUI sees as its input directory).

## Useful commands

```bash
# plain-language catalog of the shipped presets (what each does, what it takes,
# the key knobs, and a link to the canonical docs). Add a tag to filter.
python comfy_spark/comfy_launch.py --list-presets
python comfy_spark/comfy_launch.py --list-presets face      # only face-swap, etc.

# list node ids/types of a workflow (to build a preset map or find --set targets)
python comfy_spark/comfy_launch.py --inspect my_workflow_api.json

# patch + print the workflow and submit plan WITHOUT submitting
python comfy_spark/comfy_launch.py --preset ltx_Obscura_Remova clip.mp4 --prompt "..." --dry-run

python comfy_spark/comfy_launch.py --list-skus
python comfy_spark/comfy_launch.py --list-jobs          # only cpfx_comfy jobs
python comfy_spark/comfy_launch.py --logs   <JOB_ID>
python comfy_spark/comfy_launch.py --cancel <JOB_ID>
```

## The `ltx_Obscura_Remova` preset

The actual Obscura graph is bundled as
[`presets/ltx_Obscura_Remova.workflow.json`](presets/ltx_Obscura_Remova.workflow.json)
(the UI/graph export from
[axiomgraph/ComfyUIWorkflow](https://github.com/axiomgraph/ComfyUIWorkflow)),
and the preset's `params` node ids are already mapped to it.

You do **not** need a local ComfyUI. `comfy_run.py` converts the UI graph to API
format on Spark using the image's own `/object_info` schema, then applies the
preset's edits. The converted `workflow_api.json` is written to ShareSync each
run — pull it once, drop it in as `ltx_Obscura_Remova.api.json`, and point the
preset's `workflow` at it to skip conversion on later runs.

### No build required (default)

The preset runs on a **public** ComfyUI base image (`yanwk/comfyui-boot:latest`).
On startup `comfy_run.py` clones the 7 node packs (`node_packs`) and downloads the
5 LTX-2.3 weights + the Obscura LoRA (`models` / `lora_default`) **onto the Spark
node** — nothing is built or pushed, nothing large touches your computer. So
there's **no required setup** beyond `.env` credentials; you can run immediately.

Trade-off: ~30 GB is fetched to the node on each **cold** start. Within an
`--idle-hold` window the warm node is reused, so iterations are fast. (Want fast
cold starts instead? Bake the [Dockerfile](Dockerfile) into an image once and set
`image` in the preset to it — same node packs + weights, baked in.)

First, convert the graph and inspect it — this **skips** the 30 GB model pull
(conversion only needs the node classes), so it's quick:

```bash
python comfy_spark/comfy_launch.py --preset ltx_Obscura_Remova --convert-only
# → pull workflow_api.json from the job's ShareSync output, eyeball it
```

Then the real render (the clip is the positional arg; no --image needed):

```bash
python comfy_spark/comfy_launch.py --preset ltx_Obscura_Remova \
    shot.mov --prompt "remove the wooden fence from the foreground" --strength 2.3 \
    --idle-hold 300 --download ./renders
```

Outputs are written to the job's **ShareSync** folder, not your disk. The submit
line prints that folder's URL, and `--download DIR` pulls everything in it (the
rendered mp4, plus the converted `workflow_api.json`) to a local directory once
the job finishes. Without `--download`, fetch from ShareSync's web UI / mounted
drive.

UI→API conversion handles active nodes, scalar/combo widgets, named or
positional `widgets_values`, and seed/`control_after_generate` offsets. Muted
and bypassed nodes are dropped — if a kept node depended on one, `/prompt`
fails loudly rather than rendering something wrong. (This graph has none.)

### What the graph maps to (node ids → flags)

| flag | node | field | note |
|------|------|-------|------|
| `--prompt`   | 432 | `inputs.value` | `PrimitiveStringMultiline` feeding the positive CLIPTextEncode |
| `--negative` | 315 | `inputs.text`  | negative CLIPTextEncode |
| `--strength` | 427 | `inputs.strength_model` | built-in Obscura strength (graph default **2.3**) |
| `--lora-url` / `--lora-name` | 427 | `inputs.lora_name` | override the built-in Obscura LoRA source (`LTX23_Obscura_Remova_v1.safetensors`) |
| `--lora` (stack) | — | spliced after 427 | extra catalog/URL LoRAs (see *Stacking LoRAs*) |
| (positional) | 372 | `inputs.video` | `VHS_LoadVideo` input clip |
| `--fps`      | 302 | `inputs.value` | `PrimitiveInt "Frame Rate"` (drives load + output) |

Other knobs via `--set`: length/frames `303.inputs.value` (graph 81), width
`314.inputs.value` (720), height `301.inputs.value` (1280). (These are also
exposed as `--prompt`-style params now — run `--dry-run` to see them.)

> **Note:** `--lora` no longer overrides the built-in Obscura LoRA — it now
> *stacks* extra LoRAs (see below). To swap Obscura's source use `--lora-url` /
> `--lora-name`.

## Stacking LoRAs (the catalog)

LTX-2 has a huge variety of community LoRAs. The **drop-in** kind — effect,
style, camera, motion — only patch the model and can be **stacked**. comfy_spark
exposes them through a catalog plus a repeatable flag:

```bash
# list the curated catalog (per-preset or loras/ltx2.loras.json)
python comfy_spark/comfy_launch.py --list-loras

# stack two by name (strength from the catalog, or override with :N), on the
# generation preset where effect LoRAs belong:
python comfy_spark/comfy_launch.py --preset ltx2_generate shot.png \
    --prompt "a koi pond at dawn" \
    --lora transition:1.0 --lora vbvr

# anything not in the catalog: pass a direct .safetensors URL (+ optional :N)
python comfy_spark/comfy_launch.py --preset ltx2_generate shot.png \
    --lora https://huggingface.co/<repo>/resolve/main/<file>.safetensors:0.9
```

How it works: each `--lora` becomes a `LoraLoaderModelOnly` node that
`comfy_run.py` **splices in series** onto the preset's `lora_chain` anchor (the
MODEL output that feeds the sampler), repointing every downstream consumer to the
end of the chain. Catalog **trigger words** are auto-appended to `--prompt`
(disable with `--no-triggers`). A combined strength over ~2.0 warns (per the
[LTX LoRA docs](https://docs.ltx.video/open-source-model/usage-guides/lo-ra)).

**Adding to the catalog:** edit [`loras/ltx2.loras.json`](loras/ltx2.loras.json)
— one entry per LoRA (`url`, `file`, `base`, `kind`, `trigger`, `strength`,
`note`). Only **Tier-1 drop-in** LoRAs belong here; control / IC-LoRAs (depth,
pose, canny, inpaint, …) need conditioning wired in and ship as their own
presets instead. The catalog is `base`-tagged (`ltx2.3-22b`); a LoRA whose base
doesn't match the preset's model warns rather than failing silently.

## Desktop UI

```bash
python comfy_spark/comfy_ui.py
```

A small tkinter window that is **driven by the presets**: pick a preset and it
renders only the knobs that preset declares in its `ui` metadata — each with a
`[?]` tooltip — plus a Browse button for the input file (and a **mask** picker
for presets that take one), a LoRA-stack picker fed by the catalog, **Convert
only** / **No trigger words** toggles, and the compute options (GPU / mode /
idle-hold / smart-retries / job name / tags / download dir). **Dry run** prints
the plan; **Run on Spark** streams the live job log into the pane at the bottom.
It just shells out to `comfy_launch.py`, so auth, billing tags, and behaviour are
identical to the CLI. (Eventually this moves into tunet-web; the catalog + preset
`ui` metadata are designed to drive that too.)

> **Parity rule:** every CLI feature must also be reachable in the GUI. Per-preset
> params flow automatically *if* they carry a `ui` block; new **global** flags
> need an explicit control wired into `comfy_ui.py`.

## The `ltx2_generate` preset (template)

LTX-2.3 22B image-to-video (and text-to-video) **generation** — the right place
to stack effect/style LoRAs. Graph: the official
[`LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full`](https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.3)
(the cloud `GemmaAPITextEncode` nodes are bypassed, so the local encoder path
runs — no API key needed). LoRAs stack on the checkpoint (`lora_chain` anchor
`3940`), affecting both sampler passes.

It ships as a **template**: two weight URLs (`ltx-2.3-22b-dev` checkpoint and the
gemma text encoder) are `REPLACE_*` placeholders, so `comfy_launch.py` refuses to
submit until you fill them in. The third base weight — the distillation LoRA — is
already pinned to the **cond-safe** variant (see below). Validate the graph first
— `--convert-only` skips the weight pull, so it works even with the placeholders
still in:

```bash
python comfy_spark/comfy_launch.py --preset ltx2_generate --convert-only --download ./gen_check
# fill the three model URLs in presets/ltx2_generate.preset.json, then render:
python comfy_spark/comfy_launch.py --preset ltx2_generate shot.png \
    --prompt "slow dolly across a misty forest" --lora transition --idle-hold 300 --download ./renders
```

### Cond-safe distill LoRA

All three LTX-2.3 distilled presets (`ltx2_generate`, `ltx_control`, `ltx_hdr`)
use the **cond-safe rank-72** distillation LoRA
([`TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments`](https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments),
`...ceil72_condsafe.safetensors`) **instead of** Lightricks' official rank-384
distilled LoRA.

Why: the official rank-384 LoRA is high-capacity enough to **dampen conditioning
signals** on I2V / reference-driven inputs — exactly what these presets feed it
(an init frame, a canny/depth/pose reference, an SDR clip). The cond-safe build
zeroes the conditioning-related layers (cross-attention bridges, adaLN
scale-shift tables, gate logits, prompt scale-shift), so the distillation speed-up
applies while your conditioning passes through cleanly. Per the model card, rank
≤72 is safe up to **strength 1.0** on a generation pass; these graphs keep their
tuned strengths (`ltx2_generate` 0.5 / 0.2 across its two guiders, the IC-LoRA
presets 0.5). To go back to the official rank-384 LoRA, point each preset's distill
`models` entry (and the in-graph `lora_name`) at
`Lightricks/LTX-2.3/.../ltx-2.3-22b-distilled-lora-384-1.1.safetensors`.

## IC-LoRA presets (Tier-2: control)

Not every LTX-2 LoRA is a drop-in. **IC-LoRAs** (in-context) ride a model patch
*and* a guide-conditioning path (`LTXICLoRALoaderModelOnly` +
`LTXAddVideoICLoRAGuide`), often with a preprocessor — so they change graph
topology and can't be free-stacked. They ship as their own presets, built from
the official LTX-2.3 IC-LoRA workflows:

| preset | what | input | extra deps |
|--------|------|-------|------------|
| [`ltx_hdr`](presets/ltx_hdr.preset.json) | SDR→HDR re-grade ([LTX-2.3-22b-IC-LoRA-HDR](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR)) | a video | none (cleanest example) |
| [`ltx_control`](presets/ltx_control.preset.json) | **canny / depth / pose** structural control ([Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)) | a reference video | preprocessor node packs + models |

Both are **templates**: they share the same two `REPLACE_*` base weights as
`ltx2_generate` (checkpoint + gemma encoder; the distill LoRA is pinned to the
cond-safe variant, and the IC-LoRA's own URL is real/verified). The Spark-side
LoRA download path saves them to `models/loras/ltxv/ltx2/` to match the graph. Same
drill — validate the graph cheaply first (it discovers any missing node packs),
then fill the base weights:

```bash
python comfy_spark/comfy_launch.py --preset ltx_control --convert-only --download ./ctrl_check
# canny/depth/pose are all extracted on Spark from the reference clip:
python comfy_spark/comfy_launch.py --preset ltx_control walk_ref.mp4 \
    --prompt "a knight in plate armor walking through fog" --idle-hold 300 --download ./renders
```

Notes specific to `ltx_control`: the graph wires **all three** preprocessors
(canny `4991`, pose `4986` DWPose, depth `5060/5061` VideoDepthAnything) into the
union guide; to favor one, mute the others / tune the guide via `--set` after
pulling the converted `workflow_api.json`. The preprocessor models (`yolox_l.onnx`,
`dw-ll_ucoco`, `video_depth_anything_vits.pth`) usually auto-download via their
node packs — if one errors "model not found", add it to the preset's `models`.
Other 2.3 IC-LoRAs (motion-track, refocus, outpaint, …) follow this exact
pattern: bring the workflow JSON, point `lora_chain` at the checkpoint, map the
prompt/input nodes.

## The `wan_vace_inpaint` preset

Wan2.1 **VACE 14B** video inpainting: you give it a clip and a **mask**, and the
masked region is regenerated from the text prompt + the surrounding video.
With no reference image (disconnected in this preset) that's **object removal /
cleanplate** — describe the clean background in `--prompt`. Accelerated with the
[CausVid](https://huggingface.co/Kijai/WanVideo_comfy) LoRA (4 steps, cfg 1).

Like `ltx_Obscura_Remova` it runs on the public `yanwk/comfyui-boot` image with **no
build** — all nodes are *core* ComfyUI (no `node_packs`), but recent ones
(`WanVaceToVideo`, `LoadVideo`), so the image must ship a current ComfyUI (the
`cu130` tag does). On a cold start it fetches **~42 GB** (14B fp16 ≈ 35 GB +
umt5 fp8 ≈ 6.7 GB + VAE + CausVid LoRA); use `--idle-hold` to keep a warm node.

### Mask: static image *or* per-frame video

`--mask` accepts either, auto-selected by extension. **White = the region to
inpaint/remove** (read from the red channel — ordinary white-on-black masks, no
alpha needed). The mask should match the source clip's resolution; a mask video
should match its frame count.

- `--mask mask.png` → the **static** graph: one mask region applied to every
  frame. Good for a fixed area (a logo, a parked sign).
- `--mask roto.mp4` → the **mask-video** graph
  (`wan_vace_inpaint_maskvid.workflow.json`): per-frame masks for a moving
  subject. Bring your own roto/SAM2 mask sequence.

### What the graph maps to (node ids → flags)

| flag | node | field | note |
|------|------|-------|------|
| (positional) | 209 | `inputs.file`  | source clip (`LoadVideo`) |
| `--mask`     | 215 | `inputs.image` / `inputs.file` | mask image, or mask **video** in the maskvid graph |
| `--prompt`   | 6   | `inputs.text`  | positive `CLIPTextEncode` (describe the *clean* result) |
| `--negative` | 7   | `inputs.text`  | negative `CLIPTextEncode` (graph default is a generic one) |
| `--strength` | 154 | `inputs.strength_model` | CausVid LoRA strength (default **0.3**; try 0.3–0.7) |
| `--lora`     | 154 | `inputs.lora_name` | CausVid acceleration LoRA |

Other knobs via `--set`: width `49.inputs.width` (720), height
`49.inputs.height` (720), length/frames `49.inputs.length` (81), steps
`3.inputs.steps` (4), cfg `3.inputs.cfg` (1). To run the **base model without
CausVid**: `--strength 0 --set 3.inputs.steps=20 --set 3.inputs.cfg=6`.

Validate first (skips the 42 GB pull), then render:

```bash
# convert the UI graph on Spark and eyeball the api.json (no mask needed)
python comfy_spark/comfy_launch.py --preset wan_vace_inpaint --convert-only --download ./vace_check

# remove a moving subject with a per-frame mask video
python comfy_spark/comfy_launch.py --preset wan_vace_inpaint \
    shot.mov --mask shot_mask.mp4 \
    --prompt "empty cobblestone street, clean background" \
    --idle-hold 300 --download ./renders
```

The two graphs are derived from the official ComfyUI template by
[`presets/wan_vace_inpaint.derive.py`](presets/wan_vace_inpaint.derive.py)
(reference disconnected for removal; mask read from the red channel; mask-video
variant adds a `LoadVideo → GetVideoComponents → ImageToMask` front-end).

## The `ltx_faceswap` preset

LTX-2.3 22B video **face / head swap**. You give it a **body/motion clip** (the
positional arg — its motion, scene, lighting and audio are kept) and a **face
reference image** (`--face` — the identity to swap *in*), and the face/head is
replaced while the performance is preserved. Built on
[Alissonerdx's "BFS — Best Face Swap"](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video)
**V3 head-swap LoRA** (rank-adaptive). V3 is *persistent-template*: the reference
face is composited into a reserved chroma-key side strip on **every** guide frame
(`ReservedRegionFrameComposer`), so the identity signal doesn't drift over the
clip (V1's weakness). It's a dedicated preset — like the IC-LoRA presets it rides
a guide-conditioning path, so it's not a stackable catalog LoRA.

Unlike the IC-LoRA presets, **all six weights are pinned/verified** (the distilled
2.3-22b transformer, gemma + text-projection encoders, video + audio VAE, and the
V3 head-swap LoRA), so it **runs as-is** — no `REPLACE_` URLs to fill.

```bash
# validate the converted graph cheaply first (skips the ~30 GB pull)
python comfy_spark/comfy_launch.py --preset ltx_faceswap --convert-only --download ./fs_check

# swap a face: body clip is positional, identity is --face
python comfy_spark/comfy_launch.py --preset ltx_faceswap \
    performance.mp4 --face alice.png \
    --prompt "head_swap:

FACE:
A woman, fair skin, late 20s, long auburn hair, green eyes.

ACTION:
A person faces the camera and speaks, gesturing with one hand." \
    --idle-hold 300 --download ./renders
```

Notes:

- **Body clip should have audio.** LTX-2.3 is audio-aware and this graph drives
  lip motion from the **body clip's own audio track** — there's no separate audio
  input. A silent clip still runs but lip sync suffers.
- **Face reference:** a square-ish, well-lit, frontal/¾ shot gives the most stable
  identity. **Strength 1.0** is the baseline; >1.0 captures identity/hair harder
  but can distort motion. Face-reference quality matters more than strength.
- **Structured prompt:** keep the `head_swap:` / `FACE:` / `ACTION:` blocks —
  describe the reference face's traits under FACE and the performer (as "a person",
  body/clothing/action only) under ACTION. (BFS V3 model-card format.)
- **Uncensored prompts:** the abliterated-gemma CLIP LoRA is bypassed by default
  (its weight isn't pinned). Re-enable node 424 in the graph + add its weight to
  the preset's `models` if you need it.

The shipped graph is derived from Alisson's V3 *drag-and-drop* workflow by
[`presets/ltx_faceswap.derive.py`](presets/ltx_faceswap.derive.py), which makes it
Spark-convertible: it **flattens ~68 KJNodes Set/Get virtual nodes** into direct
links (comfy_run's converter has no Set/Get resolution), reconnects through the
bypassed alt-LoRA loaders onto the head-swap LoRA, **rewires the prompt to the
manual box** (dropping the `OllamaVideoDescriber` auto-prompt path, which needs a
`localhost:11434` Ollama server that doesn't exist on Spark), and mutes the
custom-audio / comparison-preview / dead-upscaler nodes. Re-run it if the upstream
workflow moves.

## Training your own LoRA

Beyond *using* LoRAs, you can **train** one on Spark and then stack it back into
comfy_spark — closing the loop in one platform. [`lora_train.py`](lora_train.py)
is the training sibling of `comfy_launch.py`: it reuses the same auth, billing
(`cpfx_tunet`), tarball-upload and ShareSync-download spine, but submits a
*training* job (tagged `cpfx_train`) that runs the official
[Lightricks ltx-trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer)
via [`lora_train_run.py`](lora_train_run.py) on the node.

**What it does for you on the node:** clone LTX-2 → `uv sync` the trainer env →
download the base weights → caption any un-captioned image → `process_dataset.py`
(cache latents) → write the YAML config from the recipe → `scripts/train.py` →
self-upload the trained `.safetensors` to the job's ShareSync folder.

### Start with images (the easy win)

The bundled [`trainers/ltx2_style.train.json`](trainers/ltx2_style.train.json)
recipe trains an **image-only** style / identity LoRA — much easier and more
reliable than teaching motion, and the right first thing to train. You only need
a folder of stills:

```bash
# 1) fill the TWO REPLACE_ base-weight URLs in trainers/ltx2_style.train.json
#    (the same dev checkpoint + a gemma text encoder the generation presets use;
#     see the recipe's _TEMPLATE note about the trainer's text-encoder format)

# 2) validate for free — prints the dataset.json + the full training plan,
#    submits nothing, spends nothing (the cheap-validate analog of --convert-only):
python comfy_spark/lora_train.py --recipe ltx2_style --dataset ./mylook \
    --trigger raxstyle --dry-run

# 3) train (warm-hold the node so re-runs are fast):
python comfy_spark/lora_train.py --recipe ltx2_style --dataset ./mylook \
    --trigger raxstyle --idle-hold 300 --download ./loras_out
```

The trained LoRA lands in ShareSync (and in `./loras_out` with `--download`).
**Use it** by passing its `.safetensors` URL straight to the stack flag:

```bash
python comfy_spark/comfy_launch.py --preset ltx2_generate shot.png \
    --prompt "a portrait in raxstyle, golden hour" \
    --lora https://.../raxstyle_ltx2.3.safetensors
```

### The dataset (where the real work is)

- **Folder of images.** ~20–50 for a style/look; 30–100 for a specific face or
  object. Consistency matters more than quantity.
- **Captions are optional.** Drop a same-named `.txt` next to an image to caption
  it yourself; anything without one is **auto-captioned on the node** (BLIP, in
  the trainer's own env — best-effort, falls back to `--describe` / a generic
  caption). The captioning rule of thumb: *describe what should stay variable
  (lighting, framing, pose); don't describe the thing you're teaching* — your
  `--trigger` word covers that, and ltx-trainer auto-prepends it to every caption.
- **The trainer enforces** dimensions divisible by 32 (the recipe's `resolution`)
  and, for video, the 8·n+1 frame rule. Images use a frame count of 1.

### Knobs

| flag | what | default |
|------|------|---------|
| `--trigger WORD` | distinctive word that activates the LoRA (required) | — |
| `--rank N` | LoRA capacity; 16–64 is the concept/style range | 32 (recipe) |
| `--steps N` | training length; validate previews ~500–750, watch overfitting past ~1000 | 2000 |
| `--low-vram` | INT8 / 8-bit config to fit a 48 GB `l40s` (else the 96 GB `rtxpro6000` path) | off |
| `--describe "…"` | fallback caption for un-captioned images when auto-caption is off | `a photo` |
| `--no-auto-caption` | skip the BLIP step; use `.txt` / `--describe` only | — |
| `--idle-hold N` / `--download DIR` / `--dry-run` / `--gpu` / `--mode` | same as `comfy_launch.py` | — |

```bash
python comfy_spark/lora_train.py --list-recipes   # available recipes
python comfy_spark/lora_train.py --list-jobs       # only cpfx_train jobs
python comfy_spark/lora_train.py --logs   <JOB_ID>
python comfy_spark/lora_train.py --cancel <JOB_ID>
```

> **Not yet:** motion LoRAs (need short coherent video clips) and IC-LoRA training
> (depth/pose/canny control) follow the same recipe pattern with `kind: video` /
> `kind: ic-lora` — added once the image path is proven on real weights.

## The `wan22_i2v` preset

**Wan 2.2 14B image-to-video**, accelerated with the **4-step lightx2v Lightning**
LoRAs — a fast generator that sits *alongside* `wan_vace_inpaint`, not replacing
it (Wan 2.2 has no VACE-style masked inpainting, so VACE 2.1 stays the
cleanplate/removal tool). Wan 2.2 14B is a two-expert mixture: a **high-noise**
expert samples the first steps, a **low-noise** expert finishes — each carrying
its own Lightning LoRA, so a clip renders in **4 sampler steps** (cfg 1).

All weights are pinned (Comfy-Org repackaged) — it runs as-is, no template to
fill, on the same no-build public image as `wan_vace_inpaint` (all core nodes).

```bash
# validate the flattened graph for free (no GPU, no 45 GB pull):
python comfy_spark/comfy_launch.py --preset wan22_i2v --convert-only

# image-to-video:
python comfy_spark/comfy_launch.py --preset wan22_i2v start.png \
    --prompt "a slow dolly forward through neon rain, the subject turns to camera" \
    --idle-hold 300 --download ./renders
```

### Flattened from a subgraph

The official ComfyUI Wan 2.2 i2v template ships as a **subgraph** graph, and
`comfy_run.py`'s UI→API converter is flat-graph only. So
[`presets/wan22.derive.py`](presets/wan22.derive.py) **flattens** it offline into
[`presets/wan22_i2v.workflow.json`](presets/wan22_i2v.workflow.json) — inlining the
subgraph's inner nodes, dropping the promoted-widget boundary links (those values
already live on the inner nodes), and rewiring the two real data crossings
(`start_image` in, `VIDEO` out). Re-run it if the upstream template moves:
`python presets/wan22.derive.py SRC_SUBGRAPH_TEMPLATE.json presets`. (Generalizing
this into the converter — so any subgraphed workflow converts directly — is
tracked in `EZ_COMFY_TODO.md`.)

### What the graph maps to (node ids → flags)

| flag | node | field | note |
|------|------|-------|------|
| (positional) | 97 | `inputs.image` | `LoadImage` start frame |
| `--prompt`   | 107 | `inputs.text` | positive `CLIPTextEncode` (describe the motion) |
| `--negative` | 125 | `inputs.text` | negative `CLIPTextEncode` |
| width/height/length | 128 | `inputs.{width,height,length}` | `WanImageToVideo` (720p sweet spot; frames = 16·n+1, 81 ≈ 5s @16fps) |
| `strength_high` | 126 | `inputs.strength_model` | high-noise Lightning LoRA (1.0) |
| `strength_low`  | 127 | `inputs.strength_model` | low-noise Lightning LoRA (1.0) |

To run the **base model without Lightning** (more motion, slower): set both
strengths to 0 and raise the step split on the two `KSamplerAdvanced` nodes
(110 high, 111 low) via `--set`, with cfg ~3.5.

## Choosing the output format (high-bit / EXR / ProRes)

By default every preset writes an 8-bit H.264 **mp4** — fine for review, lossy for
a comp. `--output` lets you pick a **high-bit deliverable** instead of assuming
mp4, and the workflow graph is rewritten on the node to produce it. The choice is
**additive**: you still get the mp4 preview *plus* the format you asked for.

```bash
python comfy_spark/comfy_launch.py --list-outputs        # the menu

# scene-linear OpenEXR sequence (drops into Nuke/AE):
python comfy_spark/comfy_launch.py --preset wan22_i2v start.png \
    --prompt "..." --output exr32 --download ./renders

# 10-bit ProRes 422 HQ instead of 8-bit mp4:
python comfy_spark/comfy_launch.py --preset ltx2_generate shot.png \
    --prompt "..." --output prores_hq --output-fps 24
```

| `--output` | what you get | how |
|------------|--------------|-----|
| `mp4` (default) | the preset's own 8-bit mp4 | no rewrite |
| `exr32` / `exr16` | OpenEXR **frame sequence**, 32-bit float / 16-bit half, **scene-linear** | CoCoTools `ColorspaceNode`(sRGB→linear) → `SaverNode` |
| `png16` / `tiff16` | 16-bit PNG / TIFF sequence (display sRGB) | CoCoTools `SaverNode` |
| `prores_proxy…prores_4444xq` | one 10/12-bit **ProRes** `.mov` (Proxy / LT / 422 / HQ / 4444 / 4444 XQ) | core `CreateVideo` → SaveVideoHQ |

How it works: the chosen format's save node(s) are spliced as a **parallel branch**
onto the workflow's final RGB frames — auto-detected as the IMAGE feeding the
terminal `CreateVideo`/`SaveImage`/`VHS_VideoCombine` (or a preset's explicit
`output_anchor`), so it works on **any** preset with no per-preset wiring. The
format's node pack (`ComfyUI-CoCoTools_IO` for EXR/PNG/TIFF, `ComfyUI-SaveVideoHQ`
for ProRes) is cloned on the node automatically. **Colour:** EXR defaults to
scene-linear (`sRGB→linear` on save) since that's what comp expects; everything
else stays display sRGB. Override with `--colorspace linear|as-is`. The registry
lives in [`outputs/outputs.json`](outputs/outputs.json) — add a format by adding
an entry (save-node chain + node pack), no code change.

> Sequence outputs (EXR/PNG/TIFF) write `prefix.####.ext` into the job's ShareSync
> folder — `--download` pulls the whole sequence. The model still emits audio on
> the mp4 preview; the EXR/ProRes branch is video-only frames.

## Roadmap: VOID video inpainting

The [VOID](https://docs.comfy.org/tutorials/utility/void-video-inpainting)
tutorial is a **different** workflow from the VACE one above. It is **not yet
runnable** here, by design — scoping notes:

- It ships as a **subgraph** graph (`SAM3 segmentation` + a 37-node `Video
  Inpaint (VOID)` subgraph). `comfy_run.py`'s UI→API converter is flat-graph
  only; subgraphs would need a flattening pass first.
- It needs **custom nodes** (`SAM3_Detect`, `VOIDSampler`, `VOIDWarpedNoise`,
  `VOIDInpaintConditioning`, `OpticalFlowLoader`, …) and **extra weights**
  (`void_pass1/2`, `cogvideox_vae`, `t5xxl`, `sam3.1_multiplex`, RAFT).
- So enabling it means: (1) subgraph flattening in the converter, (2) a
  `node_packs` list for the VOID/SAM3/flow nodes, (3) those models. Tracked as a
  follow-up.

## Adding a new preset (workflow)

Every shipped preset must be **self-describing in plain language** so it's usable
without ComfyUI knowledge. Three keys are **required** in each `.preset.json` (the
CLI `--list-presets` and the desktop UI's "About this workflow" panel both read
them):

- **`docs`** — link(s) to the *canonical* documentation (the model card / official
  workflow page). Labelled form preferred:
  ```json
  "docs": [
    { "label": "Obscura Remova model card", "url": "https://huggingface.co/WepeNerd/Obscura_Remova" },
    { "label": "LTX-2 docs", "url": "https://docs.ltx.video" }
  ]
  ```
- **`about`** — three short, jargon-free fields. If someone who's never opened
  ComfyUI can't tell what it does and what to feed it from these alone, rewrite them:
  ```json
  "about": {
    "what":      "Cleans unwanted things out of a video and rebuilds what's behind them. (A 'clean plate'.)",
    "inputs":    "One video clip. Optionally, a short line saying what to remove.",
    "key_knobs": "Prompt — what to remove. Strength — how hard it pushes (start ~2.3, lower if kept areas distort)."
  }
  ```
- **`tags`** — flat, lowercase, hyphenated, so workflows are easy to sift
  (`--list-presets <tag>` does a substring match). Reuse the existing vocabulary:
  model (`ltx2.3`, `wan2.1`, `wan2.2`), task (`video-to-video`, `generation`,
  `inpainting`, `object-removal`, `face-swap`, `hdr`, `structural-control`, …),
  needs (`mask-required`, `two-inputs`, `lora-stacking`, `ic-lora`), and readiness
  (`ready-to-run` vs `template`).

The six shipped presets are the reference for tone and depth. Per-knob
`ui.tooltip`s should likewise stay human and cite a source for non-obvious values.
(These rules are also in [`CLAUDE.md`](CLAUDE.md) for agent edits.)

## Notes / gotchas

- **GPU:** LTX-2.3 22B and Wan2.1 VACE 14B both want `rtxpro6000` (96 GB), or
  `l40s` (48 GB) with the fp8-cast weight dtype. `t4`/`a10` will OOM.
- **VACE masks** must line up with the source: same resolution, and a mask
  *video* should have the same frame count as the clip. White = inpaint region.
  Run `--convert-only` once after editing a preset to confirm the graph converts
  before paying for the model pull.
- **Smart mode** (`--mode smart`, `--max-retries N`) is ~60% cheaper but
  preemptible. ComfyUI doesn't checkpoint mid-sample, so a preemption restarts
  the clip — fine for an overnight batch, set retries accordingly.
- LoRAs are fetched at runtime from **direct `.safetensors` URLs** (catalog
  `url`, preset `lora_default`, or `--lora <url>`), never a bare HF repo path.
- `comfy_run.py` is pure stdlib so it runs on any ComfyUI image without extra
  installs.
