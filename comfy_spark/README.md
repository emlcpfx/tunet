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
    three weight URLs once — see below).
  - **`ltx_hdr`** / **`ltx_control`** — Tier-2 **IC-LoRA** presets (HDR grade;
    canny/depth/pose structural control). Templates — see *IC-LoRA presets*.
  - **`wan_vace_inpaint`** — Wan2.1 VACE 14B video inpainting / object removal,
    guided by a **mask** (static image *or* per-frame mask video). See below.
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
| `Dockerfile` | build step | ComfyUI + LTX node-pack image (a starting point) |
| `presets/*.preset.json` | — | preset bundles (image/GPU/workflow + flag→node maps) |
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

It ships as a **template**: three weight URLs (`ltx-2.3-22b-dev` checkpoint, the
gemma text encoder, and the `distilled-lora-384`) are `REPLACE_*` placeholders,
so `comfy_launch.py` refuses to submit until you fill them in. Validate the
graph first — `--convert-only` skips the weight pull, so it works even with the
placeholders still in:

```bash
python comfy_spark/comfy_launch.py --preset ltx2_generate --convert-only --download ./gen_check
# fill the three model URLs in presets/ltx2_generate.preset.json, then render:
python comfy_spark/comfy_launch.py --preset ltx2_generate shot.png \
    --prompt "slow dolly across a misty forest" --lora transition --idle-hold 300 --download ./renders
```

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

Both are **templates**: they share the same three `REPLACE_*` base weights as
`ltx2_generate` (the IC-LoRA's own URL is real/verified). The Spark-side LoRA
download path saves them to `models/loras/ltxv/ltx2/` to match the graph. Same
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
