# comfy_spark — run ComfyUI workflows on Spark Fuse

A command-line tool to submit a [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
workflow as a batch job on **Spark Fuse**, the same way [`spark_launch.py`](../spark_launch.py)
submits TuNet training: auth → pack a tarball → `POST /api/compute/jobs` →
`PUT` the upload → stream logs. Outputs land in ShareSync.

It has two layers:

- **Generic runner** — feed it any API-format workflow (ComfyUI → *Save (API
  Format)*), the media it references, and optional per-node overrides.
- **Presets** — named bundles under [`presets/`](presets/) that supply the
  image/GPU/workflow and map friendly flags (`--prompt`, `--lora`, `--strength`,
  …) onto specific nodes. Ships with a `cleanplate_ltx` template for LTX-2.3 +
  the [Obscura Remova](https://huggingface.co/WepeNerd/Obscura_Remova) LoRA
  (video-to-video occlusion / foreground removal).

## Files

| File | Runs where | What it does |
|------|-----------|--------------|
| `comfy_launch.py` | your machine | the CLI: pack + submit + tail |
| `comfy_run.py` | inside the container | starts ComfyUI headless, converts UI→API if needed, queues the workflow, polls to completion |
| `Dockerfile` | build step | ComfyUI + LTX node-pack image (a starting point) |
| `presets/cleanplate_ltx.preset.json` | — | LTX-2.3 + Obscura preset (a template — see below) |

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
python comfy_spark/comfy_launch.py --preset cleanplate_ltx clip.mp4 --prompt "..." --dry-run

python comfy_spark/comfy_launch.py --list-skus
python comfy_spark/comfy_launch.py --list-jobs          # only cpfx_comfy jobs
python comfy_spark/comfy_launch.py --logs   <JOB_ID>
python comfy_spark/comfy_launch.py --cancel <JOB_ID>
```

## The `cleanplate_ltx` preset

The actual Obscura graph is bundled as
[`presets/cleanplate_ltx.workflow.json`](presets/cleanplate_ltx.workflow.json)
(the UI/graph export from
[axiomgraph/ComfyUIWorkflow](https://github.com/axiomgraph/ComfyUIWorkflow)),
and the preset's `params` node ids are already mapped to it.

You do **not** need a local ComfyUI. `comfy_run.py` converts the UI graph to API
format on Spark using the image's own `/object_info` schema, then applies the
preset's edits. The converted `workflow_api.json` is written to ShareSync each
run — pull it once, drop it in as `cleanplate_ltx.api.json`, and point the
preset's `workflow` at it to skip conversion on later runs.

So the **only required setup** is the image:

```bash
docker build -t <registry>/comfy-ltx23:latest comfy_spark/   # then docker push
```

Edit the `Dockerfile` to bake in the LTX-2.3 weights (or wire up a Spark model
cache) — re-downloading them every run is slow and burns compute time. Then set
`image` in `presets/cleanplate_ltx.preset.json` to what you pushed.

Recommended first run — convert the graph and inspect it before spending a
render:

```bash
python comfy_spark/comfy_launch.py --preset cleanplate_ltx --convert-only
# → pull workflow_api.json from the job's ShareSync output, eyeball it
```

Then the real thing:

```bash
python comfy_spark/comfy_launch.py --preset cleanplate_ltx \
    shot.mp4 --prompt "remove the wooden fence from the foreground" --strength 2.3
```

UI→API conversion handles active nodes, scalar/combo widgets, named or
positional `widgets_values`, and seed/`control_after_generate` offsets. Muted
and bypassed nodes are dropped — if a kept node depended on one, `/prompt`
fails loudly rather than rendering something wrong. (This graph has none.)

### What the graph maps to (node ids → flags)

| flag | node | field | note |
|------|------|-------|------|
| `--prompt`   | 432 | `inputs.value` | `PrimitiveStringMultiline` feeding the positive CLIPTextEncode |
| `--negative` | 315 | `inputs.text`  | negative CLIPTextEncode |
| `--strength` | 427 | `inputs.strength_model` | LoRA strength (graph default **2.3**) |
| `--lora`     | 427 | `inputs.lora_name` | `LTX23_Obscura_Remova_v1.safetensors` |
| (positional) | 372 | `inputs.video` | `VHS_LoadVideo` input clip |
| `--fps`      | 302 | `inputs.value` | `PrimitiveInt "Frame Rate"` (drives load + output) |

Other knobs via `--set`: length/frames `303.inputs.value` (graph 81), width
`314.inputs.value` (720), height `301.inputs.value` (1280).

## Notes / gotchas

- **GPU:** LTX-2.3 22B wants `rtxpro6000` (96 GB) or `l40s` (48 GB) with fp8.
  `t4`/`a10` will OOM on the base model.
- **Smart mode** (`--mode smart`, `--max-retries N`) is ~60% cheaper but
  preemptible. ComfyUI doesn't checkpoint mid-sample, so a preemption restarts
  the clip — fine for an overnight batch, set retries accordingly.
- The Obscura LoRA is fetched at runtime from a **direct `.safetensors` URL**
  (`--lora` / preset `lora_default`), not a bare HF repo path.
- `comfy_run.py` is pure stdlib so it runs on any ComfyUI image without extra
  installs.
