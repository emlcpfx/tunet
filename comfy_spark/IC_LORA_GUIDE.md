# LTX-2 IC-LoRA — best practices (how to set them up so they actually work)

IC-LoRAs ("in-context" LoRAs) transfer the **structure + motion** of a reference
video onto a brand-new generation: you keep the motion, you change the look. This
is the distilled playbook for the comfy_spark IC-LoRA presets (`ltx_control`,
`ltx_lipdub`, `ltx_motion_track`, `ltx_hdr`, `ltx_restore`).

Derived from: the official **LTX-2 IC-LoRA tutorial** (youtu.be/NPjTpDmTdaw) and
the matching **LTX blog** (ltx.io/.../how-to-use-ic-lora-in-ltx-2); **Next
Diffusion**'s pose-control walkthrough (nextdiffusion.ai); and a creator
character-replacement walkthrough (youtu.be/u1eA8WJeO4s). Where the official line
and practitioners disagree, both are noted.

## The one rule people get wrong: prompt the STYLE, not the motion
The LoRA already owns the motion (it's read from the reference video). So your
prompt must describe **only**: subject/appearance, background, textures, lighting,
mood. **Do NOT** write "camera pans left", "she walks forward", "slow zoom" — that
fights the reference and muddies the result. (Official + blog, unanimous.)

## Pick ONE control mode and mute the others
The control workflow ships **three** modes; running more than one at once blows
VRAM and muddies control. Choose the one that matches what you're locking:

| mode | preprocessor | locks | use for |
|------|--------------|-------|---------|
| **depth** | VideoDepthAnything | camera move + 3D layout/geometry | general "same shot, new look"; scenes |
| **pose**  | DWPose (DWPreprocessor) | human joints / body / dance | people, performance, dance, lip-talk |
| **canny** | Canny edges | hard edges / silhouettes | rigid objects, clean outlines, graphics |

You can combine **two** only if they describe *different* aspects (e.g. depth for
the camera + pose for the person). In `ltx_control` all three preprocessors are
wired into the union guide by default — for best quality/VRAM, mute the two you
aren't using (their nodes: **canny 4991, pose 4986, depth 5060/5061**) in the
converted `workflow_api.json`, or use a single-mode variant once we ship them
(tracked in EZ_COMFY_TODO).

## Reference video prep (this decides whether it works)
- **Single subject, well-lit, plain / uncluttered background.** For **pose**,
  front-facing with the full upper body visible → reliable skeleton extraction.
- Output **length ≤ the reference clip's length** (it can't exceed it); and in
  comfy_spark the length must be the next `8·n+1` **≥** the control clip's frame
  count (else "conditioning frames exceed the latent sequence").
- Control frames are auto-downscaled to **0.5× output res** before encoding (the
  `ref0.5` in the adapter filename) — a built-in VRAM saver, nothing to set.

## Resolution & frames
- **720p is the sweet spot.** 1080p is slow + VRAM-hungry; >1080p rarely worth it.
- Width/height **divisible by 64** for the comfy_spark IC-LoRA presets (a non-÷64
  size gives an odd latent and fails "latent spatial size must be divisible by 2";
  e.g. 544 fails, 576 works). Good ÷64 sizes: **1280×768** (16:9), **768×1280**
  (9:16), **768×768** (1:1), **1088×1920** (9:16 FHD).
- Frames = **8·n+1** (…, 97, 121, 161). fps 24–30 (match the source for natural motion).

## Strength (the one number worth tuning)
The guide node's **strength** (default **1.0**):
- **1.0** = strictest structure lock; Lightricks' default. *Below* 1.0 the
  reference can "pop"/**bleed through** (its actual pixels leak into the output) —
  bad for a full restyle.
- In practice many land on **0.6–0.7** for the union adapter when 1.0 feels too
  rigidly glued to the reference (Next Diffusion 0.6–0.7; a creator A/B-tested
  0.5/0.6/0.7 → **0.6 best**). Start at 1.0 for a clean restyle; lower toward 0.6
  only if motion/structure is over-locked or you want a more natural feel.
- `frame_index = 0` (guidance from the first frame) — leave it.

## Image-to-video vs text-to-video
- A **reference image is required even in T2V mode** (the workflow won't run without one).
- **I2V** (default): the start frame is your custom image. **Generate that first
  frame FROM the reference video's first frame** (a ControlNet / image model),
  keeping the subject in the **same position** — otherwise you get a jump-cut /
  artifacts at frame 1. (This is the make-or-break for character replacement.)
- **T2V**: generates pixels from scratch but still follows the reference's structure.

## Talking / lip-sync IC-LoRA
- Prompt the speaker's look + voice ("a woman talking to camera, confident,
  feminine voice") **and the EXACT words the reference says**, verbatim
  (including any stutters/repeats) — that's what syncs the mouth to new speech.
- Let it **generate the soundtrack from the text prompt**; do NOT feed the
  original audio track alongside a text prompt, or the mouth "melts."

## comfy_spark quick starts
```bash
# structural control (depth/pose/canny) — describe the LOOK, motion comes from the clip
python comfy_spark/comfy_launch.py --preset ltx_control walk_ref.mp4 \
  --prompt "a knight in ornate plate armor in a misty forest at dawn, volumetric light, shallow depth of field" \
  --idle-hold 300 --download ./out
# lower the guide strength only if it's too glued to the reference:
#   --set 5012.inputs.strength=0.65
```

> Sources: youtu.be/NPjTpDmTdaw (official LTX-2 IC-LoRA tutorial),
> ltx.io/model/model-blog/how-to-use-ic-lora-in-ltx-2,
> nextdiffusion.ai (LTX-2.3 pose control), youtu.be/u1eA8WJeO4s (creator walkthrough).
