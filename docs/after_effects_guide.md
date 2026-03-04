# TuNet for After Effects Users

## What Is TuNet?

TuNet is a machine-learning tool that learns image transformations from examples.
You show it "before" and "after" frames, it learns the pattern, then applies that
pattern to new frames automatically.

Think of it like teaching an intern: you hand-paint corrections on 200–500 frames,
TuNet studies them, then does the rest of the shot for you.

---

## What Can It Do?

| Task | What You'd Normally Do in AE | What TuNet Does Instead |
|------|------------------------------|-------------------------|
| **Beauty** | Clone stamp, frequency separation, skin plugins | Learns your retouching style from examples and applies it to every frame |
| **Roto / Matte** | Rotoscope by hand or use Rotobrush | Learns object boundaries from painted mattes and generates the rest |
| **Cleanup** | Paint out wires, rigs, tracking markers frame by frame | Learns what "clean" looks like and removes the same objects across the shot |
| **Color / Grade** | Apply LUTs or manually match grades | Learns the color transform from graded examples and applies it everywhere |
| **Denoise** | Neat Video, Remove Grain | Learns the difference between noisy and clean frames and removes noise |

---

## The Basic Idea

```
You provide:                  TuNet learns:              You get:
┌──────────┐                                            ┌──────────┐
│  src/    │  200–500 paired  ───►  A model that   ───► │  output/ │
│ (before) │  before/after          transforms           │ (result) │
│  frames  │  examples              any new frame        │  frames  │
├──────────┤                                            └──────────┘
│  dst/    │
│ (after)  │
│  frames  │
└──────────┘
```

**Source (src/)** = your original plate or "before" state.
**Destination (dst/)** = your hand-corrected or "after" state.

The filenames must match — `frame_001.png` in src/ pairs with `frame_001.png` in dst/.

---

## Step by Step

### 1. Prepare Your Training Data

In After Effects:

1. Pick 200–500 representative frames from your shot (more variety = better results).
2. Export the **original plate** as a PNG sequence into a folder called `src/`.
3. Apply your corrections (retouching, paint, grade — whatever the task is).
4. Export the **corrected result** as a matching PNG sequence into `dst/`.

Your project folder should look like this:

```
my_project/
├── src/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
└── dst/
    ├── frame_0001.png
    ├── frame_0002.png
    └── ...
```

**Tips for good training data:**
- Use frames that cover the range of motion, lighting, and variation in the shot.
- The more consistent your corrections, the better the model learns.
- 200 frames is the minimum; 400–500 is ideal.
- Supported formats: PNG, JPG, TIFF, EXR, BMP, WebP.

### 2. Launch TuNet

```
python tunet_simple.py
```

This opens the simplified UI designed for preset-based workflows.

### 3. Pick a Preset

Select the preset that matches your task:

- **Beauty** — Skin retouching, blemish removal, frequency separation.
  Uses the MSRN model with attention for fine skin detail.
- **Roto / Matte** — Generating black-and-white mattes from plates.
- **Cleanup / Paint** — Wire removal, rig removal, paint fixes.
- **Color / Grade** — Color correction, look transfer, grade matching.
- **Denoise** — Noise reduction while preserving texture.
- **General** — A balanced starting point if nothing else fits.

Each preset configures the loss function, learning rate, augmentation, and model
architecture automatically. You don't need to touch these settings.

### 4. Select Your Project Folder

Click **Select Project Folder** and point to your `my_project/` directory.

TuNet will:
- Find and count images in `src/` and `dst/`
- Auto-detect image resolution
- Recommend optimal training settings (resolution, model size, batch size)

### 5. Train

Click **Train**.

- Watch the console output — the loss value should decrease over time.
- A preview image updates periodically so you can see how it's learning.
- Training takes anywhere from 30 minutes to several hours depending on
  your GPU, dataset size, and resolution.
- Click **Stop** when you're satisfied with the preview quality.
  The model checkpoint is saved automatically.

**What to look for:**
- Loss going down = the model is learning.
- Preview looking like your dst/ frames = it's working.
- Loss stuck or going up = something is wrong (check your data pairing).

### 6. Run Inference

Once training is done:

1. Place the frames you want to process in the `src/` folder
   (or keep the same ones — up to you).
2. Click **Run Inference**.
3. TuNet processes every image in `src/` and writes results to `my_project/output/`.

### 7. Bring Results Back Into After Effects

Import the `output/` PNG sequence into your After Effects comp. Done.

```
After Effects ──► export frames ──► TuNet training ──► TuNet inference ──► import back to AE
```

---

## Choosing the Right Preset

### Beauty
Best for: Skin retouching across an entire shot.

Typical workflow:
1. Hand-retouch 200–500 hero frames in Photoshop or AE (frequency separation,
   clone stamp, healing brush).
2. Export before/after pairs.
3. Train with the Beauty preset.
4. Run inference on the full shot.

Beauty uses the **MSRN** architecture (attention + recurrence) instead of the
standard UNet, because skin retouching requires preserving pore-level detail.

### Roto / Matte
Best for: Generating mattes when you have some hand-painted examples.

Typical workflow:
1. `src/` = your original plate frames.
2. `dst/` = hand-painted black & white mattes (white = foreground).
3. Train — the model learns to segment similar objects.
4. Run inference on the full sequence.

### Cleanup / Paint
Best for: Removing wires, rigs, tracking markers, or other unwanted elements.

Typical workflow:
1. `src/` = plate with wires/rigs visible.
2. `dst/` = hand-painted clean plate (wires removed).
3. Train — the model learns what "clean" looks like.
4. Inference removes the same type of element across all frames.

### Color / Grade
Best for: Applying a consistent color grade across a shot or matching between shots.

Typical workflow:
1. `src/` = original grade.
2. `dst/` = target grade (from DaVinci, Baselight, or AE color work).
3. Train — learns the color transform.
4. Apply to the full shot or to other similar shots.

### Denoise
Best for: Cleaning up noisy footage while keeping texture.

Typical workflow:
1. `src/` = noisy frames.
2. `dst/` = clean or denoised reference frames.
3. Train and apply.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "CUDA not available" | Your GPU drivers need updating, or no NVIDIA GPU is present. TuNet can run on CPU but will be very slow. |
| Out of memory errors | Reduce batch size (try 2 or 1). Or reduce resolution. |
| Loss isn't decreasing | Check that src/ and dst/ filenames match. Make sure the images are actually different. Try more training data. |
| Output looks blurry | Train longer. If still blurry, the model may need a higher resolution setting. |
| Visible tile seams in output | This is rare — the inference uses overlapping tiles with blending. If it happens, increase the overlap factor. |
| Training is very slow | Make sure you have a CUDA-capable GPU. A modern NVIDIA GPU (RTX 3060 or better) is recommended. |

---

## GPU and Performance

TuNet requires an NVIDIA GPU for practical use. Approximate VRAM usage:

| Resolution | Batch Size | Model Size | VRAM Needed |
|------------|-----------|------------|-------------|
| 512        | 4         | 128        | ~6 GB       |
| 512        | 2         | 128        | ~4 GB       |
| 768        | 2         | 128        | ~9 GB       |
| 1024       | 1         | 64         | ~11 GB      |

The UI auto-detects your available VRAM and adjusts batch size accordingly.

---

## Key Concepts

**Model** — The neural network TuNet trains. It's saved as a `.pth` file in
your project's `model/` folder.

**Checkpoint** — A snapshot of the model at a point during training. You can
stop and resume training; the latest checkpoint is used for inference.

**Loss** — A number that measures how far the model's output is from your
target. Lower = better. Watch this in the console during training.

**Inference** — Running the trained model on new images. This is the fast part —
training is where the time goes.

**UNet vs MSRN** — Two model architectures. UNet is the default: fast and
reliable. MSRN adds attention and recurrence for tasks that need fine detail
(like beauty work). The preset picks the right one for you.

**Epoch** — One complete pass through all your training images. Training
typically runs for many epochs.
