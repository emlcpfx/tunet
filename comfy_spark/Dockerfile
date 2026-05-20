# ComfyUI image for Spark Fuse batch jobs (comfy_launch.py / comfy_run.py),
# sized for the LTX-2.3 Obscura_Remova v2v graph (presets/cleanplate_ltx.*).
#
# Two things you must get right: (a) the custom-node packs the graph uses, and
# (b) getting the LTX-2.3 weights into the image (or a Spark model cache) so jobs
# don't re-download tens of GB every run. Pin node-pack revisions once you've
# validated a working combo.
#
# Build & push to a registry Spark can pull, then set it in the preset (or pass
# --image):
#   docker build -t <registry>/comfy-ltx23:latest comfy_spark/
#   docker push  <registry>/comfy-ltx23:latest

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    COMFY_HOME=/ComfyUI \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ffmpeg curl ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# --- ComfyUI core ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI $COMFY_HOME
WORKDIR $COMFY_HOME
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt

# --- Custom node packs (every pack the Obscura graph references) ---
#   ComfyUI-LTXVideo .......... LTXV*, EmptyLTXVLatentVideo, LTXAddVideoICLoRAGuide
#   VideoHelperSuite .......... VHS_LoadVideo / VHS_VideoCombine
#   ComfyUI-KJNodes ........... ImageResizeKJv2, VAELoaderKJ
#   rgthree-comfy ............. Fast Groups Muter (rgthree)
#   ComfyUI-Easy-Use .......... easy compare, easy seed
#   ComfyMath ................. ComfyMathExpression, ComfySwitchNode
#   ComfyUI-Inspire-Pack ...... SeedLogger //Inspire
WORKDIR $COMFY_HOME/custom_nodes
RUN for repo in \
        https://github.com/Lightricks/ComfyUI-LTXVideo \
        https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
        https://github.com/kijai/ComfyUI-KJNodes \
        https://github.com/rgthree/rgthree-comfy \
        https://github.com/yolain/ComfyUI-Easy-Use \
        https://github.com/evanspearman/ComfyMath \
        https://github.com/ltdrdata/ComfyUI-Inspire-Pack ; do \
        git clone --depth 1 "$repo" && \
        ( pip install -r "$(basename "$repo")/requirements.txt" || true ) ; \
    done

# --- Model weights (the exact files this graph loads, ~30 GB total) ---
# UNETLoader -> models/diffusion_models ; DualCLIPLoader -> models/text_encoders ;
# VAELoaderKJ -> models/vae . Baking them in avoids a multi-GB re-download per run.
# Public repos, so no HF token needed (curl -fL follows the Xet/CDN redirect).
# One RUN per file = one layer per file = a failed/slow pull doesn't redo the rest.
# NOTE: fp8-scaled requires an Ada/Blackwell GPU (L40S/RTX PRO 6000); the preset
# defaults to l40s. gemma-12B (~13 GB) + the 16 GB UNET is tight on 24 GB cards —
# bump to --gpu rtxpro6000 if L40S OOMs.
WORKDIR $COMFY_HOME
RUN mkdir -p models/diffusion_models models/text_encoders models/vae
RUN curl -fL --retry 4 -o models/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
RUN curl -fL --retry 4 -o models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"
RUN curl -fL --retry 4 -o models/text_encoders/ltx-2.3_text_projection_bf16.safetensors \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors"
RUN curl -fL --retry 4 -o models/vae/LTX23_video_vae_bf16.safetensors \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors"
RUN curl -fL --retry 4 -o models/vae/LTX23_audio_vae_bf16.safetensors \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors"
# The Obscura LoRA (LTX23_Obscura_Remova_v1.safetensors) is small and fetched at
# runtime by comfy_run.py (COMFY_LORA_URL), so it need not be baked in.

# comfy_run.py arrives in /input from the job tarball; nothing to COPY here.
# The Spark agent runs:  python /input/comfy_run.py
WORKDIR $COMFY_HOME
