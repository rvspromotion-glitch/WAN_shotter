# ============================================================
# ACG — AnimateDiff / Wan 2.2 T2V Pod
# Base: RunPod PyTorch 2.8, CUDA 12.8, Python 3.11
# ============================================================
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# ── Build-time env ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV COMFYUI_PATH=/workspace/ComfyUI
ENV COMFYUI_BAKED=/opt/ComfyUI

# ── System deps (single layer) ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl aria2 \
        libgl1 libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ── Python: pin numpy first so nothing can clobber it ───────
RUN pip install --no-cache-dir "numpy<2"

# ── Python: remaining deps (single pip call = faster layer) ─
RUN pip install --no-cache-dir --prefer-binary \
        triton \
        jupyterlab \
        sentencepiece \
        "protobuf<5" \
        "mediapipe==0.10.14" \
        sageattention \
        "packaging"

# ── Bake ComfyUI into /opt (survives /workspace mount) ──────
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

# ── Entrypoint ───────────────────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8188 8888
CMD ["/start.sh"]
