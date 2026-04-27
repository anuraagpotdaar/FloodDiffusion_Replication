# FloodDiffusion replication — HF Spaces Docker image.
#
# Base: NVIDIA CUDA 12.4 runtime (HF Spaces with t4-small expose a T4).
# Falls back to CPU/MPS at runtime if no CUDA is detected (model_manager picks
# the right device automatically).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps. Pinned where it matters:
#   - torch 2.x w/ CUDA 12.4 wheels.
#   - transformers >= 4.40 (covers the FloodDiffusionTiny custom-code model).
#     Our code is also tested with 5.x; either works on the inference path.
#   - lightning + omegaconf + safetensors required by the streaming model.
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
        flask flask-cors numpy "transformers>=4.40,<6" accelerate sentencepiece protobuf \
        lightning omegaconf safetensors diffusers ftfy regex einops huggingface_hub

# Non-root user (HF Spaces runs as uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app
COPY --chown=user:user . .

EXPOSE 7860
CMD ["python3", "app.py", "--port", "7860"]
