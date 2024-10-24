FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.30.3 transformers==4.44.0 accelerate==0.33.0 && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large/resolve/main/clip_g.safetensors -d /content/ComfyUI/models/clip -o clip_g.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large/resolve/main/clip_l.safetensors -d /content/ComfyUI/models/clip -o clip_l.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large/resolve/main/t5xxl_fp16.safetensors -d /content/ComfyUI/models/clip -o t5xxl_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors -d /content/ComfyUI/models/checkpoints -o sd3.5_large.safetensors
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large-turbo/resolve/main/clip_g.safetensors -d /content/ComfyUI/models/clip -o clip_g.safetensors && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large-turbo/resolve/main/clip_l.safetensors -d /content/ComfyUI/models/clip -o clip_l.safetensors && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large-turbo/resolve/main/t5xxl_fp16.safetensors -d /content/ComfyUI/models/clip -o t5xxl_fp16.safetensors && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/stable-diffusion-3.5-large-turbo/resolve/main/sd3.5_large_turbo.safetensors -d /content/ComfyUI/models/checkpoints -o sd3.5_large.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py