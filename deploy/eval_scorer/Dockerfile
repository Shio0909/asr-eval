# syntax=docker/dockerfile:1.7
ARG SCORER_PLATFORM=linux/amd64
FROM --platform=$SCORER_PLATFORM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl ffmpeg libsndfile1 libsox-fmt-all python3 python3-pip python3-venv sox \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/asr-eval-scorer/venv \
    PATH=/opt/asr-eval-scorer/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/cache/huggingface \
    UTMOSV2_CHACHE=/data/cache/utmosv2 \
    WESPEAKER_HOME=/data/cache/wespeaker \
    SCORER_DATA_ROOT=/data \
    SCORER_GPU_LOCK=/tmp/asr-eval-scorer-gpu.lock \
    XCOMET_MODEL=Unbabel/XCOMET-XL \
    UTMOS_CONFIG=fusion_stage3 \
    UTMOS_CHECKPOINT=/app/models/utmosv2/fusion_stage3/fold0_s42_best_model.pth \
    WESPEAKER_MODEL=vblinkp \
    WESPEAKER_DEVICE=cpu \
    WHISPER_MODEL=/app/models/faster-whisper-large-v3 \
    WHISPER_MODEL_LABEL=large-v3

WORKDIR /app

COPY requirements.txt /tmp/asr-eval-scorer-requirements.txt
RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir --upgrade pip 'setuptools<81' wheel \
    && python -m pip install --no-cache-dir \
       torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
       --index-url "$PYTORCH_INDEX_URL" \
    && python -m pip install --no-cache-dir -r /tmp/asr-eval-scorer-requirements.txt \
    && python -c "import comet, fastapi, faster_whisper, torch, utmosv2, wespeaker"

COPY facebook /app/facebook
ARG WAV2VEC2_COMMIT=0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
RUN curl -fsSL \
      "https://huggingface.co/facebook/wav2vec2-base/resolve/${WAV2VEC2_COMMIT}/config.json" \
      -o /app/facebook/wav2vec2-base/config.json \
    && curl -fsSL \
      "https://huggingface.co/facebook/wav2vec2-base/resolve/${WAV2VEC2_COMMIT}/pytorch_model.bin" \
      -o /app/facebook/wav2vec2-base/pytorch_model.bin \
    && echo "4937977e24d12d1bba70cdce8709c3c04807a8e4ae8ddac4229c48c436ae99ae  /app/facebook/wav2vec2-base/config.json" | sha256sum -c - \
    && echo "3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21  /app/facebook/wav2vec2-base/pytorch_model.bin" | sha256sum -c -
ARG UTMOSV2_MODEL_COMMIT=506474f2b33dc77c234d668cc419be1861899cad
RUN mkdir -p /app/models/utmosv2/fusion_stage3 \
    && curl -fsSL \
      "https://huggingface.co/sarulab-speech/UTMOSv2/resolve/${UTMOSV2_MODEL_COMMIT}/fold0_s42_best_model.pth" \
      -o /app/models/utmosv2/fusion_stage3/fold0_s42_best_model.pth \
    && echo "c8149d988e4bbf3f347e6966b5d769de347a5f8c59ffca1dc4bd4bf5b8585e57  /app/models/utmosv2/fusion_stage3/fold0_s42_best_model.pth" | sha256sum -c -
ARG FASTER_WHISPER_COMMIT=edaa852ec7e145841d8ffdb056a99866b5f0a478
RUN mkdir -p /app/models/faster-whisper-large-v3 \
    && curl -fsSL --retry 5 \
      "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/${FASTER_WHISPER_COMMIT}/config.json" \
      -o /app/models/faster-whisper-large-v3/config.json \
    && curl -fsSL --retry 5 \
      "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/${FASTER_WHISPER_COMMIT}/model.bin" \
      -o /app/models/faster-whisper-large-v3/model.bin \
    && curl -fsSL --retry 5 \
      "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/${FASTER_WHISPER_COMMIT}/preprocessor_config.json" \
      -o /app/models/faster-whisper-large-v3/preprocessor_config.json \
    && curl -fsSL --retry 5 \
      "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/${FASTER_WHISPER_COMMIT}/tokenizer.json" \
      -o /app/models/faster-whisper-large-v3/tokenizer.json \
    && curl -fsSL --retry 5 \
      "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/${FASTER_WHISPER_COMMIT}/vocabulary.json" \
      -o /app/models/faster-whisper-large-v3/vocabulary.json \
    && echo "a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9  /app/models/faster-whisper-large-v3/config.json" | sha256sum -c - \
    && echo "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1  /app/models/faster-whisper-large-v3/model.bin" | sha256sum -c - \
    && echo "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711  /app/models/faster-whisper-large-v3/preprocessor_config.json" | sha256sum -c - \
    && echo "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca  /app/models/faster-whisper-large-v3/tokenizer.json" | sha256sum -c - \
    && echo "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1  /app/models/faster-whisper-large-v3/vocabulary.json" | sha256sum -c -
COPY server.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh \
    && python -c "import server; probe = server.health(); assert all(probe['packages'].values())"

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=6 \
    CMD curl -fsS http://127.0.0.1:8090/ready || exit 1

# Preserve NVIDIA's base-image entrypoint so its driver compatibility checks run.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/app/entrypoint.sh"]
