# ASR Eval GPU Scorer

One scoring service for the production evaluation dashboard.  It is deployed
separately from the CPU dashboard and exposes these score-only endpoints:

- `POST /v1/score/xcomet`: `Unbabel/XCOMET-XL` score and MQM error spans.
- `POST /v1/score/utmos`: UTMOSv2 naturalness MOS prediction.
- `POST /v1/score/speaker`: WeSpeaker source/generated voice cosine similarity.
- `POST /v1/score/whisper`: faster-whisper transcript for TTS CER/WER scoring.
- `POST /v1/warmup`: load one metric asynchronously; `POST /v1/release` frees it.
- `POST /v1/ping`, `GET /health`, `GET /ready`: deployment diagnostics.

Audio inputs are paths below the shared `/data` mount.  The service does
not accept uploads or arbitrary host paths.  A cross-process file lock
serializes scoring, and at most one heavyweight model remains resident.
WeSpeaker runs on CPU because its public fbank frontend produces CPU tensors;
XCOMET, UTMOS and faster-whisper use the RTX 5090.

## Deployment

- Port: `8090`
- Resources: one GPU (tested on a 32 GB RTX 5090), roughly 14 CPU / 120 GiB RAM.
- Shared storage: mount the same volume as the dashboard at `/data`.

Point the dashboard at the service with:

```text
EVAL_SCORER_URL=http://<scorer-host>:8090
EVAL_SCORER_TIMEOUT_S=3600
```

Do not expose port 8090 publicly.  The dashboard selects paths from its own
manifests and the scorer independently checks that every path stays under the
shared `/data` root.

## Reproducibility and model files

The image pins the CUDA 12.8 base-image digest and installs PyTorch 2.7.1 cu128
for Blackwell support.  It also pins and checksum-verifies the offline files
needed by UTMOSv2 and faster-whisper:

- `facebook/wav2vec2-base` commit
  `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`.
- UTMOSv2 `fusion_stage3/fold0_s42_best_model.pth` from commit
  `506474f2b33dc77c234d668cc419be1861899cad`.
- `Systran/faster-whisper-large-v3` commit
  `edaa852ec7e145841d8ffdb056a99866b5f0a478`.

WeSpeaker `vblinkp` is cached on the persistent shared volume.  XCOMET-XL is
gated and is intentionally not bundled: accept its Hugging Face terms, review
its CC-BY-NC-SA model license, then add `HF_TOKEN` as a masked secret in your deployment platform.
Without that token the other three metrics remain available and XCOMET reports
an explicit diagnostic instead of silently falling back to another model.

Keep this as a separate GPU service.  COMET 2.2.7 resolves to protobuf 4.25.9,
while the CPU evaluation image requires protobuf 6.31.1 or newer.  Combining
them would create an actual dependency conflict.

## Build

```bash
docker buildx build \
  --platform linux/amd64 \
  -f deploy/eval_scorer/Dockerfile \
  -t asr-eval-scorer:latest \
  --load \
  deploy/eval_scorer
```

## Verify

After deployment, run one real sample through speaker, UTMOS and Whisper, and
check the reported version with `POST /v1/ping`.  Package imports and `/ready`
only prove that the runtime starts; they do not prove that all model artifacts
load or that GPU kernels execute.
