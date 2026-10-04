#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  "${HF_HOME:-/data/cache/huggingface}" \
  "${UTMOSV2_CHACHE:-/data/cache/utmosv2}" \
  "${WESPEAKER_HOME:-/data/cache/wespeaker}"

exec uvicorn server:app --app-dir /app --host 0.0.0.0 --port 8090 --workers 1
