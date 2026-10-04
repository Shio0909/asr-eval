"""Batch a manifest through a FireRedLID-compatible ``POST /lid`` endpoint.

This is intentionally inference-only: it preserves reference language/dialect
fields for later analysis but does not assume a label mapping or calculate
metrics.

Example:
  python eval/lid_infer.py \
    --manifest manifests/kespeech_jianghuai_full.jsonl \
    --url http://fireredasr2-aed-lid:8000 \
    --out infer/fireredlid__kespeech_jianghuai_full.jsonl
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime

import requests

from adapters import load_wav_bytes
from config import abs_path
from logconf import redact
from util import code_revision, manifest_sha_v2


def _endpoint(url):
    value = url.rstrip("/")
    return value if value.endswith("/lid") else value + "/lid"


def _load_output(path):
    done, metas = set(), []
    if not os.path.exists(path):
        return done, metas
    with open(path, encoding="utf-8") as src:
        for lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是有效 JSON") from exc
            if row.get("id") == "__meta__":
                metas.append(row)
            elif row.get("ok"):
                done.add(str(row.get("id")))
    return done, metas


def run(manifest, url, out, *, limit=0, timeout=120.0, retries=2, experiment=""):
    endpoint = _endpoint(url)
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
    rows = [row for row in rows if row.get("id") != "__meta__"]
    if limit:
        rows = rows[:limit]

    current_sha = manifest_sha_v2(manifest)
    done, metas = _load_output(out)
    if metas:
        previous = metas[-1]
        if (previous.get("manifest_sha_v2"), previous.get("endpoint")) != (current_sha, endpoint):
            raise ValueError("已有 LID 输出的 manifest 指纹或 endpoint 不同，请更换 --out")
    todo = [row for row in rows if str(row.get("id")) not in done]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    dst = open(out, "a", encoding="utf-8")
    try:
        if todo:
            dst.write(json.dumps({
                "id": "__meta__",
                "schema_version": 1,
                "task": "lid",
                "run_id": uuid.uuid4().hex,
                "manifest": os.path.basename(manifest),
                "manifest_sha_v2": current_sha,
                "endpoint": endpoint,
                "code_revision": code_revision(),
                "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                "n_total": len(rows),
                "n_todo": len(todo),
                "retries": retries,
                "experiment": experiment or None,
            }, ensure_ascii=False) + "\n")
            dst.flush()

        for index, row in enumerate(todo, 1):
            started = time.monotonic()
            error = ""
            result = None
            for attempt in range(retries + 1):
                try:
                    wav = load_wav_bytes(abs_path(row.get("audio_path", "")))
                    response = requests.post(
                        endpoint,
                        files={"file": (f"{row['id']}.wav", wav, "audio/wav")},
                        timeout=timeout,
                    )
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as exc:
                        body = str(getattr(response, "text", "") or "").strip()
                        detail = f" · response: {body[:500]}" if body else ""
                        raise requests.HTTPError(f"{exc}{detail}", response=response) from exc
                    result = response.json()
                    if not isinstance(result, dict):
                        raise ValueError("LID response must be a JSON object")
                    if not (result.get("label") or result.get("language")):
                        raise ValueError("LID response has no label or language")
                    break
                except Exception as exc:
                    result = None
                    error = redact(str(exc))
                    if attempt < retries:
                        time.sleep(0.5 * (2 ** attempt))

            elapsed = round(time.monotonic() - started, 3)
            if result is None:
                record = {
                    "id": row.get("id"),
                    "ok": False,
                    "error": error,
                    "elapsed_s": elapsed,
                    "audio_path": row.get("audio_path"),
                    "ref_text": row.get("ref_text"),
                    "ref_language": row.get("lang"),
                    "ref_dialect": row.get("dialect"),
                }
            else:
                confidence = result.get("confidence")
                if confidence is None:
                    confidence = result.get("language_confidence")
                record = {
                    "id": row.get("id"),
                    "ok": True,
                    "pred_label": result.get("label") or result.get("lid_label"),
                    "pred_language": result.get("language"),
                    "pred_dialect": result.get("dialect"),
                    "confidence": confidence,
                    "service_duration_s": result.get("duration"),
                    "service_rtf": result.get("rtf"),
                    "elapsed_s": elapsed,
                    "audio_path": row.get("audio_path"),
                    "ref_text": row.get("ref_text"),
                    "ref_language": row.get("lang"),
                    "ref_dialect": row.get("dialect"),
                    "ref_label": row.get("ref_label") or row.get("lid_label"),
                }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            dst.flush()
            if index % 10 == 0 or index == len(todo):
                print(f"  {index}/{len(todo)}…", flush=True)
    finally:
        dst.close()

    print(f"LID 完成: 新跑 {len(todo)}，已复用 {len(done)} → {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--url", required=True, help="服务根地址或完整 /lid URL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--experiment", default="", help="结果溯源标签，如 fp16/fp32")
    args = parser.parse_args()
    run(args.manifest, args.url, args.out, limit=args.limit,
        timeout=args.timeout, retries=args.retries, experiment=args.experiment)


if __name__ == "__main__":
    main()
