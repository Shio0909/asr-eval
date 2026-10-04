"""Client for the optional GPU scoring service.

The dashboard and scorer mount the same shared volume at ``/data``.  Audio paths
are resolved before sending so dashboard symlinks such as ``/app/audio_out``
become scorer-visible ``/data/...`` paths.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any


_REQUEST_STATE = threading.local()


def _set_last_error(value: str | None) -> None:
    _REQUEST_STATE.last_error = value


def scorer_last_error() -> str | None:
    return getattr(_REQUEST_STATE, "last_error", None)


def _sanitize_error(value: str) -> str:
    value = re.sub(r"/data(?:/[^\s'\"]*)?", "<data-path>", value)
    value = re.sub(r"\bhf_[A-Za-z0-9_-]+\b", "<redacted>", value)
    return value[:320]


def scorer_url() -> str:
    return os.environ.get("EVAL_SCORER_URL", "").strip().rstrip("/")


def scorer_configured() -> bool:
    return bool(scorer_url())


def shared_audio_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _request(method: str, path: str, payload: dict | None = None,
             timeout: float | None = None) -> dict[str, Any] | None:
    base = scorer_url()
    if not base:
        _set_last_error("GPU scorer 未配置")
        return None
    try:
        import requests

        response = requests.request(
            method,
            base + path,
            json=payload,
            timeout=timeout or float(os.environ.get("EVAL_SCORER_TIMEOUT_S", "3600")),
        )
        response.raise_for_status()
        data = response.json()
        _set_last_error(None)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        detail = None
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = body.get("detail") or body.get("error")
            except (TypeError, ValueError):
                pass
        _set_last_error(_sanitize_error(str(detail or exc) or type(exc).__name__))
        return None


def scorer_health(timeout: float = 5.0) -> dict[str, Any] | None:
    return _request("GET", "/health", timeout=timeout)


def scorer_runtime_available() -> bool:
    health = scorer_health()
    return bool(health and health.get("ok"))


def scorer_ping() -> dict[str, Any] | None:
    return _request("POST", "/v1/ping", {})


def scorer_warmup(metric: str) -> dict[str, Any] | None:
    return _request("POST", "/v1/warmup", {"metric": metric}, timeout=15.0)


def xcomet_score(triples: list[tuple[str, str, str]], batch_size: int = 4) -> dict | None:
    if not triples:
        return None
    response = _request(
        "POST",
        "/v1/score/xcomet",
        {
            "items": [
                {"id": str(i), "src": src or "", "ref": ref or "", "hyp": hyp or ""}
                for i, (src, ref, hyp) in enumerate(triples)
            ],
            "batch_size": max(1, min(int(batch_size), 8)),
        },
    )
    if not response:
        return None
    by_id = {str(item.get("id")): item for item in response.get("items") or []}
    if any(str(i) not in by_id for i in range(len(triples))):
        return None
    ordered = [by_id[str(i)] for i in range(len(triples))]
    try:
        scores = [round(float(item["score"]), 4) for item in ordered]
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "model": response.get("model"),
        "system_score": response.get("system_score"),
        "scores": scores,
        "error_spans": [item.get("error_spans") or [] for item in ordered],
    }


def whisper_transcribe(items: list[tuple[str, str, str | None]]) -> dict | None:
    if not items:
        return None
    response = _request(
        "POST",
        "/v1/score/whisper",
        {
            "items": [
                {
                    "id": str(item_id),
                    "audio_path": shared_audio_path(path),
                    "language": language,
                }
                for item_id, path, language in items
            ]
        },
    )
    if not response:
        return None
    rows = response.get("items") or []
    if not isinstance(rows, list):
        return None
    return {"model": response.get("model"), "items": rows}


def utmos_score(items: list[tuple[str, str]]) -> dict | None:
    if not items:
        return None
    response = _request(
        "POST",
        "/v1/score/utmos",
        {"items": [
            {"id": str(item_id), "audio_path": shared_audio_path(path)}
            for item_id, path in items
        ]},
    )
    if not response or not isinstance(response.get("items"), list):
        return None
    return {"model": response.get("model"), "items": response["items"]}


def speaker_score(items: list[tuple[str, str, str]]) -> dict | None:
    if not items:
        return None
    response = _request(
        "POST",
        "/v1/score/speaker",
        {"items": [
            {
                "id": str(item_id),
                "source_audio_path": shared_audio_path(source),
                "generated_audio_path": shared_audio_path(generated),
            }
            for item_id, source, generated in items
        ]},
    )
    if not response or not isinstance(response.get("items"), list):
        return None
    return {"model": response.get("model"), "items": response["items"]}
