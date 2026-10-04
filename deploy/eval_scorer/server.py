"""Single-GPU scoring service for expensive offline evaluation metrics."""

from __future__ import annotations

import fcntl
import gc
import importlib.util
import json
import math
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


DATA_ROOT = Path(os.environ.get("SCORER_DATA_ROOT", "/data")).resolve()
GPU_LOCK_PATH = Path(
    os.environ.get("SCORER_GPU_LOCK", "/tmp/asr-eval-scorer-gpu.lock")
)
MAX_ITEMS = int(os.environ.get("SCORER_MAX_ITEMS", "4096"))

app = FastAPI(title="ASR Eval GPU Scorer", version="0.1.10")


class XCometItem(BaseModel):
    id: str
    src: str
    ref: str
    hyp: str


class XCometRequest(BaseModel):
    items: list[XCometItem]
    batch_size: int = Field(default=4, ge=1, le=8)


class AudioItem(BaseModel):
    id: str
    audio_path: str


class AudioRequest(BaseModel):
    items: list[AudioItem]


class SpeakerItem(BaseModel):
    id: str
    source_audio_path: str
    generated_audio_path: str


class SpeakerRequest(BaseModel):
    items: list[SpeakerItem]


class WhisperItem(AudioItem):
    language: str | None = None


class WhisperRequest(BaseModel):
    items: list[WhisperItem]


class WarmupRequest(BaseModel):
    metric: str


def _check_items(items: list[Any]) -> None:
    if not items:
        raise HTTPException(status_code=422, detail="items must not be empty")
    if len(items) > MAX_ITEMS:
        raise HTTPException(
            status_code=422, detail=f"items exceeds limit {MAX_ITEMS}"
        )


def _resolve_audio_path(value: str, root: Path | None = None) -> Path:
    allowed_root = (root or DATA_ROOT).resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = allowed_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(allowed_root)
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=422,
            detail=f"audio path must be an existing file under {allowed_root}",
        ) from None
    if not resolved.is_file():
        raise HTTPException(status_code=422, detail="audio path is not a file")
    return resolved


def _runtime_probe() -> dict[str, Any]:
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("comet", "faster_whisper", "utmosv2", "wespeaker")
    }
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        capability = list(torch.cuda.get_device_capability(0)) if cuda else None
        total_gib = (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if cuda
            else None
        )
        torch_version = torch.__version__
    except Exception:
        cuda, gpu, capability, total_gib, torch_version = False, None, None, None, None
    return {
        "cuda": cuda,
        "gpu": gpu,
        "compute_capability": capability,
        "gpu_memory_total_gib": total_gib,
        "torch": torch_version,
        "packages": packages,
    }


class _SingleModelSlot:
    """Keep at most one heavyweight scorer resident on the 32 GiB GPU."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.model: Any = None
        self.lock = threading.Lock()

    def get(self, name: str, factory) -> Any:
        with self.lock:
            if self.name == name and self.model is not None:
                return self.model
            self.release()
            self.model = factory()
            self.name = name
            return self.model

    def release(self) -> None:
        self.model = None
        self.name = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


_SLOT = _SingleModelSlot()
_BUSY_LOCK = threading.Lock()
_BUSY_METRIC: str | None = None
_ERROR_LOCK = threading.Lock()
_LAST_ERROR: dict[str, str] | None = None


def _sanitize_error(exc: Exception) -> dict[str, str]:
    message = str(exc).strip() or "no error message"
    message = re.sub(r"/data(?:/[^\s'\"]*)?", "<data-path>", message)
    message = re.sub(r"\bhf_[A-Za-z0-9_-]+\b", "<redacted>", message)
    message = re.sub(
        r"(?i)\b(token|secret|password|api[_-]?key)\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        message,
    )
    return {"type": type(exc).__name__, "message": message[:300]}


def _clear_last_error() -> None:
    global _LAST_ERROR
    with _ERROR_LOCK:
        _LAST_ERROR = None


def _record_metric_error(metric: str, exc: Exception) -> dict[str, str]:
    global _LAST_ERROR
    error = _sanitize_error(exc)
    error["message"] = f"{metric}: {error['message']}"[:300]
    with _ERROR_LOCK:
        _LAST_ERROR = error
    return error


def _metric_http_error(metric: str, exc: Exception) -> HTTPException:
    error = _record_metric_error(metric, exc)
    return HTTPException(
        status_code=500,
        detail=f"{error['type']}: {error['message']}",
    )


@app.exception_handler(Exception)
async def _internal_error(_request: Request, exc: Exception):
    global _LAST_ERROR
    error = _sanitize_error(exc)
    with _ERROR_LOCK:
        _LAST_ERROR = error
    return JSONResponse(
        status_code=500,
        content={"detail": f"{error['type']}: {error['message']}"},
    )


@contextmanager
def _gpu_guard(metric: str):
    global _BUSY_METRIC
    GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        with _BUSY_LOCK:
            _BUSY_METRIC = metric
        try:
            yield
        finally:
            with _BUSY_LOCK:
                _BUSY_METRIC = None
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _need_cuda() -> None:
    if not _runtime_probe()["cuda"]:
        raise HTTPException(status_code=503, detail="CUDA GPU is unavailable")


def _xcomet_model():
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    model_name = os.environ.get("XCOMET_MODEL", "Unbabel/XCOMET-XL")
    local_dir = Path(os.environ.get(
        "XCOMET_LOCAL_DIR", "/data/cache/huggingface/xcomet-xl"
    ))
    local_checkpoint = local_dir / "checkpoints" / "model.ckpt"
    if local_checkpoint.is_file():
        return load_from_checkpoint(str(local_checkpoint))
    # comet.download_model() catches every Hub exception and rewrites it as the
    # misleading "model not supported" KeyError.  Resolve the gated repository
    # explicitly so HF_TOKEN is unambiguous and access/network errors survive.
    model_dir = Path(snapshot_download(
        repo_id=model_name,
        token=os.environ.get("HF_TOKEN") or None,
    ))
    checkpoint = model_dir / "checkpoints" / "model.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"XCOMET checkpoint not found: {checkpoint}")
    return load_from_checkpoint(str(checkpoint))


def _utmos_model():
    import timm
    import utmosv2

    checkpoint = Path(os.environ.get(
        "UTMOS_CHECKPOINT",
        "/app/models/utmosv2/fusion_stage3/fold0_s42_best_model.pth",
    ))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UTMOS checkpoint not found: {checkpoint}")

    # UTMOSv2 hard-codes ``pretrained=True`` for its timm backbones even though
    # the full UTMOS checkpoint loaded below replaces those initialization
    # weights.  Disable only that redundant construction-time download so the
    # scorer remains offline and deterministic.
    original_create_model = timm.create_model

    def create_model_offline(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_offline
    try:
        return utmosv2.create_model(
            pretrained=True,
            config=os.environ.get("UTMOS_CONFIG", "fusion_stage3"),
            checkpoint_path=checkpoint,
            device="cuda:0",
        )
    finally:
        timm.create_model = original_create_model


def _speaker_model():
    import wespeaker

    model = wespeaker.load_model(os.environ.get("WESPEAKER_MODEL", "vblinkp"))
    model.set_vad(True)
    # WeSpeaker's fbank frontend returns a CPU tensor.  Its public
    # ``set_device('cuda')`` moves only the model and then fails at inference
    # with a CPU-input/CUDA-weight mismatch, so keep this lightweight scorer on
    # CPU while the other three metrics use the colocated GPU.
    model.set_device(os.environ.get("WESPEAKER_DEVICE", "cpu"))
    return model


def _whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        _whisper_model_ref(),
        device="cuda",
        compute_type="float16",
        download_root=os.environ.get("HF_HOME"),
    )


def _whisper_model_ref() -> str:
    return os.environ.get("WHISPER_MODEL", "large-v3")


def _whisper_model_label() -> str:
    return os.environ.get("WHISPER_MODEL_LABEL") or Path(_whisper_model_ref()).name


def _error_spans(output: Any, n: int) -> list[list[dict[str, Any]]]:
    metadata = getattr(output, "metadata", None)
    spans = getattr(metadata, "error_spans", None)
    if not isinstance(spans, list) or len(spans) != n:
        return [[] for _ in range(n)]
    return _to_builtin(spans)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


@app.get("/health")
def health() -> dict[str, Any]:
    probe = _runtime_probe()
    with _BUSY_LOCK:
        busy = _BUSY_METRIC
    with _ERROR_LOCK:
        last_error = dict(_LAST_ERROR) if _LAST_ERROR else None
    return {
        "ok": bool(probe["cuda"] and all(probe["packages"].values())),
        "version": app.version,
        "busy_metric": busy,
        "resident_model": _SLOT.name,
        "last_error": last_error,
        **probe,
    }


@app.get("/ready")
def ready() -> dict[str, bool]:
    probe = _runtime_probe()
    if not probe["cuda"] or not all(probe["packages"].values()):
        raise HTTPException(status_code=503, detail="GPU scorer runtime is not ready")
    return {"ok": True}


@app.post("/v1/ping")
def ping() -> dict[str, Any]:
    return {"ok": True, "version": app.version}


def _warmup(metric: str) -> None:
    model_specs = {
        "xcomet": (
            f"xcomet:{os.environ.get('XCOMET_MODEL', 'Unbabel/XCOMET-XL')}",
            _xcomet_model,
        ),
        "utmos": (
            f"utmos:{os.environ.get('UTMOS_CONFIG', 'fusion_stage3')}",
            _utmos_model,
        ),
        "speaker": (
            f"speaker:{os.environ.get('WESPEAKER_MODEL', 'vblinkp')}",
            _speaker_model,
        ),
        "whisper": (
            f"whisper:{_whisper_model_ref()}",
            _whisper_model,
        ),
    }
    try:
        _clear_last_error()
        _need_cuda()
        name, factory = model_specs[metric]
        with _gpu_guard(metric):
            _SLOT.get(name, factory)
    except Exception as exc:
        _record_metric_error(metric, exc)


@app.post("/v1/warmup", status_code=202)
def warmup(request: WarmupRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    metric = request.metric.strip().lower()
    if metric not in {"xcomet", "utmos", "speaker", "whisper"}:
        raise HTTPException(status_code=422, detail="unsupported warmup metric")
    with _BUSY_LOCK:
        if _BUSY_METRIC:
            raise HTTPException(
                status_code=409, detail=f"GPU scorer is busy with {_BUSY_METRIC}"
            )
    background_tasks.add_task(_warmup, metric)
    return {"ok": True, "metric": metric, "status": "starting"}


@app.post("/v1/score/xcomet")
def score_xcomet(request: XCometRequest) -> dict[str, Any]:
    _check_items(request.items)
    _need_cuda()
    _clear_last_error()
    model_name = os.environ.get("XCOMET_MODEL", "Unbabel/XCOMET-XL")
    data = [
        {"src": item.src, "ref": item.ref, "mt": item.hyp}
        for item in request.items
    ]
    with _gpu_guard("xcomet"):
        model = _SLOT.get(f"xcomet:{model_name}", _xcomet_model)
        output = model.predict(
            data,
            batch_size=request.batch_size,
            gpus=1,
            num_workers=1,
            progress_bar=False,
        )
    scores = getattr(output, "scores", None)
    if scores is None or len(scores) != len(request.items):
        raise HTTPException(status_code=500, detail="XCOMET returned invalid scores")
    spans = _error_spans(output, len(request.items))
    return {
        "metric": "xcomet",
        "model": model_name,
        "system_score": float(getattr(output, "system_score", sum(scores) / len(scores))),
        "items": [
            {"id": item.id, "score": float(score), "error_spans": item_spans}
            for item, score, item_spans in zip(request.items, scores, spans)
        ],
    }


@app.post("/v1/score/utmos")
def score_utmos(request: AudioRequest) -> dict[str, Any]:
    _check_items(request.items)
    _need_cuda()
    _clear_last_error()
    paths = [_resolve_audio_path(item.audio_path) for item in request.items]
    config = os.environ.get("UTMOS_CONFIG", "fusion_stage3")
    with _gpu_guard("utmos"):
        model = _SLOT.get(f"utmos:{config}", _utmos_model)
        scores = [
            model.predict(
                input_path=path,
                device="cuda:0",
                num_workers=0,
                batch_size=1,
                verbose=False,
            )
            for path in paths
        ]
    return {
        "metric": "utmos_v2",
        "model": config,
        "items": [
            {"id": item.id, "score": float(score)}
            for item, score in zip(request.items, scores)
        ],
    }


@app.post("/v1/score/speaker")
def score_speaker(request: SpeakerRequest) -> JSONResponse:
    _check_items(request.items)
    _need_cuda()
    _clear_last_error()
    pairs = [
        (
            _resolve_audio_path(item.source_audio_path),
            _resolve_audio_path(item.generated_audio_path),
        )
        for item in request.items
    ]
    model_name = os.environ.get("WESPEAKER_MODEL", "vblinkp")
    try:
        with _gpu_guard("speaker"):
            model = _SLOT.get(f"speaker:{model_name}", _speaker_model)
            scores = []
            for source_path, generated_path in pairs:
                source = model.extract_embedding(str(source_path))
                generated = model.extract_embedding(str(generated_path))
                if source is None or generated is None:
                    scores.append(None)
                    continue
                import torch

                cosine = float(torch.nn.functional.cosine_similarity(
                    source.reshape(1, -1), generated.reshape(1, -1)
                ).item())
                scores.append(max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else None)
        result = {
            "metric": "speaker_cosine",
            "model": model_name,
            "items": [
                {
                    "id": item.id,
                    "cosine": score,
                    "normalized_similarity": ((score + 1.0) / 2.0 if score is not None else None),
                }
                for item, score in zip(request.items, scores)
            ],
        }
        json.dumps(result, allow_nan=False)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        raise _metric_http_error("speaker", exc) from None


@app.post("/v1/score/whisper")
def score_whisper(request: WhisperRequest) -> dict[str, Any]:
    _check_items(request.items)
    _need_cuda()
    _clear_last_error()
    paths = [_resolve_audio_path(item.audio_path) for item in request.items]
    model_ref = _whisper_model_ref()
    with _gpu_guard("whisper"):
        model = _SLOT.get(f"whisper:{model_ref}", _whisper_model)
        results = []
        for item, path in zip(request.items, paths):
            language = "zh" if item.language == "yue" else item.language
            segments, info = model.transcribe(
                str(path),
                language=language,
                beam_size=1,
                condition_on_previous_text=False,
            )
            results.append(
                {
                    "id": item.id,
                    "text": "".join(segment.text for segment in segments).strip(),
                    "language": getattr(info, "language", language),
                    "language_probability": float(
                        getattr(info, "language_probability", 0.0)
                    ),
                }
            )
    return {
        "metric": "whisper_transcript",
        "model": _whisper_model_label(),
        "items": results,
    }


@app.post("/v1/release")
def release_model() -> dict[str, Any]:
    with _gpu_guard("release"):
        _SLOT.release()
    return {"ok": True}
