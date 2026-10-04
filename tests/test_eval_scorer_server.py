import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


ROOT = Path(__file__).parents[1]
SERVER_PATH = ROOT / "deploy" / "eval_scorer" / "server.py"
SPEC = importlib.util.spec_from_file_location("eval_scorer_server", SERVER_PATH)
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


def test_audio_path_must_remain_under_shared_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    inside = root / "sample.wav"
    inside.write_bytes(b"RIFF")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")

    assert server._resolve_audio_path("sample.wav", root) == inside
    with pytest.raises(HTTPException) as exc:
        server._resolve_audio_path(str(outside), root)
    assert exc.value.status_code == 422


def test_error_spans_fall_back_when_model_has_no_metadata():
    assert server._error_spans(object(), 2) == [[], []]


def test_error_spans_preserve_xcomet_metadata():
    span = {"start": 1, "end": 4, "severity": "major", "text": "bad"}
    output = type(
        "Output",
        (),
        {"metadata": type("Metadata", (), {"error_spans": [[span]]})()},
    )()
    assert server._error_spans(output, 1) == [[span]]


def test_to_builtin_converts_scalar_objects():
    scalar = type("Scalar", (), {"item": lambda self: 0.25})()
    assert server._to_builtin({"confidence": scalar}) == {"confidence": 0.25}


def test_runtime_error_is_sanitized():
    error = server._sanitize_error(
        RuntimeError("failed /data/private/sample.wav token=hf_secretvalue")
    )
    assert error["type"] == "RuntimeError"
    assert "/data/private" not in error["message"]
    assert "hf_secretvalue" not in error["message"]
    assert "<data-path>" in error["message"]


def test_single_model_slot_releases_previous_model(monkeypatch):
    slot = server._SingleModelSlot()
    released = []
    monkeypatch.setattr(slot, "release", lambda: released.append(slot.name))

    first = slot.get("first", object)
    assert first is slot.get("first", object)
    slot.get("second", object)
    assert released == [None, "first"]


def test_health_reports_runtime_and_resident_model(monkeypatch):
    monkeypatch.setattr(
        server,
        "_runtime_probe",
        lambda: {
            "cuda": True,
            "gpu": "test-gpu",
            "compute_capability": [12, 0],
            "gpu_memory_total_gib": 32.0,
            "torch": "2.7.1",
            "packages": {
                "comet": True,
                "faster_whisper": True,
                "utmosv2": True,
                "wespeaker": True,
            },
        },
    )
    monkeypatch.setattr(server._SLOT, "name", "xcomet:test")
    result = server.health()
    assert result["ok"] is True
    assert result["resident_model"] == "xcomet:test"


def test_ready_rejects_cpu_only_runtime(monkeypatch):
    monkeypatch.setattr(
        server,
        "_runtime_probe",
        lambda: {"cuda": False, "packages": {"comet": True}},
    )
    with pytest.raises(HTTPException) as exc:
        server.ready()
    assert exc.value.status_code == 503


def test_ping_reports_service_version():
    assert server.ping() == {"ok": True, "version": "0.1.10"}


def test_xcomet_factory_passes_hf_token_and_loads_hub_checkpoint(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    calls = []
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("XCOMET_MODEL", "Unbabel/XCOMET-XL")
    monkeypatch.setenv("XCOMET_LOCAL_DIR", str(tmp_path / "missing-local"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: calls.append(("download", kwargs))
            or str(tmp_path)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "comet",
        SimpleNamespace(
            load_from_checkpoint=lambda path: calls.append(("load", path)) or "model"
        ),
    )

    assert server._xcomet_model() == "model"
    assert calls == [
        ("download", {"repo_id": "Unbabel/XCOMET-XL", "token": "hf_test_token"}),
        ("load", str(checkpoint)),
    ]


def test_xcomet_factory_prefers_shared_afs_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    calls = []
    monkeypatch.setenv("XCOMET_LOCAL_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: pytest.fail("Hub used")),
    )
    monkeypatch.setitem(
        sys.modules,
        "comet",
        SimpleNamespace(
            load_from_checkpoint=lambda path: calls.append(path) or "local-model"
        ),
    )

    assert server._xcomet_model() == "local-model"
    assert calls == [str(checkpoint)]


def test_utmos_factory_disables_redundant_timm_download(monkeypatch, tmp_path):
    checkpoint = tmp_path / "utmos.pth"
    checkpoint.write_bytes(b"weights")
    timm_calls = []
    create_calls = []
    fake_timm = SimpleNamespace(
        create_model=lambda *args, **kwargs: timm_calls.append((args, kwargs))
    )

    def fake_create_model(**kwargs):
        fake_timm.create_model("backbone", pretrained=True)
        create_calls.append(kwargs)
        return "utmos"

    monkeypatch.setenv("UTMOS_CHECKPOINT", str(checkpoint))
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setitem(
        sys.modules, "utmosv2", SimpleNamespace(create_model=fake_create_model)
    )

    assert server._utmos_model() == "utmos"
    assert timm_calls[0][1]["pretrained"] is False
    assert create_calls[0]["checkpoint_path"] == checkpoint


def test_utmos_offline_feature_extractor_config_is_bundled():
    path = SERVER_PATH.parent / "facebook" / "wav2vec2-base" / "preprocessor_config.json"
    config = json.loads(path.read_text())
    assert config["sampling_rate"] == 16000
    assert config["do_normalize"] is True


def test_speaker_model_stays_on_cpu(monkeypatch):
    calls = []
    model = SimpleNamespace(
        set_vad=lambda value: calls.append(("vad", value)),
        set_device=lambda value: calls.append(("device", value)),
    )
    monkeypatch.delenv("WESPEAKER_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "wespeaker",
        SimpleNamespace(load_model=lambda name: model),
    )

    assert server._speaker_model() is model
    assert calls == [("vad", True), ("device", "cpu")]


def test_warmup_runs_selected_factory(monkeypatch):
    loaded = []
    tasks = SimpleNamespace(add_task=lambda fn, metric: fn(metric))
    monkeypatch.setattr(server, "_need_cuda", lambda: None)
    monkeypatch.setattr(server, "_speaker_model", lambda: object())
    monkeypatch.setattr(
        server._SLOT, "get", lambda name, factory: loaded.append((name, factory()))
    )

    result = server.warmup(server.WarmupRequest(metric="speaker"), tasks)

    assert result["status"] == "starting"
    assert loaded[0][0].startswith("speaker:")


def test_xcomet_endpoint_maps_scores_and_error_spans(monkeypatch):
    output = SimpleNamespace(
        scores=[0.75],
        system_score=0.75,
        metadata=SimpleNamespace(
            error_spans=[
                [{"start": 0, "end": 3, "severity": "minor", "text": "bad"}]
            ]
        ),
    )
    model = SimpleNamespace(predict=lambda *args, **kwargs: output)
    monkeypatch.setattr(server, "_need_cuda", lambda: None)
    monkeypatch.setattr(server._SLOT, "get", lambda *args, **kwargs: model)

    result = server.score_xcomet(
        server.XCometRequest(
            items=[server.XCometItem(id="one", src="a", ref="b", hyp="c")]
        )
    )

    assert result["system_score"] == 0.75
    assert result["items"][0] == {
        "id": "one",
        "score": 0.75,
        "error_spans": [
            {"start": 0, "end": 3, "severity": "minor", "text": "bad"}
        ],
    }


def test_xcomet_batch_size_is_capped_for_32_gib_gpu():
    with pytest.raises(ValidationError):
        server.XCometRequest(items=[], batch_size=9)


def test_utmos_endpoint_resolves_audio_and_returns_mos(monkeypatch, tmp_path):
    audio = tmp_path / "one.wav"
    audio.write_bytes(b"RIFF")
    model = SimpleNamespace(predict=lambda **kwargs: 3.8)
    monkeypatch.setattr(server, "_need_cuda", lambda: None)
    monkeypatch.setattr(server, "_resolve_audio_path", lambda value: audio)
    monkeypatch.setattr(server._SLOT, "get", lambda *args, **kwargs: model)

    result = server.score_utmos(
        server.AudioRequest(items=[server.AudioItem(id="one", audio_path="one.wav")])
    )

    assert result["items"] == [{"id": "one", "score": 3.8}]


def test_whisper_endpoint_collects_segments(monkeypatch, tmp_path):
    audio = tmp_path / "one.wav"
    audio.write_bytes(b"RIFF")

    class FakeWhisper:
        def transcribe(self, *args, **kwargs):
            return (
                iter([SimpleNamespace(text=" hello"), SimpleNamespace(text=" world")]),
                SimpleNamespace(language="en", language_probability=0.99),
            )

    monkeypatch.setattr(server, "_need_cuda", lambda: None)
    monkeypatch.setattr(server, "_resolve_audio_path", lambda value: audio)
    monkeypatch.setattr(server._SLOT, "get", lambda *args, **kwargs: FakeWhisper())

    result = server.score_whisper(
        server.WhisperRequest(
            items=[server.WhisperItem(id="one", audio_path="one.wav", language="en")]
        )
    )

    assert result["items"] == [
        {
            "id": "one",
            "text": "hello world",
            "language": "en",
            "language_probability": 0.99,
        }
    ]


def test_whisper_reports_public_label_for_bundled_model(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "/app/models/faster-whisper-large-v3")
    monkeypatch.setenv("WHISPER_MODEL_LABEL", "large-v3")

    assert server._whisper_model_ref() == "/app/models/faster-whisper-large-v3"
    assert server._whisper_model_label() == "large-v3"


def test_speaker_endpoint_returns_raw_and_normalized_similarity(
    monkeypatch, tmp_path
):
    audio = tmp_path / "one.wav"
    audio.write_bytes(b"RIFF")

    class Embedding:
        def reshape(self, *args):
            return self

    model = SimpleNamespace(extract_embedding=lambda path: Embedding())
    cosine = SimpleNamespace(item=lambda: 0.6)
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(
            functional=SimpleNamespace(cosine_similarity=lambda *args: cosine)
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(server, "_need_cuda", lambda: None)
    monkeypatch.setattr(server, "_resolve_audio_path", lambda value: audio)
    monkeypatch.setattr(server._SLOT, "get", lambda *args, **kwargs: model)

    response = server.score_speaker(
        server.SpeakerRequest(
            items=[
                server.SpeakerItem(
                    id="one",
                    source_audio_path="source.wav",
                    generated_audio_path="generated.wav",
                )
            ]
        )
    )

    result = json.loads(response.body)
    assert result["items"] == [
        {"id": "one", "cosine": 0.6, "normalized_similarity": 0.8}
    ]


def test_speaker_endpoint_does_not_serialize_nan(monkeypatch, tmp_path):
    audio = tmp_path / "one.wav"
    audio.write_bytes(b"RIFF")

    class Embedding:
        def reshape(self, *args):
            return self

    model = SimpleNamespace(extract_embedding=lambda path: Embedding())
    cosine = SimpleNamespace(item=lambda: math.nan)
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(
            functional=SimpleNamespace(cosine_similarity=lambda *args: cosine)
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(server, "_need_cuda", lambda: None)
    monkeypatch.setattr(server, "_resolve_audio_path", lambda value: audio)
    monkeypatch.setattr(server._SLOT, "get", lambda *args, **kwargs: model)

    response = server.score_speaker(
        server.SpeakerRequest(
            items=[server.SpeakerItem(
                id="one", source_audio_path="source.wav",
                generated_audio_path="generated.wav",
            )]
        )
    )

    result = json.loads(response.body)
    assert result["items"][0]["cosine"] is None
    assert result["items"][0]["normalized_similarity"] is None
