import os
import sys

import requests


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import scorer_client


def test_unconfigured_scorer_does_not_make_request(monkeypatch):
    monkeypatch.delenv("EVAL_SCORER_URL", raising=False)
    assert scorer_client.scorer_health() is None
    assert scorer_client.xcomet_score([("src", "ref", "hyp")]) is None


def test_http_error_detail_is_available_but_sanitized(monkeypatch):
    class FailedResponse:
        def raise_for_status(self):
            raise requests.HTTPError(response=self)

        def json(self):
            return {"detail": "RuntimeError: failed /data/private/a.wav hf_secret"}

    monkeypatch.setenv("EVAL_SCORER_URL", "http://scorer")
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: FailedResponse())

    assert scorer_client.scorer_health() is None
    detail = scorer_client.scorer_last_error()
    assert detail.startswith("RuntimeError")
    assert "/data/private" not in detail
    assert "hf_secret" not in detail


def test_xcomet_response_is_restored_to_request_order(monkeypatch):
    seen = {}

    def fake_request(method, path, payload=None, timeout=None):
        seen.update(method=method, path=path, payload=payload)
        return {
            "model": "Unbabel/XCOMET-XL",
            "system_score": 0.75,
            "items": [
                {"id": "1", "score": 0.5, "error_spans": [{"severity": "minor"}]},
                {"id": "0", "score": 1.0, "error_spans": []},
            ],
        }

    monkeypatch.setattr(scorer_client, "_request", fake_request)
    result = scorer_client.xcomet_score([("a", "b", "c"), ("d", "e", "f")], batch_size=99)

    assert seen["path"] == "/v1/score/xcomet"
    assert seen["payload"]["batch_size"] == 8
    assert result["scores"] == [1.0, 0.5]
    assert result["error_spans"][1] == [{"severity": "minor"}]


def test_ping_and_warmup_use_short_control_requests(monkeypatch):
    seen = []

    def fake_request(method, path, payload=None, timeout=None):
        seen.append((method, path, payload, timeout))
        return {"ok": True}

    monkeypatch.setattr(scorer_client, "_request", fake_request)
    assert scorer_client.scorer_ping() == {"ok": True}
    assert scorer_client.scorer_warmup("utmos") == {"ok": True}
    assert seen == [
        ("POST", "/v1/ping", {}, None),
        ("POST", "/v1/warmup", {"metric": "utmos"}, 15.0),
    ]


def test_audio_clients_resolve_shared_symlink_paths(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    audio = data / "sample.wav"
    audio.write_bytes(b"RIFF")
    link = tmp_path / "audio.wav"
    link.symlink_to(audio)
    payloads = []

    def fake_request(method, path, payload=None, timeout=None):
        payloads.append((path, payload))
        if path.endswith("whisper"):
            return {"model": "large-v3", "items": [{"id": "x", "text": "hello"}]}
        if path.endswith("utmos"):
            return {"model": "fusion_stage3", "items": [{"id": "x", "score": 4.0}]}
        return {"model": "vblinkp", "items": [{"id": "x", "cosine": 0.8}]}

    monkeypatch.setattr(scorer_client, "_request", fake_request)

    assert scorer_client.whisper_transcribe([("x", str(link), "en")])["model"] == "large-v3"
    assert scorer_client.utmos_score([("x", str(link))])["items"][0]["score"] == 4.0
    assert scorer_client.speaker_score([("x", str(link), str(link))])["items"][0]["cosine"] == 0.8
    resolved = str(audio.resolve())
    assert payloads[0][1]["items"][0]["audio_path"] == resolved
    assert payloads[1][1]["items"][0]["audio_path"] == resolved
    assert payloads[2][1]["items"][0]["source_audio_path"] == resolved
    assert payloads[2][1]["items"][0]["generated_audio_path"] == resolved
