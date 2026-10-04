import importlib.util
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))

spec = importlib.util.spec_from_file_location("lid_infer", os.path.join(ROOT, "eval", "lid_infer.py"))
lid_infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lid_infer)


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "label": "zh xinan",
            "language": "zh",
            "dialect": "xinan",
            "confidence": 0.91,
            "duration": 2.5,
            "rtf": 0.02,
        }


def test_lid_infer_writes_predictions_and_reference_fields(tmp_path, monkeypatch):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "u1", "audio_path": "audio.wav", "ref_text": "你好",
        "lang": "zh", "dialect": "Southwestern",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "lid.jsonl"
    seen = {}

    monkeypatch.setattr(lid_infer, "load_wav_bytes", lambda _path: b"RIFF")

    def post(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Response()

    monkeypatch.setattr(lid_infer.requests, "post", post)

    lid_infer.run(str(manifest), "http://lid.test:8000", str(out),
                  retries=0, experiment="fp32")

    meta, record = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert meta["task"] == "lid" and meta["endpoint"] == "http://lid.test:8000/lid"
    assert meta["experiment"] == "fp32"
    assert record["id"] == "u1" and record["ok"] is True
    assert record["pred_label"] == "zh xinan"
    assert record["pred_language"] == "zh" and record["pred_dialect"] == "xinan"
    assert record["ref_language"] == "zh" and record["ref_dialect"] == "Southwestern"
    assert record["ref_text"] == "你好"
    assert seen["files"]["file"][1] == b"RIFF"


def test_lid_infer_resumes_successful_rows_without_calling_service(tmp_path, monkeypatch):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "u1", "audio_path": "audio.wav", "ref_text": "你好", "lang": "zh",
    }) + "\n", encoding="utf-8")
    out = tmp_path / "lid.jsonl"

    monkeypatch.setattr(lid_infer, "load_wav_bytes", lambda _path: b"RIFF")
    monkeypatch.setattr(lid_infer.requests, "post", lambda *_args, **_kwargs: Response())
    lid_infer.run(str(manifest), "http://lid.test:8000/lid", str(out), retries=0)
    before = out.read_text(encoding="utf-8")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("成功样本不应重复请求")

    monkeypatch.setattr(lid_infer.requests, "post", unexpected)
    lid_infer.run(str(manifest), "http://lid.test:8000", str(out), retries=0)

    assert out.read_text(encoding="utf-8") == before


def test_lid_infer_preserves_http_error_response_body(tmp_path, monkeypatch):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "u1", "audio_path": "audio.wav", "lang": "en",
    }) + "\n", encoding="utf-8")
    out = tmp_path / "lid.jsonl"

    class FailedResponse:
        text = '{"detail":"LID decoder produced NaN"}'

        def raise_for_status(self):
            response = type("Raw", (), {"status_code": 500, "url": "http://lid.test/lid"})()
            raise lid_infer.requests.HTTPError("500 Server Error", response=response)

    monkeypatch.setattr(lid_infer, "load_wav_bytes", lambda _path: b"RIFF")
    monkeypatch.setattr(lid_infer.requests, "post",
                        lambda *_args, **_kwargs: FailedResponse())

    lid_infer.run(str(manifest), "http://lid.test:8000", str(out), retries=0)

    record = json.loads(out.read_text(encoding="utf-8").splitlines()[1])
    assert record["ok"] is False
    assert "LID decoder produced NaN" in record["error"]
