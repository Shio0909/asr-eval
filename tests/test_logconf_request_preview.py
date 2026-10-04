import importlib.util
import os
import json


ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "preview_logconf", os.path.join(ROOT, "eval", "logconf.py")
)
logconf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logconf)


def test_request_preview_keeps_form_fields_and_hides_token_and_audio():
    preview = logconf.build_request_preview(
        "POST",
        "http://h.example.test/api/asr/std",
        {
            "headers": {
                "Authorization": "Bearer real-secret-token",
                "X-Appid": "private-app-id",
            },
            "data": {
                "enable_text_edit": "false",
                "language": "auto",
                "max_tokens": 1024,
            },
            "files": {
                "file": (
                    "/private/data/real-audio.wav",
                    b"secret-audio-content",
                    "audio/wav",
                )
            },
        },
        timeout=90,
    )

    rendered = preview["curl"]
    assert "enable_text_edit=false" in rendered
    assert "language=auto" in rendered
    assert "max_tokens=1024" in rendered
    assert "Bearer token hidden" in rendered
    assert "X-Appid: ***" in rendered
    assert "audio-file-hidden" in rendered
    assert "real-secret-token" not in rendered
    assert "private-app-id" not in rendered
    assert "real-audio.wav" not in rendered
    assert "secret-audio-content" not in rendered


def test_response_preview_hides_secrets_audio_and_large_lists():
    class Response:
        status_code = 200
        reason = "OK"
        headers = {"content-type": "application/json", "x-request-id": "req-1"}
        content = json.dumps({
            "result": {"text": "识别文本", "token": "private-token-value"},
            "audio_base64": "A" * 1024,
            "utterances": [{"text": str(i)} for i in range(80)],
        }).encode()

    preview = logconf.build_response_preview(Response())
    rendered = json.dumps(preview, ensure_ascii=False)

    assert preview["body_type"] == "json"
    assert preview["body"]["result"]["text"] == "识别文本"
    assert preview["body"]["result"]["token"] == "***"
    assert "private-token-value" not in rendered
    assert "audio/base64 omitted" in rendered
    assert len(preview["body"]["utterances"]) == 41
