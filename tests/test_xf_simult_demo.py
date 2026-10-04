import base64
import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SERVER_PATH = Path(__file__).parents[1] / "demos" / "xf_simult" / "server.py"
SPEC = importlib.util.spec_from_file_location("xf_simult_demo", SERVER_PATH)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def test_signed_url_contains_documented_auth_fields_without_plain_secret():
    url = demo.signed_url("api-key", "api-secret", "Mon, 13 Dec 2021 03:37:23 GMT")
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "ws-api.xf-yun.com"
    assert parsed.path == "/v1/private/simult_interpretation"
    assert query["host"] == ["ws-api.xf-yun.com"]
    assert query["serviceId"] == ["simult_interpretation"]
    assert "api-key" in base64.b64decode(query["authorization"][0]).decode()
    assert "api-secret" not in url


def test_first_audio_frame_declares_fixed_zh_to_en_contract():
    frame = demo.audio_frame("app", b"\x00\x01", seq=0, status=0)

    assert frame["header"] == {"app_id": "app", "status": 0}
    assert frame["payload"]["data"] == {
        "audio": "AAE=", "encoding": "raw", "sample_rate": 16000,
        "seq": 0, "status": 0,
    }
    assert frame["parameter"]["ist"] == {
        "accent": "mandarin", "domain": "ist_ed_open", "language": "zh_cn",
        "vto": 15000, "eos": 150000,
    }
    assert frame["parameter"]["streamtrans"] == {"from": "cn", "to": "en"}
    assert frame["parameter"]["tts"]["vcn"] == "x2_john"


def test_middle_audio_frame_does_not_repeat_business_parameters():
    frame = demo.audio_frame("app", b"pcm", seq=3, status=1)

    assert "parameter" not in frame
    assert frame["payload"]["data"]["seq"] == 3
    assert frame["payload"]["data"]["status"] == 1


def test_translation_response_is_decoded_and_terminal_event_is_preserved():
    encoded = base64.b64encode(json.dumps({
        "src": "你好", "dst": "Hello", "wb": 20, "we": 800, "is_final": 1,
    }, ensure_ascii=False).encode()).decode()
    raw = json.dumps({
        "header": {"code": 0, "status": 2},
        "payload": {"streamtrans_results": {"text": encoded}},
    })

    assert demo.response_events(raw) == [
        {
            "type": "translation", "source": "你好", "translation": "Hello",
            "is_final": True, "begin_ms": 20, "end_ms": 800,
        },
        {"type": "done"},
    ]


def test_raw_tts_response_is_decoded_before_terminal_event():
    raw = json.dumps({
        "header": {"code": 0, "status": 2},
        "payload": {"tts_results": {
            "audio": base64.b64encode(b"\x00\x01\x02\x03").decode(),
            "encoding": "raw",
            "sample_rate": "16000",
            "status": "1",
        }},
    })

    assert demo.response_events(raw) == [
        {"type": "tts_audio", "audio": b"\x00\x01\x02\x03"},
        {"type": "done"},
    ]


def test_non_pcm_tts_is_not_forwarded_to_pcm_player():
    raw = json.dumps({
        "header": {"code": 0, "status": 1},
        "payload": {"tts_results": {
            "audio": base64.b64encode(b"mp3").decode(),
            "encoding": "lame",
        }},
    })

    assert demo.response_events(raw) == []


def test_vendor_error_is_forwarded_without_response_payload():
    raw = json.dumps({"header": {"code": 11200, "message": "invalid param", "status": 2}})

    assert demo.response_events(raw) == [
        {"type": "error", "code": 11200, "message": "invalid param"},
    ]
