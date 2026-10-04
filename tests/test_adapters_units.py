import json
import os
import sys
import types
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import adapters


class FakeResp:
    ok = True
    status_code = 200

    def json(self):
        return {"text": "ok"}

    def raise_for_status(self):
        pass


class FakeChatResp:
    ok = True
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "测试转写"}}]}


def test_longform_progress_is_opt_in_and_structured(capsys):
    adapters.emit_longform_progress({"id": "short"}, audio_s=1, audio_total_s=2)
    assert capsys.readouterr().out == ""

    adapters.emit_longform_progress(
        {"id": "talk-1", "_stream_progress": {"sample_index": 2, "sample_total": 5}},
        phase="segment", audio_s=507.26, audio_total_s=682.0,
        segments=69, latest_text="最新一段译文",
    )
    line = capsys.readouterr().out.strip()
    assert line.startswith(adapters.LONGFORM_PROGRESS_MARKER)
    payload = json.loads(line[len(adapters.LONGFORM_PROGRESS_MARKER):])
    assert payload == {
        "phase": "segment", "sample_id": "talk-1", "sample_index": 2,
        "sample_total": 5, "audio_s": 507.3, "audio_total_s": 682.0,
        "segments": 69, "latest_text": "最新一段译文",
    }


def test_xf_simult_reports_longform_progress_and_does_not_wait_for_optional_tts(
        monkeypatch, capsys, tmp_path):
    def response(dst, is_final, status=1):
        encoded = base64.b64encode(json.dumps({
            "dst": dst, "is_final": is_final,
        }).encode()).decode()
        return json.dumps({
            "header": {"code": 0, "status": status},
            "payload": {"streamtrans_results": {"text": encoded}},
        })

    class FakeWS:
        def __init__(self):
            self.responses = [
                response("Hel", False),
                response("Hellx", False),
                response("Hello", False),       # 替换 Hellx：真实 revision
                response("Hello", True, 2),     # 翻译已终态，但没有 tts_results.status=2
            ]
            self.sent = []

        def send(self, data):
            self.sent.append(data)

        def recv(self):
            return self.responses.pop(0)

        def close(self):
            pass

    ws = FakeWS()
    monkeypatch.setattr(adapters, "ws_connect", lambda *args, **kwargs: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x00" * 2560)
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: None)

    ad = adapters.XfSimultAdapter(
        appid="app", api_key="key", api_secret="secret", audio_out_dir=str(tmp_path)
    )
    res = ad.generate({
        "id": "talk-xf", "audio_path": "a.wav", "lang": "zh-en",
        "target_lang": "English", "ref_text": "Hello",
        "_stream_progress": {"sample_index": 1, "sample_total": 2},
    })

    assert res.ok and res.text == "Hello"
    assert "tts_audio" not in res.extra
    assert [s["text"] for s in res.extra["translation_segments"]] == ["Hello"]
    markers = [json.loads(line[len(adapters.LONGFORM_PROGRESS_MARKER):])
               for line in capsys.readouterr().out.splitlines()
               if line.startswith(adapters.LONGFORM_PROGRESS_MARKER)]
    phases = [item["phase"] for item in markers]
    assert phases[0] == "started"
    assert {"streaming", "partial", "revision", "segment", "completed"} <= set(phases)
    final = next(item for item in markers if item["phase"] == "segment")
    assert final["segments"] == 1 and final["latest_text"] == "Hello"


def test_timed_text_segments_keeps_final_text_and_source_audio_clock():
    assert adapters._timed_text_segments([(1.23456, " first "), (9, ""), (12.5, "second")]) == [
        {"end_s": 1.235, "text": "first"},
        {"end_s": 12.5, "text": "second"},
    ]


def test_openai_audio_adapter_does_not_send_hotwords_by_default(monkeypatch):
    seen = {}

    def fake_post(url, **kwargs):
        seen.update(kwargs)
        return FakeResp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)

    ad = adapters.OpenAIAudioAdapter(base_url="https://example.test", prefix="/v1", model="whisper-1")
    res = ad.transcribe("a.wav", hotwords="foo bar")

    assert res.ok
    assert "hot_words" not in seen["data"]


def test_openai_audio_adapter_can_explicitly_send_hotwords(monkeypatch):
    seen = {}

    def fake_post(url, **kwargs):
        seen.update(kwargs)
        return FakeResp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)

    ad = adapters.OpenAIAudioAdapter(base_url="https://example.test", supports_hotwords=True)
    res = ad.transcribe("a.wav", hotwords="foo bar")

    assert res.ok
    assert seen["data"]["hot_words"] == "foo,bar"


def test_openai_audio_adapter_forwards_arbitrary_non_auto_language(monkeypatch):
    seen = {}

    def fake_post(url, **kwargs):
        seen.update(kwargs)
        return FakeResp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)

    ad = adapters.OpenAIAudioAdapter(base_url="https://example.test")
    assert ad.transcribe("a.wav", language="Chinese,English").ok
    assert seen["data"]["language"] == "Chinese,English"

    assert ad.transcribe("a.wav", language="auto").ok
    assert "language" not in seen["data"]


def test_platform_asr_adapter_never_switches_normal_identity_for_hotwords(monkeypatch):
    seen = []

    class Resp(FakeResp):
        def json(self):
            return {"result": {"text": "测试"}}

    def fake_post(url, **kwargs):
        seen.append((url, kwargs["data"], kwargs["headers"]))
        return Resp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)

    ad = adapters.PlatformASRAdapter(
        base_url="http://example.test", endpoint="normal", api_key="secret-token"
    )
    assert ad.transcribe("a.wav", language="Chinese,English").ok
    assert seen[-1][0].endswith("/api/v2/asr/normal")
    assert seen[-1][1]["language"] == "Chinese,English"
    assert seen[-1][2] == {"Authorization": "Bearer secret-token"}

    before = len(seen)
    rejected = ad.transcribe("a.wav", language="English", hotwords="Qwen")
    assert rejected.ok is False
    assert "pro-domain" in rejected.error
    assert len(seen) == before

    translated = ad.transcribe("a.wav", target_lang="English")
    assert translated.ok is False
    assert "normal 不支持语音翻译" in translated.error
    assert len(seen) == before

    domain = adapters.PlatformProDomainAdapter(
        base_url="http://example.test", domain="legal", api_key="secret-token"
    )
    assert domain.transcribe("a.wav", language="English", hotwords="Qwen").ok
    assert seen[-1][0].endswith("/api/v2/asr/pro-domain")
    assert seen[-1][1]["domain"] == "legal"
    assert seen[-1][1]["language"] == "English"


def test_minutes_forwards_managed_language_and_target(monkeypatch):
    seen = {}

    class Resp(FakeResp):
        def json(self):
            return {"render": {"final_summary": {"markdown": "# summary"},
                               "live_transcript": []}}

    def fake_post(url, **kwargs):
        seen.update(kwargs)
        return Resp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)
    result = adapters.PlatformMinutesAdapter(base_url="http://example.test", api_key="token").generate({
        "audio_path": "a.wav", "request_language": "Chinese,English", "target_lang": "英文",
    })

    assert result.ok
    assert seen["data"]["language"] == "Chinese,English"
    assert seen["data"]["target_lang"] == "English"


def test_diarization_forwards_managed_language(monkeypatch):
    seen = {}

    class Resp(FakeResp):
        def json(self):
            return {"result": {"utterances": [
                {"start_time": 0, "end_time": 1000, "speaker_id": "0"},
            ]}}

    def fake_post(url, **kwargs):
        seen.update(kwargs)
        return Resp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)
    result = adapters.PlatformDiarAdapter(base_url="http://example.test", api_key="token").transcribe(
        "a.wav", language="Chinese,English",
    )

    assert result.ok
    assert seen["data"]["language"] == "Chinese,English"


def test_plat_simult_builds_declared_voice_input_capabilities():
    ad = adapters.PlatformWSSimultAdapter(base_url="http://example.test", api_key="token")
    ad.request_schema = adapters.BUILTIN_REQUEST_CONTRACTS["plat-simult"]["request_schema"]
    ad.request_params = {
        "abbreviations": {"AI": "artificial intelligence"},
        "pipeline_mode": "multi_stage",
        "return_asr_text": True,
        "incremental_enabled": True,
        "incremental_interval_ms": 800,
        "incremental_holdback_tokens": 2,
    }

    setting = ad._voice_input_setting({
        "target_lang": "French", "request_language": "Chinese,English",
        "hotwords": "ExampleTerm,示例词",
    })

    assert setting == {
        "language": "Chinese,English",
        "target_language": "French",
        "hot_words": ["ExampleTerm", "示例词"],
        "abbreviations": {"AI": "artificial intelligence"},
        "pipeline_mode": "multi_stage",
        "return_asr_text": True,
        "incremental_enabled": True,
        "incremental_interval_ms": 800,
        "incremental_holdback_tokens": 2,
    }
    ad.audio_out_dir = "/tmp/audio"
    ad.request_params.update({
        "tts_mode": "clone", "tts_output_sample_rate": 24000, "tts_voice_id": "voice-x",
        "vad_threshold": 0.7, "vad_silence_duration_ms": 900,
        "vad_min_speech_duration_ms": 400, "vad_soft_max_duration_ms": 12000,
        "vad_hard_max_duration_ms": 24000, "vad_soft_silence_duration_ms": 250,
    })
    assert ad._tts_setting() == {
        "enable": True, "mode": "clone", "output_sample_rate": 24000, "voice_id": "voice-x",
    }
    assert ad._vad_setting() == {
        "threshold": 0.7, "silence_duration": 900, "min_speech_duration": 400,
        "soft_max_duration": 12000, "hard_max_duration": 24000,
        "soft_silence_duration": 250,
    }


def test_plat_simult_persists_deployed_segment_fields_and_stage_timings(monkeypatch):
    class FakeWS:
        def __init__(self):
            self.responses = [json.dumps(event) for event in [
                {"event": "connected_success"},
                {"event": "task_started", "data": {
                    "backend": "translation-v4", "route": "gpu-a", "token": "server-secret",
                }},
                {"event": "si_incremental_update", "data": {
                    "segment_epoch": 0, "segment_key": "s:0", "update_index": 0,
                    "audio_ms": 1000, "phase": "asr",
                    "original_confirmed": "", "original_draft": "source wrong",
                    "translation_confirmed": "", "translation_draft": "",
                    "timing": {"asr_ms": 30, "text_translate_ms": None, "total_ms": 31},
                }},
                {"event": "si_incremental_update", "data": {
                    "segment_epoch": 0, "segment_key": "s:0", "update_index": 0,
                    "audio_ms": 1000, "phase": "translated",
                    "original_confirmed": "", "original_draft": "source wrong",
                    "translation_confirmed": "", "translation_draft": "旧译",
                    "timing": {"asr_ms": 30, "text_translate_ms": 40, "total_ms": 71},
                }},
                {"event": "si_incremental_update", "data": {
                    "segment_epoch": 0, "segment_key": "s:0", "update_index": 1,
                    "audio_ms": 2000, "phase": "asr",
                    "original_confirmed": "source", "original_draft": "transcript",
                    "translation_confirmed": "", "translation_draft": "",
                    "timing": {"asr_ms": 25, "text_translate_ms": None, "total_ms": 26},
                }},
                {"event": "si_incremental_update", "data": {
                    "segment_epoch": 0, "segment_key": "s:0", "update_index": 1,
                    "audio_ms": 2000, "phase": "translated",
                    "original_confirmed": "source", "original_draft": "transcript",
                    "translation_confirmed": "译", "translation_draft": "文",
                    "timing": {"asr_ms": 25, "text_translate_ms": 35, "total_ms": 61},
                }},
                {"event": "si_token", "data": {"content": "译"}},
                {"event": "si_segment_done", "data": {
                    "segment_id": 0, "text": "译文", "original_text": "source transcript",
                    "segment_reason": "probability_valley",
                    "pipeline_mode": "multi_stage", "timestamp_start": 200,
                    "timestamp_end": 9800, "timing": {
                        "queue_ms": 50, "asr_ms": 210, "text_translate_ms": 132,
                        "text_ttft_ms": 32, "text_decode_ms": 100,
                        "total_ms": 343, "e2e_ms": 393,
                    },
                }},
                {"event": "task_finished"},
            ]]
            self.sent = []

        def send(self, data):
            self.sent.append(json.loads(data))

        def send_binary(self, data):
            pass

        def recv(self):
            return self.responses.pop(0)

        def close(self):
            pass

    ws = FakeWS()
    monkeypatch.setattr(adapters, "ws_connect", lambda *args, **kwargs: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x00" * 3200)
    ad = adapters.PlatformWSSimultAdapter(
        base_url="http://example.test", api_key="token", pace=False,
    )
    ad.request_schema = adapters.BUILTIN_REQUEST_CONTRACTS["plat-simult"]["request_schema"]
    ad.request_params = {
        "pipeline_mode": "multi_stage", "return_asr_text": True,
        "incremental_enabled": True, "incremental_interval_ms": 1000,
        "incremental_holdback_tokens": 1,
    }

    result = ad.generate({
        "id": "talk", "audio_path": "a.wav", "target_lang": "Chinese",
        "request_language": "English", "ref_text": "译文",
    })

    assert result.ok and result.text == "译文"
    assert result.extra["pipeline_mode"] == "multi_stage"
    assert result.extra["asr_text"] == "source transcript"
    assert result.extra["asr_ms_mean"] == 210
    assert result.extra["text_translate_ms_mean"] == 132
    assert result.extra["e2e_ms_mean"] == 393
    assert result.extra["segment_done_count"] == 1
    assert result.extra["empty_segment_count"] == 0
    assert result.extra["segment_error_count"] == 0
    assert result.extra["incremental_update_count"] == 4
    assert result.extra["incremental_phase_counts"] == {"asr": 2, "translated": 2}
    assert result.extra["incremental_asr_ttfb_s"] is not None
    assert result.extra["incremental_translation_ttfb_s"] is not None
    assert result.extra["incremental_asr_trace"]["summary"]["stability_observable"] is True
    assert result.extra["incremental_translation_trace"]["summary"]["stability_observable"] is True
    assert result.extra["incremental_translation_churn_rate"] > 0
    assert result.extra["finish_tail_s"] >= 0
    assert result.extra["ws_control"]["task_start"]["model"] == "simult-interpreting"
    assert result.extra["ws_control"]["task_started"]["data"] == {
        "backend": "translation-v4", "route": "gpu-a", "token": "<redacted>",
    }
    assert len(result.extra["ws_control"]["task_start_sha256"]) == 64
    assert len(result.extra["ws_control"]["task_started_sha256"]) == 64
    assert len(result.extra["ws_control"]["task_started_stable_sha256"]) == 64
    assert result.extra["translation_segments"] == [{
        "segment_id": 0, "start_s": 0.2, "end_s": 9.8, "emitted_at_s": 0.1,
        "text": "译文", "asr_text": "source transcript",
        "segment_reason": "probability_valley",
        "pipeline_mode": "multi_stage", "timing": {
            "queue_ms": 50, "asr_ms": 210, "text_translate_ms": 132,
            "text_ttft_ms": 32, "text_decode_ms": 100,
            "total_ms": 343, "e2e_ms": 393,
        },
    }]
    task_start = next(item for item in ws.sent if item.get("event") == "task_start")
    assert task_start["voice_input_setting"]["pipeline_mode"] == "multi_stage"
    assert task_start["voice_input_setting"]["return_asr_text"] is True
    assert task_start["voice_input_setting"]["incremental_enabled"] is True
    assert task_start["voice_input_setting"]["incremental_interval_ms"] == 1000
    assert task_start["voice_input_setting"]["incremental_holdback_tokens"] == 1
    assert result.extra["ws_control"]["task_start"]["voice_input_setting"][
        "incremental_holdback_tokens"
    ] == 1


def test_plat_voice_input_applies_start_frame_params_and_persists_ack(monkeypatch):
    class FakeWS:
        def __init__(self):
            self.responses = [json.dumps(event) for event in [
                {"event": "connected_success", "data": {"protocol_version": "2"}},
                {"event": "task_started", "data": {"route": "voice-input-a"}},
                {"event": "result_final", "data": {"translation": "译文"}},
                {"event": "task_finished"},
            ]]
            self.sent = []

        def send(self, data):
            self.sent.append(json.loads(data))

        def send_binary(self, data):
            pass

        def recv(self):
            return self.responses.pop(0)

        def close(self):
            pass

    ws = FakeWS()
    monkeypatch.setattr(adapters, "ws_connect", lambda *args, **kwargs: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x00" * 3200)
    ad = adapters.PlatformWSVoiceInputAdapter(
        base_url="http://example.test", api_key="token", pace=False,
    )
    ad.request_schema = adapters.BUILTIN_REQUEST_CONTRACTS["plat-simult-ws"]["request_schema"]
    ad.request_params = {
        "vad_threshold": 0.65,
        "vad_silence_duration_ms": 850,
        "vad_min_speech_duration_ms": 400,
        "vad_soft_max_duration_ms": 12000,
        "vad_hard_max_duration_ms": 24000,
        "vad_soft_silence_duration_ms": 250,
        "summary_interval_s": 15,
    }

    result = ad.generate({
        "id": "talk", "audio_path": "a.wav", "target_lang": "zh",
        "request_language": "en", "ref_text": "译文",
    })

    assert result.ok and result.text == "译文"
    task_start = next(item for item in ws.sent if item.get("event") == "task_start")
    assert task_start["vad_setting"] == {
        "silence_duration": 850, "min_speech_duration": 400,
        "soft_max_duration": 12000, "hard_max_duration": 24000,
        "soft_silence_duration": 250, "threshold": 0.65,
    }
    assert task_start["voice_input_setting"]["summary_interval_s"] == 15
    assert result.extra["ws_control"]["task_started"]["data"]["route"] == "voice-input-a"
    assert result.extra["finish_tail_s"] >= 0


def test_ws_control_stable_ack_hash_ignores_session_ids_but_keeps_route():
    first = adapters._ws_control_record(
        "ws://example.test/ws", {}, {"event": "task_start"},
        {"event": "task_started", "session_id": "s1", "meeting_id": "m1",
         "data": {"route": "gpu-a"}},
    )
    second = adapters._ws_control_record(
        "ws://example.test/ws", {}, {"event": "task_start"},
        {"event": "task_started", "session_id": "s2", "meeting_id": "m2",
         "data": {"route": "gpu-a"}},
    )
    other_route = adapters._ws_control_record(
        "ws://example.test/ws", {}, {"event": "task_start"},
        {"event": "task_started", "session_id": "s3", "meeting_id": "m3",
         "data": {"route": "gpu-b"}},
    )

    assert first["task_started_sha256"] != second["task_started_sha256"]
    assert first["task_started_stable_sha256"] == second["task_started_stable_sha256"]
    assert first["task_started_stable_sha256"] != other_route["task_started_stable_sha256"]


def test_custom_http_adapter_routes_task_fields_to_registered_wire_locations(monkeypatch):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(kwargs)
        return FakeResp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_request", fake_request)
    ad = adapters.CustomHTTPASRAdapter({
        "id": "x-task-fields", "base_url": "https://example.test/asr",
        "body_type": "multipart", "response_type": "json", "text_path": "text",
    })
    ad.request_schema = [
        {"name": "language", "in": "query", "type": "string", "managed_by": "language"},
        {"name": "target_language", "in": "form", "type": "string", "managed_by": "target_lang"},
        {"name": "X-Hotwords", "in": "header", "type": "string", "managed_by": "hotwords"},
    ]

    result = ad.transcribe(
        "a.wav", language="Chinese", target_lang="English", hotwords="ExampleTerm",
    )

    assert result.ok
    assert seen["params"]["language"] == "Chinese"
    assert seen["data"]["target_language"] == "English"
    assert seen["headers"]["X-Hotwords"] == "ExampleTerm"


def test_custom_http_adapter_strips_speaker_labels_before_scoring(monkeypatch):
    class Resp(FakeResp):
        def json(self):
            return {"text": "Speaker 0:你好 Speaker 1:世界"}

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_request", lambda *a, **k: Resp())
    ad = adapters.make_custom_adapter({
        "id": "x-custom", "template": "custom-http-asr",
        "base_url": "http://custom.test/v1/transcribe",
        "body_type": "json_base64", "audio_field": "audio_base64",
        "response_type": "json", "text_path": "text",
        "strip_regex": r"Speaker\s*\d+\s*:",
    })

    result = ad.transcribe("a.wav")

    assert result.ok
    assert result.text == "你好 世界"


def test_plat_multipart_template_resolves_bearer_token_from_env(monkeypatch):
    monkeypatch.setenv("ASR_PLATFORM_TOKEN", "env-token")
    ad = adapters.make_custom_adapter({
        "id": "x-normal-8000",
        "template": "plat-multipart",
        "base_url": "http://h.example.test/api/v2/asr/normal",
        "path": "normal",
        "key_env": "ASR_PLATFORM_TOKEN",
    })

    assert isinstance(ad, adapters.PlatformASRAdapter)
    assert ad.base == "http://h.example.test"
    assert ad.url == "http://h.example.test/api/v2/asr/normal"
    assert ad.api_key == "env-token"


def test_custom_plat_adapter_merges_registered_runtime_form_params(monkeypatch):
    seen = {}

    class Resp(FakeResp):
        def json(self):
            return {"result": {"text": "测试"}}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Resp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)
    ad = adapters.make_custom_adapter({
        "id": "x-pro",
        "template": "plat-multipart",
        "base_url": "http://asr.test",
        "path": "pro",
        "api_key": "fixture",
        "request_schema": [
            {"name": "enable_word_timestamps", "in": "form", "type": "boolean"},
            {"name": "speaker_merge_gap_ms", "in": "form", "type": "integer"},
            {"name": "trace_id", "in": "query", "type": "string"},
        ],
    }, request_params={
        "enable_word_timestamps": True,
        "speaker_merge_gap_ms": 1200,
        "trace_id": "run-1",
    })

    result = ad.transcribe("a.wav", language="auto")

    assert result.ok
    assert seen["data"]["enable_word_timestamps"] == "true"
    assert seen["data"]["speaker_merge_gap_ms"] == "1200"
    assert seen["params"] == {"trace_id": "run-1"}
    assert seen["headers"] == {"Authorization": "Bearer fixture"}


def test_custom_adapter_infers_hotword_capability_from_registered_schema():
    ad = adapters.make_custom_adapter({
        "id": "x-schema-hotwords",
        "template": "custom-http-asr",
        "base_url": "http://asr.test",
        "request_schema": [
            {"name": "hot_words", "in": "form", "type": "string",
             "managed_by": "hotwords", "default": ""},
        ],
    })

    assert ad.supports_hotwords is True


def test_builtin_plat_adapters_default_to_platform_token(monkeypatch):
    monkeypatch.setenv("ASR_PLATFORM_TOKEN", "env-token")

    asr = adapters.PlatformASRAdapter()
    sse = adapters.PlatformSSEAdapter()
    minutes = adapters.PlatformMinutesAdapter()
    diar = adapters.PlatformDiarAdapter()
    formula = adapters.PlatformFormulaAdapter()
    voice_ws = adapters.PlatformWSVoiceInputAdapter()
    simult_ws = adapters.PlatformWSSimultAdapter()

    assert asr.base == "http://platform.test:8000"
    assert asr.api_key == "env-token"
    for adapter in (sse, minutes, diar, formula):
        assert adapter.headers == {"Authorization": "Bearer env-token"}
    for adapter in (voice_ws, simult_ws):
        assert adapter.ws_headers == ["Authorization: Bearer env-token"]


def test_custom_plat_adapter_without_key_does_not_leak_builtin_token(monkeypatch):
    monkeypatch.setenv("ASR_PLATFORM_TOKEN", "must-not-leak")
    ad = adapters.make_custom_adapter({
        "id": "x-keyless",
        "template": "plat-sse",
        "base_url": "http://keyless.example.test",
    })

    assert ad.headers == {}


def test_asr_lite_adapter_sends_task_language_itn_and_timestamps(monkeypatch):
    seen = {}

    class Resp(FakeResp):
        def json(self):
            return {"text": "你好", "language": "sichuan",
                    "timestamps": [{"text": "你好", "start_time": 0, "end_time": 1}]}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Resp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)

    ad = adapters.LiteASRAdapter(
        "http://example.test", default_itn=True, return_timestamps=True,
    )
    res = ad.transcribe("a.wav", language="sichuan")

    assert res.ok and res.text == "你好"
    assert seen["url"] == "http://example.test/asr_lite"
    assert seen["data"] == {
        "language": "sichuan", "itn": "true", "return_timestamps": "true",
    }
    assert res.extra["language"] == "sichuan"
    assert res.extra["timestamps"][0]["text"] == "你好"


def test_builtin_lite_surface_does_not_send_unpublished_timestamp_field(monkeypatch):
    seen = {}

    class Resp(FakeResp):
        def json(self):
            return {"text": "你好"}

    def fake_post(url, **kwargs):
        seen.update(kwargs)
        return Resp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)
    ad = adapters.LiteASRAdapter("http://example.test", default_itn=False)

    assert ad.transcribe("a.wav", language="auto").ok
    assert seen["data"] == {"language": "auto", "itn": "false"}


def test_asr_mlt_adapter_uses_mlt_route(monkeypatch):
    seen = {}

    class Resp(FakeResp):
        def json(self):
            return {"text": "مرحبا"}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Resp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)

    res = adapters.lite_mlt_nano().transcribe("a.wav", language="阿拉伯语")

    assert res.ok and res.text == "مرحبا"
    assert seen["url"] == "http://lite.test:8001/asr_mlt_nano"
    assert seen["data"] == {"language": "阿拉伯语", "itn": "false"}


def test_qwen3_1_7b_remote_provider_exposes_asr_lite_parameters():
    provider = adapters.PROVIDERS["qwen3-asr-1.7b"]

    assert provider["base_url"] == "http://qwen3.test:8005"
    assert provider["template"] == "asr_lite"
    assert "language" not in provider
    assert provider["itn"] is False
    assert provider["return_timestamps"] is False
    assert "sichuan" in provider["language_spec"]["values"]
    assert "yue_hk" in provider["language_spec"]["values"]


def test_openai_chat_audio_adapter_sends_input_audio_data_uri(monkeypatch):
    seen = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return FakeChatResp()

    monkeypatch.setattr(adapters, "load_wav_bytes", lambda path: b"RIFFxxxx")
    monkeypatch.setattr(adapters, "http_post", fake_post)

    ad = adapters.OpenAIChatAudioAdapter(
        base_url="https://api.example.test", prefix="/v1", model="mimo-v2.5-asr", api_key="secret"
    )
    res = ad.transcribe("a.wav", language="zh")

    assert res.ok and res.text == "测试转写"
    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer secret"}
    body = seen["json"]
    assert body["model"] == "mimo-v2.5-asr"
    assert body["asr_options"] == {"language": "zh"}
    audio = body["messages"][0]["content"][0]
    assert audio["type"] == "input_audio"
    assert audio["input_audio"]["data"].startswith("data:audio/wav;base64,")


def test_mimo_provider_uses_key_env_and_asr_only():
    provider = adapters.PROVIDERS["mimo"]
    assert provider["template"] == "openai-chat-audio"
    assert provider["key_env"] == "MIMO_API_KEY"
    assert provider["models"] == ["mimo-v2.5-asr"]
    assert adapters.TEMPLATE_SCENARIOS["openai-chat-audio"] == ["ASR"]


def test_doubao_streams_pcm_chunks_not_wav_container(monkeypatch):
    sent_payloads = []

    class FakeType:
        StartSession = 1
        SessionStarted = 2
        TaskRequest = 3
        FinishSession = 4
        SessionFinished = 5
        SessionFailed = 6
        SessionCanceled = 7
        TranslationSubtitleResponse = 8
        TranslationSubtitleEnd = 9
        TTSResponse = 10

    class FakeMeta:
        Message = ""

    class FakeRequest:
        def __init__(self):
            self.request_meta = types.SimpleNamespace(SessionID="")
            self.event = None
            self.user = types.SimpleNamespace(uid="", did="")
            self.source_audio = types.SimpleNamespace(format="", rate=0, bits=0, channel=0, binary_data=b"")
            self.target_audio = types.SimpleNamespace(format="", rate=0)
            self.request = types.SimpleNamespace(mode="", source_language="", target_language="")

        def SerializeToString(self):
            if self.event == FakeType.TaskRequest:
                sent_payloads.append((self.source_audio.format, self.source_audio.binary_data))
            return bytes([self.event or 0])

    class FakeResponse:
        def __init__(self):
            self.event = None
            self.text = ""
            self.data = b""
            self.response_meta = FakeMeta()

        def ParseFromString(self, raw):
            self.event = raw[0]

    class FakeWS:
        def __init__(self):
            self.responses = [bytes([FakeType.SessionStarted]), bytes([FakeType.SessionFinished])]

        def send_binary(self, data):
            pass

        def recv(self):
            return self.responses.pop(0)

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(create_connection=lambda *a, **k: FakeWS()))
    monkeypatch.setattr(adapters.DoubaoSimultAdapter, "_pb", lambda self: (FakeRequest, FakeResponse, FakeType))
    monkeypatch.setattr(adapters, "ws_connect", lambda *args, **kwargs: FakeWS())
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x01\x02" * 4000)
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: None)

    ad = adapters.DoubaoSimultAdapter(appid="app", token="tok", pace=False)
    res = ad.generate({"id": "u0", "audio_path": "a.wav", "target_lang": "英文", "lang": "zh-en", "ref_text": "hello"})

    assert res.ok
    assert sent_payloads
    assert sent_payloads[0][0] == "pcm"
    assert not sent_payloads[0][1].startswith(b"RIFF")


# ── plat-realtime（8000 /v1/realtime 流式 ASR）：模块级 fake 骨架与假时钟 ──────────
class _RealtimeIdle(Exception):
    """假 recv 的静默超时（照 websocket 的真实超时异常语义）。"""


class _FakeRealtimeWS:
    """按序吐预设事件、并记下所有送出报文；无预设后共 recv 抛静默异常。"""
    def __init__(self, events):
        self.events = list(events)
        self.sent = []

    def send(self, data):
        self.sent.append(data)

    def send_binary(self, data):
        raise AssertionError("plat-realtime 不得发二进制帧")

    def recv(self):
        if not self.events:
            raise _RealtimeIdle("idle")
        return self.events.pop(0)

    def close(self):
        pass


class _FakeClock:
    """步进假时钟：每次调用 +0.5s。不赌 per_cf_counter 的精确调用位——那会让测试
    随 adapter 内部加一次探点就碎。只保证单调推进，静默阈(6s)能被走到。"""
    def __init__(self, start=0.0, step=0.5):
        self.t = start
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


def _frams_of(ws):
    """取出 input_audio_buffer.append 帧并解 JSON。"""
    out = []
    for m in ws.sent:
        d = json.loads(m)
        if d.get("type") == "input_audio_buffer.append":
            out.append(d)
    return out


def _setup_realtime(monkeypatch, events, pcm, clock, pace=False):
    """pace=False 让时序确定：paced 分支内部也取时钟，容易数错探点。"""
    ws = _FakeRealtimeWS(events)
    monkeypatch.setattr(adapters, "ws_connect", lambda *a, **k: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: pcm)
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    monkeypatch.setattr(adapters.time, "perf_counter", clock)
    monkeypatch.setattr(adapters.PlatformRealtimeAdapter, "_RECV_TIMEOUT", 3.0)
    return ws


def _ev(t, **kw):
    return json.dumps({"type": t, **kw})


def test_plat_realtime_sends_base64_json_frames_not_binary(monkeypatch):
    """流式 ASR 端点只收 Base64 JSON 帧：二进制帧被服务端拒（unsupported_audio_frame）。"""
    ws = _setup_realtime(
        monkeypatch,
        [_ev("session.created", session={"id": "sess_x"}),
         _ev("session.updated", session={"type": "transcription"}),
         _ev("conversation.item.input_audio_transcription.completed", transcript="你好世界")],
        b"\x11" * 6400,
        _FakeClock(),
    )
    res = adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    assert res.ok and res.text == "你好世界"
    frames = _frams_of(ws)
    assert frames, "必须发 Base64 JSON 音频帧"
    assert all(f["audio"] for f in frames), "audio 字段装 Base64 文本"
    for f in frames:
        assert base64.b64decode(f["audio"]), "Base64 能解回字节"   # 注：项目 base64 为 b64encode/b64decode


def test_plat_realtime_omits_context_in_session_update_by_default(monkeypatch):
    """跨 item 上下文默认关：不在 session.update 里新增 context 字段，保持历史跑分口径。"""
    ws = _setup_realtime(
        monkeypatch,
        [_ev("session.created", session={"id": "sess_x"}),
         _ev("session.updated", session={"type": "transcription"}),
         _ev("conversation.item.input_audio_transcription.completed", transcript="你好")],
        b"\x11" * 6400,
        _FakeClock(),
    )
    adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    update = json.loads(ws.sent[0])
    assert "context" not in update["session"]["x_platform"]


def test_plat_realtime_context_enabled_lands_in_session_update(monkeypatch):
    """构造参数开启后，session.update 必须带 x_platform.context 及预算。"""
    ws = _setup_realtime(
        monkeypatch,
        [_ev("session.created", session={"id": "sess_x"}),
         _ev("session.updated", session={"type": "transcription"}),
         _ev("conversation.item.input_audio_transcription.completed", transcript="你好")],
        b"\x11" * 6400,
        _FakeClock(),
    )
    adapters.PlatformRealtimeAdapter(
        context_enabled=True, context_max_items=3, context_max_chars=128,
    ).transcribe("a.wav", language="zh")

    update = json.loads(ws.sent[0])
    assert update["session"]["x_platform"]["context"] == {
        "enabled": True, "max_items": 3, "max_chars": 128,
    }


def test_plat_realtime_context_can_be_enabled_from_request_params(monkeypatch):
    """CLI/看板经 request_params 传字符串开关也能开启，且预算沿用传参。"""
    ws = _setup_realtime(
        monkeypatch,
        [_ev("session.created", session={"id": "sess_x"}),
         _ev("session.updated", session={"type": "transcription"}),
         _ev("conversation.item.input_audio_transcription.completed", transcript="你好")],
        b"\x11" * 6400,
        _FakeClock(),
    )
    adapter = adapters.PlatformRealtimeAdapter()
    adapter.request_params = {
        "context_enabled": "true", "context_max_items": "2", "context_max_chars": "64",
    }
    adapter.transcribe("a.wav", language="zh")

    update = json.loads(ws.sent[0])
    assert update["session"]["x_platform"]["context"] == {
        "enabled": True, "max_items": 2, "max_chars": 64,
    }


def test_plat_realtime_pads_tail_chunk_to_min_frame(monkeypatch):
    """末尾残块不足 10ms 须补齐，否则服务端报 audio_chunk_too_short 丢样本。"""
    ws = _setup_realtime(
        monkeypatch,
        [_ev("session.created"), _ev("session.updated"),
         _ev("conversation.item.input_audio_transcription.completed", transcript="短")],
        b"\x22" * 3270,     # 3200 + 70：末帧仅 70 字节(<320) → 须补零到 320
        _FakeClock(),
    )
    # 关尾部补静音：本条只验「音频切帧 + 残块补齐」，补静音会多出帧干扰计数
    res = adapters.PlatformRealtimeAdapter(trail_silence_s=0).transcribe("a.wav", language="zh")

    assert res.ok
    frames = _frams_of(ws)
    assert len(frames) == 2, "3270 字节应切成 2 帧"
    assert len(base64.b64decode(frames[0]["audio"])) == 3200
    assert len(base64.b64decode(frames[1]["audio"])) == 320, "残块须补到最小帧 320 字节"


def test_plat_realtime_appends_trailing_silence_for_vad(monkeypatch):
    """离线评测须补尾部静音：server_vad 要有静音才收段，否则短音频被丢弃/截尾。

    实测：0.17s 样本不补静音 item_count=0 无结果；补后正常出字；正常样本 CER 不受影响。
    """
    ws = _setup_realtime(
        monkeypatch,
        [_ev("session.created"), _ev("session.updated"),
         _ev("conversation.item.input_audio_transcription.completed", transcript="对"),
         _ev("platform.session.finished", item_count=1, status="completed")],
        b"\x11" * 3200,          # 0.1s 音频
        _FakeClock(),
    )
    res = adapters.PlatformRealtimeAdapter(trail_silence_s=1.0).transcribe("a.wav", language="zh")

    assert res.ok
    total = sum(len(base64.b64decode(f["audio"])) for f in _frams_of(ws))
    # 原 3200 + 补 1.0s(32000) = 35200
    assert total == 3200 + 32000, f"应补足 1s 静音，实得 {total} 字节"
    assert res.extra["src_dur_s"] == 0.1, "src_dur 记的是真实音频时长，不含补的静音"


def test_plat_realtime_joins_multi_segment_without_losing_segments(monkeypatch):
    """无终止事件：逐段 .completed 后静默收尾；两段都要收到，不能只取最后一段。"""
    _setup_realtime(
        monkeypatch,
        [_ev("session.created"), _ev("session.updated"),
         _ev("conversation.item.input_audio_transcription.completed", transcript="第一段"),
         _ev("input_audio_buffer.committed"),
         _ev("conversation.item.input_audio_transcription.completed", transcript="第二段")],
        b"\x33" * 3200,
        # t0@0.0 → loop-base@0.5 → commit@1.0 → 段1@2.0 → 段2@3.0 → 静默到 9.5(过 6s 阈)
        _FakeClock(),
    )
    res = adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    assert res.ok
    assert res.text == "第一段第二段", f"两段都须保留，实得 {res.text!r}"
    assert res.extra["n_seg"] == 2


def test_plat_realtime_reports_error_event_as_failure(monkeypatch):
    """错误事件 → ok=False 且 error 非空（不得当空 hyp 静默收尾）。"""
    _setup_realtime(
        monkeypatch,
        [_ev("session.created"), _ev("session.updated"),
         _ev("error", error={"code": "unsupported_audio_frame",
                             "message": "Binary frames are not supported"})],
        b"\x44" * 3200,
        _FakeClock(),
    )
    res = adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    assert not res.ok
    assert res.error and "unsupported_audio_frame" in res.error


def test_plat_realtime_elapsed_excludes_client_idle_tail(monkeypatch):
    """延迟记到最后一字落地；客户端静默等待(收尾策略)不算进 elapsd_s。"""
    _setup_realtime(
        monkeypatch,
        [_ev("session.created"), _ev("session.updated"),
         _ev("conversation.item.input_audio_transcription.completed", transcript="一字")],
        b"\x55" * 3200,
        # t0@0.0 → loop-base@0.5 → commit@1.0 → 最后一字@2.0；收尾在 9.0 → elapsd 2.0、尾 7.0
        _FakeClock(),
    )
    res = adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    assert res.ok
    assert res.extra["fanel_tail_s"] > 0.0, "静默尾须单独记入 fanel_tail_s"
    assert res.elapsed_s < res.extra["fanel_tail_s"] + res.elapsed_s, "elapsd 是最后一字时刻"
    assert res.extra["recv_timeout_s"] == 3.0, "短超时是静默分辩率的来源"


# ── qwen3-asr-ws（8007 真流式 WS）：协议与参数面 ──────────────────────────────
class _FakeQwen3WS:
    """按序吐预设事件、记下送出的报文。audio 走 send_binary（裸 float32le）。"""
    def __init__(self, events):
        self.events = list(events)
        self.sent = []
        self.binary = []

    def send(self, data):
        self.sent.append(json.loads(data))

    def send_binary(self, data):
        self.binary.append(data)

    def recv(self):
        if not self.events:
            raise RuntimeError("qwen3-asr-ws 测试：预设事件耗尽")
        return json.dumps(self.events.pop(0))

    def close(self):
        pass


def _setup_qwen3(monkeypatch, events, pcm=None):
    ws = _FakeQwen3WS(events)
    monkeypatch.setattr(adapters, "ws_connect", lambda *a, **k: ws)
    # 2 个 chunk 的料（每 chunk 1000ms=16000 samples=32000 bytes）
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: pcm or (b"\x11\x00" * 32000))
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    return ws


def _qwen_ev(t, **kw):
    return {"type": t, **kw}


def test_qwen3_asr_ws_sends_start_then_binary_float32_and_finish(monkeypatch):
    """协议：start → started → 裸二进制 float32le chunk → finish；不是 Base64 JSON。"""
    ws = _setup_qwen3(
        monkeypatch,
        [_qwen_ev("started", session_id="s1", language="Chinese"),
         _qwen_ev("result", text="宋芳对。"),
         _qwen_ev("result", text="宋芳对北京银行。"),
         _qwen_ev("result", final=True, text="宋芳对北京银行的财富顾问说。")],
    )
    res = adapters.Qwen3ASRWSAdapter(trail_silence_s=0).transcribe("a.wav", language="zh")

    assert res.ok and res.text == "宋芳对北京银行的财富顾问说。"
    assert ws.sent[0]["type"] == "start"
    assert ws.sent[0]["language"] == "Chinese", "zh 应映射为服务端语言名 Chinese"
    assert ws.sent[-1]["type"] == "finish"
    assert ws.binary, "音频须走 send_binary 裸 float32 帧"
    assert len(ws.binary[0]) % 4 == 0, "float32 帧长应为 4 的倍数"


def test_qwen3_asr_ws_keeps_incremental_segments_and_ttfb(monkeypatch):
    """真流式：逐 chunk 增量文本都保留，且记首字 TTFB；终稿取 final。"""
    _setup_qwen3(
        monkeypatch,
        [_qwen_ev("started", session_id="s2", language="Chinese"),
         _qwen_ev("result", text="一"),
         _qwen_ev("result", text="一二"),
         _qwen_ev("result", final=True, text="一二三")],
    )
    res = adapters.Qwen3ASRWSAdapter(pace=False, trail_silence_s=0).transcribe("a.wav", language="zh")

    assert res.ok and res.text == "一二三", "终稿取 finish 的 final 结果"
    assert res.extra["final_event"] is True
    assert res.extra["n_seg"] >= 2, "增量段应被记录"
    assert res.extra["ttfb_s"] is not None, "首字延迟须记下（同传延迟口径要）"


def test_qwen3_asr_ws_reports_error_event_as_failure(monkeypatch):
    """错误事件 → ok=False 且 error 非空。"""
    _setup_qwen3(
        monkeypatch,
        [_qwen_ev("error", error="invalid session_id")],
    )
    res = adapters.Qwen3ASRWSAdapter(trail_silence_s=0).transcribe("a.wav", language="zh")

    assert not res.ok
    assert res.error and "invalid session_id" in res.error


def test_qwen3_asr_ws_declares_concurrency_cap_of_two(monkeypatch):
    """线上同传共用 8007 → 评测并发上限须为 2，不得被当普通同传接口打满。"""
    ad = adapters.Qwen3ASRWSAdapter()
    assert ad._MAX_WORKERS == 2
    caps = adapters.capability_contract_for("qwen3-asr-ws")
    assert caps["features"]["concurrency_cap"] == 2
    assert "同传" in caps["features"]["concurrency_note"]


def test_qwen3_asr_ws_normalaises_base_url_to_ws_path(monkeypatch):
    """只给 host:port 时应自动补 /ws；给了 /ws 不重复拼。"""
    assert adapters.Qwen3ASRWSAdapter(base_url="ws://h:8007").ws_url == "ws://h:8007/ws"
    assert adapters.Qwen3ASRWSAdapter(base_url="ws://h:8007/ws").ws_url == "ws://h:8007/ws"


def test_infor_preses_worker_cap_for_shared_endpoints():
    """端点自有并发上限时，infer 必须压回上限，绝不静默超用把线上同传产品打满。"""
    import infer

    ad = adapters.Qwen3ASRWSAdapter()
    assert ad._MAX_WORKERS == 2, "8007 线上同传共用 → 上限 2"

    assert infer._aply_worker_cap("qwen3-asr-ws", ad, 1) == 1, "未超上限不动"
    assert infer._aply_worker_cap("qwen3-asr-ws", ad, 2) == 2, "刚好到上限不动"
    assert infer._aply_worker_cap("qwen3-asr-ws", ad, 8) == 2, "超上限须压回"
    assert infer._aply_worker_cap("qwen3-asr-ws", ad, 64) == 2, "远超上限也压回"

    class _Uncaped:
        pass  # 无 _MAX_WORKERS

    assert infer._aply_worker_cap("pro", _Uncaped(), 16) == 16, "无上限的端点不受影响"


def test_plat_realtime_gives_up_when_server_returns_no_text(monkeypatch):
    """无语音样本服务端一个文本都不回 → 必须早退，不能耗到 60s 硬上限×重试。"""
    ws = _FakeRealtimeWS([
        _ev("session.created"), _ev("session.updated"),
        _ev("input_audio_buffer.speech_started"),
        _ev("input_audio_buffer.speech_stopped"),
        _ev("input_audio_buffer.committed"),
    ])

    def recv_no_text():
        if ws.events:
            return json.dumps(ws.events.pop(0))
        _FakeClock().t += 0.5          # 推进时钟
        raise _RealtimeIdle("idle")

    monkeypatch.setattr(adapters, "ws_connect", lambda *a, **k: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x11" * 3200)
    monkeypatch.setattr(ws, "recv", recv_no_text)
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    clock = _FakeClock(step=0.5)
    monkeypatch.setattr(adapters.time, "perf_counter", clock)
    monkeypatch.setattr(adapters.PlatformRealtimeAdapter, "_NO_TEXT_TIMEOUT_S", 2.0)

    res = adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    assert not res.ok and res.error, "无文本应判失败且带 error"
    # 关键：不能跑到硬上限(timeout=60s)。时钟被推进的量应远小于 60s
    assert clock.t < 60.0, f"应早退，实际推进到 {clock.t}s"


def test_plat_realtime_sends_session_finish_and_not_commit(monkeypatch):
    """server_vad 下：必须发 platform.session.finish，且**不能**再发 input_audio_buffer.commit。

    依据平台示例客户端 —— server_vad 由服务端自动分段，
    重复 commit 会回 input_audio_buffer_commit_empty 并触发无谓重试（实测踩到）。
    """
    ws = _FakeRealtimeWS([
        _ev("session.created"), _ev("session.updated"),
        _ev("input_audio_buffer.committed"),
        _ev("conversation.item.input_audio_transcription.completed", transcript="好"),
        _ev("platform.session.finished", item_count=1, status="completed", usage_seconds=1.0),
    ])
    monkeypatch.setattr(adapters, "ws_connect", lambda *a, **k: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x11" * 3200)
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    monkeypatch.setattr(adapters.time, "perf_counter", _FakeClock())

    res = adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    types = [json.loads(m)["type"] for m in ws.sent]
    assert "platform.session.finish" in types, "须发 session.finish 请服务端排空"
    assert "input_audio_buffer.commit" not in types, "server_vad 下不得重复 commit"
    assert res.ok and res.text == "好"
    assert res.extra["server_finished"] is True
    assert res.extra["item_count"] == 1


def test_plat_realtime_terminates_on_session_finished(monkeypatch):
    """收到 platform.session.finished 即权威终态 → 立刻收，不等任何静默超时。"""
    ws = _FakeRealtimeWS([
        _ev("session.created"), _ev("session.updated"),
        _ev("conversation.item.input_audio_transcription.completed", transcript="一"),
        _ev("platform.session.finished", item_count=1, status="completed",
            usage_seconds=1.2, degraded_stages=[]),
    ])
    monkeypatch.setattr(adapters, "ws_connect", lambda *a, **k: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x11" * 3200)
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    clock = _FakeClock(step=0.5)
    monkeypatch.setattr(adapters.time, "perf_counter", clock)

    res = adapters.PlatformRealtimeAdapter().transcribe("a.wav", language="zh")

    assert res.extra["server_finished"] is True
    assert res.extra["server_status"] == "completed"
    assert clock.t < 5.0, f"应在 session.finished 立刻收，实际推进 {clock.t}s"


def test_plat_realtime_paced_does_not_terminate_mid_stream(monkeypatch):
    """pace=True 时服务端会在喂音频途中就提交前几段 → 不得因配平而提前收工丢内容。"""
    ws = _FakeRealtimeWS([
        _ev("session.created"), _ev("session.updated"),
        # 音频还在喂，服务端已提交并完成第 1 段（配平达成！）
        _ev("input_audio_buffer.committed"),
        _ev("conversation.item.input_audio_transcription.completed", transcript="前半"),
        # 后面还有第 2 段
        _ev("input_audio_buffer.committed"),
        _ev("conversation.item.input_audio_transcription.completed", transcript="后半"),
        _ev("platform.session.finished", item_count=2, status="completed"),
    ])
    monkeypatch.setattr(adapters, "ws_connect", lambda *a, **k: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x11" * 64000)
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    monkeypatch.setattr(adapters.time, "perf_counter", _FakeClock(step=0.05))

    res = adapters.PlatformRealtimeAdapter(pace=True).transcribe("a.wav", language="zh")

    assert res.text == "前半后半", f"两段都要，实得 {res.text!r}"


def test_clean_asr_scaffolding_strips_prompt_leak_but_keeps_real_text():
    """模型偶发吐出 chat 模板脚手架时，只保留 <asr_text> 之后的正文。

    实测样本(8007, ascend_00243)：把 'system/user/assistant/language Chinese<asr_text>' 整段吐了出来。
    处理口径照服务端参考客户端实现。
    """
    leak = ("啊，走过的 professor �system\n\nuser\n\nassistant\nlanguage Chinese"
            "<asr_text>啊，走过的 professor �大概有一百多位。")
    out = adapters._clean_asr_scaffolding(leak)
    assert "<asr_text>" not in out and "assistant" not in out and "system" not in out
    assert out == "啊，走过的 professor �大概有一百多位。"


def test_clean_asr_scaffolding_never_touches_normal_transcript():
    """无协议标记时一律不动 —— 正常语音里本来就可能出现 system 等词。"""
    for t in ("宋芳对北京银行私人银行的财富顾问说",
              "i mean for android system",      # 真实内容，不是脚手架
              "system", "user 说的对", ""):
        assert adapters._clean_asr_scaffolding(t) == t, t


def test_plat_realtime_orders_segments_by_speech_start_not_arrival(monkeypatch):
    """分段必须按 speech_started 的时间轴拼，不能按 completed 到达序。

    实测服务端推 completed 的顺序不确定（同段音频两次跑可能正序/反序）；按到达序拼会把
    文本打乱（ref『其市管…政策也均进行调整』→ hyp『也均进行调整。其市管…政策』）。
    """
    ws = _FakeRealtimeWS([
        _ev("session.created"), _ev("session.updated"),
        # 时间轴上：第 1 段 start=200ms，第 2 段 start=5900ms
        _ev("input_audio_buffer.speech_started", item_id="i1", audio_start_ms=200),
        _ev("input_audio_buffer.committed", item_id="i1"),
        _ev("input_audio_buffer.speech_started", item_id="i2", audio_start_ms=5900),
        _ev("input_audio_buffer.committed", item_id="i2"),
        # ⚠️ completed 以**反序**到达（这正是服务端的非确定性行为）
        _ev("conversation.item.input_audio_transcription.completed",
            item_id="i2", transcript="也均进行调整。"),
        _ev("conversation.item.input_audio_transcription.completed",
            item_id="i1", transcript="其市管国管住房公积金政策"),
        _ev("platform.session.finished", item_count=2, status="completed"),
    ])
    monkeypatch.setattr(adapters, "ws_connect", lambda *a, **k: ws)
    monkeypatch.setattr(adapters, "load_pcm_bytes", lambda path: b"\x11" * 3200)
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    monkeypatch.setattr(adapters.time, "perf_counter", _FakeClock())

    res = adapters.PlatformRealtimeAdapter(trail_silence_s=0).transcribe("a.wav", language="zh")

    assert res.text == "其市管国管住房公积金政策也均进行调整。", \
        f"须按 audio_start_ms 排序，实得 {res.text!r}"


