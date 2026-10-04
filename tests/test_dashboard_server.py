import importlib.util
import base64
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("dashboard_server", os.path.join(ROOT, "dashboard", "server.py"))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


def _reset_jobs(monkeypatch):
    server.JOBS.clear()
    server.PROCS.clear()
    monkeypatch.setattr(server, "_flush_jobs", lambda: None)
    monkeypatch.setattr(server.threading, "Thread", NoopThread)


def test_run_dedupe_distinguishes_limit_seed_and_comet(monkeypatch):
    _reset_jobs(monkeypatch)
    first = server.RunReq(model="light", dataset="aishell", tier="lite", limit=50, seed=1, comet=False)
    second = server.RunReq(model="light", dataset="aishell", tier="lite", limit=300, seed=42, comet=True)

    r1 = server.run(first)
    r2 = server.run(second)

    assert "job_id" in r1
    assert r2.get("dedup") is not True
    assert r2["job_id"] != r1["job_id"]


def test_run_dedupes_identical_normalized_request(monkeypatch):
    _reset_jobs(monkeypatch)
    req = server.RunReq(model="light", dataset="aishell", tier="lite", limit=50, seed=1)

    r1 = server.run(req)
    r2 = server.run(req)

    assert r2 == {"job_id": r1["job_id"], "dedup": True}


def test_text_and_speech_simultaneous_runs_have_distinct_fingerprints(monkeypatch):
    _reset_jobs(monkeypatch)
    common = dict(
        model="plat-simult", dataset="mcif_long_en_zh", tier="lite",
        limit=1, seed=7, language="English", target_lang="Chinese",
        request_params={"pipeline_mode": "single_stage"},
    )

    text_run = server.run(server.RunReq(**common, speech_eval=False))
    speech_run = server.run(server.RunReq(**common, speech_eval=True))
    text_job = server.JOBS[text_run["job_id"]]
    speech_job = server.JOBS[speech_run["job_id"]]

    assert speech_run.get("dedup") is not True
    assert text_job["config_hash"] != speech_job["config_hash"]
    assert text_job["speech_eval"] is False
    assert speech_job["speech_eval"] is True


def test_accuracy_mode_defaults_to_four_workers_and_model_scoped_locks(monkeypatch):
    _reset_jobs(monkeypatch)
    server._EP_LOCKS.clear()

    result = server.run(server.RunReq(
        model="light", dataset="aishell", tier="lite", limit=5,
    ))
    job = server.JOBS[result["job_id"]]

    assert job["run_mode"] == "accuracy"
    assert job["workers"] == 4
    assert server._ep_lock("std", "accuracy") is not server._ep_lock("adv", "accuracy")
    assert server._ep_lock("std", "latency") is server._ep_lock("adv", "latency")


def test_latency_mode_forces_single_worker(monkeypatch):
    _reset_jobs(monkeypatch)

    result = server.run(server.RunReq(
        model="std", dataset="aishell", tier="lite", limit=5,
        run_mode="latency", workers=8,
    ))
    job = server.JOBS[result["job_id"]]

    assert job["run_mode"] == "latency"
    assert job["workers"] == 1


def test_simultaneous_model_forces_latency_mode(monkeypatch):
    _reset_jobs(monkeypatch)

    result = server.run(server.RunReq(
        model="plat-simult-ws", dataset="acl6060_long_en_zh", tier="lite", limit=5,
        run_mode="accuracy", workers=8,
    ))
    job = server.JOBS[result["job_id"]]

    assert job["run_mode"] == "latency"
    assert job["workers"] == 1


def test_artifact_delete_endpoints_are_scoped_to_runtime_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", str(tmp_path))
    infer_dir = tmp_path / "infer"
    audio_dir = tmp_path / "audio_out" / "one-run"
    infer_dir.mkdir()
    audio_dir.mkdir(parents=True)
    (infer_dir / "one.jsonl").write_text("{}\n", encoding="utf-8")
    (audio_dir / "one.wav").write_bytes(b"RIFF")

    assert server.del_infer("one.jsonl") == {"ok": True, "file": "one.jsonl"}
    assert server.del_audio_artifacts("one-run") == {"ok": True, "directory": "one-run"}
    assert not (infer_dir / "one.jsonl").exists()
    assert not audio_dir.exists()
    assert server.del_infer("../outside.jsonl").status_code == 400
    assert server.del_audio_artifacts("../outside").status_code == 400


def test_run_exposes_ws_request_plan_before_worker_or_result(monkeypatch, tmp_path):
    _reset_jobs(monkeypatch)
    monkeypatch.setattr(server, "JOBS_LOG_DIR", str(tmp_path))
    req = server.RunReq(
        model="plat-simult-ws", dataset="acl6060_long_en_zh", tier="lite",
        limit=5, language="auto", target_lang="Chinese",
        request_params={"vad_silence_duration_ms": 850},
    )

    started = server.run(req)
    preview = server.job_request(started["job_id"])

    assert preview["source"] == "planned_contract"
    assert preview["method"] == "WEBSOCKET"
    assert preview["url"].startswith("ws://")
    assert '"event": "task_start"' in preview["curl"]
    assert '"target_language": "Chinese"' in preview["curl"]
    assert '"silence_duration": 850' in preview["curl"]
    assert "SEND BINARY <16 kHz mono PCM chunks" in preview["curl"]


def test_openapi_operation_schema_resolves_multipart_ref_and_filters_transport_fields():
    doc = {
        "openapi": "3.1.0",
        "paths": {
            "/api/asr/adv": {
                "post": {
                    "summary": "Advanced ASR",
                    "parameters": [
                        {"name": "trace_id", "in": "query", "schema": {"type": "string"}},
                        {"name": "Authorization", "in": "header", "schema": {"type": "string"}},
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {"$ref": "#/components/schemas/ProRequest"}
                            }
                        },
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "ProRequest": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "language": {
                            "type": "string", "default": "auto",
                            "enum": ["auto", "Chinese", "English"],
                        },
                        "enable_text_edit": {"type": "boolean", "default": False},
                        "speaker_merge_gap_ms": {
                            "type": "integer", "default": 2000, "minimum": 0,
                        },
                    },
                }
            }
        },
    }

    operations = server._openapi_operations(doc)

    assert len(operations) == 1
    op = operations[0]
    assert op["path"] == "/api/asr/adv"
    assert op["template"] == "plat-multipart"
    assert op["template_path"] == "adv"
    assert op["audio_field"] == "file"
    names = {field["name"] for field in op["request_schema"]}
    assert names == {"language", "enable_text_edit", "speaker_merge_gap_ms", "trace_id"}
    language = next(field for field in op["request_schema"] if field["name"] == "language")
    assert language["enum"] == ["auto", "Chinese", "English"]
    assert language["default"] == "auto"
    gap = next(field for field in op["request_schema"] if field["name"] == "speaker_merge_gap_ms")
    assert gap["type"] == "integer" and gap["minimum"] == 0


def test_runtime_request_params_are_allowlisted_typed_and_defaulted():
    schema = [
        {"name": "enable_text_edit", "in": "form", "type": "boolean", "default": False},
        {"name": "speaker_merge_gap_ms", "in": "form", "type": "integer",
         "default": 2000, "minimum": 0},
        {"name": "mode", "in": "form", "type": "string", "enum": ["fast", "accurate"]},
    ]

    normalized = server._normalize_runtime_request_params(
        schema, {"enable_text_edit": "true", "speaker_merge_gap_ms": "1200", "mode": "fast"},
    )

    assert normalized == {
        "enable_text_edit": True,
        "speaker_merge_gap_ms": 1200,
        "mode": "fast",
    }
    assert server._normalize_runtime_request_params(schema, {}) == {
        "enable_text_edit": False,
        "speaker_merge_gap_ms": 2000,
    }

    try:
        server._normalize_runtime_request_params(schema, {"unknown": "x"})
    except ValueError as exc:
        assert "未登记" in str(exc)
    else:
        raise AssertionError("未登记参数必须拒绝")

    try:
        server._sanitize_request_schema([
            {"name": "api_key", "in": "form", "type": "string"},
        ])
    except ValueError as exc:
        assert "敏感" in str(exc)
    else:
        raise AssertionError("敏感参数不能登记为运行时参数")

    enriched = server._sanitize_request_schema([{
        "name": "hotwords", "wire_name": "hot_words", "in": "form", "type": "string",
        "managed_by": "hotwords", "api_default": "", "eval_default": "",
        "depends_on": "domain is set",
    }])[0]
    assert enriched["wire_name"] == "hot_words"
    assert enriched["api_default"] == enriched["eval_default"] == ""
    assert enriched["depends_on"] == "domain is set"


def test_interface_config_fingerprint_changes_with_runtime_params_but_not_secret(monkeypatch):
    cfg = {
        "id": "x-pro",
        "name": "Advanced",
        "base_url": "http://asr.test",
        "template": "plat-multipart",
        "path": "adv",
        "api_key": "secret-a",
        "key_env": "ASR_TOKEN",
        "request_schema": [
            {"name": "enable_text_edit", "in": "form", "type": "boolean", "default": False},
        ],
    }
    monkeypatch.setattr(server, "_load_ifaces", lambda: [dict(cfg)])

    first = server._runtime_config("x-pro", {"enable_text_edit": False})
    changed = server._runtime_config("x-pro", {"enable_text_edit": True})
    cfg["api_key"] = "secret-b"
    rotated = server._runtime_config("x-pro", {"enable_text_edit": False})

    assert first["hash"] != changed["hash"]
    assert first["hash"] == rotated["hash"]
    assert "secret" not in json.dumps(first)


def test_target_language_changes_runtime_fingerprint():
    en = server._runtime_config("adv", {}, {"target_lang": "英文"})
    zh = server._runtime_config("adv", {}, {"target_lang": "中文"})

    assert en["hash"] and zh["hash"]
    assert en["hash"] != zh["hash"]


def test_custom_hotwords_change_runtime_fingerprint():
    first = server._runtime_config("sse", {}, {"hotwords_text": "ExampleTerm"})
    second = server._runtime_config("sse", {}, {"hotwords_text": "示例词"})

    assert first["hash"] and second["hash"]
    assert first["hash"] != second["hash"]


def test_all_visible_builtin_interfaces_have_explicit_contract_status():
    payloads = [server._model_payload(m) for m in server.MODELS if m.get("group") != "custom"]

    assert payloads
    assert all(m["request_contract"]["status"] != "unverified" for m in payloads)
    assert all(m["capability_contract"].get("tasks") for m in payloads)
    assert all(m["capability_contract"].get("input_modalities") for m in payloads)
    assert all(m["capability_contract"].get("output_modalities") for m in payloads)
    assert next(m for m in payloads if m["id"] == "adv")["request_contract"]["source"] == "platform-openapi"
    assert {x["name"] for x in next(m for m in payloads if m["id"] == "adv")["request_schema"]} >= {
        "language", "target_lang", "enable_text_edit", "speaker_merge_gap_ms",
    }
    pro_schema = next(m for m in payloads if m["id"] == "adv")["request_schema"]
    assert "英文逗号" in next(x for x in pro_schema if x["name"] == "language")["description"]
    assert "仅在说话人识别开启时生效" in next(
        x for x in pro_schema if x["name"] == "speaker_merge_gap_ms"
    )["description"]


def test_own_service_contracts_match_verified_public_surfaces():
    lite = server._model_payload(server._find_model("light"))
    assert {x["name"] for x in lite["request_schema"]} == {
        "audio", "language", "itn", "hotwords",
    }
    itn = next(x for x in lite["request_schema"] if x["name"] == "itn")
    assert itn["api_default"] is True and itn["eval_default"] is False
    assert lite["capability_contract"]["output_modalities"] == ["final_text"]
    assert lite["capability_contract"]["features"]["timestamps"] is False

    normal = server._model_payload(server._find_model("std"))
    assert {x["name"] for x in normal["request_schema"]} == {
        "file", "enable_word_timestamps", "language",
    }

    pro = server._model_payload(server._find_model("adv"))
    edit = next(x for x in pro["request_schema"] if x["name"] == "enable_text_edit")
    assert edit["api_default"] is True and edit["eval_default"] is False
    assert {"raw_result", "alt_result", "pipeline_status", "speaker_overlap"} <= set(
        pro["capability_contract"]["output_modalities"]
    )

    domain = server._model_payload(server._find_model("adv-domain"))
    hotwords = next(x for x in domain["request_schema"] if x.get("managed_by") == "hotwords")
    assert hotwords["wire_name"] == "hot_words"

    sse = server._model_payload(server._find_model("sse"))["capability_contract"]
    assert sse["streaming"]["granularity"] == "stage_event"
    assert sse["streaming"]["token_delta"] is False
    assert sse["response_events"] == ["start", "asr_result", "sop", "done", "error"]

    formula = server._model_payload(server._find_model("plat-formula"))
    text = next(x for x in formula["request_schema"] if x["name"] == "text")
    assert text["required"] is True and text["managed_by"] == "source_text"


def test_own_streaming_profiles_declare_fixed_requests_and_full_outputs():
    simult = server._model_payload(server._find_model("plat-simult"))
    assert simult["request_contract"]["kind"] == "endpoint"
    assert "vad_setting" in simult["request_contract"]["fixed_request"]
    assert {"tts_mode", "tts_output_sample_rate", "tts_voice_id"} <= {
        x["name"] for x in simult["request_schema"]
    }
    assert {"pipeline_mode", "return_asr_text", "incremental_enabled",
            "incremental_interval_ms", "incremental_holdback_tokens", "vad_threshold",
            "vad_silence_duration_ms", "vad_hard_max_duration_ms"} <= {
        x["name"] for x in simult["request_schema"]
    }
    assert {"translated_audio", "source_audio", "segment_timings"} <= set(
        simult["capability_contract"]["output_modalities"]
    )
    assert {"si_incremental_update", "si_token", "si_segment_done",
            "si_tts_file", "si_source_audio"} <= set(
        simult["capability_contract"]["response_events"]
    )

    voice_profile = server._model_payload(server._find_model("plat-simult-ws"))
    assert voice_profile["request_contract"]["kind"] == "evaluation_profile"
    assert voice_profile["request_contract"]["profile_of"] == "voice-input"
    assert voice_profile["request_contract"]["fixed_request"][
        "voice_input_setting.hot_words"
    ] == []
    assert {"vad_threshold", "vad_silence_duration_ms", "vad_hard_max_duration_ms",
            "summary_interval_s"} <= {item["name"] for item in voice_profile["request_schema"]}

    diar = server._model_payload(server._find_model("plat-diar"))
    assert diar["request_contract"]["kind"] == "evaluation_profile"
    assert diar["request_contract"]["profile_of"] == "adv"
    assert diar["request_contract"]["fixed_request"] == {
        "enable_speaker_diarization": True,
    }

    hidden = server.request_contract_for("plat-simult-post")
    assert hidden["kind"] == "hidden_legacy_endpoint"
    assert "plat-simult-post" not in {m["id"] for m in server.MODELS}


def test_custom_interface_infers_hotword_cap_from_registered_schema():
    caps = server._model_caps({
        "id": "x-schema-hotwords", "group": "custom", "tmpl": "custom-http-asr",
        "request_schema": [
            {"name": "hot_words", "in": "form", "type": "string",
             "managed_by": "hotwords", "default": ""},
        ],
    })

    assert "hotwords" in caps


def test_endpoint_capabilities_do_not_leak_across_normal_pro_domain():
    normal = server._model_caps(server._find_model("std"))
    pro = server._model_caps(server._find_model("adv"))
    domain = server._model_caps(server._find_model("adv-domain"))

    assert "hotwords" not in normal and "domain" not in normal
    assert "hotwords" not in pro and "domain" not in pro
    assert {"asr", "translate_audio", "hotwords", "domain"} <= set(domain)


def test_sse_declares_translation_and_dual_result():
    caps = server._model_caps(server._find_model("sse"))

    assert {"asr", "translate_audio", "hotwords", "dual_result"} <= set(caps)
    assert "hotwords" in server._model_caps(server._find_model("plat-simult"))


def test_custom_template_caps_follow_selected_scenarios():
    caps = server._model_caps({
        "id": "x-mt-only", "group": "custom", "tmpl": "openai-chat",
        "scenarios": ["翻译"],
    })

    assert "translate_text" in caps
    assert "summarize" not in caps


def test_simult_language_pair_contract_rejects_known_invalid_direction():
    acl = next(d for d in server.DATASETS if d["id"] == "acl6060")
    multilingual = next(d for d in server.DATASETS if d["id"] == "fleurs_multi")

    assert "未登记语向 en→zh" in server._capability_compat_error("xf-simult", acl)
    assert server._capability_compat_error("qwen-simult", acl) == ""
    assert server._capability_compat_error("qwen-simult", multilingual) == ""
    assert "已停用" in server._capability_compat_error("doubao-simult", acl)


def test_longform_simult_datasets_are_registered_as_fixed_language_pairs():
    longform = [d for d in server.DATASETS if d.get("longform")]
    ids = {d["id"] for d in longform}

    assert len([x for x in ids if x.startswith("acl6060_long_en_")]) == 10
    assert {"mcif_long_en_zh", "mcif_long_en_de", "mcif_long_en_it"} <= ids
    assert {"realsi_en_zh", "realsi_zh_en"} <= ids
    assert all(d.get("simul") and d.get("required_paths") for d in longform)
    assert all(server._ds_requires(d)["hard"] == ["translate_audio"] for d in longform)


def test_aishell2_official_test_channels_are_registered_separately():
    channels = [d for d in server.DATASETS if d.get("builder") == "aishell2_eval"]

    assert {d["id"] for d in channels} == {
        "aishell2_test_ios", "aishell2_test_android", "aishell2_test_mic"}
    assert {d["subset"] for d in channels} == {"ios", "android", "mic"}
    assert all(d["count"] == 5000 and d.get("required_paths") for d in channels)


def test_qwen3_multilingual_asr_datasets_are_registered_full_split():
    fleurs = [d for d in server.DATASETS if d.get("builder") == "fleurs_asr"]
    commonvoice = [d for d in server.DATASETS if d.get("builder") == "commonvoice"]

    assert len(fleurs) == 12
    assert {d["subset"] for d in fleurs} == {
        "en", "zh", "yue", "ar", "de", "es", "fr", "it", "ja", "ko", "pt", "ru",
    }
    assert len(commonvoice) == 13
    assert {d["subset"] for d in commonvoice} == {
        "en", "zh-CN", "yue", "zh-TW", "ar", "de", "es", "fr", "it", "ja", "ko", "pt", "ru",
    }
    assert sum(d["count"] for d in commonvoice) == 134731
    assert all(d["runnable"] and d.get("required_paths") for d in commonvoice)


def test_librispeech_other_and_all_local_wmt_directions_are_registered():
    libri_other = next(d for d in server.DATASETS if d["id"] == "librispeech_other")
    wmt = [d for d in server.DATASETS if d.get("builder", d["id"]) == "wmt"]

    assert libri_other["subset"] == "other" and libri_other["count"] == 2939
    assert {(d.get("subset") or "wmt22-zh-en", d["count"]) for d in wmt} == {
        ("wmt22-zh-en", 1875), ("wmt22-en-zh", 2037),
        ("wmt23-zh-en", 1976), ("wmt23-en-zh", 2074),
    }


def test_longform_dataset_preview_returns_complete_text_and_navigation_metadata(monkeypatch):
    import build_manifest

    source = "source " * 120
    reference = "参考译文" * 180
    row = {
        "id": "long-1", "task": "translate", "lang": "en-zh",
        "audio_path": "/tmp/long.wav", "source_text": source, "ref_text": reference,
        "segments": [{"source_text": "a", "ref_text": "甲"}] * 7,
        "reference_type": "written_translation",
    }
    monkeypatch.setattr(server, "DATASETS", [{
        "id": "long-preview", "name": "Long Preview", "scenario": "翻译·语音(长音频)",
        "lang": "en-zh", "trait": "完整演讲", "runnable": True, "longform": True,
        "builder": "preview_test",
    }])
    monkeypatch.setitem(build_manifest.BUILDERS, "preview_test", lambda limit=0: [row])
    server.PREVIEW_CACHE.clear()

    payload = server.dataset_preview("long-preview", n=8)

    assert payload["longform"] is True
    assert payload["dataset"]["name"] == "Long Preview"
    assert payload["items"][0]["source_text"] == source
    assert payload["items"][0]["ref_text"] == reference
    assert payload["items"][0]["source_chars"] == len(source)
    assert payload["items"][0]["ref_chars"] == len(reference)
    assert payload["items"][0]["segment_count"] == 7
    assert payload["items"][0]["reference_type"] == "written_translation"

    short = server._dataset_preview_item(row, longform=False)
    assert len(short["source_text"]) == 400
    assert short["source_chars"] == len(source)


def test_dataset_availability_uses_declarative_required_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", str(tmp_path))
    required = tmp_path / "datasets" / "translation" / "long" / "ref.xml"
    ds = {"id": "declarative", "required_paths": ["translation/long/ref.xml"]}

    ok, reason = server._dataset_available(ds)
    assert ok is False and "ref.xml" in reason

    required.parent.mkdir(parents=True)
    required.write_text("ok", encoding="utf-8")
    assert server._dataset_available(ds) == (True, "")


def test_offline_qwen_is_translation_baseline_not_simult_capability():
    caps = server._model_caps(server._find_model("qwen-simult-offline"))

    assert "translate_audio" in caps
    assert "simult" not in caps
    contract = server._model_payload(server._find_model("qwen-simult-offline"))["capability_contract"]
    assert contract["streaming"]["realtime"] is False


def test_legacy_normal_hotword_result_is_preserved_but_invalidated(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    path = results / "normal_seaco_n50_hw.json"
    path.write_text(json.dumps({"summary": {
        "model": "std", "dataset": "seaco", "tier": "lite",
        "meta": {"hotwords": True}, "low_coverage": False,
    }}), encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", str(tmp_path))

    row = server.read_results()[0]

    assert row["invalid_identity"] is True
    assert row["low_coverage"] is True
    assert "adv-domain" in row["invalid_reason"]


def test_dashboard_run_dedupes_same_request_params_and_separates_changed_params(monkeypatch):
    _reset_jobs(monkeypatch)
    cfg = {
        "id": "x-runtime",
        "base_url": "http://asr.test",
        "template": "plat-multipart",
        "path": "adv",
        "request_schema": [
            {"name": "enable_text_edit", "in": "form", "type": "boolean", "default": False},
        ],
    }
    monkeypatch.setattr(server, "_load_ifaces", lambda: [cfg])

    first = server.RunReq(
        model="x-runtime", dataset="aishell", tier="lite",
        request_params={"enable_text_edit": False},
    )
    same = server.RunReq(
        model="x-runtime", dataset="aishell", tier="lite",
        request_params={"enable_text_edit": "false"},
    )
    changed = server.RunReq(
        model="x-runtime", dataset="aishell", tier="lite",
        request_params={"enable_text_edit": True},
    )

    r1 = server.run(first)
    r2 = server.run(same)
    r3 = server.run(changed)

    assert r2 == {"job_id": r1["job_id"], "dedup": True}
    assert r3["job_id"] != r1["job_id"]
    assert server.JOBS[r1["job_id"]]["request_params"] == {"enable_text_edit": False}
    assert server.JOBS[r3["job_id"]]["config_hash"] != server.JOBS[r1["job_id"]]["config_hash"]


def test_result_model_parses_legacy_lite_filename():
    assert server._result_model("/x/results/aishell_light_lite_n300_s42.json", {}) == "light"
    assert server._result_model("/x/results/fleurs_multi_qwen-simult_lite_n80_s42.json", {}) == "qwen-simult"
    assert server._result_model("/x/results/seaco_ext-adv_full_hw.json", {}) == "ext-adv"


def test_jobs_marks_external_dashboard_running_as_interrupted(monkeypatch):
    server.JOBS.clear()
    old = {"id": "deadbeef", "origin": "dashboard", "status": "running",
           "stage": "infer+score", "created_ts": 1.0, "started_ts": 1.0}
    monkeypatch.setattr(server, "_read_jobs_file", lambda: {"deadbeef": old})

    out = server.jobs()
    job = out["jobs"][0]

    assert job["status"] != "running"
    assert "重启" in job["stage"] or "丢失" in job["stage"]


def test_jobs_exposes_live_cli_as_started_and_marks_dead_pid(monkeypatch):
    server.JOBS.clear()
    cli = {"id": "cli_live", "origin": "cli", "status": "running",
           "stage": "CLI 运行中", "created_at": "2026-08-05T18:34:28",
           "started_ts": 100.0, "pid": 123}
    monkeypatch.setattr(server, "_read_jobs_file", lambda: {"cli_live": cli})
    monkeypatch.setattr(server.time, "time", lambda: 120.0)
    monkeypatch.setattr(server, "_pid_alive", lambda pid: True)

    live = server.jobs()["jobs"][0]
    assert live["status"] == "running"
    assert live["started_at"] == cli["created_at"]
    assert live["elapsed"] == 20

    monkeypatch.setattr(server, "_pid_alive", lambda pid: False)
    dead = server.jobs()["jobs"][0]
    assert dead["status"] == "error"
    assert "退出" in dead["stage"]


def test_jobs_hydrates_legacy_cli_result_for_clickable_detail(monkeypatch):
    server.JOBS.clear()
    cli = {"id": "cli_done", "origin": "cli", "status": "done",
           "stage": "完成", "out": "aishell_lite_full.json"}
    summary = {"metric": "CER", "CER": 0.02, "n_ok": 10, "n_total": 10,
               "meta": {"run_spec": {"model": "light", "workers": 4,
                                        "language": "auto", "config_hash": "abc123",
                                        "request_params": {"itn": False},
                                        "endpoint": "http://example.test/asr"}}}
    monkeypatch.setattr(server, "_read_jobs_file", lambda: {"cli_done": cli})
    monkeypatch.setattr(
        server, "_cli_result_record",
        lambda filename: ("aishell_lite_full.json", summary),
    )

    job = server.jobs()["jobs"][0]
    assert job["result_file"] == "aishell_lite_full.json"
    assert job["summary"] == summary
    assert job["tier"] == "full"
    assert job["run_mode"] == "accuracy"
    assert job["workers"] == 4
    assert job["language"] == "auto"
    assert job["config_hash"] == "abc123"
    assert job["request_params"] == {"itn": False}
    assert job["request_preview"]["url"] == "http://example.test/asr"


def test_parse_longform_progress_and_expose_heartbeat_age(monkeypatch):
    line = ('prefix @@LONGFORM_PROGRESS {"phase":"segment","sample_id":"talk-1",'
            '"sample_index":1,"sample_total":5,"audio_s":507.26,'
            '"audio_total_s":682,"segments":69,"latest_text":"译文"}')
    progress = server._parse_longform_progress(line)

    assert progress["audio_s"] == 507.3
    assert progress["sample_index"] == 1
    assert progress["latest_text"] == "译文"
    assert server._parse_longform_progress("ordinary log") is None
    assert server._parse_longform_progress("@@LONGFORM_PROGRESS not-json") is None

    server.JOBS.clear()
    server.JOBS["live"] = {
        "id": "live", "origin": "dashboard", "status": "running",
        "stage": "同传流式处理中", "created_ts": 100.0, "started_ts": 101.0,
        "_stream_heartbeat_ts": 115.0, "stream_progress": progress,
    }
    monkeypatch.setattr(server, "_read_jobs_file", lambda: {})
    monkeypatch.setattr(server.time, "time", lambda: 120.0)

    job = server.jobs()["jobs"][0]
    assert job["stream_progress"]["heartbeat_age_s"] == 5
    assert "_stream_heartbeat_ts" not in job


def test_longform_job_log_only_persists_state_changes():
    base = {"sample_index": 1, "sample_total": 2, "audio_s": 125,
            "audio_total_s": 682, "segments": 15, "latest_text": "同一段译文"}

    assert server._longform_progress_log_line({**base, "phase": "streaming"}, "T") is None
    assert server._longform_progress_log_line({**base, "phase": "partial"}, "T") is None
    assert server._longform_progress_log_line({**base, "phase": "revision"}, "T") is None
    segment = server._longform_progress_log_line({**base, "phase": "segment"}, "T")
    assert segment == ("T INFO [segment] 场次 1/2 · 音频 02:05/11:22"
                       " · 第 15 段定稿 · 译文 同一段译文\n")
    assert "开始" in server._longform_progress_log_line({**base, "phase": "started"}, "T")


def test_doctor_distinguishes_required_and_optional_dependencies(monkeypatch):
    import importlib.util as importlib_util
    import shutil

    monkeypatch.setattr(server, "MODELS", [])
    monkeypatch.setattr(server, "DATASETS", [])
    monkeypatch.setattr(importlib_util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(server, "_speex_available", lambda: False)
    monkeypatch.setattr(server, "comet_runtime_available", lambda: False)
    monkeypatch.setattr(server, "scorer_health", lambda timeout=5.0: None)
    out = server.doctor()

    required = {"websocket-client(同传)", "protobuf(豆包同传)",
                "GPU scorer / faster-whisper(同传tts_err)", "pyannote.metrics(DER)"}
    selected = [dep for dep in out["deps"] if dep["name"] in required]
    assert {dep["name"] for dep in selected} == required
    assert all(dep["ok"] is False and dep["required"] is True and dep["severity"] == "error"
               and dep["reason"] for dep in selected)
    comet = next(dep for dep in out["deps"] if dep["name"].startswith("GPU XCOMET-XL"))
    assert comet["ok"] is False and comet["required"] is False and comet["severity"] == "warning"
    ffmpeg = next(item for item in out["bins"] if item["name"].startswith("ffmpeg"))
    assert ffmpeg["ok"] is False and ffmpeg["required"] is True and ffmpeg["severity"] == "error"


def test_runtime_dependency_check_is_task_and_model_specific(monkeypatch):
    import shutil

    available = {"websocket", "faster_whisper", "google.protobuf", "pyannote.metrics", "comet"}
    monkeypatch.setattr(server, "_module_available", lambda name: name in available)
    monkeypatch.setattr(server, "_speex_available", lambda: True)
    monkeypatch.setattr(server, "comet_runtime_available", lambda: "comet" in available)
    monkeypatch.setattr(server, "scorer_runtime_available", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    assert server._missing_runtime_dependencies(
        "qwen-simult", {"metric": "chrF"}, comet=False) == []
    assert server._missing_runtime_dependencies(
        "doubao-simult", {"metric": "chrF"}, comet=False) == []
    assert server._missing_runtime_dependencies(
        "plat-diar", {"metric": "DER"}, comet=False) == []

    available.remove("google.protobuf")
    assert server._missing_runtime_dependencies(
        "qwen-simult", {"metric": "chrF"}, comet=False) == []
    assert server._missing_runtime_dependencies(
        "doubao-simult", {"metric": "chrF"}, comet=False) == ["protobuf>=6.31.1"]

    available.remove("pyannote.metrics")
    assert server._missing_runtime_dependencies(
        "plat-diar", {"metric": "DER"}, comet=False) == ["pyannote.metrics"]

    available.remove("comet")
    assert server._missing_runtime_dependencies(
        "gemma-text", {"metric": "chrF"}, comet=True) == ["GPU XCOMET-XL scorer / unbabel-comet"]

    monkeypatch.setattr(server, "scorer_runtime_available", lambda: True)
    assert server._missing_runtime_dependencies(
        "gemma-text", {"metric": "chrF"}, comet=True) == []
    assert server._missing_runtime_dependencies(
        "qwen-simult", {"metric": "chrF"}, comet=False) == []


def test_scorer_smoke_uses_manifest_audio_and_never_accepts_a_path(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    audio_dir = tmp_path / "datasets"
    manifests.mkdir()
    audio_dir.mkdir()
    audio = audio_dir / "sample.wav"
    audio.write_bytes(b"RIFF")
    (manifests / "m.jsonl").write_text(
        json.dumps({"id": "a0", "audio_path": "datasets/sample.wav"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "ROOT", str(tmp_path))
    monkeypatch.setattr(server, "remote_utmos_score", lambda items: {
        "model": "fusion_stage3", "items": [{"id": "smoke", "score": 4.1}],
    })

    response = server.scorer_smoke(server.ScorerSmokeReq(metric="utmos"))

    assert response["ok"] is True
    assert response["sample_id"] == "a0"
    assert response["result"] == {"model": "fusion_stage3", "score": 4.1}


def test_formula_availability_uses_builder_golden_path(tmp_path, monkeypatch):
    golden = tmp_path / "datasets" / "golden"
    golden.mkdir(parents=True)
    (golden / "tts_formula.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", str(tmp_path))

    assert server._dataset_available({"id": "formula"}) == (True, "")


def test_dashboard_reads_legacy_result_without_plat_metadata(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    legacy = {"summary": {"metric": "CER", "CER": 0.1, "err_rate": 0.1,
                           "n_ok": 1, "n_total": 1}, "samples": []}
    (results / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", str(tmp_path))

    rows = server.read_results()
    assert len(rows) == 1 and rows[0]["CER"] == 0.1
    assert rows[0]["_file"] == "legacy.json" and "meta" not in rows[0]


def test_lid_result_aggregates_distribution_and_filters_samples(tmp_path, monkeypatch):
    infer_dir = tmp_path / "infer"
    infer_dir.mkdir()
    path = infer_dir / "fireredlid__demo.jsonl"
    rows = [
        {"id": "__meta__", "task": "lid", "manifest": "demo.jsonl",
         "endpoint": "http://lid.test/lid", "n_total": 3},
        {"id": "a", "ok": True, "pred_label": "zh xinan", "confidence": .9,
         "service_rtf": .02, "audio_path": "datasets/a.wav", "ref_dialect": "Southwestern"},
        {"id": "b", "ok": True, "pred_label": "zh north", "confidence": .7,
         "service_rtf": .04, "audio_path": "datasets/b.wav", "ref_dialect": "Ji-Lu"},
        {"id": "c", "ok": False, "error": "timeout", "audio_path": "datasets/c.wav"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", str(tmp_path))

    all_rows = server.lid_result(path.name)

    assert all_rows["summary"]["n_ok"] == 2
    assert all_rows["summary"]["n_fail"] == 1
    assert all_rows["summary"]["coverage"] == 0.666667
    assert all_rows["summary"]["labels"][0]["count"] == 1
    assert all_rows["summary"]["confidence_mean"] == .8

    filtered = server.lid_result(path.name, label="zh xinan")
    assert filtered["filter"]["total"] == 1
    assert filtered["samples"][0]["id"] == "a"
    assert filtered["samples"][0]["audio_path"] == str(tmp_path / "datasets/a.wav")

    failed = server.lid_result(path.name, label="__failed__")
    assert failed["filter"]["total"] == 1 and failed["samples"][0]["id"] == "c"


def test_lid_config_lists_only_runnable_asr_datasets_and_prefills_url(monkeypatch):
    datasets = [
        {"id": "zh", "name": "中文", "scenario": "ASR·普通话",
         "count": 10, "metric": "CER", "runnable": True},
        {"id": "mt", "name": "翻译", "scenario": "翻译",
         "count": 10, "metric": "chrF", "runnable": True},
        {"id": "off", "name": "关闭", "scenario": "ASR·英文",
         "count": 10, "metric": "WER", "runnable": False},
    ]
    monkeypatch.setattr(server, "DATASETS", datasets)
    monkeypatch.setattr(server, "_dataset_available",
                        lambda ds: (ds["id"] == "zh", "missing" if ds["id"] != "zh" else ""))
    monkeypatch.setenv("FIRERED_LID_URL", "http://lid.internal:8000")

    out = server.lid_config()

    assert [row["id"] for row in out["datasets"]] == ["zh"]
    assert out["datasets"][0]["available"] is True
    assert out["default_url"] == "http://lid.internal:8000"
    assert out["defaults"]["limit"] == 100


def test_lid_run_uses_endpoint_and_experiment_as_independent_result_dimensions(monkeypatch):
    _reset_jobs(monkeypatch)
    monkeypatch.setattr(server, "_dataset_available", lambda ds: (True, ""))

    first = server.start_lid_run(server.LidRunReq(
        dataset="aishell", limit=20, seed=7,
        url="http://lid.internal:8000", experiment="fp16",
    ))
    duplicate = server.start_lid_run(server.LidRunReq(
        dataset="aishell", limit=20, seed=7,
        url="http://lid.internal:8000", experiment="fp16",
    ))
    second = server.start_lid_run(server.LidRunReq(
        dataset="aishell", limit=20, seed=7,
        url="http://lid.internal:8000", experiment="fp32",
    ))

    assert duplicate == {"job_id": first["job_id"], "dedup": True}
    assert second["job_id"] != first["job_id"]
    first_job = server.JOBS[first["job_id"]]
    second_job = server.JOBS[second["job_id"]]
    assert first_job["task"] == "lid"
    assert first_job["endpoint"] == "http://lid.internal:8000"
    assert first_job["experiment"] == "fp16"
    _, first_out = server._lid_job_paths("aishell", 20, 7, first_job["endpoint"], "fp16")
    _, second_out = server._lid_job_paths("aishell", 20, 7, second_job["endpoint"], "fp32")
    assert first_out != second_out
    assert "fp16" in first_out and "fp32" in second_out


def test_lid_run_rejects_non_http_or_credentialed_service_urls(monkeypatch):
    monkeypatch.setattr(server, "_dataset_available", lambda ds: (True, ""))

    missing_scheme = server.start_lid_run(server.LidRunReq(
        dataset="aishell", url="lid.internal:8000",
    ))
    credentialed = server.start_lid_run(server.LidRunReq(
        dataset="aishell", url="http://user:secret@lid.internal:8000",
    ))

    assert missing_scheme.status_code == 400
    assert credentialed.status_code == 400


def test_interfaces_are_saved_as_valid_json(tmp_path, monkeypatch):
    target = tmp_path / "interfaces.json"
    monkeypatch.setattr(server, "IFACE_FILE", str(target))
    server._save_ifaces([{"id": "x-safe", "name": "测试"}])
    assert json.loads(target.read_text(encoding="utf-8")) == [{"id": "x-safe", "name": "测试"}]


def test_edit_custom_plat_recovers_display_url_saved_as_base_url(tmp_path, monkeypatch):
    target = tmp_path / "interfaces.json"
    broken = {
        "id": "x-normal-8000",
        "name": "旧名称",
        "base_url": "http://h.example.test/api/asr/std",
        "url": "h.example.test/api/asr/std",
        "template": "plat-multipart",
        "path": "std",
        "scenarios": ["ASR"],
        "enabled": True,
    }
    target.write_text(json.dumps([broken]), encoding="utf-8")
    monkeypatch.setattr(server, "IFACE_FILE", str(target))
    monkeypatch.setattr(server, "MODELS", [{
        "id": broken["id"], "name": broken["name"], "group": "custom",
        "url": broken["url"], "base_url": broken["base_url"], "enabled": True,
    }])
    monkeypatch.setattr(server, "_BUILTIN_MODELS", {})

    out = server.edit_iface(
        broken["id"],
        server.EditIfaceReq(
            name="新名称",
            base_url="http://h.example.test/api/asr/std",
        ),
    )

    saved = json.loads(target.read_text(encoding="utf-8"))[0]
    assert out == {"ok": True, "builtin": False, "schema_reset": False}
    assert saved["base_url"] == "http://h.example.test"
    assert saved["url"] == "h.example.test/api/asr/std"
    assert server.MODELS[0]["base_url"] == "http://h.example.test"


def test_edit_custom_interface_clears_stale_request_schema_when_endpoint_changes(tmp_path, monkeypatch):
    target = tmp_path / "interfaces.json"
    cfg = {
        "id": "x-schema",
        "name": "带参数接口",
        "base_url": "http://old.test",
        "url": "old.test/api/asr/adv",
        "template": "plat-multipart",
        "path": "adv",
        "scenarios": ["ASR"],
        "enabled": True,
        "request_schema": [
            {"name": "enable_text_edit", "in": "form", "type": "boolean", "default": False},
        ],
    }
    target.write_text(json.dumps([cfg]), encoding="utf-8")
    monkeypatch.setattr(server, "IFACE_FILE", str(target))
    monkeypatch.setattr(server, "MODELS", [{
        "id": cfg["id"], "name": cfg["name"], "group": "custom",
        "url": cfg["url"], "base_url": cfg["base_url"], "enabled": True,
        "request_schema": cfg["request_schema"],
    }])
    monkeypatch.setattr(server, "_BUILTIN_MODELS", {})

    out = server.edit_iface(
        cfg["id"],
        server.EditIfaceReq(name=cfg["name"], base_url="http://new.test"),
    )

    saved = json.loads(target.read_text(encoding="utf-8"))[0]
    assert out["schema_reset"] is True
    assert saved["request_schema"] == []
    assert server.MODELS[0]["request_schema"] == []


def test_edit_custom_interface_can_refresh_path_and_request_schema(tmp_path, monkeypatch):
    target = tmp_path / "interfaces.json"
    cfg = {
        "id": "x-refresh",
        "name": "Standard",
        "base_url": "http://asr.test",
        "url": "asr.test/api/asr/std",
        "template": "plat-multipart",
        "path": "std",
        "scenarios": ["ASR"],
        "enabled": True,
        "request_schema": [
            {"name": "enable_word_timestamps", "in": "form", "type": "boolean"},
        ],
    }
    target.write_text(json.dumps([cfg]), encoding="utf-8")
    monkeypatch.setattr(server, "IFACE_FILE", str(target))
    monkeypatch.setattr(server, "MODELS", [{
        "id": cfg["id"], "name": cfg["name"], "group": "custom",
        "url": cfg["url"], "base_url": cfg["base_url"], "enabled": True,
        "path": cfg["path"], "request_schema": cfg["request_schema"],
    }])
    monkeypatch.setattr(server, "_BUILTIN_MODELS", {})
    refreshed = [
        {"name": "enable_text_edit", "in": "form", "type": "boolean", "default": False},
        {"name": "speaker_merge_gap_ms", "in": "form", "type": "integer", "default": 2000},
    ]

    out = server.edit_iface(
        cfg["id"],
        server.EditIfaceReq(
            name="Advanced", base_url="http://asr.test", path="adv",
            request_schema=refreshed,
        ),
    )

    saved = json.loads(target.read_text(encoding="utf-8"))[0]
    assert out["schema_reset"] is False
    assert saved["path"] == "adv"
    assert saved["url"] == "asr.test/api/asr/adv"
    assert [x["name"] for x in saved["request_schema"]] == [
        "enable_text_edit", "speaker_merge_gap_ms",
    ]
    assert server.MODELS[0]["path"] == "adv"
    assert len(server.MODELS[0]["request_schema"]) == 2


def test_job_request_reads_sanitized_preview_from_job_log(tmp_path, monkeypatch):
    preview = {
        "method": "POST",
        "url": "http://asr.internal/api/asr/std",
        "headers": {"Authorization": "<Bearer token hidden>"},
        "form": {"language": "auto"},
        "files": {"file": {"value": "<audio omitted>", "content_type": "audio/wav"}},
        "curl": "curl --header 'Authorization: <Bearer token hidden>' --form 'file=@<audio-file-hidden>'",
        "audio_hidden": True,
    }
    token = base64.urlsafe_b64encode(json.dumps(preview).encode()).decode().rstrip("=")
    monkeypatch.setattr(server, "JOBS_LOG_DIR", str(tmp_path))
    (tmp_path / "job-123.log").write_text(
        f"2026-07-31 INFO [req] {server.logconf.REQUEST_PREVIEW_MARKER}{token}\n",
        encoding="utf-8",
    )

    out = server.job_request("job-123")

    assert out["method"] == "POST"
    assert out["audio_hidden"] is True
    assert "Bearer token hidden" in out["curl"]
    assert "secret" not in json.dumps(out)


def test_job_response_reads_bounded_preview_from_job_log(tmp_path, monkeypatch):
    preview = {
        "source": "http", "status": 200, "content_type": "application/json",
        "body_type": "json", "body": {"result": {"text": "你好"}},
        "response_bytes": 48, "truncated": False,
    }
    token = base64.urlsafe_b64encode(json.dumps(preview).encode()).decode().rstrip("=")
    monkeypatch.setattr(server, "JOBS_LOG_DIR", str(tmp_path))
    (tmp_path / "job-response.log").write_text(
        f"2026-07-31 INFO [req] {server.logconf.RESPONSE_PREVIEW_MARKER}{token}\n",
        encoding="utf-8",
    )

    out = server.job_response("job-response")

    assert out["source"] == "http"
    assert out["body"]["result"]["text"] == "你好"


def test_every_visible_builtin_interface_has_a_test_strategy():
    configs = {}
    for model in server._BUILTIN_MODELS.values():
        current = dict(model)
        if current["id"] == "plat-simult":
            current["url"] = "ws://simult.internal/ws/audio/simult-interpreting"
        configs[current["id"]] = server._builtin_test_cfg(current["id"], current)

    assert set(configs) == set(server._BUILTIN_MODELS)
    for iid in server._BUILTIN_SPECIAL_TESTS:
        assert configs[iid]["template"] == iid


def test_lite_mlt_builtin_test_uses_mlt_route():
    model = next(m for m in server.MODELS if m["id"] == "light-mlt")

    cfg = server._builtin_test_cfg("light-mlt", model)

    assert cfg["base_url"] == "http://lite.test:8001"
    assert cfg["template"] == "asr_lite"
    assert cfg["path"] == "/asr_mlt_nano"


def test_builtin_special_test_config_normalizes_plat_base_and_keeps_key_override():
    model = {
        "id": "plat-simult-ws", "name": "同传",
        "url": "wss://asr.internal/ws/audio/voice-input",
    }
    cfg = server._builtin_test_cfg(
        "plat-simult-ws", model,
        {"api_key": "temporary", "key_env": "SIMULT_KEY"},
    )

    assert cfg["base_url"] == "https://asr.internal"
    assert cfg["api_key"] == "temporary"
    assert cfg["key_env"] == "SIMULT_KEY"


def test_all_visible_plat_builtins_use_platform_and_token_env():
    ids = {
        "std", "adv", "sse", "plat-simult", "plat-simult-ws",
        "plat-minutes", "plat-formula", "plat-diar",
    }
    models = {m["id"]: m for m in server.MODELS if m["id"] in ids}

    assert set(models) == ids
    assert all("platform.test:8000" in m["url"] for m in models.values())
    assert all(m["endpoint"] == "platform" for m in models.values())
    assert all(m["key_env"] == "ASR_PLATFORM_TOKEN" for m in models.values())


def test_stale_builtin_override_migrates_to_default(monkeypatch):
    monkeypatch.setattr(server, "_RETIRED_BUILTIN_URLS", frozenset({"old.example/api/asr/std"}))
    cfg = server._normalize_custom_iface({
        "kind": "builtin",
        "id": "std",
        "url": "old.example/api/asr/std",
        "endpoint": "retired",
        "key_env": "",
    })

    assert cfg["url"] != "old.example/api/asr/std"
    assert cfg["endpoint"] == "platform"
    assert cfg["key_env"] == "ASR_PLATFORM_TOKEN"


def test_unconfigured_builtin_test_reports_actionable_url_error():
    model = {
        "id": "plat-simult", "name": "同传",
        "url": "未配置——接口管理编辑此接口填WS地址",
    }

    try:
        server._builtin_test_cfg("plat-simult", model)
    except RuntimeError as exc:
        assert "接口管理" in str(exc)
        assert "URL" in str(exc)
    else:
        raise AssertionError("未配置地址不应进入网络冒烟")


def test_simult_builtin_smoke_uses_translation_task_and_labels_output(monkeypatch):
    from types import SimpleNamespace

    seen = {}

    class FakeSimult:
        def generate(self, item):
            seen.update(item)
            return SimpleNamespace(ok=True, text="Hello", elapsed_s=0.12, error="")

    monkeypatch.setattr(server, "_builtin_smoke_adapter",
                        lambda cfg, audio_out_dir=None: FakeSimult())
    monkeypatch.setattr(server, "_smoke_sample", lambda: ("/tmp/smoke.wav", "参考文本"))

    result, meta = server._iface_smoke({
        "id": "plat-simult-ws", "template": "plat-simult-ws",
        "base_url": "http://asr.internal", "scenarios": ["同传"],
    })

    assert result.text == "Hello"
    assert seen["task"] == "translate"
    assert seen["target_lang"] == "英文"
    assert meta["output_label"] == "译文"
    assert meta["mode"] == "audio"


def test_formula_builtin_smoke_uses_text_not_audio(monkeypatch):
    from types import SimpleNamespace

    seen = {}

    class FakeFormula:
        def generate(self, item):
            seen.update(item)
            return SimpleNamespace(ok=True, text="x的平方", elapsed_s=0.08, error="")

    monkeypatch.setattr(server, "_builtin_smoke_adapter",
                        lambda cfg, audio_out_dir=None: FakeFormula())

    _, meta = server._iface_smoke({
        "id": "plat-formula", "template": "plat-formula",
        "base_url": "http://asr.internal", "scenarios": ["其他"],
    })

    assert "x^2" in seen["source_text"]
    assert meta["mode"] == "text"
    assert meta["output_label"] == "公式口语化结果"


def test_simult_interface_smoke_forwards_tts_output_directory(monkeypatch, tmp_path):
    from types import SimpleNamespace

    seen = {}

    class FakeSimult:
        def generate(self, item):
            return SimpleNamespace(ok=True, text="Hello", elapsed_s=0.12, error="", extra={})

    def fake_adapter(cfg, audio_out_dir=None):
        seen["audio_out_dir"] = audio_out_dir
        return FakeSimult()

    monkeypatch.setattr(server, "_builtin_smoke_adapter", fake_adapter)
    monkeypatch.setattr(server, "_smoke_sample", lambda: ("/tmp/smoke.wav", "参考文本"))

    server._iface_smoke({
        "id": "plat-simult", "template": "plat-simult",
        "base_url": "http://asr.internal", "scenarios": ["同传"],
    }, audio_out_dir=str(tmp_path))

    assert seen["audio_out_dir"] == str(tmp_path)


def test_smoke_tts_audio_is_stable_and_confined_to_audio_out(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(server, "ROOT", str(tmp_path))
    cfg = {"template": "xf-simult"}
    first = server._smoke_audio_dir("xf-simult", cfg)
    second = server._smoke_audio_dir("xf-simult", cfg)
    assert first == second
    assert first.startswith(str(tmp_path / "audio_out"))
    assert server._smoke_audio_dir("plat-simult-ws", {"template": "plat-simult-ws"}) is None

    audio = tmp_path / "audio_out" / "smoke" / "translated.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF")
    assert server._smoke_audio_payload(
        SimpleNamespace(extra={"tts_audio": str(audio)})) == {"smoke_audio": str(audio)}
    assert server._smoke_audio_payload(
        SimpleNamespace(extra={"tts_audio": str(tmp_path / "secret.wav")})) == {"smoke_audio": None}


def test_audio_preview_allows_dataset_and_saved_tts_but_rejects_other_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", str(tmp_path))
    source = tmp_path / "datasets" / "source.wav"
    translated = tmp_path / "audio_out" / "translated.wav"
    outside = tmp_path / "private.wav"
    source.parent.mkdir(parents=True)
    translated.parent.mkdir(parents=True)
    for path in (source, translated, outside):
        path.write_bytes(b"RIFF")

    assert server.audio_file(str(source)).path == str(source)
    assert server.audio_file(str(translated)).path == str(translated)
    assert server.audio_file(str(outside)).status_code == 403


def test_longform_timeline_uses_exact_dataset_times_and_saved_model_events():
    source, source_timing = server._longform_source_timeline({"segments": [
        {"start_ms": 1000, "end_ms": 3500, "source_text": "hello", "ref_text": "你好"},
    ]}, 10)
    events, model_timing = server._longform_translation_events({"extra": {
        "translation_segments": [{
            "segment_id": 0, "start_s": 1.0, "end_s": 4.25,
            "emitted_at_s": 5.1, "text": "译文", "original_text": "hello",
            "pipeline_mode": "multi_stage", "timing": {"asr_ms": 210},
        }],
    }}, 10)

    assert source_timing == "exact"
    assert source[0]["start_s"] == 1 and source[0]["end_s"] == 3.5
    assert model_timing == "event"
    assert events == [{
        "end_s": 4.25, "text": "译文", "start_s": 1.0, "emitted_at_s": 5.1,
        "segment_id": 0, "pipeline_mode": "multi_stage", "asr_text": "hello",
        "timing": {"asr_ms": 210},
    }]


def test_plat_simult_runtime_rejects_invalid_special_parameter_combinations():
    with pytest.raises(ValueError, match="return_asr_text"):
        server._runtime_config("plat-simult", {
            "pipeline_mode": "single_stage", "return_asr_text": True,
        })
    with pytest.raises(ValueError, match="hard_max"):
        server._runtime_config("plat-simult", {
            "pipeline_mode": "multi_stage", "vad_soft_max_duration_ms": 30000,
            "vad_hard_max_duration_ms": 15000,
        })
    with pytest.raises(ValueError, match="hard_max"):
        server._runtime_config("plat-simult-ws", {
            "vad_soft_max_duration_ms": 30000,
            "vad_hard_max_duration_ms": 15000,
        })
    with pytest.raises(ValueError, match="incremental_interval_ms"):
        server._runtime_config("plat-simult", {
            "pipeline_mode": "multi_stage", "incremental_enabled": True,
            "incremental_interval_ms": 100,
        })


def test_longform_sample_reads_full_manifest_and_infer_instead_of_score_excerpt(tmp_path, monkeypatch):
    manifest = tmp_path / "manifests" / "long.jsonl"
    manifest.parent.mkdir(parents=True)
    source = "First complete source sentence. Second complete source sentence."
    reference = "第一句完整参考。第二句完整参考。"
    row = {"id": "talk-1", "task": "translate", "lang": "en-zh", "longform": True,
           "audio_path": "datasets/talk.wav", "source_text": source, "ref_text": reference,
           "segments": [
               {"source_text": "First complete source sentence.", "ref_text": "第一句完整参考。"},
               {"source_text": "Second complete source sentence.", "ref_text": "第二句完整参考。"},
           ]}
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    result_data = {"summary": {"meta": {"model": "plat-simult-ws"}}, "samples": [
        {"id": "talk-1", "hyp_norm": "被 score 截断", "err_rate": .2},
    ]}
    infer_row = {"id": "talk-1", "hyp": "第一段模型译文。 第二段模型译文。", "audio_s": 40,
                 "extra": {"translation_segments": [
                     {"end_s": 10, "text": "第一段模型译文。"},
                     {"end_s": 30, "text": "第二段模型译文。"},
                 ]}}
    monkeypatch.setattr(server, "ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_review_result", lambda file: ("result.json", result_data))
    monkeypatch.setattr(server, "_review_manifest", lambda data: (str(manifest), "sha"))
    monkeypatch.setattr(server, "_review_infer_records", lambda data: {"talk-1": infer_row})

    payload = server.longform_sample("result.json", "talk-1")

    assert payload["source_timing"] == "estimated"
    assert payload["translation_timing"] == "event"
    assert "First complete source sentence" in payload["segments"][0]["source_text"]
    assert payload["segments"][0]["hyp_text"] == "第一段模型译文。"
    assert payload["segments"][1]["hyp_text"] == "第二段模型译文。"
    assert payload["model_segment_count"] == 2
    assert payload["model_segments"] == [
        {"end_s": 10.0, "text": "第一段模型译文。"},
        {"end_s": 30.0, "text": "第二段模型译文。"},
    ]


def test_asr_lite_interface_config_keeps_request_parameters():
    language_spec = {"supported": True, "values": ["auto", "zh", "sichuan"]}
    req = server.AddIfaceReq(
        name="qwen3-asr-1.7b", base_url="http://h.example.test",
        template="asr_lite", language_spec=language_spec,
        itn=True, return_timestamps=True,
    )

    cfg = server._iface_cfg_from_req(req, "x-qwen3-asr-1-7b")

    assert "language" not in cfg
    assert cfg["language_spec"] == language_spec
    assert cfg["itn"] is True
    assert cfg["return_timestamps"] is True


def test_model_payload_explains_language_values_without_binding_them():
    lite = server._model_payload(next(m for m in server.MODELS if m["id"] == "light"))
    normal = server._model_payload(next(m for m in server.MODELS if m["id"] == "std"))

    assert lite["language_spec"]["values"] == ["auto", "中文", "英文", "日文"]
    assert lite["language_spec"]["supported"] is True
    assert normal["language_spec"]["supported"] is True
    assert "Chinese,English" in normal["language_spec"]["values"]


def test_plat_sse_custom_interface_declares_candidate_language_passthrough():
    model = {
        "id": "x-sse", "name": "新sse", "group": "custom",
        "endpoint": "http://custom.test:9000", "url": "custom.test:9000",
        "tmpl": "plat-sse", "enabled": True,
    }

    payload = server._model_payload(model)

    assert payload["language_spec"]["supported"] is True
    assert "Chinese,English" in payload["language_spec"]["values"]
    assert "英文逗号" in payload["language_spec"]["note"]


def test_legacy_custom_interface_gets_language_help_from_matching_provider():
    model = {
        "id": "x-existing-qwen", "name": "existing qwen", "group": "custom",
        "endpoint": "http://qwen3.test:8005", "url": "qwen3.test:8005",
        "tmpl": "asr_lite", "enabled": True,
    }

    payload = server._model_payload(model)

    assert "sichuan" in payload["language_spec"]["values"]


def test_run_keeps_language_override_per_model_request(monkeypatch):
    _reset_jobs(monkeypatch)

    r1 = server.run(server.RunReq(
        model="light", dataset="aishell", tier="lite", limit=5, language="auto",
    ))
    r2 = server.run(server.RunReq(
        model="std", dataset="aishell", tier="lite", limit=5, language="sichuan",
    ))

    assert server.JOBS[r1["job_id"]]["language"] == "auto"
    assert server.JOBS[r2["job_id"]]["language"] == "sichuan"


def test_run_records_custom_and_dataset_hotwords_separately(monkeypatch):
    _reset_jobs(monkeypatch)

    result = server.run(server.RunReq(
        model="light", dataset="seaco", tier="lite", limit=5,
        hotwords=True, hotwords_text="ExampleTerm 示例词",
    ))
    job = server.JOBS[result["job_id"]]

    assert job["hotwords"] is True
    assert job["dataset_hotwords"] is True
    assert job["hotwords_text"] == "ExampleTerm 示例词"
    assert job["config_hash"]


def test_mimo_domain_sniff_selects_chat_audio_without_network(monkeypatch):
    def unexpected_get(*args, **kwargs):
        raise AssertionError("MiMo 域名应由指纹识别，不应发起探测请求")

    monkeypatch.setattr(server.requests, "get", unexpected_get)
    out = server.sniff_iface(server.SniffReq(base_url="https://api.xiaomimimo.com"))

    assert out["template"] == "openai-chat-audio"
    assert out["path"] == "/v1"
    assert out["models"] == ["mimo-v2.5-asr"]


def test_self_hosted_mimo_model_sniff_selects_chat_audio(monkeypatch):
    class ModelsResp:
        ok = True
        status_code = 200

        def json(self):
            return {"data": [{"id": "mimo-v2.5-asr"}, {"id": "text-model"}]}

    monkeypatch.setattr(server.requests, "get", lambda *args, **kwargs: ModelsResp())
    out = server.sniff_iface(server.SniffReq(base_url="http://127.0.0.1:8000"))

    assert out["template"] == "openai-chat-audio"
    assert out["models"][0] == "mimo-v2.5-asr"


def _review_fixture(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    infer = tmp_path / "infer"
    drafts = tmp_path / "reviews" / "drafts"
    manifests.mkdir()
    results.mkdir()
    infer.mkdir()
    manifest = manifests / "demo.jsonl"
    manifest.write_text(json.dumps({
        "id": "s1", "audio_path": "datasets/demo.wav", "ref_text": "他今天开会", "lang": "zh",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    sha = server.manifest_sha_v2(str(manifest))
    infer_path = infer / "demo_model.jsonl"
    infer_path.write_text(json.dumps({
        "id": "s1", "hyp": "它今天开会?", "ok": True, "extra": {},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {
        "summary": {"task": "asr", "metric": "CER", "err_rate": 0.25,
                    "manifest": "manifests/demo.jsonl", "infer_file": "infer/demo_model.jsonl",
                    "meta": {"manifest_sha_v2": sha}},
        "samples": [{"id": "s1", "ref_norm": "他今天开会", "hyp_norm": "它今天开会", "err_rate": 0.2}],
    }
    result_path = results / "demo_model.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", str(tmp_path))
    monkeypatch.setattr(server, "REVIEW_DRAFT_DIR", str(drafts))
    return manifest, result_path, drafts, sha


def test_review_json_draft_is_isolated_from_original_results(tmp_path, monkeypatch):
    manifest, result_path, drafts, sha = _review_fixture(tmp_path, monkeypatch)
    before_manifest = manifest.read_bytes()
    before_result = result_path.read_bytes()
    before = server.review_context(result_path.name)
    score_issue = before["samples"][0]["score_diff"][0]
    raw_format_issue = next(x for x in before["samples"][0]["raw_diff"] if x["hyp_text"] == "?")

    out = server.save_review(server.ReviewSaveReq(
        file=result_path.name, sample_id="s1", data_decision="ref_correct",
        issues=[
            server.ReviewIssueReq(issue_id=score_issue["issue_id"], decision="recognition_error"),
            server.ReviewIssueReq(issue_id=raw_format_issue["issue_id"], decision="format_only"),
        ], comment="一个识别错误 + 一个格式问题",
    ))

    assert out["ok"] is True and out["status"] == "reviewed"
    assert manifest.read_bytes() == before_manifest
    assert result_path.read_bytes() == before_result
    draft = json.loads((drafts / f"{sha}.json").read_text(encoding="utf-8"))
    row = draft["samples"]["s1"]
    assert row["data_review"]["decision"] == "ref_correct"
    saved = row["output_reviews"][result_path.name]
    assert saved["schema_version"] == 2 and saved["scope_decision"] == "unreviewed"
    assert {x["decision"] for x in saved["issues"]} == {"recognition_error", "format_only"}

    ctx = server.review_context(result_path.name)
    assert ctx["original_metric"] == "CER" and ctx["original_err_rate"] == 0.25
    assert ctx["counts"] == {"total": 1, "reviewed": 1, "remaining": 0}
    assert ctx["samples"][0]["review_complete"] is True
    assert ctx["samples"][0]["hyp_raw"] == "它今天开会?"


def test_review_ref_correction_requires_text(tmp_path, monkeypatch):
    _, result_path, _, _ = _review_fixture(tmp_path, monkeypatch)
    out = server.save_review(server.ReviewSaveReq(
        file=result_path.name, sample_id="s1", data_decision="ref_incorrect",
        scope_decision="all_recognition_error",
    ))
    assert out.status_code == 400
    assert "修订参考" in json.loads(out.body)["error"]


def test_review_context_reports_non_destructive_exclusion(tmp_path, monkeypatch):
    _, result_path, _, _ = _review_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "exclusions_for_rows",
                        lambda _rows, _root: {"s1": {"reason": "audio_reference_mismatch"}})

    ctx = server.review_context(result_path.name)

    assert ctx["original_err_rate"] == 0.25
    assert ctx["reviewed"]["n_excluded"] == 1
    assert ctx["samples"][0]["excluded_from_reviewed"] is True
    assert ctx["samples"][0]["exclusion_reason"] == "audio_reference_mismatch"


def test_review_rejects_changed_manifest(tmp_path, monkeypatch):
    manifest, result_path, drafts, _ = _review_fixture(tmp_path, monkeypatch)
    manifest.write_text(json.dumps({"id": "s1", "ref_text": "标注已变"}, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    out = server.review_context(result_path.name)
    assert out.status_code == 409
    assert "manifest_sha_v2" in json.loads(out.body)["error"]
    assert not drafts.exists()


def test_diff_blocks_merge_nearby_character_edits_but_keep_large_span_one_block():
    near = server._diff_blocks("甲乙丙丁", "甲X丙Y", "score")
    assert len(near) == 1
    assert near[0]["ref_text"] == "乙丙丁" and near[0]["hyp_text"] == "X丙Y"

    large = server._diff_blocks("这是一段完全不同的参考文本", "模型输出了另一大段无关内容", "score")
    assert any(block["large"] for block in large)
    assert len(large) < 8  # 不退化成每个字一条审核任务


def test_scope_decision_completes_large_error_without_reviewing_each_block(tmp_path, monkeypatch):
    _, result_path, _, _ = _review_fixture(tmp_path, monkeypatch)
    out = server.save_review(server.ReviewSaveReq(
        file=result_path.name, sample_id="s1", data_decision="ref_correct",
        scope_decision="large_omission",
    ))
    assert out["ok"] is True and out["status"] == "reviewed"
    saved = out["review"]["output_reviews"][result_path.name]
    assert saved["scope_decision"] == "large_omission" and saved["issues"] == []


def test_review_result_picker_groups_current_runs_and_hides_noise(monkeypatch):
    rows = [
        {"_file": "aishell_pro_old.json", "_mtime": 1, "task": "asr", "metric": "CER",
         "model": "adv", "dataset": "aishell", "manifest": "manifests/a.jsonl",
         "n_total": 300, "err_rate": .1, "meta": {"manifest_sha_v2": "a" * 12}},
        {"_file": "aishell_pro_new.json", "_mtime": 2, "task": "asr", "metric": "CER",
         "model": "adv", "dataset": "aishell", "manifest": "manifests/a.jsonl",
         "n_total": 300, "err_rate": .09, "meta": {"manifest_sha_v2": "a" * 12}},
        {"_file": "_smoke.json", "_mtime": 3, "task": "asr", "metric": "CER",
         "model": "adv", "dataset": "aishell", "manifest": "manifests/a.jsonl",
         "n_total": 1, "err_rate": 0, "meta": {"manifest_sha_v2": "a" * 12}},
        {"_file": "legacy.json", "_mtime": 3, "task": "asr", "metric": "CER",
         "model": "legacy-pro", "dataset": "aishell", "manifest": "manifests/a.jsonl",
         "n_total": 300, "err_rate": .1, "meta": {"manifest_sha_v2": "a" * 12}},
    ]
    monkeypatch.setattr(server, "read_results", lambda: rows)
    monkeypatch.setattr(server, "MODELS", [{"id": "adv"}])
    monkeypatch.setattr(server, "RETIRED_MODELS", ["legacy-pro"])
    monkeypatch.setattr(server, "DATASETS", [{"id": "aishell", "name": "AISHELL-1"}])
    out = server.review_results()
    assert len(out["batches"]) == 1
    assert out["batches"][0]["dataset_name"] == "AISHELL-1"
    assert [x["file"] for x in out["batches"][0]["models"]] == ["aishell_pro_new.json"]


def test_bruteforce_guard_locks_after_max_failures(monkeypatch):
    # 防撞库：真实凭据尝试连续失败达上限 → 锁定该 (ip,user)；锁定期间直接拒绝，过期后解锁
    ip, user = "10.9.9.9", "admin"
    monkeypatch.setattr(server, "AUTH_MAX_FAILURES", 3)
    monkeypatch.setattr(server, "AUTH_LOCKOUT_S", 900)

    for _ in range(server.AUTH_MAX_FAILURES - 1):
        assert server._auth_guard(ip, user) == 0.0
    remaining = server._auth_guard(ip, user)
    assert remaining == server.AUTH_LOCKOUT_S
    # 锁定期间：key 在锁定表里，中间件会直接 429
    assert server._AUTH_LOCKED_UNTIL.get((ip, user), 0) > 0
    # 同 IP 不同用户互不牵连
    assert server._AUTH_LOCKED_UNTIL.get((ip, "other"), 0) == 0

    # 模拟锁定过期：把到期时间拨回过去 → 解锁
    server._AUTH_LOCKED_UNTIL[(ip, user)] = 0.0
    assert server._auth_guard(ip, user) == 0.0


def test_bruteforce_guard_ignores_headerless_probes(monkeypatch):
    # 无凭据的 401（探活/扫描）不参与撞库记账：attempted_user 为空时不调用记账
    import asyncio

    calls = {"guard": 0}
    real_guard = server._auth_guard

    def spy(ip, user):
        calls["guard"] += 1
        return real_guard(ip, user)

    monkeypatch.setattr(server, "_auth_guard", spy)

    from starlette.responses import PlainTextResponse

    async def call_next(request):
        return PlainTextResponse("ok")

    request = type("R", (), {"client": type("C", (), {"host": "1.2.3.4"})(),
                             "url": type("U", (), {"path": "/"})(),
                             "headers": {}})()
    asyncio.run(server.basic_auth(request, call_next))
    assert calls["guard"] == 0   # 无 Authorization 头 → 不记账
