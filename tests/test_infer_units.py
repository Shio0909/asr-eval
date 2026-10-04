"""infer.py 纯函数黄金用例（热词串构造、场景 a/b 文件隔离）。

注意：导入 infer 会连带 soundfile/requests，测试命令需带这俩依赖（见 README）。
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import pytest

import infer
from infer import infer_file_path, item_hotwords
from util import manifest_sha_v2


def test_hotwords_from_plain_strings():
    assert item_hotwords({"keywords": ["万龙滑雪场", "三藩市"]}) == "万龙滑雪场 三藩市"


def test_infer_rejects_duplicate_manifest_ids_before_model_setup(tmp_path):
    manifest = tmp_path / "duplicate.jsonl"
    row = {"id": "dup", "audio_path": "x.wav", "ref_text": "hello", "lang": "en"}
    manifest.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复样本 ID: dup"):
        infer.run(str(manifest), "unused")


def test_hotwords_from_alias_dicts_dedup_and_skip_empty():
    r = {"keywords": [{"aliases": ["阿司匹林", "阿斯匹林"]}, {"surface": "布洛芬"},
                      "阿司匹林", {"surface": ""}]}
    # 别名全部进热词表，去重保序，空串丢弃
    assert item_hotwords(r) == "阿司匹林 阿斯匹林 布洛芬"


def test_hotwords_empty_keywords():
    assert item_hotwords({"keywords": []}) == ""
    assert item_hotwords({}) == ""


def test_infer_path_isolates_scenario_b():
    # 场景 b(带热词)与场景 a 的 hyp 必须分文件，否则续跑互相污染
    a = infer_file_path("ext-adv", "manifests/seaco_lite.jsonl", hotwords=False)
    b = infer_file_path("ext-adv", "manifests/seaco_lite.jsonl", hotwords=True)
    assert a == "infer/ext-adv__seaco_lite.jsonl"
    assert b == "infer/ext-adv__seaco_lite__hw.jsonl"


def test_infer_path_isolates_language_experiment_variants():
    auto = infer_file_path("qwen", "manifests/wsc_lite.jsonl", language="auto")
    zh = infer_file_path("qwen", "manifests/wsc_lite.jsonl", language="zh")
    sichuan = infer_file_path("qwen", "manifests/wsc_lite.jsonl", language="sichuan")

    assert len({auto, zh, sichuan}) == 3
    assert "__lang-auto-" in auto
    assert "__lang-zh-" in zh
    assert "__lang-sichuan-" in sichuan


def test_infer_path_isolates_runtime_config_variants():
    first = infer_file_path(
        "x-pro", "manifests/aishell_lite.jsonl", language="auto", config_hash="a1b2c3d4",
    )
    second = infer_file_path(
        "x-pro", "manifests/aishell_lite.jsonl", language="auto", config_hash="deadbeef",
    )

    assert first != second
    assert "__cfg-a1b2c3d4" in first
    assert "__cfg-deadbeef" in second


def test_safe_request_params_keep_token_counts_but_drop_credentials():
    assert infer._safe_request_params({
        "incremental_holdback_tokens": 2,
        "max_tokens": 512,
        "access_token": "secret",
        "api_key": "secret",
    }) == {
        "incremental_holdback_tokens": 2,
        "max_tokens": 512,
    }


def _legacy_meta(manifest, *, model="qwen", endpoint="http://asr.test/v1",
                 language="auto", hotwords=False, domain=None):
    return {
        "id": "__meta__",
        "schema_version": 2,
        "run_id": "legacy-run",
        "code_revision": "legacy-code",
        "started": "2026-07-01T12:00:00+08:00",
        "note": "legacy note",
        "run_spec": {
            "model": model,
            "endpoint": endpoint,
            "manifest_sha_v2": manifest_sha_v2(str(manifest)),
            "language": language,
            "hotwords": hotwords,
            "domain": domain,
        },
        "model": model,
        "endpoint": endpoint,
        "manifest_sha_v2": manifest_sha_v2(str(manifest)),
        "language": language,
        "hotwords": hotwords,
        "domain": domain,
    }


def test_legacy_auto_infer_is_migrated_without_model_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a1", "audio_path": "missing.wav", "ref_text": "你好", "lang": "zh",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    class Adapter:
        url = "http://asr.test/v1"

        def transcribe(self, *_args, **_kwargs):
            raise AssertionError("安全迁移后不应重新调用模型")

    monkeypatch.setitem(infer.ADAPTERS, "qwen", Adapter)
    legacy = tmp_path / infer_file_path("qwen", str(manifest), language="")
    legacy.write_text(
        json.dumps(_legacy_meta(manifest), ensure_ascii=False) + "\n"
        + json.dumps({"id": "a1", "hyp": "你好", "elapsed_s": 0.1,
                      "audio_s": 1.0, "ok": True, "workers": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    out, _ = infer.run(str(manifest), "qwen", language="auto")

    target = tmp_path / out
    assert target.exists()
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["migration"]["kind"] == "legacy_untagged_auto"
    assert rows[-1]["migration"]["source"] == legacy.name
    assert rows[-1]["run_id"] == "legacy-run"
    assert rows[-1]["code_revision"] == "legacy-code"
    assert rows[-1]["started"] == "2026-07-01T12:00:00+08:00"
    assert rows[-1]["note"] == "legacy note"
    assert rows[-1]["migration"]["migrated_at"]
    assert legacy.exists()


@pytest.mark.parametrize("mutate", [
    lambda meta: meta.update(language="zh"),
    lambda meta: meta.update(endpoint="http://other.test/v1"),
    lambda meta: meta.update(manifest_sha_v2="wrong"),
    lambda meta: meta.pop("language"),
])
def test_legacy_auto_migration_rejects_unproven_metadata(tmp_path, mutate):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"id": "a1", "ref_text": "你好"}) + "\n", encoding="utf-8")
    legacy = tmp_path / "legacy.jsonl"
    target = tmp_path / "target.jsonl"
    meta = _legacy_meta(manifest)
    mutate(meta)
    if "language" not in meta:
        meta["run_spec"].pop("language", None)
    legacy.write_text(json.dumps(meta) + "\n", encoding="utf-8")

    migrated = infer.migrate_legacy_auto_infer(
        str(legacy), str(target), str(manifest), "qwen", "http://asr.test/v1",
    )

    assert migrated is False
    assert not target.exists()


def test_legacy_auto_migration_rejects_mixed_sessions(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"id": "a1", "ref_text": "你好"}) + "\n", encoding="utf-8")
    legacy = tmp_path / "legacy.jsonl"
    target = tmp_path / "target.jsonl"
    auto = _legacy_meta(manifest)
    zh = _legacy_meta(manifest, language="zh")
    legacy.write_text(json.dumps(auto) + "\n" + json.dumps(zh) + "\n", encoding="utf-8")

    migrated = infer.migrate_legacy_auto_infer(
        str(legacy), str(target), str(manifest), "qwen", "http://asr.test/v1",
    )

    assert migrated is False
    assert not target.exists()


def test_adapter_constructor_internal_typeerror_is_not_swallowed(tmp_path, monkeypatch):
    mani = tmp_path / "m.jsonl"
    mani.write_text('', encoding="utf-8")

    class BuggyAdapter:
        def __init__(self, audio_out_dir=None):
            if audio_out_dir is not None:
                raise TypeError("internal constructor bug")

    monkeypatch.setitem(infer.ADAPTERS, "buggy", BuggyAdapter)
    with pytest.raises(TypeError, match="internal constructor bug"):
        infer.run(str(mani), "buggy", save_audio=True, out=str(tmp_path / "i.jsonl"), resume=False)


def test_adapter_constructor_drops_unsupported_optional_kwargs(tmp_path, monkeypatch):
    mani = tmp_path / "m.jsonl"
    mani.write_text('', encoding="utf-8")
    seen = {}

    class PlainAdapter:
        def __init__(self):
            seen["constructed"] = True

    monkeypatch.setitem(infer.ADAPTERS, "plain", PlainAdapter)
    infer.run(str(mani), "plain", save_audio=True, out=str(tmp_path / "i.jsonl"), resume=False)
    assert seen["constructed"] is True


def test_saved_audio_directory_isolated_by_language_and_config(tmp_path, monkeypatch):
    mani = tmp_path / "m.jsonl"
    mani.write_text('', encoding="utf-8")
    seen = {}

    class AudioAdapter:
        def __init__(self, audio_out_dir=None):
            seen["audio_out_dir"] = audio_out_dir

    monkeypatch.setitem(infer.ADAPTERS, "audio", AudioAdapter)
    infer.run(
        str(mani), "audio", language="English", save_audio=True,
        out=str(tmp_path / "i.jsonl"), resume=False, config_hash="abc123",
    )

    assert os.path.basename(seen["audio_out_dir"]) == "audio__m__lang-English-649df0__cfg-abc123"


def test_parallel_run_resumes_only_missing_successful_ids(tmp_path, monkeypatch):
    mani = tmp_path / "m.jsonl"
    mani.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
        {"id": "t1", "task": "translate", "source_text": "一", "ref_text": "one"},
        {"id": "t2", "task": "translate", "source_text": "二", "ref_text": "two"},
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "i.jsonl"
    out.write_text(json.dumps({
        "id": "t1", "hyp": "one", "elapsed_s": 0.1, "audio_s": 0,
        "ok": True, "workers": 1,
    }) + "\n", encoding="utf-8")
    called = []

    class Result:
        ok = True
        text = "two"
        elapsed_s = 0.01
        extra = {}
        error = ""

    class Adapter:
        url = "http://example.test/v1"

        def generate(self, item):
            called.append(item["id"])
            return Result()

    monkeypatch.setitem(infer.ADAPTERS, "resume-test", Adapter)
    infer.run(str(mani), "resume-test", workers=4, out=str(out), resume=True)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert called == ["t2"]
    assert {row["id"] for row in rows if row.get("ok")} == {"t1", "t2"}
    assert next(row for row in rows if row.get("id") == "t2")["workers"] == 4


def test_builtin_adapter_receives_runtime_request_contract():
    adapter, _ = infer._make_adapter(
        "std", {"request_params": {"enable_word_timestamps": True}},
    )

    assert adapter.request_params == {"enable_word_timestamps": True}
    assert {x["name"] for x in adapter.request_schema} >= {"language", "enable_word_timestamps"}


def test_infer_meta_v2_is_additive_and_does_not_change_jsonl_records(tmp_path, monkeypatch):
    mani = tmp_path / "m.jsonl"
    mani.write_text(json.dumps({"id": "t1", "task": "translate", "source_text": "你好",
                                "ref_text": "hello", "target_lang": "英文"}, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    class Result:
        ok = True
        text = "hello"
        elapsed_s = 0.01
        extra = {}
        error = ""

    seen = {}

    class TextAdapter:
        ws_url = "ws://example.test/v1"

        def generate(self, item):
            seen.update(item)
            return Result()

    monkeypatch.setitem(infer.ADAPTERS, "fake-text", TextAdapter)
    monkeypatch.setattr(infer, "code_revision", lambda: "abc123")
    out = tmp_path / "i.jsonl"
    infer.run(str(mani), "fake-text", out=str(out), resume=False, target_lang="中文")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    meta, record = rows

    assert meta["id"] == "__meta__" and meta["schema_version"] == 2
    assert meta["run_id"] and meta["code_revision"] == "abc123"
    assert meta["manifest_sha_v2"] == manifest_sha_v2(str(mani))
    assert meta["run_spec"]["model"] == "fake-text"
    assert meta["endpoint"] == "ws://example.test/v1"
    assert meta["run_spec"]["endpoint"] == "ws://example.test/v1"
    assert meta["target_lang"] == "中文" and meta["run_spec"]["target_lang"] == "中文"
    assert seen["target_lang"] == "中文"
    assert meta["metric_signature"] is None and meta["output_variant_used"] is None
    assert record["id"] == "t1" and record["hyp"] == "hello" and record["ok"] is True
    assert record["attempts"] == 1


def test_longform_generate_receives_sample_position_without_persisting_it(tmp_path, monkeypatch, capsys):
    mani = tmp_path / "long.jsonl"
    mani.write_text("\n".join(json.dumps({
        "id": f"talk-{i}", "task": "translate", "audio_path": f"{i}.wav",
        "source_text": "hello", "ref_text": "你好", "target_lang": "中文",
        "longform": True,
    }, ensure_ascii=False) for i in (1, 2)) + "\n", encoding="utf-8")

    class Result:
        ok = True
        text = "你好"
        elapsed_s = 0.01
        extra = {}
        error = ""

    seen = []

    class Adapter:
        url = "ws://example.test"

        def generate(self, item):
            seen.append(dict(item))
            return Result()

    monkeypatch.setitem(infer.ADAPTERS, "fake-long", Adapter)
    monkeypatch.setattr(infer, "audio_duration_s", lambda _path: 10.0)
    out = tmp_path / "long-infer.jsonl"
    infer.run(str(mani), "fake-long", out=str(out), resume=False)

    assert [item["_stream_progress"] for item in seen] == [
        {"sample_index": 1, "sample_total": 2},
        {"sample_index": 2, "sample_total": 2},
    ]
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()][1:]
    assert all("_stream_progress" not in record for record in records)
    progress = capsys.readouterr().out
    assert "1/2…" in progress
    assert "2/2…" in progress


def test_short_sample_progress_stays_batched(tmp_path, monkeypatch, capsys):
    mani = tmp_path / "short.jsonl"
    mani.write_text("\n".join(json.dumps({
        "id": f"short-{i}", "task": "translate", "source_text": "hello",
        "ref_text": "你好", "target_lang": "中文",
    }, ensure_ascii=False) for i in (1, 2)) + "\n", encoding="utf-8")

    class Result:
        ok = True
        text = "你好"
        elapsed_s = 0.01
        extra = {}
        error = ""

    class Adapter:
        url = "http://example.test"

        def generate(self, _item):
            return Result()

    monkeypatch.setitem(infer.ADAPTERS, "fake-short", Adapter)
    infer.run(str(mani), "fake-short", out=str(tmp_path / "short-infer.jsonl"), resume=False)

    progress = capsys.readouterr().out
    assert "1/2…" not in progress
    assert "2/2…" in progress


def test_simult_progress_reports_each_short_sample(tmp_path, monkeypatch, capsys):
    mani = tmp_path / "simult-short.jsonl"
    mani.write_text("\n".join(json.dumps({
        "id": f"simult-{i}", "task": "translate", "audio_path": f"{i}.wav",
        "source_text": "hello", "ref_text": "你好", "target_lang": "中文",
    }, ensure_ascii=False) for i in (1, 2)) + "\n", encoding="utf-8")

    class Result:
        ok = True
        text = "你好"
        elapsed_s = 0.01
        extra = {}
        error = ""

    class Adapter:
        url = "ws://example.test"

        def generate(self, _item):
            return Result()

    monkeypatch.setitem(infer.ADAPTERS, "plat-simult-ws", Adapter)
    monkeypatch.setattr(infer, "audio_duration_s", lambda _path: 10.0)
    infer.run(str(mani), "plat-simult-ws", out=str(tmp_path / "simult-infer.jsonl"), resume=False)

    progress = capsys.readouterr().out
    assert "1/2…" in progress
    assert "2/2…" in progress


def test_custom_hotwords_merge_with_sample_keywords_and_target_lang(tmp_path, monkeypatch):
    mani = tmp_path / "m.jsonl"
    mani.write_text(json.dumps({
        "id": "a1", "audio_path": "a.wav", "ref_text": "测试", "lang": "zh",
        "keywords": ["数据集词"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    class Result:
        ok = True
        text = "测试"
        elapsed_s = 0.01
        extra = {}
        error = ""

    seen = {}

    class Adapter:
        url = "http://example.test/asr"
        supports_hotwords = True

        def transcribe(self, audio_path, language="auto", **kwargs):
            seen.update(kwargs)
            return Result()

    monkeypatch.setitem(infer.ADAPTERS, "fake-hotwords", Adapter)
    monkeypatch.setattr(infer, "audio_duration_s", lambda _path: 1.0)
    out = tmp_path / "i.jsonl"

    infer.run(
        str(mani), "fake-hotwords", out=str(out), resume=False,
        hotwords=True, hotwords_text="固定词", target_lang="English",
    )
    meta = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    assert seen["hotwords"] == "数据集词 固定词"
    assert seen["target_lang"] == "English"
    assert meta["dataset_hotwords"] is True
    assert meta["hotwords_text"] == "固定词"
