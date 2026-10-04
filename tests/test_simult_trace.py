import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from simult_trace import (SimultTraceRecorder, export_infer, legacy_trace,
                          omnisteval_record, replay_trace)


def test_trace_preserves_partial_revision_and_commit_clocks():
    trace = SimultTraceRecorder(target_lang="en", granularity="token")
    trace.replace_partial("Hellx", 1.0, 1.2)
    trace.replace_partial("Hello", 1.4, 1.7)
    trace.commit("Hello", 1.8, 2.1)
    trace.append_partial(" world", 2.2, 2.6)
    trace.commit("world", 2.5, 3.0)
    saved = trace.finish("Hello world", 3.0, 3.5)

    assert replay_trace(saved)[1]["display_text"] == "Hello"
    assert saved["summary"] == {
        "ttfb_visible_s": 1.2,
        "ttfb_stable_s": 2.1,
        "update_events": 3,
        "revision_events": 1,
        "stability_observable": True,
        "revision_rate": 0.3333,
        "revised_chars": 1,
        "churn_rate": 0.1,
    }

    stable = omnisteval_record("/tmp/talk.wav", saved, 3.0, "stable")
    visible = omnisteval_record("/tmp/talk.wav", saved, 3.0, "visible")
    assert stable["prediction"] == "Hello world"
    assert stable["delays"] == [1800.0, 2500.0]
    assert stable["elapsed"] == [2100.0, 3000.0]
    assert visible["delays"] == [1400.0, 2200.0]
    assert visible["elapsed"] == [1700.0, 2600.0]


def test_chinese_export_is_character_level_and_legacy_omits_fake_ca():
    trace = legacy_trace("你好 世界", [
        {"end_s": 1.25, "text": "你好"},
        {"emitted_at_s": 2.5, "text": "世界"},
    ], "zh")
    record = omnisteval_record("talk.wav", trace, 3.0)

    assert record["prediction"] == "你好世界"
    assert record["delays"] == [1250.0, 1250.0, 2500.0, 2500.0]
    assert len(record["prediction"]) == len(record["delays"])
    assert "elapsed" not in record
    assert record["emission_granularity"] == "segment-final"
    assert trace["summary"]["stability_observable"] is False
    assert trace["summary"]["churn_rate"] is None


def test_word_level_export_spaces_punctuation_to_keep_delay_count_aligned():
    trace = SimultTraceRecorder(target_lang="en", granularity="token")
    trace.append_partial("Hello,", 1.0, 1.1)
    trace.append_partial(" world!", 2.0, 2.2)
    trace.commit("Hello, world!", 2.5, 2.8)
    saved = trace.finish("Hello, world!", 3.0, 3.3)

    record = omnisteval_record("talk.wav", saved, 3.0, "visible")

    assert record["prediction"] == "Hello , world !"
    assert len(record["delays"]) == len(record["prediction"].split()) == 4
    assert record["evaluation_unit"] == "word"


def test_computation_aware_clock_cannot_precede_consumed_source_audio():
    trace = SimultTraceRecorder(target_lang="en")
    trace.commit("hello", source_s=1.0, wall_s=0.94)
    saved = trace.finish("hello", source_s=1.0, wall_s=0.95)

    record = omnisteval_record("talk.wav", saved, 1.0)

    assert record["delays"] == [1000.0]
    assert record["elapsed"] == [1000.0]


def test_export_infer_writes_omnisteval_jsonl(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    infer = tmp_path / "infer.jsonl"
    out = tmp_path / "omni.jsonl"
    manifest.write_text(json.dumps({
        "id": "one", "audio_path": "/data/one.wav", "target_lang": "zh",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    infer.write_text(json.dumps({
        "id": "one", "ok": True, "hyp": "你好", "audio_s": 2,
        "extra": {"src_dur_s": 2, "translation_segments": [
            {"end_s": 1.5, "text": "你好"},
        ]},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = export_infer(str(infer), str(manifest), str(out))
    row = json.loads(out.read_text(encoding="utf-8"))
    assert summary["written"] == 1 and summary["skipped"] == 0
    assert row["source"] == "one.wav"
    assert row["prediction"] == "你好"
    assert row["delays"] == [1500.0, 1500.0]
