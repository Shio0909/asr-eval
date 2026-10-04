"""score.py 端到端黄金用例 — 构造微型 manifest+infer 文件，断言 summary 口径。

覆盖：旧/新 infer 格式兼容、混采延迟过滤(只统计串行)、manifest 指纹告警、
失败样本口径(low_coverage/fail_rate)、KRR 按语言归一、按 id 去重(优先成功)。
跑法同 test_metrics.py（见该文件头部）。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import score
import omnisteval_eval
from util import manifest_sha, manifest_sha_v2


def test_arabic_uses_word_error_rate():
    _normalizer, is_word = score._asr_norm_metric("ar")
    assert is_word is True


def test_japanese_matches_kotoba_normalization():
    normalizer, is_word = score._asr_norm_metric("ja")
    assert is_word is False
    assert normalizer("今日は、晴れです。") == "今日は晴れです"
    assert " " not in normalizer("ＡＳＲ  テスト")


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


def _zh_rows(n=4):
    return [{"id": f"u{i}", "audio_path": "x", "ref_text": "今天天气好",
             "lang": "zh", "keywords": []} for i in range(n)]


def _ok(i, hyp="今天天气好", elapsed=0.5, workers=None, audio=2.0):
    d = {"id": f"u{i}", "hyp": hyp, "elapsed_s": elapsed, "audio_s": audio,
         "ok": True, "extra": {}}
    if workers is not None:
        d["workers"] = workers
    return d


def _meta(mani, workers=1, sha=None):
    return {"id": "__meta__", "model": "ext-pro", "endpoint": "http://x/asr_lite",
            "language": "zh", "workers": workers, "manifest": mani,
            "manifest_sha": sha or manifest_sha(mani),
            "started": "2026-06-10T10:00:00+08:00", "n_total": 4, "n_todo": 4}


# ---------- 旧格式（无 __meta__、无 workers 字段）兼容 ----------

def test_legacy_format_scores_and_meta_null(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows())
    inf = _write_jsonl(tmp_path / "i.jsonl", [_ok(i) for i in range(4)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["CER"] == 0.0 and s["n_ok"] == 4
    assert s["meta"]["model"] is None  # 旧格式无 provenance
    assert s["schema_version"] == 2 and s["meta"]["schema_version"] == 2
    assert s["meta"]["run_id"] is None and s["meta"]["run_spec"] is None
    assert s["meta"]["manifest_sha_v2"] == manifest_sha_v2(mani)
    assert s["output_variant_used"] == "raw"
    assert "output=raw" in s["metric_signature"]
    assert s["latency_valid"] and s["latency_p50_s"] == 0.5


def test_score_rejects_duplicate_manifest_ids_before_loading_infer(tmp_path):
    rows = _zh_rows(1) * 2
    mani = _write_jsonl(tmp_path / "duplicate.jsonl", rows)

    with pytest.raises(ValueError, match="重复样本 ID: u0"):
        score.run(mani, str(tmp_path / "missing-infer.jsonl"))


def test_load_infer_ignores_only_malformed_final_nonempty_line(tmp_path, monkeypatch):
    inf = tmp_path / "i.jsonl"
    inf.write_text(json.dumps(_ok(0), ensure_ascii=False) + "\n" + '{"id":"u1"', encoding="utf-8")
    warnings = []
    monkeypatch.setattr(score.log, "warning", lambda *args: warnings.append(args))
    best, metas = score.load_infer(str(inf))
    assert set(best) == {"u0"} and metas == []
    assert warnings and "最后一个非空行" in warnings[0][0]


def test_load_infer_rejects_malformed_middle_line(tmp_path):
    inf = tmp_path / "i.jsonl"
    inf.write_text(json.dumps(_ok(0)) + "\n{" + "\n" + json.dumps(_ok(1)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="仅允许忽略文件尾部残行"):
        score.load_infer(str(inf))


def test_legacy_concurrent_flag_invalidates_latency(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows())
    inf = _write_jsonl(tmp_path / "i.jsonl", [_ok(i) for i in range(4)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"), concurrent=True)
    assert not s["latency_valid"] and s["latency_p50_s"] is None
    assert s["concurrent_latency_p50_s"] == 0.5
    assert s["concurrent_latency_p95_s"] == 0.5
    assert s["rtf_mean"] is None


# ---------- 新格式：记录级 workers，混采只统计串行 ----------

def test_mixed_workers_latency_serial_only(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows())
    inf = _write_jsonl(tmp_path / "i.jsonl", [
        _meta(mani),
        _ok(0, elapsed=0.5, workers=1), _ok(1, elapsed=0.7, workers=1),
        _ok(2, elapsed=0.1, workers=4), _ok(3, elapsed=0.1, workers=4)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["latency_p50_s"] == 0.6  # 只有 0.5/0.7 两条串行参与
    assert s["concurrent_latency_p50_s"] == 0.1
    assert s["concurrent_latency_p95_s"] == 0.1
    assert s["concurrent_workers"] == [4]
    assert abs(s["rtf_mean"] - 0.3) < 1e-9  # (0.25+0.35)/2，并发两条不进 RTF
    assert s["meta"]["model"] == "ext-pro" and s["meta"]["endpoint"] == "http://x/asr_lite"


def test_simult_stability_metrics_are_aggregated_and_stable_ttfb_is_serial_only(tmp_path):
    rows = [{"id": f"u{i}", "task": "translate", "source_text": "hello",
             "lang": "en-zh", "source_lang": "en",
             "ref_text": "你好", "target_lang": "zh"} for i in range(3)]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    recs = [_meta(mani)]
    for i, workers in enumerate((1, 1, 4)):
        recs.append({"id": f"u{i}", "hyp": "你好", "elapsed_s": 2.0,
                     "audio_s": 3.0, "ok": True, "workers": workers,
                     "extra": {"ttfb_s": 0.5 + i, "stable_ttfb_s": 1.0 + i,
                               "finish_tail_s": 0.3 + i,
                               "incremental_asr_ttfb_s": 0.2 + i,
                               "incremental_translation_ttfb_s": 0.4 + i,
                               "incremental_update_count": i + 1,
                               "incremental_asr_churn_rate": 0.05 * (i + 1),
                               "incremental_translation_churn_rate": 0.08 * (i + 1),
                               "asr_ms_mean": 100 + i,
                               "text_translate_ms_mean": 200 + i,
                               "e2e_ms_mean": 300 + i,
                               "asr_text": "hello",
                               "revision_rate": 0.1 * (i + 1),
                               "churn_rate": 0.2 * (i + 1),
                               "ws_control": {
                                   "endpoint": "ws://example.test/ws",
                                   "task_start": {"event": "task_start"},
                                   "task_started": {"event": "task_started", "route": f"r{i}"},
                                   "task_start_sha256": "start",
                                   "task_started_sha256": f"ack-{i}",
                                   "task_started_stable_sha256": f"ack-{i}",
                               }}})
    inf = _write_jsonl(tmp_path / "i.jsonl", recs)

    result_path = tmp_path / "r.json"
    summary = score.run(mani, inf, out=str(result_path))

    assert summary["stable_ttfb_p50_s"] == 1.5
    assert summary["finish_tail_p50_s"] == 0.8
    assert summary["finish_tail_p95_s"] == score._pct([0.3, 1.3], 95)
    assert summary["revision_rate"] == 0.2
    assert summary["churn_rate"] == 0.4
    assert summary["incremental_asr_ttfb_p50_s"] == 0.7
    assert summary["incremental_translation_ttfb_p50_s"] == 0.9
    assert summary["incremental_update_count_mean"] == 2
    assert summary["incremental_asr_churn_rate"] == 0.1
    assert summary["incremental_translation_churn_rate"] == 0.16
    assert summary["asr_stage_p50_ms"] == 100.5
    assert summary["translation_stage_p50_ms"] == 200.5
    assert summary["e2e_stage_p50_ms"] == 300.5
    assert summary["source_asr_metric"] == "WER"
    assert summary["source_asr_error_rate"] == 0
    assert summary["source_asr_coverage"] == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["samples"][0]["source_asr_hyp"] == "hello"
    assert result["samples"][0]["ref_norm"] == "你好"
    assert result["samples"][0]["hyp_norm"] == "你好"
    assert summary["meta"]["ws_control"]["endpoint"] == "ws://example.test/ws"
    assert summary["meta"]["ws_control_signature_counts"] == [
        {"signature": "start:ack-0", "n": 1},
        {"signature": "start:ack-1", "n": 1},
        {"signature": "start:ack-2", "n": 1},
    ]


def test_simult_vad_cost_retry_term_and_anomaly_metrics(tmp_path):
    rows = [{
        "id": "talk", "task": "translate", "source_text": "hello weather",
        "lang": "en-zh", "source_lang": "en", "ref_text": "北京天气很好",
        "target_lang": "Chinese", "terms": [
            {"source": "Beijing", "target": "北京"},
            {"source": "weather", "target": "天气"},
        ],
    }]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    meta = _meta(mani)
    meta["request_params"] = {
        "vad_soft_max_duration_ms": 15000,
        "vad_hard_max_duration_ms": 20000,
    }
    rec = {
        "id": "talk", "hyp": "北京天气很好", "elapsed_s": 40.0,
        "audio_s": 60.0, "ok": True, "workers": 1, "attempts": 2,
        "extra": {
            "src_dur_s": 60.0, "n_seg": 3, "segment_done_count": 4,
            "empty_segment_count": 1, "segment_error_count": 1,
            "source_text": "hello weather",
            "translation_segments": [
                {"segment_id": 0, "start_s": 0.0, "end_s": 1.0,
                 "text": "北京", "segment_reason": "silence",
                 "timing": {"queue_ms": 10, "first_token_ms": 20,
                            "total_ms": 30, "e2e_ms": 40}},
                {"segment_id": 1, "start_s": 1.0, "end_s": 17.0,
                 "text": "天气很好", "timing": {"queue_ms": 20,
                 "first_token_ms": 30, "total_ms": 40, "e2e_ms": 60}},
                {"segment_id": 2, "start_s": 17.0, "end_s": 38.0,
                 "text": "结束", "timing": {"queue_ms": 1500,
                 "first_token_ms": 40, "total_ms": 50, "e2e_ms": 6000}},
            ],
        },
    }
    inf = _write_jsonl(tmp_path / "i.jsonl", [meta, rec])
    result_path = tmp_path / "r.json"

    summary = score.run(mani, inf, out=str(result_path))

    vad = summary["simult_vad"]
    assert vad["n_segments"] == 3 and vad["n_requests"] == 4
    assert vad["segments_per_minute"] == 3 and vad["requests_per_minute"] == 4
    assert vad["segment_duration_p50_s"] == 16
    assert vad["segment_duration_max_s"] == 21
    assert vad["segment_lt_2s_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert vad["segment_gt_15s_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert vad["segment_reason_coverage"] == pytest.approx(1 / 3, abs=1e-4)
    assert vad["segment_reason_counts"] == {"silence": 1}
    assert vad["empty_segment_count"] == 1 and vad["segment_error_count"] == 1
    assert vad["anomaly_counts"] == {
        "short_phrase_island": 1, "hard_max_exceeded": 1,
        "queue_spike": 1, "e2e_spike": 1,
    }
    assert summary["retry_total"] == 1 and summary["retry_sample_rate"] == 1
    assert summary["term_recall"] == 1 and summary["term_total"] == 2
    assert summary["translation_length_ratio_mean"] == 1
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved["samples"][0]["simult"]["anomalies"]


def test_translation_literal_number_and_date_diagnostics(tmp_path):
    rows = [{
        "id": "talk", "task": "translate", "source_text": "report",
        "source_lang": "en", "ref_text": "截至2026年8月24日，共有13,000人参加。",
        "target_lang": "Chinese",
    }]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [{
        "id": "talk", "hyp": "截至2026-08-24，共有13000人参加。",
        "elapsed_s": 1.0, "audio_s": 2.0, "ok": True, "workers": 1,
        "extra": {"source_text": "report"},
    }])

    result_path = tmp_path / "r.json"
    summary = score.run(mani, inf, out=str(result_path))

    assert summary["numeric_literal_recall"] == 1
    assert summary["numeric_literal_precision"] == 1
    assert summary["date_literal_recall"] == 1
    assert summary["literal_proxy_basis"].startswith("target-reference edit")
    sample = json.loads(result_path.read_text(encoding="utf-8"))["samples"][0]
    assert sample["numeric_literal_hit"] == sample["numeric_literal_total"] == 4
    assert sample["date_literal_recall"] == 1


def test_simult_trace_automatically_adds_official_long_yaal(tmp_path, monkeypatch):
    rows = [{"id": "u0", "task": "translate", "audio_path": "talk.wav",
             "source_text": "hello", "ref_text": "你好", "target_lang": "zh"}]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [{
        "id": "u0", "hyp": "你好", "elapsed_s": 2.0, "audio_s": 3.0,
        "ok": True, "workers": 1, "extra": {"simult_trace": {"events": []}},
    }])
    monkeypatch.setattr(omnisteval_eval, "evaluate", lambda *args, **kwargs: {
        "ok": True, "version": "0.1.10", "timing_basis": "stable",
        "output_dir": str(tmp_path / "scores"),
        "assets": {"segmentation_sources": ["manifest_timestamps"],
                   "n_recordings": 1, "n_segments": 2},
        "omnisteval_chrf": 88.1234, "omnisteval_bleu": 77.4321,
        "long_yaal_cu_ms": 1850.0, "long_yaal_ca_ms": 2140.0,
    })

    summary = score.run(mani, inf, out=str(tmp_path / "r.json"))

    assert summary["long_yaal_cu_ms"] == 1850.0
    assert summary["long_yaal_ca_ms"] == 2140.0
    assert summary["omnisteval"]["ok"] is True
    assert summary["chrF"] == 88.12 and summary["chrF_document"] == 100.0
    assert summary["quality_basis"] == "omnisteval_softsegmenter"
    assert summary["err_ci95"] is None
    assert "OmniSTEval-SoftSegmenter" in summary["metric_signature"]
    assert "long_yaal_cu_ms" in summary["metric_signature"]


def test_manifest_sha_mismatch_flagged(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows())
    inf = _write_jsonl(tmp_path / "i.jsonl",
                       [_meta(mani, sha="deadbeef0000")] + [_ok(i, workers=1) for i in range(4)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["meta"]["manifest_sha_mismatch"] == "deadbeef0000"


def test_sha_match_no_flag(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows())
    inf = _write_jsonl(tmp_path / "i.jsonl",
                       [_meta(mani)] + [_ok(i, workers=1) for i in range(4)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert "manifest_sha_mismatch" not in s["meta"]


def test_manifest_sha_v2_is_recorded_without_changing_v1_behavior(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows())
    meta = _meta(mani)
    meta["manifest_sha_v2"] = manifest_sha_v2(mani)
    meta["run_id"] = "run-1"
    meta["run_spec"] = {"model": "ext-pro"}
    meta["code_revision"] = "infer-rev"
    inf = _write_jsonl(tmp_path / "i.jsonl", [meta] + [_ok(i, workers=1) for i in range(4)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["meta"]["manifest_sha"] == manifest_sha(mani)
    assert s["meta"]["manifest_sha_v2"] == manifest_sha_v2(mani)
    assert "manifest_sha_v2_mismatch" not in s["meta"]
    assert s["meta"]["run_id"] == "run-1"
    assert s["meta"]["infer_code_revision"] == "infer-rev"


def test_output_variant_records_edited_scoring_behavior(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows(1))
    rec = _ok(0, hyp="原始")
    rec["extra"] = {"edited_text": "今天天气好"}
    inf = _write_jsonl(tmp_path / "i.jsonl", [rec])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["CER"] == 0.0
    assert s["output_variant_used"] == "edited"
    assert s["meta"]["output_variant_used"] == "edited"
    assert s["meta"]["output_variant_counts"] == {"raw": 0, "edited": 1}


def test_raw_metric_keeps_empty_output_from_editing_pipeline(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows(2))
    empty = _ok(0, hyp="")
    empty["extra"] = {"edited_text": ""}
    edited = _ok(1)
    edited["extra"] = {"edited_text": "今天天气好"}
    inf = _write_jsonl(tmp_path / "i.jsonl", [empty, edited])

    s = score.run(mani, inf, out=str(tmp_path / "r.json"))

    assert s["CER"] == 0.5
    assert s["CER_raw"] == 0.5
    assert s["n_raw"] == 2


# ---------- 失败样本口径：不计精度但必须可见、覆盖率不足标记 ----------

def test_failures_set_low_coverage(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows(4))
    inf = _write_jsonl(tmp_path / "i.jsonl", [
        _meta(mani), _ok(0, workers=1), _ok(1, workers=1), _ok(2, workers=1),
        {"id": "u3", "hyp": "", "elapsed_s": 0, "audio_s": 2.0, "ok": False,
         "workers": 1, "error": "timeout"}])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["n_fail"] == 1 and s["fail_rate"] == 0.25
    assert s["low_coverage"] is True  # 75% < 95% 门槛
    assert s["CER"] == 0.0  # 精度仍只算成功样本——所以必须有 low_coverage 兜底


def test_full_coverage_not_flagged(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows())
    inf = _write_jsonl(tmp_path / "i.jsonl",
                       [_meta(mani)] + [_ok(i, workers=1) for i in range(4)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["low_coverage"] is False and s["fail_rate"] == 0.0


# ---------- KRR：别名归一化必须跟样本语言走 ----------

def test_krr_zh_keywords(tmp_path):
    rows = [{"id": "u0", "audio_path": "x", "ref_text": "阿司匹林有效", "lang": "zh",
             "keywords": [{"aliases": ["阿斯匹林", "阿司匹林"]}, {"aliases": ["布洛芬"]}]}]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [_ok(0, hyp="阿司匹林有效")])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["KRR"] == 0.5  # 命中 1/2


def test_krr_en_keywords_with_digits(tmp_path):
    # 英文样本：别名走 normalize_en，数字保留阿拉伯形式才能与 hyp 匹配
    # （旧实现用中文 normalize 会把 "15" 变 "十五"，永远匹配不上）
    rows = [{"id": "u0", "audio_path": "x", "ref_text": "flight 15 to new york",
             "lang": "en", "keywords": ["15", "New York"]}]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [_ok(0, hyp="Flight 15 to New York.")])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["KRR"] == 1.0


# ---------- 置信区间（bootstrap，固定种子可复现） ----------

def test_ci_zero_width_for_identical_samples(tmp_path):
    # 所有样本错误率相同 → 重抽样怎么抽都一样，CI 宽度为 0
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows(10))
    inf = _write_jsonl(tmp_path / "i.jsonl",
                       [_meta(mani)] + [_ok(i, hyp="今天天气坏", workers=1) for i in range(10)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["CER"] == 0.2
    assert s["err_ci95"] == [0.2, 0.2]


def test_ci_brackets_point_estimate_and_reproducible(tmp_path):
    # 一半全对一半错 1 字：CI 应包住点估计，且同种子两次打分完全一致
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows(20))
    recs = [_meta(mani)] + [_ok(i, hyp=("今天天气好" if i % 2 else "今天天气坏"), workers=1)
                            for i in range(20)]
    inf = _write_jsonl(tmp_path / "i.jsonl", recs)
    s1 = score.run(mani, inf, out=str(tmp_path / "r1.json"))
    s2 = score.run(mani, inf, out=str(tmp_path / "r2.json"))
    lo, hi = s1["err_ci95"]
    assert lo <= s1["CER"] <= hi and lo < hi
    assert s1["err_ci95"] == s2["err_ci95"]


def test_ci_none_for_single_sample(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows(1))
    inf = _write_jsonl(tmp_path / "i.jsonl", [_meta(mani), _ok(0, workers=1)])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["err_ci95"] is None


def test_asr_error_breakdown_and_sentence_rate(tmp_path):
    rows = [
        {"id": "u0", "audio_path": "x", "ref_text": "甲乙丙", "lang": "zh", "keywords": []},
        {"id": "u1", "audio_path": "x", "ref_text": "甲乙", "lang": "zh", "keywords": []},
        {"id": "u2", "audio_path": "x", "ref_text": "甲乙", "lang": "zh", "keywords": []},
    ]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [
        _meta(mani),
        {"id": "u0", "hyp": "甲丁丙", "elapsed_s": 0.1, "audio_s": 1.0, "ok": True, "workers": 1, "extra": {}},
        {"id": "u1", "hyp": "甲", "elapsed_s": 0.1, "audio_s": 1.0, "ok": True, "workers": 1, "extra": {}},
        {"id": "u2", "hyp": "甲乙丙", "elapsed_s": 0.1, "audio_s": 1.0, "ok": True, "workers": 1, "extra": {}},
    ])
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    data = json.load(open(tmp_path / "r.json", encoding="utf-8"))
    assert s["CER"] == 0.4286  # 3 edits / 7 ref chars
    assert s["SER"] == 1.0
    assert s["total_sub"] == 1 and s["total_del"] == 1 and s["total_ins"] == 1
    assert s["sub_rate"] == s["del_rate"] == s["ins_rate"] == 0.1429
    assert s["extra_text_rate"] == 0.1429
    assert s["total_ref_len"] == 7 and s["total_hyp_len"] == 7
    assert data["samples"][0]["sub"] == 1
    assert data["samples"][1]["del"] == 1
    assert data["samples"][2]["ins"] == 1
    assert data["samples"][2]["length_ratio"] == 1.5


def test_score_adds_reviewed_cer_but_preserves_original_cer(tmp_path, monkeypatch):
    rows = _zh_rows(2)
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [
        _ok(0, hyp="今天天气坏"),
        _ok(1),
    ])
    monkeypatch.setattr(score, "exclusions_for_rows",
                        lambda _rows, _root: {"u0": {"reason": "audio_reference_mismatch"}})

    summary = score.run(mani, inf, out=str(tmp_path / "r.json"))
    data = json.load(open(tmp_path / "r.json", encoding="utf-8"))

    assert summary["CER"] == 0.1
    assert summary["reviewed"]["CER"] == 0.0
    assert summary["reviewed"]["n_excluded"] == 1
    assert data["samples"][0]["excluded_from_reviewed"] is True
    assert data["samples"][0]["exclusion_reason"] == "audio_reference_mismatch"


# ---------- 按 id 去重：优先成功记录 ----------

def test_dedup_prefers_ok_record(tmp_path):
    mani = _write_jsonl(tmp_path / "m.jsonl", _zh_rows(1))
    inf = _write_jsonl(tmp_path / "i.jsonl", [
        {"id": "u0", "hyp": "", "elapsed_s": 0, "audio_s": 2.0, "ok": False, "error": "x"},
        _ok(0)])  # 重试成功的后写记录应生效
    s = score.run(mani, inf, out=str(tmp_path / "r.json"))
    assert s["n_ok"] == 1 and s["n_fail"] == 0


# ---------- 同传 TTS 保真：必须拿 infer 里的完整译文 ----------

def test_asr_bleu_uses_full_infer_hyp(tmp_path, monkeypatch):
    long_hyp = "the quick brown fox jumps over the lazy dog " * 8
    rows = [{"id": "t0", "task": "translate", "source_text": "你好", "ref_text": long_hyp,
             "target_lang": "Chinese"}]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [
        {"id": "t0", "hyp": long_hyp, "elapsed_s": 0.1, "ok": True,
         "extra": {"tts_audio": "tts.wav"}}
    ])
    seen = {}

    def fake_tts(summary, samples, lang_by_id, hyp_by_id=None):
        seen["hyp_by_id"] = hyp_by_id
        seen["lang_by_id"] = lang_by_id

    monkeypatch.setattr(score, "_add_tts_fidelity", fake_tts)
    score.run(mani, inf, out=str(tmp_path / "r.json"), asr_bleu=True)
    assert seen["hyp_by_id"] == {"t0": long_hyp}
    assert seen["lang_by_id"] == {"t0": "zh"}


def test_remote_xcomet_is_preferred_and_keeps_error_spans(tmp_path, monkeypatch):
    rows = [
        {"id": "t0", "task": "translate", "source_text": "hello", "ref_text": "你好"},
        {"id": "t1", "task": "translate", "source_text": "world", "ref_text": "世界"},
    ]
    mani = _write_jsonl(tmp_path / "m.jsonl", rows)
    inf = _write_jsonl(tmp_path / "i.jsonl", [
        {"id": "t0", "hyp": "您好", "elapsed_s": 0.1, "ok": True, "extra": {}},
        {"id": "t1", "hyp": "世界", "elapsed_s": 0.1, "ok": True, "extra": {}},
    ])
    monkeypatch.setattr(score, "remote_xcomet_score", lambda triples: {
        "model": "Unbabel/XCOMET-XL", "system_score": 0.8,
        "scores": [0.6, 1.0], "error_spans": [[{"severity": "minor"}], []],
    })
    monkeypatch.setattr(score, "comet_score", lambda triples: pytest.fail("local COMET must not run"))

    summary = score.run(mani, inf, out=str(tmp_path / "r.json"), comet=True)
    data = json.load(open(tmp_path / "r.json", encoding="utf-8"))

    assert summary["xcomet"] == 0.8
    assert summary["xcomet_model"] == "Unbabel/XCOMET-XL"
    assert data["samples"][0]["xcomet"] == 0.6
    assert data["samples"][0]["xcomet_error_spans"] == [{"severity": "minor"}]
    assert summary["xcomet_error_severity_counts"] == {"minor": 1}
    assert "xcomet" in summary["metric_signature"]


def test_remote_whisper_utmos_and_speaker_metrics(monkeypatch):
    summary = {}
    samples = [
        {"id": "t0", "hyp_norm": "hello world", "tts_audio": "/data/out.wav"},
    ]
    monkeypatch.setattr(score, "remote_whisper_transcribe", lambda items: {
        "model": "large-v3", "items": [{"id": "t0", "text": "hello world"}],
    })
    monkeypatch.setattr(score, "remote_utmos_score", lambda items: {
        "model": "fusion_stage3", "items": [{"id": "t0", "score": 4.25}],
    })
    monkeypatch.setattr(score, "remote_speaker_score", lambda items: {
        "model": "vblinkp",
        "items": [{"id": "t0", "cosine": 0.8, "normalized_similarity": 0.9}],
    })

    score._add_tts_fidelity(summary, samples, {"t0": "en"}, hyp_by_id={"t0": "hello world"})
    score._add_speech_metrics(summary, samples, {"t0": "/data/source.wav"})

    assert summary["tts_err"] == 0.0 and summary["tts_asr_model"] == "large-v3"
    assert summary["utmos"] == 4.25 and samples[0]["utmos"] == 4.25
    assert summary["speaker_cosine"] == 0.8
    assert summary["speaker_similarity"] == 0.9
