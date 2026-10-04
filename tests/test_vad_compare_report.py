import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import vad_compare_report


def _write_result(root, arm, dataset, *, chrf, asr_error, e2e, requests,
                  short_rate=0.0, anomaly=False):
    samples = []
    anomaly_counts = {}
    if anomaly:
        anomaly_counts = {"short_phrase_island": 1}
        samples = [{"id": "s1", "simult": {"anomalies": [
            {"type": "short_phrase_island", "duration_s": 1.0, "text": "短语"}
        ]}}]
    payload = {"summary": {
        "n_total": 10, "n_ok": 10, "fail_rate": 0.0,
        "low_coverage": False, "retry_sample_rate": 0.0,
        "chrF": chrf, "BLEU": chrf + 5,
        "source_asr_error_rate": asr_error,
        "stable_ttfb_p50_s": 5.0, "finish_tail_p95_s": 0.5,
        "simult_vad": {
            "n_requests": 100, "audio_minutes": 20,
            "requests_per_minute": requests,
            "segment_lt_2s_rate": short_rate,
            "segment_gt_hard_max_rate": 0.0,
            "segment_reason_coverage": 0.0,
            "segment_error_count": 0,
            "e2e_p95_ms": e2e, "inference_p95_ms": e2e - 100,
            "anomaly_counts": anomaly_counts,
        },
        "infer_file": f"infer/{arm}__{dataset}.jsonl",  # build_report 只认带溯源的结果文件
    }, "samples": samples}
    path = root / f"{arm}__{dataset}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_report_rejects_quality_regression_and_selects_balanced_arm(tmp_path):
    for dataset in ("mcif", "realsi"):
        _write_result(tmp_path, "online_old", dataset,
                      chrf=46.0, asr_error=0.075, e2e=1200, requests=5.8)
        _write_result(tmp_path, "local_impl_old_params", dataset,
                      chrf=45.8, asr_error=0.072, e2e=1000, requests=5.4)
        _write_result(tmp_path, "local_impl_new_params", dataset,
                      chrf=43.5, asr_error=0.09, e2e=700, requests=7.8,
                      short_rate=0.08, anomaly=True)

    report = vad_compare_report.build_report(tmp_path)

    assert report["complete"] is True
    assert report["winner"] == "local_impl_old_params"
    aggressive = report["arms"]["local_impl_new_params"]
    assert aggressive["eligible"] is False
    assert any("chrF" in reason for reason in aggressive["guardrail_failures"])
    assert len(report["anomalies"]) == 2


def test_incomplete_report_never_selects_winner(tmp_path):
    _write_result(tmp_path, "online_old", "mcif",
                  chrf=46.0, asr_error=0.075, e2e=1200, requests=5.8)
    report = vad_compare_report.build_report(tmp_path)
    assert report["complete"] is False
    assert report["winner"] is None
