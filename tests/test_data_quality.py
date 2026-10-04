import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import build_manifest
from data_quality import exclusions_for_rows, reviewed_asr_metrics


def test_aishell2_builder_skips_excluded_speaker_before_limit(tmp_path, monkeypatch):
    base = tmp_path / "datasets" / "asr" / "aishell2_ios"
    (base / "wav" / "C_BAD").mkdir(parents=True)
    (base / "wav" / "C_OK").mkdir(parents=True)
    (base / "wav" / "C_BAD" / "bad.wav").write_bytes(b"bad")
    (base / "wav" / "C_OK" / "ok.wav").write_bytes(b"ok")
    (base / "trans_subset.tsv").write_text(
        "bad\tC_BAD\t错误样本\n"
        "ok\tC_OK\t正常样本\n",
        encoding="utf-8",
    )
    (base / "exclusions.json").write_text(json.dumps({
        "speakers": {"C_BAD": {"action": "exclude", "reason": "misalignment"}},
    }), encoding="utf-8")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_aishell2(limit=1)

    assert [row["id"] for row in rows] == ["ok"]


def test_manifest_rows_find_dataset_sidecar_and_match_sample_or_speaker(tmp_path):
    base = tmp_path / "datasets" / "asr" / "demo"
    (base / "wav" / "C1").mkdir(parents=True)
    (base / "exclusions.json").write_text(json.dumps({
        "samples": {"u1": {"action": "exclude", "reason": "bad_ref"}},
        "speakers": {"C1": {"action": "exclude", "reason": "bad_speaker"}},
    }), encoding="utf-8")
    rows = [
        {"id": "u1", "audio_path": "datasets/asr/demo/wav/C0/u1.wav"},
        {"id": "u2", "audio_path": "datasets/asr/demo/wav/C1/u2.wav"},
    ]

    out = exclusions_for_rows(rows, str(tmp_path))

    assert out["u1"]["matched_by"] == "sample"
    assert out["u2"]["matched_by"] == "speaker"
    assert out["u1"]["policy"] == "datasets/asr/demo/exclusions.json"


def test_reviewed_metric_excludes_samples_without_changing_original_summary():
    summary = {"metric": "CER", "CER": 0.2, "err_rate": 0.2, "n_total": 2}
    samples = [
        {"id": "bad", "ref_len": 5, "sub": 2, "del": 0, "ins": 0},
        {"id": "good", "ref_len": 5, "sub": 0, "del": 0, "ins": 0},
    ]

    reviewed = reviewed_asr_metrics(summary, samples, {"bad": {"reason": "bad_ref"}})

    assert summary == {"metric": "CER", "CER": 0.2, "err_rate": 0.2, "n_total": 2}
    assert reviewed["CER"] == 0.0
    assert reviewed["n_total"] == 1 and reviewed["n_excluded"] == 1
    assert reviewed["excluded_ids"] == ["bad"]
