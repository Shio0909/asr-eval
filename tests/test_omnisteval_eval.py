import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import omnisteval_eval


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return str(path)


def test_prepare_assets_uses_manifest_timestamps_and_both_clocks(tmp_path):
    manifest = _jsonl(tmp_path / "manifest.jsonl", [{
        "id": "talk", "task": "translate", "audio_path": "talk.wav",
        "lang": "en-zh", "target_lang": "中文", "longform": True,
        "segments": [
            {"start_ms": 0, "end_ms": 1000, "source_text": "hello", "ref_text": "你好"},
            {"start_ms": 1000, "end_ms": 2200, "source_text": "world", "ref_text": "世界"},
        ],
    }])
    trace = {
        "schema_version": 1, "target_lang": "zh", "granularity": "token",
        "final_text": "你好世界", "events": [
            {"seq": 0, "op": "commit", "text": "你好", "source_s": 1, "wall_s": 1.2},
            {"seq": 1, "op": "commit", "text": "世界", "source_s": 2, "wall_s": 2.4},
            {"seq": 2, "op": "final", "text": "你好世界", "source_s": 2.2, "wall_s": 2.6},
        ],
    }
    infer = _jsonl(tmp_path / "infer.jsonl", [{
        "id": "talk", "ok": True, "hyp": "你好世界", "audio_s": 2.2,
        "extra": {"src_dur_s": 2.2, "simult_trace": trace},
    }])

    assets = omnisteval_eval.prepare_assets(manifest, infer, str(tmp_path / "omni"))
    hyp = json.loads(Path(assets["hypothesis"]).read_text(encoding="utf-8"))
    segments = json.loads(Path(assets["speech_segmentation"]).read_text(encoding="utf-8"))

    assert hyp["delays"] == [1000.0, 1000.0, 2000.0, 2000.0]
    assert hyp["elapsed"] == [1200.0, 1200.0, 2400.0, 2400.0]
    assert segments[1]["offset"] == 1.0 and segments[1]["duration"] == 1.2
    assert assets["segmentation_sources"] == ["manifest_timestamps"]
    assert Path(assets["references"]).read_text(encoding="utf-8") == "你好\n世界\n"


def test_evaluate_invokes_longform_and_maps_ca_cu_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(omnisteval_eval, "_command", lambda: ["omnisteval"])
    monkeypatch.setattr(omnisteval_eval, "prepare_assets", lambda *args, **kwargs: {
        "hypothesis": "hyp.jsonl", "speech_segmentation": "segments.json",
        "references": "refs.txt", "sources": "src.txt", "target_lang": "zh",
        "char_level": True, "n_recordings": 1, "n_segments": 2,
        "segmentation_sources": ["manifest_timestamps"], "timing_basis": "stable",
    })

    def fake_run(cmd, **kwargs):
        score_dir = Path(cmd[cmd.index("--output_folder") + 1])
        score_dir.mkdir(parents=True)
        (score_dir / "scores.tsv").write_text(
            "metric\tvalue\nBLEU\t22.1\nchrF\t48.2\nLongYAAL (CU)\t1800\n"
            "LongYAAL (CA)\t2100\nLongLAAL (CU)\t2000\nLongLAAL (CA)\t2300\n",
            encoding="utf-8")
        return type("Done", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(omnisteval_eval.subprocess, "run", fake_run)
    result = omnisteval_eval.evaluate("m.jsonl", "i.jsonl", str(tmp_path / "out"))

    assert result["ok"] is True
    assert result["long_yaal_cu_ms"] == 1800
    assert result["long_yaal_ca_ms"] == 2100
    assert result["long_laal_cu_ms"] == 2000
    assert result["long_laal_ca_ms"] == 2300


def test_evaluate_degrades_when_runtime_is_missing(monkeypatch):
    monkeypatch.setattr(omnisteval_eval, "_command", lambda: None)
    assert omnisteval_eval.evaluate("m", "i", "out")["ok"] is False


@pytest.mark.skipif(not omnisteval_eval.runtime_available(), reason="OmniSTEval CLI 未安装")
def test_installed_omnisteval_cli_accepts_generated_assets(tmp_path):
    manifest = _jsonl(tmp_path / "manifest.jsonl", [{
        "id": "talk", "task": "translate", "audio_path": "talk.wav",
        "lang": "en-zh", "target_lang": "zh", "duration_ms": 2200,
        "segments": [
            {"start_ms": 0, "end_ms": 1000, "source_text": "hello", "ref_text": "你好"},
            {"start_ms": 1000, "end_ms": 2200, "source_text": "world", "ref_text": "世界"},
        ],
    }])
    infer = _jsonl(tmp_path / "infer.jsonl", [{
        "id": "talk", "ok": True, "hyp": "你好世界", "audio_s": 2.2,
        "extra": {"src_dur_s": 2.2, "simult_trace": {
            "schema_version": 1, "target_lang": "zh", "granularity": "token",
            "final_text": "你好世界", "events": [
                {"seq": 0, "op": "commit", "text": "你好", "source_s": 1, "wall_s": 1.2},
                {"seq": 1, "op": "commit", "text": "世界", "source_s": 2, "wall_s": 2.4},
            ],
        }},
    }])

    result = omnisteval_eval.evaluate(manifest, infer, str(tmp_path / "official"), timeout_s=30)

    assert result["ok"] is True, result
    assert result["long_yaal_cu_ms"] >= 0
    assert result["long_yaal_ca_ms"] >= result["long_yaal_cu_ms"]
    assert Path(result["output_dir"], "instances.resegmented.jsonl").exists()
