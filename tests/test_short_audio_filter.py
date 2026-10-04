import io
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from short_audio_filter import inspect_audio, lexical_units, write_balanced_keep


def _wav(samples, sample_rate=16000):
    out = io.BytesIO()
    sf.write(out, samples, sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def test_lexical_units_handles_zh_en_and_numbers():
    assert lexical_units("我刚开始 record 123") == 6


def test_clean_short_audio_is_kept():
    t = np.arange(16000) / 16000
    samples = 0.1 * np.sin(2 * np.pi * 440 * t)

    result = inspect_audio(_wav(samples), 1.0, "zh", "你好世界")

    assert result["decision"] == "keep"
    assert result["sample_rate"] == 16000
    assert result["lexical_units"] == 4


def test_silent_audio_is_rejected():
    result = inspect_audio(_wav(np.zeros(16000)), 1.0, "zh", "你好")

    assert result["decision"] == "reject"
    assert "near_silent" in result["flags"]


def test_very_short_audio_requires_review():
    samples = np.full(6400, 0.05)

    result = inspect_audio(_wav(samples), 0.4, "zh", "好")

    assert result["decision"] == "review"
    assert "very_short_under_0.5s" in result["flags"]


def test_ultra_short_audio_is_reviewed_not_discarded():
    samples = np.full(2400, 0.05)

    result = inspect_audio(_wav(samples), 0.15, "zh", "嗯")

    assert result["decision"] == "review"
    assert "ultra_short_under_0.2s" in result["flags"]


def test_annotation_markup_requires_review():
    t = np.arange(16000) / 16000
    samples = 0.1 * np.sin(2 * np.pi * 440 * t)

    result = inspect_audio(_wav(samples), 1.0, "en", "<hesitation> NO")

    assert result["decision"] == "review"
    assert "label_markup" in result["flags"]


def test_balanced_keep_caps_repeated_transcripts(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    rows = [
        {"id": f"a{i}", "source_dataset": "demo", "transcription": "嗯",
         "decision": "keep", "actual_duration": 1.0}
        for i in range(5)
    ]
    rows.append({
        "id": "review", "source_dataset": "demo", "transcription": "你好",
        "decision": "review", "actual_duration": 1.0,
    })
    candidates.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    out = tmp_path / "balanced.jsonl"

    summary = write_balanced_keep(candidates, out, max_per_transcript=2)

    selected = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(selected) == 2
    assert summary["rows"] == 2
    assert all(row["selection"] == "balanced_keep" for row in selected)
