import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from materialize_short_audio import qwen_record, select_variant, split_rows


def _row(dataset, uid, language="zh", speaker="spk"):
    row = {
        "id": uid,
        "source_dataset": dataset,
        "language": language,
        "transcription": "测试",
        "actual_duration": 1.0,
    }
    if dataset == "ASCEND":
        row["original_speaker_id"] = speaker
    else:
        row["speaker"] = speaker
    return row


def test_qwen_record_uses_supported_base_language_prefix():
    row = qwen_record(_row("ASCEND", "a", "mixed"), "wavs/a.wav")

    assert row["text"] == "language Chinese<asr_text>测试"
    assert row["duration_bucket"] == "1-1.5s"


def test_speaker_split_has_no_overlap():
    rows = [
        _row("ASCEND", f"a{i}", speaker=f"spk{i // 2}")
        for i in range(40)
    ]

    train, validation = split_rows(rows, validation_ratio=0.2, seed=42)
    train_speakers = {row["original_speaker_id"] for row in train}
    validation_speakers = {row["original_speaker_id"] for row in validation}

    assert train_speakers
    assert validation_speakers
    assert train_speakers.isdisjoint(validation_speakers)


def test_each_dataset_with_multiple_speakers_gets_validation_group():
    rows = [
        _row(dataset, f"{dataset}-{speaker}", speaker=speaker)
        for dataset in ("ASCEND", "MED-IT")
        for speaker in ("a", "b")
    ]

    _, validation = split_rows(rows, validation_ratio=0.05, seed=42)

    assert {row["source_dataset"] for row in validation} == {"ASCEND", "MED-IT"}


def test_zh_focus_keeps_chinese_and_caps_med_it():
    rows = [
        _row("ASCEND", "zh", "zh"),
        _row("ASCEND", "mixed", "mixed"),
        _row("ASCEND", "en", "en"),
        _row("MED-IT", "med1", "en"),
        _row("MED-IT", "med2", "en"),
    ]

    selected = select_variant(rows, "zh_focus", max_med_it=1, seed=42)

    assert {row["id"] for row in selected if row["source_dataset"] == "ASCEND"} == {
        "zh",
        "mixed",
    }
    assert sum(row["source_dataset"] == "MED-IT" for row in selected) == 1
