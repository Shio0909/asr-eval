import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import util


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_atomic_json_replace_failure_keeps_old_file(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(util.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        util.atomic_write_json(target, {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_json_success_is_valid_and_complete(tmp_path):
    target = tmp_path / "nested" / "result.json"
    util.atomic_write_json(target, {"中文": [1, 2, 3]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"中文": [1, 2, 3]}


def test_manifest_v1_stays_same_but_v2_covers_routing_and_audio(tmp_path):
    base = {"id": "u1", "task": "asr", "ref_text": "你好", "audio_path": "/a/datasets/asr/set/a.wav",
            "lang": "zh", "keywords": ["甲"]}
    changed = {**base, "audio_path": "/b/datasets/asr/set/b.wav", "lang": "yue", "keywords": ["乙"]}
    p1 = _jsonl(tmp_path / "a.jsonl", [base])
    p2 = _jsonl(tmp_path / "b.jsonl", [changed])

    assert util.manifest_sha(p1) == util.manifest_sha(p2)
    assert util.manifest_sha_v2(p1) != util.manifest_sha_v2(p2)


def test_manifest_v2_ignores_machine_root_for_dataset_audio(tmp_path):
    a = {"id": "u1", "ref_text": "你好", "audio_path": "/machine-a/work/datasets/asr/set/a.wav"}
    b = {**a, "audio_path": "/machine-b/other/datasets/asr/set/a.wav"}
    assert util.manifest_sha_v2(_jsonl(tmp_path / "a.jsonl", [a])) == util.manifest_sha_v2(
        _jsonl(tmp_path / "b.jsonl", [b]))


def test_manifest_rows_reject_duplicate_ids():
    with pytest.raises(ValueError, match="重复样本 ID: u1"):
        util.ensure_unique_ids([{"id": "u1"}, {"id": "u1"}], source="fixture")
