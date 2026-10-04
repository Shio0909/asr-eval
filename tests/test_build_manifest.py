import gzip
import json
import os
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import build_manifest


def test_fleurs_asr_matches_qwen3_public_twelve_language_set():
    assert set(build_manifest._FLEURS_ASR) == {
        "en", "zh", "yue", "ar", "de", "es", "fr", "it", "ja", "ko", "pt", "ru",
    }


def test_commonvoice_builder_reads_only_requested_test_split(tmp_path, monkeypatch):
    base = tmp_path / "datasets" / "asr" / "common_voice17"
    transcript = base / "transcript" / "zh-CN"
    audio = base / "audio" / "zh-CN" / "test"
    transcript.mkdir(parents=True)
    audio.mkdir(parents=True)
    transcript.joinpath("test.tsv").write_text(
        "client_id\tpath\tsentence\nspk1\tcommon_voice_1.mp3\t今天天气很好\n",
        encoding="utf-8",
    )
    source = tmp_path / "common_voice_1.mp3"
    source.write_bytes(b"mp3")
    with tarfile.open(audio / "zh-CN_test_0.tar", "w") as archive:
        archive.add(source, arcname="common_voice_1.mp3")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_commonvoice(subset="zh-CN")

    assert rows == [{
        "id": "cv17_zh-CN_common_voice_1",
        "audio_path": str(audio / "extracted" / "common_voice_1.mp3"),
        "ref_text": "今天天气很好", "lang": "zh", "keywords": [],
        "speaker_id": "spk1",
    }]


def test_ja_benchmark_materializes_embedded_audio(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = tmp_path / "datasets" / "asr" / "ja_cv8"
    (base / "parquet").mkdir(parents=True)
    table = pa.Table.from_pylist([
        {"audio": {"bytes": b"wav-one", "path": "one.wav"}, "transcription": "今日は、晴れです。"},
        {"audio": {"bytes": b"wav-two", "path": "two.wav"}, "transcription": "音声認識です。"},
    ])
    pq.write_table(table, base / "parquet" / "0000.parquet")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))
    monkeypatch.setitem(build_manifest._JA_BENCHMARKS, "cv8", ("ja_cv8", "fixture/cv8", 2))

    rows = build_manifest.build_ja_benchmark(subset="cv8")

    assert len(rows) == 2
    assert rows[0]["lang"] == "ja"
    assert rows[0]["ref_text"] == "今日は、晴れです。"
    assert open(rows[0]["audio_path"], "rb").read() == b"wav-one"
    assert rows[0]["source_dataset"] == "fixture/cv8"


def test_open_asr_builder_materializes_requested_subset(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = tmp_path / "datasets" / "asr" / "open_asr" / "ami"
    base.mkdir(parents=True)
    table = pa.Table.from_pylist([{
        "audio": {"bytes": b"wav-bytes", "path": "meeting/clip.wav"},
        "dataset": "ami", "text": "A test sentence.", "id": "meeting/clip.wav",
        "audio_length_s": 1.0,
    }])
    pq.write_table(table, base / "test-00000-of-00001.parquet")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_open_asr(subset="ami")

    assert len(rows) == 1
    assert rows[0]["ref_text"] == "A test sentence."
    assert rows[0]["lang"] == "en"
    assert rows[0]["source_id"] == "meeting/clip.wav"
    assert rows[0]["source_dataset"] == "ami"
    assert os.path.dirname(rows[0]["audio_path"]) == str(
        tmp_path / "datasets" / "asr" / "open_asr" / "wav" / "ami")
    assert open(rows[0]["audio_path"], "rb").read() == b"wav-bytes"


def test_open_asr_builder_rejects_unknown_subset():
    with pytest.raises(ValueError, match="ami/earnings22/gigaspeech"):
        build_manifest.build_open_asr(subset="unknown")


def test_wildasr_builder_deduplicates_exact_upstream_rows(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = tmp_path / "datasets" / "asr" / "wildasr" / "data"
    base.mkdir(parents=True)
    row = {
        "audio": {"bytes": b"wav", "path": "clip.wav"},
        "transcript": "hello world", "subset": "fleurs_clipping",
        "audio_hash_id": "a" * 64,
    }
    pq.write_table(pa.Table.from_pylist([row, row]), base / "part.parquet")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_wildasr()

    assert len(rows) == 1
    assert rows[0]["id"] == "wildasr_fleurs_clipping_aaaaaaaaaaaa"
    assert open(rows[0]["audio_path"], "rb").read() == b"wav"


def test_wildasr_builder_rejects_conflicting_duplicate_ids(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = tmp_path / "datasets" / "asr" / "wildasr" / "data"
    base.mkdir(parents=True)
    shared = {"subset": "fleurs_clipping", "audio_hash_id": "a" * 64}
    pq.write_table(pa.Table.from_pylist([
        {**shared, "audio": {"bytes": b"one", "path": "one.wav"}, "transcript": "one"},
        {**shared, "audio": {"bytes": b"two", "path": "two.wav"}, "transcript": "two"},
    ]), base / "part.parquet")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="重复 ID 对应不同内容"):
        build_manifest.build_wildasr()


def test_wmt_builder_keeps_each_year_and_direction_in_separate_manifest(tmp_path, monkeypatch):
    base = tmp_path / "datasets" / "translation" / "wmt"
    base.mkdir(parents=True)
    (base / "wmt23.en-zh.src.txt").write_text("hello\n", encoding="utf-8")
    (base / "wmt23.en-zh.ref.txt").write_text("你好\n", encoding="utf-8")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_wmt(subset="wmt23-en-zh")

    assert rows == [{
        "id": "wmt23_enzh_0", "task": "translate", "lang": "en-zh",
        "source_text": "hello", "ref_text": "你好", "target_lang": "中文",
    }]


def test_aishell2_eval_builder_selects_official_channel(tmp_path, monkeypatch):
    base = (tmp_path / "datasets" / "asr" / "aishell2_eval" /
            "AISHELL-DEV-TEST-SET" / "iOS" / "test")
    (base / "wav" / "T0011").mkdir(parents=True)
    (base / "wav" / "T0011" / "IT0011W0001.wav").write_bytes(b"wav")
    (base / "wav.scp").write_text(
        "IT0011W0001\twav/T0011/IT0011W0001.wav\n", encoding="utf-8")
    (base / "trans.txt").write_text("IT0011W0001\t换一首歌\n", encoding="utf-8")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_aishell2_eval(subset="ios")

    assert rows == [{
        "id": "IT0011W0001",
        "audio_path": str(base / "wav" / "T0011" / "IT0011W0001.wav"),
        "ref_text": "换一首歌", "lang": "zh", "keywords": [],
        "channel": "ios", "speaker_id": "T0011",
    }]


def test_aishell2_eval_builder_rejects_unknown_channel():
    with pytest.raises(ValueError, match="ios/android/mic"):
        build_manifest.build_aishell2_eval(subset="windows")


def test_wsyue_long_builder_reads_text_tier(tmp_path, monkeypatch):
    base = tmp_path / "datasets" / "asr" / "wsyue" / "Long"
    (base / "wav").mkdir(parents=True)
    (base / "TextGrid").mkdir()
    (base / "wav" / "sample.wav").write_bytes(b"wav")
    (base / "TextGrid" / "sample.TextGrid").write_text(
        'item [1]:\n'
        '    name = "文本"\n'
        '    intervals [1]:\n'
        '        text = "第一句"\n'
        '    intervals [2]:\n'
        '        text = ""\n'
        '    intervals [3]:\n'
        '        text = "第二句"\n'
        'item [2]:\n'
        '    name = "性别"\n'
        '    intervals [1]:\n'
        '        text = "女"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_wsyue(subset="long")

    assert rows == [{
        "id": "sample",
        "audio_path": str(base / "wav" / "sample.wav"),
        "ref_text": "第一句第二句",
        "lang": "yue",
        "keywords": [],
    }]


def test_chuan_yu_builder_supports_chinese_and_slug_subsets(tmp_path, monkeypatch):
    base = tmp_path / "datasets" / "asr" / "chuan_yu_12city"
    for city, speaker, wav, text in (
        ("成都", "G0001", "G0001_S0001.wav", "成都参考文本"),
        ("重庆", "G0002", "G0002_S0001.wav", "重庆参考文本"),
    ):
        wav_dir = base / city / "WAV" / speaker
        wav_dir.mkdir(parents=True)
        (wav_dir / wav).write_bytes(b"wav")
        (base / city / "UTTERANCEINFO.txt").write_text(
            "CHANNEL\tUTTRANS_ID\tSPEAKER_ID\tPROMPT\tPROMTTYPE\tTRANSCRIPTION\n"
            f"C0\t{wav}\t{speaker}\t\t\t{text}\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_chuan_yu(subset="chengdu,重庆")

    assert [row["id"] for row in rows] == [
        "chuan_yu_chengdu_G0001_S0001",
        "chuan_yu_chongqing_G0002_S0001",
    ]
    assert [row["dialect"] for row in rows] == ["成都", "重庆"]
    assert rows[0]["city"] == "成都"
    assert rows[0]["speaker_id"] == "G0001"
    assert rows[0]["ref_text"] == "成都参考文本"


def test_chuan_yu_builder_rejects_unknown_city(tmp_path, monkeypatch):
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="不支持城市"):
        build_manifest.build_chuan_yu(subset="不存在")


def test_acl6060_long_builder_groups_full_talk_and_tolerates_bare_ampersand(tmp_path, monkeypatch):
    base = (tmp_path / "datasets" / "translation" / "longform_raw" / "acl6060_full" /
            "extracted" / "2" / "acl_6060" / "eval")
    (base / "full_wavs").mkdir(parents=True)
    (base / "text" / "xml").mkdir(parents=True)
    (base / "full_wavs" / "talk.1.wav").write_bytes(b"wav")
    source = ('<mteval><doc docid="talk.1"><abstract>V&L</abstract>'
              '<seg id="1">First &amp; one.</seg><seg id="2">Second.</seg>'
              '</doc></mteval>')
    target = ('<mteval><doc docid="talk.1"><abstract>V&L</abstract>'
              '<seg id="1">第一句。</seg><seg id="2">第二句。</seg>'
              '</doc></mteval>')
    (base / "text" / "xml" / "ACL.6060.eval.en-xx.en.xml").write_text(source, encoding="utf-8")
    (base / "text" / "xml" / "ACL.6060.eval.en-xx.zh.xml").write_text(target, encoding="utf-8")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_acl6060_long(subset="eval-zh")

    assert len(rows) == 1
    assert rows[0]["audio_path"] == str(base / "full_wavs" / "talk.1.wav")
    assert rows[0]["source_text"] == "First & one.\nSecond."
    assert rows[0]["ref_text"] == "第一句。\n第二句。"
    assert rows[0]["target_lang"] == "Chinese"
    assert rows[0]["reference_type"] == "written_translation"
    assert rows[0]["segment_count"] == 2


def test_mcif_long_builder_only_keeps_translation_tasks(tmp_path, monkeypatch):
    base = tmp_path / "datasets" / "translation" / "longform_raw" / "mcif"
    (base / "MCIF_DATA" / "LONG_AUDIOS").mkdir(parents=True)
    (base / "MCIF_DATA" / "LONG_AUDIOS" / "talk.wav").write_bytes(b"wav")
    xml = b'''<testset><task track="long" text_lang="de">
      <sample id="0" iid="QA_1" task="QA"><audio_path>talk.wav</audio_path><reference>ignore</reference></sample>
      <sample id="1" iid="TRANS_1" task="TRANS"><audio_path>talk.wav</audio_path>
        <reference>Deutsche Referenz.</reference><metadata><transcript>English transcript.</transcript></metadata>
      </sample></task></testset>'''
    with gzip.open(base / "MCIF.long.de.ref.xml.gz", "wb") as dst:
        dst.write(xml)
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_mcif_long(subset="de")

    assert len(rows) == 1
    assert rows[0]["id"] == "mcif_long_en_de_TRANS_1"
    assert rows[0]["lang"] == "en-de"
    assert rows[0]["target_lang"] == "German"
    assert rows[0]["source_text"] == "English transcript."


def test_realsi_builder_preserves_timing_terms_and_human_reference_type(tmp_path, monkeypatch):
    base = tmp_path / "datasets" / "translation" / "longform_raw" / "realsi" / "data" / "en2zh"
    (base / "json").mkdir(parents=True)
    (base / "wav").mkdir()
    (base / "wav" / "en2zh-01-tech.wav").write_bytes(b"wav")
    payload = {
        "vid": "en2zh-01-tech", "duration": 9000,
        "segment": [{
            "start_time": 100, "end_time": 8900,
            "src_text": "Load balancing.", "trg_text": "负载均衡。",
            "utterance": [{"term": [{"src": "load balancing", "trg": "负载均衡"}]}],
        }],
    }
    (base / "json" / "en2zh-01-tech.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_realsi(subset="en-zh")

    assert len(rows) == 1
    assert rows[0]["duration_ms"] == 9000
    assert rows[0]["segments"] == [{"start_ms": 100, "end_ms": 8900,
                                     "source_text": "Load balancing.", "ref_text": "负载均衡。"}]
    assert rows[0]["terms"] == [{"source": "load balancing", "target": "负载均衡"}]
    assert rows[0]["keywords"] == ["load balancing"]
    assert rows[0]["reference_type"] == "human_simultaneous_interpretation"


def test_bstc_long_builder_extracts_and_preserves_official_timestamps(tmp_path, monkeypatch):
    base = (tmp_path / "datasets" / "translation" / "longform_raw" / "bstc_ccmt2019" /
            "CCMT_2019_BSTC" / "data")
    base.mkdir(parents=True)
    annotations = [
        {"offset": "1.250", "duration": "0.750", "transcript": "你好。",
         "translation": "Hello."},
        {"offset": "2.500", "duration": "1.500", "transcript": "欢迎。",
         "translation": "Welcome."},
    ]
    with zipfile.ZipFile(base / "development_data.zip", "w") as archive:
        archive.writestr("42.wav", b"wav")
        archive.writestr("42.asr.json", "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in annotations))
    monkeypatch.setattr(build_manifest, "ROOT", str(tmp_path))

    rows = build_manifest.build_bstc_long()

    assert len(rows) == 1
    assert rows[0]["id"] == "bstc_dev_zh_en_42"
    assert rows[0]["segments"][0]["start_ms"] == 1250
    assert rows[0]["segments"][1]["end_ms"] == 4000
    assert rows[0]["duration_ms"] == 4000
    assert rows[0]["audio_path"].endswith("development/42.wav")
