"""OmniSTEval 官方同传指标接入。

从项目 manifest + infer 产出标准 JSONL、参考句和 speech segmentation，调用
OmniSTEval 计算 YAAL/LongYAAL 的 computation-aware / unaware 版本。依赖缺失
或输入不完整时返回结构化错误，不影响现有 chrF/BLEU/自定义 AL 结果落盘。
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

from config import abs_path
from simult_trace import legacy_trace, omnisteval_record
from util import atomic_text_writer, atomic_write_json, validate_jsonl_file


def runtime_available() -> bool:
    return bool(_command())


def _command() -> list[str] | None:
    configured = os.environ.get("OMNISTEVAL_BIN")
    if configured:
        return [configured]
    executable = shutil.which("omnisteval")
    return [executable] if executable else None


def _read_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def _read_infer(path: str) -> dict[str, dict]:
    best = {}
    with open(path, encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") == "__meta__":
                continue
            if row.get("id") not in best or row.get("ok"):
                best[row["id"]] = row
    return best


def _lang_code(value: str) -> str:
    value = str(value or "").strip().lower()
    names = {
        "中文": "zh", "简体中文": "zh", "chinese": "zh",
        "英文": "en", "english": "en", "日文": "ja", "japanese": "ja",
        "韩文": "ko", "korean": "ko", "德文": "de", "german": "de",
        "意大利文": "it", "italian": "it",
    }
    return names.get(value, value.split("-")[-1] if "-" in value else value)


def _audio_duration(path: str, row: dict) -> float:
    if row.get("duration_ms"):
        return float(row["duration_ms"]) / 1000
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", path],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return float(proc.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        import soundfile as sf
        return float(sf.info(path).duration)


def _acl_sidecar_segments(row: dict) -> list[dict] | None:
    audio = Path(abs_path(row.get("audio_path", "")))
    if audio.parent.name != "full_wavs":
        return None
    sidecar_dir = audio.parent.parent / "segmented_wavs" / "shas"
    candidates = sorted(sidecar_dir.glob("*.yaml"))
    if not candidates:
        return None
    try:
        import yaml
        entries = yaml.safe_load(candidates[0].read_text(encoding="utf-8")) or []
    except Exception:
        return None
    return [entry for entry in entries if Path(str(entry.get("wav", ""))).name == audio.name]


def _row_segments(row: dict) -> tuple[list[dict], str]:
    """返回 OmniSTEval speech segmentation 和来源标签。"""
    audio = abs_path(row.get("audio_path", ""))
    wav = os.path.basename(audio)
    annotated = row.get("segments") or []
    timed = []
    refs = []
    for segment in annotated:
        start_ms, end_ms = segment.get("start_ms"), segment.get("end_ms")
        if start_ms is None and segment.get("start_s") is not None:
            start_ms = float(segment["start_s"]) * 1000
        if end_ms is None and segment.get("end_s") is not None:
            end_ms = float(segment["end_s"]) * 1000
        if start_ms is None or end_ms is None:
            timed = []
            break
        timed.append({
            "wav": wav, "offset": round(float(start_ms) / 1000, 4),
            "duration": round(max(0.0, float(end_ms) - float(start_ms)) / 1000, 4),
            "speaker_id": str(segment.get("speaker_id") or "NA"),
            "reference": segment.get("ref_text", ""),
            "source_text": segment.get("source_text", ""),
        })
    if timed and len(timed) == len(annotated):
        return timed, "manifest_timestamps"

    sidecar = _acl_sidecar_segments(row)
    if sidecar and annotated and len(sidecar) == len(annotated):
        for timing, text in zip(sidecar, annotated):
            refs.append({
                "wav": wav, "offset": round(float(timing["offset"]), 4),
                "duration": round(float(timing["duration"]), 4),
                "speaker_id": str(timing.get("speaker_id") or "NA"),
                "reference": text.get("ref_text", ""),
                "source_text": text.get("source_text", ""),
            })
        return refs, "official_shas"

    duration = _audio_duration(audio, row)
    return [{
        "wav": wav, "offset": 0.0, "duration": round(duration, 4), "speaker_id": "NA",
        "reference": row.get("ref_text", ""), "source_text": row.get("source_text", ""),
    }], "document_fallback"


def _one_line(value) -> str:
    return " ".join(str(value or "").split())


def prepare_assets(manifest_path: str, infer_path: str, output_dir: str,
                   timing: str = "stable") -> dict:
    rows, infer = _read_manifest(manifest_path), _read_infer(infer_path)
    selected = []
    for row in rows:
        prediction = infer.get(row.get("id"))
        if not prediction or not prediction.get("ok"):
            continue
        extra = prediction.get("extra") or {}
        trace = extra.get("simult_trace") or legacy_trace(
            prediction.get("hyp", ""), extra.get("translation_segments") or [],
            row.get("target_lang") or row.get("lang", "").split("-")[-1],
        )
        if not trace:
            continue
        record = omnisteval_record(
            abs_path(row.get("audio_path", "")), trace,
            extra.get("src_dur_s") or prediction.get("audio_s") or _audio_duration(
                abs_path(row.get("audio_path", "")), row),
            timing,
        )
        selected.append((row, record))
    if not selected:
        raise ValueError("没有带同传 trace/translation_segments 的成功样本")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hypothesis = out / "hypothesis.jsonl"
    with atomic_text_writer(hypothesis, validator=validate_jsonl_file) as dst:
        for _, record in selected:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    speech, references, sources, segmentation_sources = [], [], [], set()
    for row, _ in selected:
        segments, source = _row_segments(row)
        segmentation_sources.add(source)
        for segment in segments:
            speech.append({key: segment[key] for key in ("wav", "offset", "duration", "speaker_id")})
            references.append(_one_line(segment["reference"]))
            sources.append(_one_line(segment["source_text"]))
    segmentation = out / "speech_segments.json"
    refs = out / "references.txt"
    srcs = out / "sources.txt"
    atomic_write_json(segmentation, speech, indent=2)  # JSON 是合法 YAML，OmniSTEval 可直接读取
    with atomic_text_writer(refs) as dst:
        dst.write("\n".join(references) + "\n")
    with atomic_text_writer(srcs) as dst:
        dst.write("\n".join(sources) + "\n")

    target_lang = _lang_code(selected[0][0].get("target_lang") or selected[0][0].get("lang"))
    char_level = selected[0][1].get("evaluation_unit") == "char"
    return {
        "hypothesis": str(hypothesis), "speech_segmentation": str(segmentation),
        "references": str(refs), "sources": str(srcs), "target_lang": target_lang,
        "char_level": char_level, "n_recordings": len(selected), "n_segments": len(speech),
        "segmentation_sources": sorted(segmentation_sources), "timing_basis": timing,
    }


def _parse_scores(path: str) -> dict:
    display_names = {
        "BLEU": "bleu", "chrF": "chrf", "COMET": "comet",
        "LongYAAL (CU)": "long_yaal", "LongYAAL (CA)": "ca_long_yaal",
        "LongAL (CU)": "long_al", "LongAL (CA)": "ca_long_al",
        "LongLAAL (CU)": "long_laal", "LongLAAL (CA)": "ca_long_laal",
        "LongAP (CU)": "long_ap", "LongAP (CA)": "ca_long_ap",
        "LongDAL (CU)": "long_dal", "LongDAL (CA)": "ca_long_dal",
    }
    scores = {}
    with open(path, encoding="utf-8") as src:
        next(src, None)
        for line in src:
            key, sep, value = line.rstrip("\n").partition("\t")
            if not sep:
                continue
            key = display_names.get(key, key)
            try:
                scores[key] = float(value)
            except ValueError:
                scores[key] = value
    return scores


def evaluate(manifest_path: str, infer_path: str, output_dir: str,
             timing: str = "stable", comet_model: str | None = None,
             timeout_s: float | None = None) -> dict:
    command = _command()
    if not command:
        return {"ok": False, "error": "OmniSTEval 未安装或 omnisteval 不在 PATH"}
    try:
        assets = prepare_assets(manifest_path, infer_path, output_dir, timing)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    score_dir = os.path.join(output_dir, "scores")
    cmd = command + [
        "longform", "--speech_segmentation", assets["speech_segmentation"],
        "--ref_sentences_file", assets["references"],
        "--source_sentences_file", assets["sources"],
        "--hypothesis_file", assets["hypothesis"], "--hypothesis_format", "jsonl",
        "--lang", assets["target_lang"] or "en",
        "--bleu_tokenizer", "zh" if assets["target_lang"] == "zh" else "13a",
        "--char_level" if assets["char_level"] else "--word_level",
        "--output_folder", score_dir,
    ]
    if comet_model:
        cmd += ["--comet", "--comet_model", comet_model]
    try:
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True,
            timeout=timeout_s or float(os.environ.get("OMNISTEVAL_TIMEOUT_S", "3600")),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": f"OmniSTEval 启动失败: {exc}", "assets": assets}
    if proc.returncode:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-1200:],
                "returncode": proc.returncode, "assets": assets}
    raw = _parse_scores(os.path.join(score_dir, "scores.tsv"))
    try:
        version = importlib.metadata.version("OmniSTEval")
    except importlib.metadata.PackageNotFoundError:
        version = None
    mapped = {
        "omnisteval_bleu": raw.get("bleu"), "omnisteval_chrf": raw.get("chrf"),
        "long_yaal_cu_ms": raw.get("long_yaal", raw.get("yaal")),
        "long_yaal_ca_ms": raw.get("ca_long_yaal", raw.get("ca_yaal")),
        "long_al_cu_ms": raw.get("long_al"), "long_al_ca_ms": raw.get("ca_long_al"),
        "long_laal_cu_ms": raw.get("long_laal"), "long_laal_ca_ms": raw.get("ca_long_laal"),
        "long_dal_cu_ms": raw.get("long_dal"), "long_dal_ca_ms": raw.get("ca_long_dal"),
    }
    if "comet" in raw:
        mapped["omnisteval_comet"] = raw["comet"]
    return {
        "ok": True, "version": version, "timing_basis": timing,
        "output_dir": score_dir, "assets": assets, "raw_scores": raw,
        **{key: value for key, value in mapped.items() if value is not None},
    }
