"""Screen embedded-audio parquet rows for short-ASR fine-tuning candidates.

The tool is deliberately conservative: it never rewrites labels or audio.  It
emits a metadata-only JSONL with ``keep`` / ``review`` / ``reject`` decisions,
plus a compact aggregate report.

Example:
  python eval/short_audio_filter.py \
    --input 'datasets/asr/ascend/main/train-*.parquet' \
    --out reports/short_audio/ascend_train_candidates.jsonl \
    --report reports/short_audio/ascend_train_report.json
"""

import argparse
import collections
import glob
import hashlib
import io
import json
import math
import os
import re
import tarfile

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_MARKUP_RE = re.compile(r"<[^>]+>|\[[^\]]+\]")


def lexical_units(text):
    """Count Chinese characters, English words and numbers as lexical units."""
    text = text or ""
    return len(_HAN_RE.findall(text)) + len(_WORD_RE.findall(text)) + len(_NUMBER_RE.findall(text))


def _dbfs(value):
    return 20 * math.log10(max(float(value), 1e-12))


def inspect_audio(audio_bytes, expected_duration, language, text):
    """Return acoustic/text diagnostics and a conservative triage decision."""
    hard = []
    review = []
    metrics = {}
    try:
        samples, sample_rate = sf.read(
            io.BytesIO(audio_bytes), dtype="float32", always_2d=True
        )
    except Exception as exc:
        return {
            "decision": "reject",
            "flags": ["decode_error"],
            "decode_error": f"{type(exc).__name__}: {exc}",
        }

    mono = samples.mean(axis=1)
    actual_duration = len(mono) / sample_rate if sample_rate else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if len(mono) else 0.0
    clip_ratio = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 0.0
    zero_ratio = float(np.mean(np.abs(mono) < 1e-7)) if len(mono) else 1.0
    units = lexical_units(text)
    rate = units / actual_duration if actual_duration else 0.0

    metrics.update({
        "actual_duration": round(actual_duration, 4),
        "sample_rate": sample_rate,
        "channels": samples.shape[1],
        "peak_dbfs": round(_dbfs(peak), 2),
        "rms_dbfs": round(_dbfs(rms), 2),
        "clip_ratio": round(clip_ratio, 6),
        "zero_ratio": round(zero_ratio, 6),
        "lexical_units": units,
        "lexical_units_per_s": round(rate, 3),
    })

    duration_delta = abs(actual_duration - float(expected_duration or 0))
    metrics["duration_delta"] = round(duration_delta, 4)

    if actual_duration < 0.2:
        review.append("ultra_short_under_0.2s")
    elif actual_duration < 0.5:
        review.append("very_short_under_0.5s")
    if duration_delta > max(0.1, actual_duration * 0.05):
        hard.append("duration_mismatch")
    if units == 0:
        hard.append("no_lexical_label")
    if _MARKUP_RE.search(text or ""):
        review.append("label_markup")
    if rms < 10 ** (-55 / 20):
        hard.append("near_silent")
    elif rms < 10 ** (-42 / 20):
        review.append("low_volume")
    if clip_ratio > 0.01:
        review.append("heavy_clipping")
    if zero_ratio > 0.25:
        review.append("many_zero_samples")

    max_rate = 8.0 if language == "en" else 12.0
    if units and rate < 0.35:
        review.append("label_rate_too_low")
    if rate > max_rate:
        review.append("label_rate_too_high")
    if sample_rate < 8000:
        review.append("low_sample_rate")

    flags = hard + review
    decision = "reject" if hard else ("review" if review else "keep")
    return {"decision": decision, "flags": flags, **metrics}


def _duration_bucket(duration):
    if duration < 0.5:
        return "0-0.5s"
    if duration < 1.0:
        return "0.5-1s"
    if duration < 1.5:
        return "1-1.5s"
    return "1.5-2s"


def screen_parquets(inputs, out_path, report_path, max_duration=2.0):
    files = sorted({path for pattern in inputs for path in glob.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No parquet matched: {inputs}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    decisions = collections.Counter()
    languages = collections.Counter()
    buckets = collections.Counter()
    flags = collections.Counter()
    hours = collections.Counter()
    seen = 0
    selected = 0

    columns = [
        "id", "audio", "transcription", "duration", "language",
        "original_speaker_id", "session_id", "topic",
    ]
    with open(out_path, "w", encoding="utf-8") as dst:
        for source in files:
            parquet = pq.ParquetFile(source)
            row_index = 0
            for batch in parquet.iter_batches(batch_size=128, columns=columns):
                names = batch.schema.names
                for offset in range(batch.num_rows):
                    seen += 1
                    duration = float(batch.column(names.index("duration"))[offset].as_py() or 0)
                    if duration > max_duration:
                        row_index += 1
                        continue
                    values = {
                        name: batch.column(names.index(name))[offset].as_py()
                        for name in names
                    }
                    audio = values.pop("audio") or {}
                    result = inspect_audio(
                        audio.get("bytes") or b"",
                        duration,
                        values.get("language") or "",
                        values.get("transcription") or "",
                    )
                    record = {
                        "id": values.pop("id"),
                        "source_dataset": "ASCEND",
                        "source_split": "train",
                        "source_parquet": os.path.relpath(source),
                        "source_row": row_index,
                        "embedded_audio_path": audio.get("path"),
                        **values,
                        **result,
                    }
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")

                    selected += 1
                    decision = result["decision"]
                    language = record.get("language") or "unknown"
                    decisions[decision] += 1
                    languages[(decision, language)] += 1
                    buckets[(decision, _duration_bucket(result.get("actual_duration", duration)))] += 1
                    hours[decision] += result.get("actual_duration", duration) / 3600
                    flags.update(result["flags"])
                    row_index += 1

    report = {
        "source_files": [os.path.relpath(path) for path in files],
        "rows_scanned": seen,
        "rows_at_or_under_max_duration": selected,
        "max_duration_s": max_duration,
        "decision_counts": dict(sorted(decisions.items())),
        "decision_hours": {key: round(value, 4) for key, value in sorted(hours.items())},
        "language_counts": {
            f"{decision}/{language}": count
            for (decision, language), count in sorted(languages.items())
        },
        "duration_bucket_counts": {
            f"{decision}/{bucket}": count
            for (decision, bucket), count in sorted(buckets.items())
        },
        "flag_counts": dict(flags.most_common()),
        "notes": [
            "keep is automatic screening only, not a substitute for human label review",
            "review rows need listening; flags do not prove that a sample is unusable",
            "background speech, overlap, reverberation and semantic label correctness are not automatically verified",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as dst:
        json.dump(report, dst, ensure_ascii=False, indent=2)
        dst.write("\n")
    return report


def screen_med_it(root, out_path, report_path, max_duration=2.0):
    """Screen MED-IT train tar members without extracting the complete corpus."""
    transcript_path = os.path.join(root, "train.txt")
    transcripts = {}
    with open(transcript_path, encoding="utf-8") as src:
        for line in src:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1].strip()

    tar_paths = sorted(glob.glob(os.path.join(root, "train", "*.tar")))
    if not tar_paths:
        raise FileNotFoundError(f"No MED-IT train tar found under: {root}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    decisions = collections.Counter()
    buckets = collections.Counter()
    flags = collections.Counter()
    hours = collections.Counter()
    seen = 0
    selected = 0

    with open(out_path, "w", encoding="utf-8") as dst:
        for tar_path in tar_paths:
            with tarfile.open(tar_path) as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith(".wav"):
                        continue
                    seen += 1
                    uid = os.path.splitext(os.path.basename(member.name))[0]
                    audio_file = archive.extractfile(member)
                    try:
                        info = sf.info(audio_file)
                        duration = info.frames / info.samplerate
                    except Exception:
                        duration = 0.0
                    if duration > max_duration:
                        continue
                    audio_file.seek(0)
                    result = inspect_audio(
                        audio_file.read(), duration, "en", transcripts.get(uid, "")
                    )
                    record = {
                        "id": uid,
                        "source_dataset": "MED-IT",
                        "source_split": "train",
                        "source_tar": os.path.relpath(tar_path),
                        "tar_member": member.name,
                        "speaker": uid.split("-")[0],
                        "transcription": transcripts.get(uid, ""),
                        "duration": round(duration, 4),
                        "language": "en",
                        **result,
                    }
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")

                    selected += 1
                    decision = result["decision"]
                    decisions[decision] += 1
                    buckets[(decision, _duration_bucket(result.get("actual_duration", duration)))] += 1
                    hours[decision] += result.get("actual_duration", duration) / 3600
                    flags.update(result["flags"])

    report = {
        "source_files": [os.path.relpath(path) for path in tar_paths],
        "rows_scanned": seen,
        "rows_at_or_under_max_duration": selected,
        "max_duration_s": max_duration,
        "decision_counts": dict(sorted(decisions.items())),
        "decision_hours": {key: round(value, 4) for key, value in sorted(hours.items())},
        "duration_bucket_counts": {
            f"{decision}/{bucket}": count
            for (decision, bucket), count in sorted(buckets.items())
        },
        "flag_counts": dict(flags.most_common()),
        "notes": [
            "MED-IT is English medical dialogue; use it as a minority domain component",
            "keep is automatic screening only, not a substitute for human label review",
            "background speech, overlap, reverberation and semantic label correctness are not automatically verified",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as dst:
        json.dump(report, dst, ensure_ascii=False, indent=2)
        dst.write("\n")
    return report


def write_balanced_keep(candidate_path, out_path, max_per_transcript=20, seed=42):
    """Write deterministic keep-only rows while capping repeated exact labels."""
    groups = collections.defaultdict(list)
    with open(candidate_path, encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            if row.get("decision") != "keep":
                continue
            key = re.sub(r"\s+", " ", (row.get("transcription") or "").strip().lower())
            groups[key].append(row)

    selected = []
    for rows in groups.values():
        rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('source_dataset')}:{row.get('id')}:"
            f"{row.get('source_parquet') or row.get('source_tar')}".encode("utf-8")
        ).hexdigest())
        selected.extend(rows[:max_per_transcript])
    selected.sort(key=lambda row: (
        row.get("source_parquet") or row.get("source_tar") or "",
        str(row.get("source_row", row.get("tar_member") or "")),
    ))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as dst:
        for row in selected:
            dst.write(json.dumps({**row, "selection": "balanced_keep"}, ensure_ascii=False) + "\n")
    return {
        "rows": len(selected),
        "hours": round(sum(row.get("actual_duration", 0) for row in selected) / 3600, 4),
        "unique_transcripts": len(groups),
        "max_per_transcript": max_per_transcript,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", action="append", help="ASCEND parquet glob; repeatable")
    source.add_argument("--med-it-root", help="MED-IT root containing train.txt and train/*.tar")
    parser.add_argument("--out", required=True, help="Metadata-only candidate JSONL")
    parser.add_argument("--report", required=True, help="Aggregate JSON report")
    parser.add_argument("--balanced-out", help="Optional keep-only, transcript-capped JSONL")
    parser.add_argument("--max-per-transcript", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=2.0)
    args = parser.parse_args()
    if args.med_it_root:
        report = screen_med_it(
            args.med_it_root, args.out, args.report, args.max_duration
        )
    else:
        report = screen_parquets(args.input, args.out, args.report, args.max_duration)
    if args.balanced_out:
        report["balanced_keep"] = write_balanced_keep(
            args.out, args.balanced_out, args.max_per_transcript, args.seed
        )
        with open(args.report, "w", encoding="utf-8") as dst:
            json.dump(report, dst, ensure_ascii=False, indent=2)
            dst.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
