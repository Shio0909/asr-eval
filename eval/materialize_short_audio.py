"""Materialize screened short-audio candidates for Qwen3-ASR fine-tuning.

The source candidate JSONL files point into embedded-audio parquet shards or
tar archives.  This tool extracts only selected WAV files, creates a
speaker-disjoint train/validation split, and writes the JSONL format expected
by Qwen3-ASR's official fine-tuning script.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import tarfile

import pyarrow.parquet as pq


def _stable_fraction(value, seed):
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _speaker_group(row):
    speaker = row.get("original_speaker_id") or row.get("speaker") or row["id"]
    return f"{row['source_dataset']}:{speaker}"


def split_rows(rows, validation_ratio=0.05, seed=42):
    groups_by_dataset = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        groups_by_dataset[row["source_dataset"]][_speaker_group(row)].append(row)

    train = []
    validation = []
    for dataset, groups in groups_by_dataset.items():
        group_names = sorted(
            groups,
            key=lambda group: _stable_fraction(f"{dataset}:{group}", seed),
        )
        validation_groups = 0
        if validation_ratio > 0 and len(group_names) > 1:
            validation_groups = max(1, round(len(group_names) * validation_ratio))
            validation_groups = min(validation_groups, len(group_names) - 1)
        selected = set(group_names[:validation_groups])
        for group, group_rows in groups.items():
            (validation if group in selected else train).extend(group_rows)
    return train, validation


def _duration_bucket(duration):
    if duration < 0.5:
        return "0-0.5s"
    if duration < 1.0:
        return "0.5-1s"
    if duration < 1.5:
        return "1-1.5s"
    return "1.5-2s"


def _qwen_language(row):
    return "English" if row.get("language") == "en" else "Chinese"


def qwen_record(row, audio_path):
    language = _qwen_language(row)
    transcript = (row.get("transcription") or "").strip()
    return {
        "audio": audio_path,
        "text": f"language {language}<asr_text>{transcript}",
        "prompt": "",
        "source_dataset": row["source_dataset"],
        "source_id": row["id"],
        "source_language": row.get("language"),
        "duration": row.get("actual_duration", row.get("duration")),
        "duration_bucket": _duration_bucket(
            row.get("actual_duration", row.get("duration", 0))
        ),
    }


def select_variant(rows, variant, max_med_it=400, seed=42):
    if variant == "all":
        return list(rows)
    if variant != "zh_focus":
        raise ValueError(f"Unknown variant: {variant}")

    selected = [
        row
        for row in rows
        if row["source_dataset"] == "ASCEND"
        and row.get("language") in {"zh", "mixed"}
    ]
    med_it = [row for row in rows if row["source_dataset"] == "MED-IT"]
    med_it.sort(key=lambda row: _stable_fraction(row["id"], seed))
    selected.extend(med_it[:max_med_it])
    return selected


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _audio_relpath(row):
    dataset = row["source_dataset"].lower().replace("-", "_")
    return os.path.join("wavs", dataset, f"{_safe_name(row['id'])}.wav")


def _write_audio_bytes(output_root, row, audio_bytes):
    relpath = _audio_relpath(row)
    output_path = os.path.join(output_root, relpath)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        with open(output_path, "wb") as dst:
            dst.write(audio_bytes)
    row["materialized_audio"] = relpath


def materialize_parquet_rows(rows, repo_root, output_root):
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["source_dataset"] == "ASCEND":
            grouped[row["source_parquet"]].append(row)

    for source, source_rows in grouped.items():
        wanted = {int(row["source_row"]): row for row in source_rows}
        parquet = pq.ParquetFile(os.path.join(repo_root, source))
        row_index = 0
        for batch in parquet.iter_batches(batch_size=256, columns=["audio"]):
            for offset in range(batch.num_rows):
                row = wanted.get(row_index + offset)
                if row is None:
                    continue
                audio = batch.column(0)[offset].as_py() or {}
                _write_audio_bytes(output_root, row, audio.get("bytes") or b"")
            row_index += batch.num_rows


def materialize_tar_rows(rows, repo_root, output_root):
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["source_dataset"] == "MED-IT":
            grouped[row["source_tar"]].append(row)

    for source, source_rows in grouped.items():
        wanted = {row["tar_member"]: row for row in source_rows}
        with tarfile.open(os.path.join(repo_root, source)) as archive:
            for member_name, row in wanted.items():
                audio_file = archive.extractfile(member_name)
                if audio_file is None:
                    raise FileNotFoundError(f"{source}: {member_name}")
                _write_audio_bytes(output_root, row, audio_file.read())


def _load_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as src:
            rows.extend(json.loads(line) for line in src if line.strip())
    return rows


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(rows):
    languages = collections.Counter()
    buckets = collections.Counter()
    datasets = collections.Counter()
    hours = 0.0
    for row in rows:
        languages[row.get("source_language") or "unknown"] += 1
        buckets[row["duration_bucket"]] += 1
        datasets[row["source_dataset"]] += 1
        hours += float(row.get("duration") or 0) / 3600
    return {
        "rows": len(rows),
        "hours": round(hours, 4),
        "datasets": dict(sorted(datasets.items())),
        "languages": dict(sorted(languages.items())),
        "duration_buckets": dict(sorted(buckets.items())),
    }


def build_pack(
    candidate_paths,
    repo_root,
    output_root,
    validation_ratio=0.05,
    max_med_it=400,
    seed=42,
):
    source_rows = _load_rows(candidate_paths)
    materialize_parquet_rows(source_rows, repo_root, output_root)
    materialize_tar_rows(source_rows, repo_root, output_root)

    report = {}
    for variant in ("all", "zh_focus"):
        variant_rows = select_variant(source_rows, variant, max_med_it, seed)
        train, validation = split_rows(variant_rows, validation_ratio, seed)
        rendered_train = [
            qwen_record(row, row["materialized_audio"]) for row in train
        ]
        rendered_validation = [
            qwen_record(row, row["materialized_audio"]) for row in validation
        ]
        _write_jsonl(
            os.path.join(output_root, f"train.{variant}.jsonl"), rendered_train
        )
        _write_jsonl(
            os.path.join(output_root, f"validation.{variant}.jsonl"),
            rendered_validation,
        )
        report[variant] = {
            "train": _summary(rendered_train),
            "validation": _summary(rendered_validation),
        }

    report.update({
        "seed": seed,
        "validation_ratio": validation_ratio,
        "max_med_it_in_zh_focus": max_med_it,
        "qwen_note": (
            "Run training from output_root so relative wavs/... paths resolve. "
            "Dialect metadata should remain separate; Qwen's public forced "
            "language prefixes use Chinese/English rather than Sichuan."
        ),
    })
    with open(os.path.join(output_root, "report.json"), "w", encoding="utf-8") as dst:
        json.dump(report, dst, ensure_ascii=False, indent=2)
        dst.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--max-med-it", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    report = build_pack(
        args.candidates,
        os.path.abspath(args.repo_root),
        os.path.abspath(args.output_root),
        args.validation_ratio,
        args.max_med_it,
        args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
