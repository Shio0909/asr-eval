#!/usr/bin/env python3
"""Score OmniSTEval-resegmented long-form translations with remote XCOMET."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from scorer_client import scorer_last_error, xcomet_score  # noqa: E402


def _dataset_name(result_path: Path) -> str:
    return result_path.stem.split("__", 1)[0]


def _load_items(result_path: Path, arm: str) -> list[dict]:
    dataset = _dataset_name(result_path)
    omni_dir = result_path.with_name(result_path.stem + "__omnisteval")
    instances_path = omni_dir / "scores" / "instances.resegmented.jsonl"
    sources_path = omni_dir / "sources.txt"
    segments_path = omni_dir / "speech_segments.json"
    if not instances_path.exists() or not sources_path.exists():
        return []

    instances = [
        json.loads(line) for line in instances_path.read_text().splitlines() if line
    ]
    sources = sources_path.read_text().splitlines()
    if len(instances) != len(sources):
        raise ValueError(
            f"source/instance count mismatch for {dataset}: "
            f"{len(sources)} != {len(instances)}"
        )
    recordings = json.loads(segments_path.read_text()) if segments_path.exists() else []

    items = []
    for source, instance in zip(sources, instances, strict=True):
        docid = int(instance["docid"])
        recording_id = str(docid)
        if docid < len(recordings):
            recording_id = Path(recordings[docid].get("wav", recording_id)).stem
        items.append(
            {
                "id": f"{dataset}|{arm}|{docid}|{instance['segid']}",
                "src": source,
                "ref": instance["reference"],
                "hyp": instance["prediction"],
                "dataset": dataset,
                "arm": arm,
                "docid": docid,
                "segid": int(instance["segid"]),
                "recording_id": recording_id,
                "ref_chars": len(instance["reference"]),
            }
        )
    return items


def _summarize(items: list[dict], response: dict) -> dict:
    response_by_id = {
        str(item["id"]): item for item in response.get("items") or []
    }
    recording_scores: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    dataset_scores: dict[str, list[tuple[float, int]]] = defaultdict(list)
    severities: Counter[str] = Counter()
    for item in items:
        scored = response_by_id[item["id"]]
        score = float(scored["score"])
        key = (item["dataset"], item["recording_id"])
        recording_scores[key].append((score, item["ref_chars"]))
        dataset_scores[item["dataset"]].append((score, item["ref_chars"]))
        severities.update(
            span.get("severity", "unknown")
            for span in scored.get("error_spans") or []
        )

    datasets = {}
    recording_means_by_dataset: dict[str, list[float]] = defaultdict(list)
    for (dataset, _recording), values in recording_scores.items():
        recording_means_by_dataset[dataset].append(
            statistics.mean(score for score, _weight in values)
        )
    for dataset, values in dataset_scores.items():
        weight_total = sum(weight for _score, weight in values)
        datasets[dataset] = {
            "recording_macro": statistics.mean(recording_means_by_dataset[dataset]),
            "char_weighted": (
                sum(score * weight for score, weight in values) / weight_total
                if weight_total else None
            ),
            "segment_micro": statistics.mean(score for score, _weight in values),
            "n_segments": len(values),
            "n_recordings": len(recording_means_by_dataset[dataset]),
        }
    return {
        "dataset_macro_recording_mean": statistics.mean(
            value["recording_macro"] for value in datasets.values()
        ),
        "all_recordings_mean": statistics.mean(
            score
            for scores in recording_means_by_dataset.values()
            for score in scores
        ),
        "datasets": datasets,
        "severity_counts": dict(severities),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--datasets", nargs="+", default=[
        "realsi_en_zh_n5", "realsi_zh_en_n5", "bstc_zh_en_n5"
    ])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    if not os.environ.get("EVAL_SCORER_URL"):
        raise SystemExit("EVAL_SCORER_URL is required")
    items = []
    for path in sorted(args.results_dir.glob("*.json")):
        if _dataset_name(path) in args.datasets:
            items.extend(_load_items(path, args.arm))
    if not items:
        raise SystemExit("no resegmented items found")

    request = {
        "items": [
            {key: item[key] for key in ("id", "src", "ref", "hyp")}
            for item in items
        ],
        "batch_size": args.batch_size,
    }
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_name(args.out_prefix.name + "_index.json").write_text(
        json.dumps({"method": "official reference segments + OmniSTEval resegmented hypotheses", "items": items}, ensure_ascii=False, indent=2)
    )
    args.out_prefix.with_name(args.out_prefix.name + "_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2)
    )

    response = xcomet_score(
        [(item["src"], item["ref"], item["hyp"]) for item in items],
        batch_size=args.batch_size,
    )
    if response is None:
        raise SystemExit(f"XCOMET failed: {scorer_last_error()}")
    response_items = []
    for item, score, spans in zip(
        items, response["scores"], response["error_spans"], strict=True
    ):
        response_items.append(
            {"id": item["id"], "score": score, "error_spans": spans}
        )
    raw_response = {
        "metric": "xcomet",
        "model": response.get("model"),
        "system_score": response.get("system_score"),
        "items": response_items,
    }
    args.out_prefix.with_name(args.out_prefix.name + "_response.json").write_text(
        json.dumps(raw_response, ensure_ascii=False, indent=2)
    )
    summary = {
        "metric": "xcomet",
        "model": response.get("model"),
        "arm": args.arm,
        "n_items": len(items),
        "summary": _summarize(items, raw_response),
    }
    args.out_prefix.with_name(args.out_prefix.name + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
