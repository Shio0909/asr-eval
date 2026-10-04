#!/usr/bin/env python3
"""Evaluate the installed TEN VAD on TEN's released frame labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import wave

import numpy as np
from ten_vad import TenVad


def _read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def _frame_labels(path: Path, hop_size: int) -> np.ndarray:
    values = path.read_text().strip().split(",")[1:]
    labels: list[int] = []
    frame_s = hop_size / 16_000
    for offset in range(0, len(values), 3):
        start_s = float(values[offset])
        end_s = float(values[offset + 1])
        label = int(values[offset + 2])
        labels.extend([label] * round((end_s - start_s) / frame_s))
    return np.asarray(labels, dtype=np.int8)


def _probabilities(path: Path, hop_size: int) -> np.ndarray:
    audio = _read_pcm(path)
    vad = TenVad(hop_size=hop_size, threshold=0.5)
    probabilities = []
    for offset in range(0, len(audio) - hop_size + 1, hop_size):
        probability, _flag = vad.process(audio[offset : offset + hop_size])
        probabilities.append(float(probability))
    return np.asarray(probabilities, dtype=np.float32)


def _load_pair(path: Path, hop_size: int) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _probabilities(path, hop_size)
    labels = _frame_labels(path.with_suffix(".scv"), hop_size)
    frame_count = min(len(probabilities), len(labels))
    # Match TEN's official plot_pr_curves.py alignment: skip the first output.
    return probabilities[1:frame_count], labels[: frame_count - 1]


def _metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    speech = labels == 1
    tp = int(np.sum(predictions & speech))
    fp = int(np.sum(predictions & ~speech))
    fn = int(np.sum(~predictions & speech))
    tn = int(np.sum(~predictions & ~speech))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 2),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fp / (fp + tn), 6) if fp + tn else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 6) if fn + tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _concat(
    files: list[Path],
    pairs: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([pairs[path][0] for path in files]),
        np.concatenate([pairs[path][1] for path in files]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", type=Path)
    parser.add_argument("--hop-size", type=int, default=256, choices=(160, 256))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    files = sorted(args.testset.glob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files found in {args.testset}")
    pairs = {path: _load_pair(path, args.hop_size) for path in files}

    shuffled = files.copy()
    random.Random(args.seed).shuffle(shuffled)
    split = len(shuffled) // 2
    tune_files, eval_files = shuffled[:split], shuffled[split:]
    tune_probabilities, tune_labels = _concat(tune_files, pairs)
    eval_probabilities, eval_labels = _concat(eval_files, pairs)

    thresholds = [value / 100 for value in range(20, 81)]
    tune_metrics = [
        _metrics(tune_probabilities, tune_labels, threshold)
        for threshold in thresholds
    ]
    best_tune = max(tune_metrics, key=lambda item: (item["f1"], item["threshold"]))
    report_thresholds = sorted({0.35, 0.4, 0.45, 0.5, 0.55, best_tune["threshold"]})
    payload = {
        "testset": str(args.testset),
        "files": len(files),
        "hop_size": args.hop_size,
        "seed": args.seed,
        "tune_files": [path.name for path in tune_files],
        "eval_files": [path.name for path in eval_files],
        "tune_frames": len(tune_labels),
        "eval_frames": len(eval_labels),
        "best_tune": best_tune,
        "eval_at_tune_best": _metrics(
            eval_probabilities,
            eval_labels,
            best_tune["threshold"],
        ),
        "eval_selected_thresholds": [
            _metrics(eval_probabilities, eval_labels, threshold)
            for threshold in report_thresholds
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
