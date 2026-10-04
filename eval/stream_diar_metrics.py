"""Metrics for streaming speaker timelines and speaker-attributed TTS.

The offline DER scorer answers whether the final timeline is correct.  This
module preserves every revision and measures the user-visible streaming
contract separately.  It accepts one JSON object per line, either a wrapped
WS event (``{"event": ..., "data": ...}``) or a normalized event object.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL events, ignoring only blank lines and a trailing partial line."""
    events: list[dict[str, Any]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if isinstance(value, dict):
            events.append(value)
    return events


def _event_parts(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str(event.get("event") or event.get("type") or "")
    data = event.get("data")
    if not isinstance(data, dict):
        data = event
    return name, data


def _speaker_snapshot(data: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    rows = []
    for row in data.get("segments") or []:
        if not isinstance(row, dict):
            continue
        try:
            rows.append((str(row["segment_id"]), int(row["start_time"]), int(row["end_time"])))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(rows)


def _speaker_count(data: dict[str, Any]) -> int:
    ids = set()
    for row in data.get("segments") or []:
        if isinstance(row, dict) and row.get("speaker_id") is not None:
            ids.add(str(row["speaker_id"]))
    return len(ids)


def _tts_attributions(events: Iterable[dict[str, Any]]) -> dict[str, list[tuple[int | None, str | None]]]:
    values: dict[str, list[tuple[int | None, str | None]]] = defaultdict(list)
    for event in events:
        name, data = _event_parts(event)
        if name not in {"si_tts_start", "si_tts_file", "si_segment_done"}:
            continue
        key = str(data.get("segment_key") or data.get("segment_id") or "")
        if not key:
            continue
        speaker = data.get("tts_speaker_id", data.get("speaker_id"))
        voice = data.get("tts_voice", data.get("voice"))
        values[key].append((int(speaker) if isinstance(speaker, int) else None, str(voice) if voice else None))
    return values


def evaluate_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return streaming stability and speaker-to-TTS contract metrics.

    ``received_ms`` is preferred for latency; otherwise event order is used
    and latency is reported as unavailable.  Revision rate counts a published
    segment whose snapshot speaker assignment changes in a later snapshot.
    """
    materialized = list(events)
    snapshots: list[tuple[int | None, tuple[tuple[str, int, int], ...], dict[str, Any]]] = []
    text_done: dict[str, dict[str, Any]] = {}
    segment_order: dict[str, list[str]] = defaultdict(list)
    for event in materialized:
        name, data = _event_parts(event)
        if name == "si_speaker_update":
            revision = data.get("revision")
            snapshots.append((int(revision) if isinstance(revision, int) else None, _speaker_snapshot(data), data))
        elif name == "si_text_done":
            key = str(data.get("segment_key") or data.get("segment_id") or "")
            if "received_ms" not in data and isinstance(event.get("received_ms"), (int, float)):
                data = dict(data)
                data["received_ms"] = event["received_ms"]
            if key:
                text_done[key] = data
        if name and data.get("segment_id") is not None:
            key = str(data.get("segment_key") or data["segment_id"])
            segment_order[key].append(name)

    first_assignment: dict[str, str] = {}
    changed: set[str] = set()
    for _revision, snapshot, _data in snapshots:
        for segment_id, _start, _end in snapshot:
            # Speaker IDs may be in a companion assignment list; snapshots
            # themselves are still counted for publication/stability timing.
            first_assignment.setdefault(segment_id, "published")
    # A change is observable from segment_assignments, which carries the
    # public speaker ID for each text/TTS segment.
    assignment_history: dict[str, list[Any]] = defaultdict(list)
    for event in materialized:
        name, data = _event_parts(event)
        if name != "si_speaker_update":
            continue
        for row in data.get("segment_assignments") or []:
            if isinstance(row, dict) and row.get("segment_key"):
                assignment_history[str(row["segment_key"])].append(row.get("speaker_id"))
    for key, values in assignment_history.items():
        if len(set(str(value) for value in values if value is not None)) > 1:
            changed.add(key)

    latencies = []
    for key, data in text_done.items():
        received = data.get("received_ms")
        if isinstance(received, (int, float)):
            latencies.append(float(received))
    key_by_id = {}
    for key, data in text_done.items():
        if data.get("segment_id") is not None:
            key_by_id[str(data["segment_id"])] = key
    normalized_events = []
    for event in materialized:
        name, data = _event_parts(event)
        if name.startswith("si_tts_") and not data.get("segment_key") and data.get("segment_id") is not None:
            event = dict(event)
            payload = dict(data)
            payload["segment_key"] = key_by_id.get(str(data["segment_id"]), str(data["segment_id"]))
            event["data"] = payload
        normalized_events.append(event)
    tts = _tts_attributions(normalized_events)
    attribution_total = 0
    attribution_consistent = 0
    for values in tts.values():
        pairs = {(speaker, voice) for speaker, voice in values if speaker is not None or voice is not None}
        if pairs:
            attribution_total += 1
            if len(pairs) == 1:
                attribution_consistent += 1

    # Older WS events omit segment_key on TTS events; merge those numeric keys
    # into the canonical text_done key before checking lifecycle order.
    merged_order: dict[str, list[str]] = defaultdict(list)
    for key, names in segment_order.items():
        merged_order[key_by_id.get(key, key)].extend(names)
    orders = []
    for names in merged_order.values():
        required = ["si_text_done", "si_tts_start", "si_tts_end", "si_segment_done"]
        positions = [names.index(item) for item in required if item in names]
        orders.append(len(positions) == len(required) and positions == sorted(positions))
    return {
        "speaker_update_count": len(snapshots),
        "published_segment_count": len(first_assignment),
        "published_label_revision_count": len(changed),
        "published_label_revision_rate": round(len(changed) / len(first_assignment), 4) if first_assignment else 0.0,
        "text_done_count": len(text_done),
        "speaker_result_latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "tts_attribution_consistency": round(attribution_consistent / attribution_total, 4) if attribution_total else None,
        "tts_attribution_segments": attribution_total,
        "event_order_correct_rate": round(sum(orders) / len(orders), 4) if orders else None,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score streaming diarization JSONL events")
    parser.add_argument("events")
    args = parser.parse_args()
    print(json.dumps(evaluate_events(load_events(args.events)), ensure_ascii=False, indent=2))
