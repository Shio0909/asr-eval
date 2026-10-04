"""Dataset-side exclusion policies and non-destructive reviewed ASR metrics."""

import hashlib
import json
import os


def load_exclusion_policy(path):
    """Load one dataset's exclusions.json; a missing file means no exclusions."""
    if not path or not os.path.exists(path):
        return {"samples": {}, "speakers": {}}
    with open(path, encoding="utf-8") as src:
        data = json.load(src)
    data.setdefault("samples", {})
    data.setdefault("speakers", {})
    data["_path"] = os.path.abspath(path)
    return data


def _speaker_id(row):
    if row.get("speaker"):
        return str(row["speaker"])
    path = str(row.get("audio_path") or "").replace("\\", "/")
    return path.rsplit("/", 2)[-2] if path.count("/") >= 2 else ""


def exclusion_for_row(row, policy):
    """Return the matching exclusion record, preferring a sample-specific record."""
    sample = (policy.get("samples") or {}).get(str(row.get("id")))
    if sample and sample.get("action", "exclude") == "exclude":
        return {**sample, "matched_by": "sample"}
    speaker_id = _speaker_id(row)
    speaker = (policy.get("speakers") or {}).get(speaker_id)
    if speaker and speaker.get("action", "exclude") == "exclude":
        return {**speaker, "matched_by": "speaker", "speaker": speaker_id}
    return None


def _policy_path_for_row(row, root):
    audio = row.get("audio_path")
    if not audio:
        return None
    path = audio if os.path.isabs(audio) else os.path.join(root, audio)
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace("\\", "/")
    parts = rel.split("/")
    if len(parts) >= 3 and parts[0] == "datasets":
        central = os.path.join(root, "datasets", "exclusions", parts[2] + ".json")
        if os.path.exists(central):
            return central
    current = os.path.dirname(os.path.abspath(path))
    datasets_root = os.path.abspath(os.path.join(root, "datasets"))
    while current == datasets_root or current.startswith(datasets_root + os.sep):
        candidate = os.path.join(current, "exclusions.json")
        if os.path.exists(candidate):
            return candidate
        if current == datasets_root:
            break
        current = os.path.dirname(current)
    return None


def exclusions_for_rows(rows, root):
    """Resolve dataset sidecars from manifest audio paths and return id -> evidence."""
    policies = {}
    excluded = {}
    for row in rows:
        path = _policy_path_for_row(row, root)
        if not path:
            continue
        policy = policies.setdefault(path, load_exclusion_policy(path))
        record = exclusion_for_row(row, policy)
        if record:
            excluded[str(row.get("id"))] = {**record, "policy": os.path.relpath(path, root)}
    return excluded


def reviewed_asr_metrics(summary, samples, excluded):
    """Calculate a second CER/WER after exclusions without changing original totals."""
    excluded_ids = {str(x) for x in excluded}
    affected = [s for s in samples if str(s.get("id")) in excluded_ids]
    if not affected:
        return None
    kept = [s for s in samples
            if str(s.get("id")) not in excluded_ids and s.get("ref_len") is not None]
    ref_len = sum(s.get("ref_len") or 0 for s in kept)
    edits = sum((s.get("sub") or 0) + (s.get("del") or 0) + (s.get("ins") or 0) for s in kept)
    metric = summary.get("metric") or "CER"
    signature_input = json.dumps(
        {sid: excluded[sid] for sid in sorted(excluded_ids & {str(s.get('id')) for s in samples})},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    value = round(edits / ref_len, 4) if ref_len else None
    return {
        "metric": metric,
        metric: value,
        "err_rate": value,
        "n_total": max((summary.get("n_total") or len(samples)) - len(affected), 0),
        "n_scored": len(kept),
        "n_excluded": len(affected),
        "excluded_ids": sorted(str(s.get("id")) for s in affected),
        "total_ref_len": ref_len,
        "total_edits": edits,
        "exclusion_signature": hashlib.sha256(signature_input.encode("utf-8")).hexdigest()[:12],
    }
