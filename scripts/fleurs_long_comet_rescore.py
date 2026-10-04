#!/usr/bin/env python3
"""Re-score the translation arms of a concatenated-FLEURS streaming report with COMET.

The source report scores translation with 译中CER (character edit distance between
the model's Chinese output and the human Chinese reference).  That metric only
rewards literal wording and, on 180 s concatenated clips, cannot be computed as a
single unit by COMET either: every clip is 790-2187 XLM-R tokens, well past the
512-token limit of wmt22-comet-da.

So this script re-segments each clip into translation units and scores every unit:

* the human Chinese reference carries one *missing space* at each FLEURS sentence
  join (chars are otherwise space-separated), which recovers real sentence
  boundaries;
* the model's Chinese output and the source transcript are cut at the same
  relative positions, snapped to the nearest word boundary for the source.

Two source variants are scored, because they answer different questions:

``gold``  src = human source transcript -> end-to-end quality as delivered
``asr``   src = model's own transcript  -> MT quality given what the system heard

Usage:
    .venv-comet/bin/python scripts/fleurs_long_comet_rescore.py \
        --report ./report.html --out scores.json
"""

from __future__ import annotations

import argparse
import html as H
import json
import re
import statistics
from pathlib import Path

CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))
MIN_SEG_CHARS = 24  # merge segments shorter than this; COMET degenerates on tiny inputs


def _is_cjk(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in CJK)


def parse_report(path: Path) -> list[dict]:
    """Pull section 5 (逐条明细) out of the report HTML."""
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"data:[a-z0-9/+;=,\-]{200,}", "[AUDIO]", raw)
    section = raw[raw.find("5. 逐条明细"):]
    rows = re.findall(r"<tr([^>]*)>(.*?)</tr>", section, re.S)[1:]

    def text(fragment: str) -> str:
        fragment = re.sub(r"<br\s*/?>", "\n", fragment)
        return H.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()

    def labelled(fragment: str) -> dict:
        return {
            text(m.group(1)): text(m.group(2))
            for m in re.finditer(
                r'<div class="lab">(.*?)</div><div class="txt">(.*?)</div>', fragment, re.S
            )
        }

    out = []
    for attrs, row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        head = [x for x in text(cells[0]).split("\n") if x]
        verdict = [x for x in text(cells[1]).split("\n") if x]
        metrics = text(cells[2])
        src_labels = labelled(cells[3])
        mt_labels = labelled(cells[4])

        def num(pattern: str) -> float | None:
            hit = re.search(pattern, metrics)
            return float(hit.group(1)) if hit else None

        f1 = re.search(r"F1 ([\d.]+)/([\d.]+)", metrics)
        out.append({
            "lang_name": head[0],
            "id": head[1],
            "dur_desc": head[2] if len(head) > 2 else "",
            "grade": verdict[0] if verdict else "",
            "comment": " ".join(verdict[1:]),
            "cls": (re.search(r'class="([^"]*)"', attrs) or [None, ""])[1],
            "report_cer": num(r"CER ([\d.]+)%"),
            "report_wer": num(r"WER ([\d.]+)%"),
            "report_zh_cer": num(r"译中CER ([\d.]+)%"),
            "report_f1_asr": float(f1.group(1)) if f1 else None,
            "report_f1_zh": float(f1.group(2)) if f1 else None,
            "gold_src": src_labels.get("金标源语", ""),
            "hyp_asr": src_labels.get("模型转写", ""),
            "gold_zh": mt_labels.get("金标中文", ""),
            "hyp_mt": mt_labels.get("模型译文", ""),
        })
    return out


def ref_spans(gold_zh: str) -> list[tuple[int, int]]:
    """Char spans of the space-stripped reference, split at sentence joins."""
    compact = re.sub(r"\s+", "", gold_zh)
    # The reference spaces every character *within* a sentence, so a FLEURS
    # sentence join shows up as two characters that are adjacent in the raw
    # string.  Both checks must run on the raw string -- comparing consecutive
    # non-space characters instead would flag every pair.
    bounds = [
        k
        for k, i in enumerate(
            (i for i, ch in enumerate(gold_zh) if not ch.isspace())
        )
        if i and not gold_zh[i - 1].isspace()
        and _is_cjk(gold_zh[i - 1]) and _is_cjk(gold_zh[i])
    ]

    # Cut at every sentence join once the accumulated run is long enough; a run
    # of very short sentences folds into the previous unit rather than splitting.
    spans, start = [], 0
    for b in [*bounds, len(compact)]:
        if b - start >= MIN_SEG_CHARS:
            spans.append((start, b))
            start = b
    if start < len(compact):
        spans[-1] = (spans[-1][0], len(compact)) if spans else (start, len(compact))
    return spans


def _align_map(ref: str, hyp: str) -> list[int]:
    """Map every reference char index to the matching hypothesis char index.

    A constant ref:hyp length ratio drifts over a 180 s clip (free translation
    expands and contracts sentence by sentence), which pushes late segment cuts
    off by a whole sentence.  Both sides are Chinese, so a character-level diff
    recovers the correspondence directly; unmatched runs are interpolated.
    """
    import difflib

    mapping = [0] * (len(ref) + 1)
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        else:
            span_ref, span_hyp = i2 - i1, j2 - j1
            for k in range(span_ref):
                mapping[i1 + k] = j1 + (span_hyp * k // span_ref if span_ref else 0)
    mapping[len(ref)] = len(hyp)
    return mapping


def _cut_at_word(text: str, target: int) -> int:
    """Nearest whitespace to ``target`` in ``text`` (source speech has real spaces)."""
    if target <= 0 or target >= len(text):
        return max(0, min(len(text), target))
    left = text.rfind(" ", 0, target + 1)
    right = text.find(" ", target)
    if left < 0:
        return right if right >= 0 else target
    if right < 0:
        return left
    return left if target - left <= right - target else right


def segment(row: dict) -> list[dict]:
    """Cut (src, ref, hyp) of one clip into aligned translation units."""
    ref_compact = re.sub(r"\s+", "", row["gold_zh"])
    hyp_compact = re.sub(r"\s+", "", row["hyp_mt"])
    src = row["gold_src"]
    total = len(ref_compact)
    if not total:
        return []

    spans = ref_spans(row["gold_zh"])
    hyp_at = _align_map(ref_compact, hyp_compact)

    # The source transcript and the model's own transcript are both in the source
    # language, so neither can be diffed against the Chinese reference; cut them
    # at the same reference-relative positions, snapped to a word boundary.
    tsr = row["hyp_asr"].strip()
    src_cuts = _source_cuts(src, spans, total)
    tsr_cuts = _source_cuts(tsr, spans, total)

    units = []
    for i, (start, end) in enumerate(spans):
        h0, h1 = hyp_at[start], hyp_at[end]
        units.append({
            "ref": ref_compact[start:end],
            "hyp": hyp_compact[h0:h1],
            "gold_src": src[src_cuts[i]:src_cuts[i + 1]].strip(),
            "hyp_asr": tsr[tsr_cuts[i]:tsr_cuts[i + 1]].strip(),
        })
    return units


def _source_cuts(text: str, spans: list[tuple[int, int]], total: int) -> list[int]:
    """Cut positions in ``text`` matching the reference spans' relative offsets."""
    cuts = [0]
    for _, end in spans[:-1]:
        cuts.append(_cut_at_word(text, round(len(text) * end / total)))
    cuts.append(len(text))
    # keep the cuts monotone; word snapping can otherwise collide on short units
    return [max(a, b) for a, b in zip(cuts, [cuts[0]] + cuts[:-1])]


def rescore(rows: list[dict], batch: int, device: str) -> None:
    import torch
    from comet import download_model, load_from_checkpoint

    checkpoint = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(checkpoint)
    if device:
        model.to(device)

    jobs = {"gold": [], "asr": []}
    index = {"gold": [], "asr": []}
    for row in rows:
        units = segment(row)
        row["n_units"] = len(units)
        for variant in ("gold", "asr"):
            for i, unit in enumerate(units):
                src = unit["gold_src"] if variant == "gold" else unit["hyp_asr"]
                jobs[variant].append({"src": src, "ref": unit["ref"], "mt": unit["hyp"]})
                index[variant].append((row["id"], i))

    for variant, samples in jobs.items():
        print(f"[{variant}] {len(samples)} units", flush=True)
        result = model.predict(
            samples, batch_size=batch, gpus=0, accelerator=device or "cpu",
            num_workers=1, progress_bar=False, length_batching=True,
        )
        scores = result["scores"] if isinstance(result, dict) else result.scores
        per_row: dict[str, list[float]] = {}
        for (row_id, _), score in zip(index[variant], scores):
            per_row.setdefault(row_id, []).append(float(score))
        for row in rows:
            values = per_row.get(row["id"]) or []
            row[f"comet_{variant}"] = statistics.mean(values) if values else None
            row[f"comet_{variant}_units"] = [round(v, 4) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps", help="cpu / mps / cuda")
    parser.add_argument("--no-score", action="store_true", help="only re-segment, skip COMET")
    args = parser.parse_args()

    rows = parse_report(args.report)
    print(f"parsed {len(rows)} rows")
    if args.no_score:
        for row in rows:
            row["comet_gold"] = row["comet_asr"] = None
            row["n_units"] = len(segment(row))
    else:
        rescore(rows, args.batch, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
