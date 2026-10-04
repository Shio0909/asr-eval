"""同传流式事件的统一记录与 OmniSTEval JSONL 导出。

适配器只记录服务端真实返回的 partial / commit 事件；这里负责把不同厂商的
增量、累积修订和段定稿统一成可回放 trace。官方指标由 OmniSTEval 计算，
本模块不重复实现 YAAL/LongYAAL。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field


_CJK_LANGS = {"zh", "yue", "ja", "ko", "中文", "简体中文", "日文", "韩文"}


def _join_segments(left: str, right: str) -> str:
    left, right = str(left or "").strip(), str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def _common_prefix_len(a, b) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _is_char_level(text: str, target_lang: str = "") -> bool:
    lang = str(target_lang or "").lower()
    return lang in _CJK_LANGS or any("\u3400" <= c <= "\u9fff" for c in str(text or ""))


def _units(text: str, target_lang: str = "") -> list[str]:
    if _is_char_level(text, target_lang):
        return [c for c in str(text or "") if not c.isspace()]
    # 输出显式空格化的词/标点，避免 OmniSTEval 的 Moses tokenizer 再切标点后
    # prediction 单元数与 delays 数组不一致。
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", str(text or ""), flags=re.UNICODE)


@dataclass
class SimultTraceRecorder:
    """记录真实流式 UI 状态，不合成厂商没有提供的 token 时间。"""

    target_lang: str = ""
    granularity: str = "token"
    events: list[dict] = field(default_factory=list)
    committed_text: str = ""
    draft_text: str = ""

    def _event(self, op: str, text: str, source_s: float, wall_s: float | None):
        event = {
            "seq": len(self.events),
            "op": op,
            "source_s": round(float(source_s or 0.0), 3),
            "text": str(text or ""),
        }
        if wall_s is not None:
            event["wall_s"] = round(float(wall_s), 3)
        self.events.append(event)

    def append_partial(self, text: str, source_s: float, wall_s: float | None):
        if not text:
            return
        self.draft_text += str(text)
        self._event("append", text, source_s, wall_s)

    def replace_partial(self, text: str, source_s: float, wall_s: float | None):
        text = str(text or "")
        if text == self.draft_text:
            return
        self.draft_text = text
        self._event("replace", text, source_s, wall_s)

    def commit(self, text: str, source_s: float, wall_s: float | None):
        text = str(text or "").strip()
        if not text:
            return
        self.committed_text = _join_segments(self.committed_text, text)
        self.draft_text = ""
        self._event("commit", text, source_s, wall_s)

    def finish(self, final_text: str, source_s: float, wall_s: float | None) -> dict:
        final_text = str(final_text or "").strip()
        # final 是权威输出；即使和最后一次 commit 相同也保留，确保末 token 有明确终态。
        self._event("final", final_text, source_s, wall_s)
        return {
            "schema_version": 1,
            "clock_unit": "second",
            "target_lang": self.target_lang,
            "granularity": self.granularity,
            "final_text": final_text,
            "events": self.events,
            "summary": trace_summary({"final_text": final_text, "events": self.events}),
        }


def replay_trace(trace: dict) -> list[dict]:
    """回放压缩事件，返回每一步的屏幕全文和已定稿全文。"""
    committed = draft = ""
    out = []
    for raw in trace.get("events") or []:
        event = dict(raw)
        op, text = event.get("op"), str(event.get("text") or "")
        if op == "append":
            draft += text
        elif op == "replace":
            draft = text
        elif op == "commit":
            committed = _join_segments(committed, text)
            draft = ""
        elif op == "final":
            committed, draft = text.strip(), ""
        else:
            continue
        event["display_text"] = _join_segments(committed, draft)
        event["committed_text"] = committed
        out.append(event)
    return out


def trace_summary(trace: dict) -> dict:
    replayed = replay_trace(trace)
    first_visible = next((e.get("wall_s") for e in replayed if e.get("display_text")), None)
    first_stable = next((e.get("wall_s") for e in replayed if e.get("committed_text")), None)
    old_draft = ""
    updates = revisions = revised_chars = 0
    for event in trace.get("events") or []:
        op, text = event.get("op"), str(event.get("text") or "")
        if op == "append":
            updates += 1
            old_draft += text
        elif op == "replace":
            updates += 1
            prefix = _common_prefix_len(old_draft, text)
            if prefix < len(old_draft):
                revisions += 1
                revised_chars += len(old_draft) - prefix
            old_draft = text
        elif op in {"commit", "final"}:
            old_draft = ""
    final_chars = len(str(trace.get("final_text") or "").replace(" ", ""))
    observable = updates > 0
    return {
        "ttfb_visible_s": round(first_visible, 3) if first_visible is not None else None,
        "ttfb_stable_s": round(first_stable, 3) if first_stable is not None else None,
        "update_events": updates,
        "revision_events": revisions,
        "stability_observable": observable,
        "revision_rate": round(revisions / updates, 4) if observable else None,
        "revised_chars": revised_chars,
        "churn_rate": round(revised_chars / final_chars, 4)
                      if observable and final_chars else None,
    }


def omnisteval_record(source: str, trace: dict, source_length_s: float,
                      timing: str = "stable") -> dict:
    """生成 OmniSTEval/SimulEval JSONL 单条记录。

    ``stable`` 只使用厂商明确 commit/final 的文本；``visible`` 使用屏幕临时文本。
    delays=已消费源音频(CU)，elapsed=客户端墙钟(CA)，均为毫秒。
    """
    if timing not in {"stable", "visible"}:
        raise ValueError("timing 只支持 stable/visible")
    final_text = str(trace.get("final_text") or "").strip()
    target_lang = trace.get("target_lang") or ""
    final_units = _units(final_text, target_lang)
    char_level = _is_char_level(final_text, target_lang)
    delays = [None] * len(final_units)
    elapsed = [None] * len(final_units)
    for event in replay_trace(trace):
        snapshot = event["committed_text" if timing == "stable" else "display_text"]
        prefix = _common_prefix_len(_units(snapshot, target_lang), final_units)
        for i in range(prefix):
            if delays[i] is None:
                delays[i] = round(float(event.get("source_s") or 0.0) * 1000, 3)
                if event.get("wall_s") is not None:
                    # 真 1x 推流按帧发送时，发送线程会在本帧 sleep 前更新 source_s，
                    # 墙钟可能因一个帧宽的竞态略小于 CU。CA 必须包含而不能早于已消费源时间。
                    elapsed[i] = max(delays[i], round(float(event["wall_s"]) * 1000, 3))
    if final_units and any(v is None for v in delays):
        raise ValueError("trace 终态无法覆盖 final_text 的全部评测单元")
    record = {
        "source": os.path.basename(source),
        # OmniSTEval 的 char-level loader 直接 ``list(prediction)``，空格也会被
        # 当作评测单元；本模块的字符单元明确排除空白，所以导出时必须同步去掉。
        "prediction": "".join(final_units) if char_level else " ".join(final_units),
        "delays": delays,
        "source_length": round(float(source_length_s or 0.0) * 1000, 3),
        "timing_basis": timing,
        "trace_schema_version": trace.get("schema_version", 1),
        "emission_granularity": trace.get("granularity", "unknown"),
        "evaluation_unit": "char" if char_level else "word",
    }
    if elapsed and all(v is not None for v in elapsed):
        record["elapsed"] = elapsed
    return record


def legacy_trace(final_text: str, segments: list[dict], target_lang: str = "") -> dict | None:
    """旧结果只有段定稿源时钟：可导出 CU，绝不伪造 CA。"""
    recorder = SimultTraceRecorder(target_lang=target_lang, granularity="segment-final")
    for segment in segments or []:
        text = segment.get("text")
        source_s = segment.get("emitted_at_s", segment.get("end_s"))
        if text and source_s is not None:
            recorder.commit(text, source_s, None)
    if not recorder.events:
        return None
    return recorder.finish(final_text, max(e["source_s"] for e in recorder.events), None)


def export_infer(infer_path: str, manifest_path: str, out_path: str,
                 timing: str = "stable") -> dict:
    manifest = {}
    with open(manifest_path, encoding="utf-8") as src:
        for line in src:
            if line.strip():
                row = json.loads(line)
                manifest[row["id"]] = row
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as dst, open(infer_path, encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            if row.get("id") == "__meta__" or not row.get("ok"):
                continue
            item = manifest.get(row.get("id")) or {}
            extra = row.get("extra") or {}
            trace = extra.get("simult_trace") or legacy_trace(
                row.get("hyp", ""), extra.get("translation_segments") or [],
                item.get("target_lang") or item.get("lang", "").split("-")[-1],
            )
            if not trace:
                skipped += 1
                continue
            record = omnisteval_record(
                item.get("audio_path") or row.get("id", "audio"), trace,
                extra.get("src_dur_s") or row.get("audio_s") or 0.0, timing,
            )
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return {"written": written, "skipped": skipped, "out": out_path, "timing": timing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer JSONL → OmniSTEval/SimulEval JSONL")
    parser.add_argument("--infer", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timing", choices=("stable", "visible"), default="stable")
    args = parser.parse_args()
    print(json.dumps(export_infer(args.infer, args.manifest, args.out, args.timing),
                     ensure_ascii=False, indent=2))
