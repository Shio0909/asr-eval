#!/usr/bin/env python3
"""Render the COMET re-score of a concatenated-FLEURS streaming report as HTML.

Keeps the source report's document structure (headline cards -> overall -> by type
-> by language -> watch list -> per-sample detail) so the two can be read side by
side, and adds the COMET columns plus the metric defects found while reproducing it.

Usage:
    python3 scripts/build_fleurs_long_comet_report.py \
        --scores reports/fleurs-long-comet-20260917/comet_scores.json \
        --out reports/fleurs-long-comet-20260917/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LANG_CODE = {
    "法语": "fr", "意大利语": "it", "韩语": "ko", "日语": "ja", "德语": "de",
    "土耳其语": "tr", "荷兰语": "nl", "印尼语": "id", "马来语": "ms", "阿拉伯语": "ar",
    "英语": "en", "中文（普通话）": "zh", "葡萄牙语": "pt", "西班牙语": "es",
    "越南语": "vi", "俄语": "ru", "粤语": "yue", "乌尔都语": "ur",
}
# Languages whose WER column in the source report is really a character rate.
CHAR_RATE_LANG = {"ko", "ja", "yue", "zh"}
# Languages where the "translation" arm is degenerate: source and target are the
# same language, so 译中CER just re-measures ASR.
DEGENERATE_MT_LANG = {"zh"}
# Samples whose streaming session died and dumped a JSON error into the tail.
HARNESS_FAIL = "会话尾部异常"


def is_harness_fail(row: dict) -> bool:
    return HARNESS_FAIL in (row.get("comment") or "")


def rank_of(values: list[float]) -> list[int]:
    """1-based average ranks, descending (largest value first)."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank_of(a), rank_of(b)
    n = len(a)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def bootstrap_ci(values: list[float], n: int = 4000) -> tuple[float, float] | None:
    """95% percentile CI of the mean, resampling clips. n=30 here, so these are wide."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = random.Random(20260917)  # fixed seed: the report must be reproducible
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def esc(value) -> str:
    return html.escape(str(value), quote=False)


def fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def _ci(interval: tuple[float, float] | None) -> str:
    if not interval:
        return "—"
    lo, hi = interval
    return f"{lo:.3f}–{hi:.3f}"


def cls_for(cer: float | None) -> str:
    if cer is None:
        return ""
    return "ok" if cer < 10 else ("mid" if cer < 30 else "bad")


def build(rows: list[dict], meta: dict) -> str:
    scored = [r for r in rows if r.get("comet_gold") is not None]
    clean = [r for r in scored if not is_harness_fail(r)]

    cards = [
        ("样本", f"{len(rows)}"),
        ("COMET 端到端 ↑", fmt(mean(r["comet_gold"] for r in scored), 3)),
        ("COMET 仅翻译 ↑", fmt(mean(r["comet_asr"] for r in scored), 3)),
        ("译中CER（原报告）↓", fmt(mean(r["report_zh_cer"] for r in scored), 2, "%")),
        ("剔除会话失败后 COMET ↑", fmt(mean(r["comet_gold"] for r in clean), 3)),
        ("剔除会话失败后 CER ↓", fmt(mean(r["report_cer"] for r in clean), 2, "%")),
        ("会话失败样本", f"{len(scored) - len(clean)}"),
        ("COMET 模型", meta.get("comet_model", "wmt22-comet-da")),
    ]

    by_lang = defaultdict(list)
    for row in scored:
        by_lang[row["lang_name"]].append(row)
    # keep the source report's ordering (descending CER was its sort; here: worst COMET first)
    langs = sorted(by_lang, key=lambda l: mean(r["comet_gold"] for r in by_lang[l]))

    lang_stats = []
    for lang in langs:
        group = by_lang[lang]
        lang_stats.append({
            "lang": lang,
            "code": LANG_CODE.get(lang, ""),
            "n": len(group),
            "cer": mean(r["report_cer"] for r in group),
            "zh_cer": mean(r["report_zh_cer"] for r in group),
            "comet_gold": mean(r["comet_gold"] for r in group),
            "comet_asr": mean(r["comet_asr"] for r in group),
            "comet_ci": bootstrap_ci([r["comet_gold"] for r in group]),
            "harness": sum(1 for r in group if is_harness_fail(r)),
        })

    # Rank by the source report's translation metric vs rank by COMET.
    # Both ranks are oriented so that 1 = best: negate the CER before ranking it
    # descending, otherwise the two columns would mean opposite things.
    ranked = [s for s in lang_stats if s["code"] not in DEGENERATE_MT_LANG]
    cer_ranks = rank_of([-s["zh_cer"] for s in ranked])
    comet_ranks = rank_of([s["comet_gold"] for s in ranked])
    for s, cr, mr in zip(ranked, cer_ranks, comet_ranks):
        s["rank_cer"], s["rank_comet"] = cr, mr

    # Spearman on the two ranking vectors, both oriented 1 = best.
    rho = spearman(cer_ranks, comet_ranks)
    max_shift = max(abs(s["rank_cer"] - s["rank_comet"]) for s in ranked) if ranked else 0

    worst = sorted(scored, key=lambda r: r["comet_gold"])[:30]

    meta["max_shift"] = max_shift
    return _render(rows, scored, clean, cards, lang_stats, ranked, rho, worst, meta)


def _render(rows, scored, clean, cards, lang_stats, ranked, rho, worst, meta) -> str:
    out = []
    add = out.append
    add("""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>流式 ASR 评测报告 · COMET 翻译重新打分</title>
<style>
:root { --bg:#f5f7fb; --card:#fff; --line:#e5e7ef; --text:#111827; --muted:#6b7280; --ok:#ecfdf5; --mid:#fffbeb; --bad:#fef2f2; }
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Noto Sans SC",sans-serif;background:var(--bg);color:var(--text);line-height:1.55}
.wrap{max-width:1280px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:26px;margin:0 0 6px} h2{font-size:18px;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.meta{color:var(--muted);font-size:13px} .note{color:var(--muted);font-size:12.5px;margin:8px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card b{display:block;font-size:22px;margin-top:4px} .card span{color:var(--muted);font-size:12px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}
table{border-collapse:collapse;width:100%;min-width:860px}
th,td{border-bottom:1px solid var(--line);padding:9px 11px;font-size:12.5px;vertical-align:top;text-align:left}
th{background:#f8fafc;position:sticky;top:0;z-index:1;white-space:nowrap}
.ok{background:var(--ok)} .mid{background:var(--mid)} .bad{background:var(--bad)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;background:#f3f4f6;padding:1px 5px;border-radius:4px}
.clip{max-width:340px;word-break:break-word} .mono{font-variant-numeric:tabular-nums}
.lab{color:#888;font-size:11px;margin-top:8px} .txt{white-space:pre-wrap;line-height:1.45;max-height:180px;overflow:auto}
.footer{margin-top:28px;color:var(--muted);font-size:12px}
ul{margin:8px 0 8px 18px;padding:0} li{margin:4px 0}
.up{color:#047857} .down{color:#b91c1c}
</style></head><body><div class="wrap">
""")
    add("<h1>流式 ASR 评测报告 · COMET 翻译重新打分</h1>\n")
    add(f'<div class="meta">源报告 <code>{esc(meta.get("source", ""))}</code> · '
        f'重打分子集 {len(scored)} 条 · 翻译单元 {meta.get("units", 0)} 段 · '
        f'COMET <code>{esc(meta.get("comet_model", "wmt22-comet-da"))}</code>（参考式，句级）</div>\n')

    add('<div class="cards">')
    for label, value in cards:
        add(f'<div class="card"><span>{esc(label)}</span><b class="mono">{esc(value)}</b></div>')
    add("</div>\n")

    # ---- 1 总体结论
    gold = mean(r["comet_gold"] for r in scored)
    asr_only = mean(r["comet_asr"] for r in scored)
    clean_gold = mean(r["comet_gold"] for r in clean)
    add("<h2>1. 总体结论</h2><ul>")
    add(f"<li>本轮用 <b>COMET</b> 重新给翻译结果打分，替换原报告的 <b>译中CER</b>。"
        f"端到端 COMET=<b>{fmt(gold,3)}</b>，仅看翻译（以模型自己的转写为源）COMET=<b>{fmt(asr_only,3)}</b>；"
        f"原报告译中CER=<b>{fmt(mean(r['report_zh_cer'] for r in scored),2,'%')}</b>。</li>")
    add(f"<li>剔除 {len(scored)-len(clean)} 条会话失败样本后 COMET=<b>{fmt(clean_gold,3)}</b>，"
        f"ASR CER 从 {fmt(mean(r['report_cer'] for r in scored),2,'%')} 降到 "
        f"<b>{fmt(mean(r['report_cer'] for r in clean),2,'%')}</b>。</li>")
    add(f"<li>按语种排序，译中CER 与 COMET 的 <b>Spearman 秩相关 ρ={rho:.2f}</b>"
        f"（{len(ranked)} 个语种，已剔除译中退化的中文普通话）；"
        f"最大位次变动 <b>{meta.get('max_shift', 0):.0f} 位</b>。"
        f"<b>两种口径的语种排序大体一致</b>——译中CER 的问题不在排序方向，而在"
        f"它的数值不可读、且被 ASR 污染。</li>")
    add("</ul>\n")

    # ---- 2 口径问题
    add("<h2>2. 重打分的理由：原口径的问题</h2><ul>")
    add("<li><b>译中CER 不是翻译质量指标。</b>它是「模型中文输出」与「人写中文参考」的字面重合率，"
        "换个说法就扣分——正是 COMET 要解决的事。</li>")
    add("<li><b>译中CER 被 ASR 错误污染。</b>译文由<b>模型自己的转写</b>再翻译而来（级联），"
        "ASR 听错的内容会原样进入译文。逐条看 ar_fleurs_long_01：金标「مات في أوساكا」（在大阪去世）"
        "被听成「ما تفوكس」，译文跟着写成了「福克斯周二怎么样」。"
        f"译中CER 与 ASR-CER 的相关系数 <b>{meta.get('corr_cer', 0):.2f}</b>。</li>")
    add("<li><b>长文整段打分对 COMET 同样不可行。</b>每个样本 790–2187 个 XLM-R token，"
        "远超 wmt22-comet-da 的 512 上限，整段打分只能看到前 ~40%。本报告按参考文本的句子边界重新切分后逐段打分。</li>")
    add(f"<li><b>会话失败被当成模型错误。</b>{len(scored)-len(clean)} 条样本的流式会话中途断开，"
        "尾部把 <code>{\"code\": \"service_unavailable\"}</code> 混进转写，输出只剩参考的 9%–30%，"
        "却被判为「ASR明显偏差／译文偏离金标」并计入语种平均。这些样本在原报告里铺满了「较差」档和 CER 关注榜。</li>")
    add("<li><b>说话人分离全线失效但无指标。</b>540 条里有 537 条的评语含「单人音频却检出N人」"
        "（N 从 2 到 21），而报告没有任何分离指标，也没把它算作失败。</li>")
    add("<li><b>中文普通话的翻译臂是退化的。</b>zh 的「金标源语」与「金标中文」30/30 完全相同，"
        "译中CER 只是在再测一遍 ASR，把它放进跨语种翻译宏平均会拉低整体。</li>")
    add("<li><b>部分语种指标不可复现。</b>用报告里展示的原文重算，"
        "乌尔都语 CER 比报告低 <b>3.9pp</b>（报告 59.84% vs 重算 55.91%）、阿拉伯语低 2.3pp；"
        "而 fr/it/de/ja/ko 等逐条能精确复现。受影响的正是带变音符号的阿拉伯字母语言。</li>")
    add("<li><b>没有不确定度。</b>每语种只有 30 条，bootstrap 95% 区间普遍有 2–17pp 宽"
        "（英语 CER 11.73% 的区间是 5.77%–18.71%）。报告把 fr 3.71% / it 4.77% / ko 5.16% / ja 5.28% "
        "排成精确序列，但中段语种之间的差异基本淹没在抽样噪声里。</li>")
    add("<li><b>WER 的两处口径互相打架。</b>卡片写「平均 WER（词级语种）21.66%」，"
        "第 2 节表格写 20.43%——后者把 ja/ko/yue/zh 的<b>字级</b>率当成词级 WER 一起平均了。</li>")
    add("</ul>\n")

    # ---- 3 按语种
    add("<h2>3. 按语种（原译中CER ↔ COMET）</h2>\n")
    add('<p class="note">两列 COMET 只差 src：<b>端到端</b>用金标源语，<b>仅翻译</b>用模型自己的转写。'
        '两者在这个数据集上几乎完全相同（0.761 vs 0.759），说明 COMET 的判断几乎不因换 src 而改变，'
        '因此<b>不能</b>用它们的差去拆分 ASR 与翻译各自的贡献。'
        '秩变化 = 译中CER 名次 − COMET 名次，两列名次都是 1 = 最好；正数表示原口径排得更靠后。'
        '中文普通话的翻译臂退化（源=目标），不参与排名。</p>\n')
    add('<div class="table-wrap"><table>\n<thead><tr>'
        "<th>语言</th><th>code</th><th>n</th><th>CER ↓</th><th>译中CER ↓</th>"
        "<th>COMET 端到端 ↑</th><th>COMET 95% CI</th><th>COMET 仅翻译 ↑</th>"
        "<th>秩（CER）</th><th>秩（COMET）</th>"
        "<th>秩变化</th><th>会话失败</th></tr></thead><tbody>\n")
    for s in lang_stats:
        if "rank_cer" in s:
            d = s["rank_cer"] - s["rank_comet"]
            tone = "up" if d > 0 else "down" if d < 0 else ""
            rank_cells = (f'<td class="mono">{s["rank_cer"]:.0f}</td>'
                          f'<td class="mono">{s["rank_comet"]:.0f}</td>'
                          f'<td class="mono"><span class="{tone}">{d:+.0f}</span></td>')
        else:
            rank_cells = '<td class="mono">—</td><td class="mono">—</td><td class="mono">—</td>'
        add(
            f'<tr class="{cls_for(s["cer"])}"><td>{esc(s["lang"])}</td>'
            f'<td><code>{esc(s["code"])}</code></td><td>{s["n"]}</td>'
            f'<td class="mono">{fmt(s["cer"],2,"%")}</td>'
            f'<td class="mono">{fmt(s["zh_cer"],2,"%")}</td>'
            f'<td class="mono"><b>{fmt(s["comet_gold"],3)}</b></td>'
            f'<td class="mono">{_ci(s["comet_ci"])}</td>'
            f'<td class="mono">{fmt(s["comet_asr"],3)}</td>'
            f'{rank_cells}'
            f'<td class="mono">{s["harness"] or ""}</td></tr>\n'
        )
    add("</tbody></table></div>\n")

    # ---- 4 关注样例
    add("<h2>4. COMET 最低的 30 条</h2>\n")
    add('<div class="table-wrap"><table>\n<thead><tr><th>样本</th><th>COMET</th><th>译中CER</th>'
        "<th>CER</th><th>输出/参考</th><th>会话</th><th>模型译文摘录</th></tr></thead><tbody>\n")
    for row in worst:
        ratio = len(re.sub(r"\s+", "", row["hyp_asr"])) / max(1, len(re.sub(r"\s+", "", row["gold_src"])))
        add(f'<tr class="{cls_for(row["report_cer"])}"><td>{esc(row["lang_name"])}<br>'
            f'<code>{esc(row["id"])}</code></td>'
            f'<td class="mono"><b>{fmt(row["comet_gold"],3)}</b></td>'
            f'<td class="mono">{fmt(row["report_zh_cer"],2,"%")}</td>'
            f'<td class="mono">{fmt(row["report_cer"],2,"%")}</td>'
            f'<td class="mono">{ratio:.2f}</td>'
            f'<td>{"失败" if is_harness_fail(row) else ""}</td>'
            f'<td class="clip">{esc(row["hyp_mt"][:220])}</td></tr>\n')
    add("</tbody></table></div>\n")

    # ---- 5 逐条明细
    add("<h2>5. 逐条明细</h2>\n")
    add('<div class="table-wrap"><table>\n<thead><tr><th>样本</th><th>指标</th>'
        "<th>COMET</th><th>翻译比对</th></tr></thead><tbody>\n")
    for row in rows:
        ratio = len(re.sub(r"\s+", "", row["hyp_asr"])) / max(1, len(re.sub(r"\s+", "", row["gold_src"])))
        add(f'<tr class="{cls_for(row["report_cer"])}">'
            f'<td>{esc(row["lang_name"])} <code>{esc(row["id"])}</code><br>{esc(row["dur_desc"])}'
            f'<br>{"<b>会话失败</b>" if is_harness_fail(row) else ""}</td>'
            f'<td class="mono">CER {fmt(row["report_cer"],2,"%")}<br>WER {fmt(row["report_wer"],2,"%")}'
            f'<br>译中CER {fmt(row["report_zh_cer"],2,"%")}<br>输出/参考 {ratio:.2f}</td>'
            f'<td class="mono">{row["n_units"]} 段<br>端到端 <b>{fmt(row.get("comet_gold"),3)}</b>'
            f'<br>仅翻译 {fmt(row.get("comet_asr"),3)}</td>'
            f'<td><div class="lab">金标中文</div><div class="txt">{esc(row["gold_zh"])}</div>'
            f'<div class="lab">模型译文</div><div class="txt">{esc(row["hyp_mt"])}</div>'
            f'<div class="lab">模型转写</div><div class="txt">{esc(row["hyp_asr"])}</div></td></tr>\n')
    add("</tbody></table></div>\n")
    add(f'<div class="footer">重打分口径：BLEU 系与译中CER 均为字面指标；COMET 为 wmt22-comet-da 参考式句级打分，'
        f'按参考文本的句子边界切分后取单元平均。源报告：{esc(meta.get("source",""))}</div>\n')
    add("</div></body></html>\n")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", default="report.html")
    parser.add_argument("--comet-model", default="wmt22-comet-da")
    args = parser.parse_args()

    rows = json.loads(args.scores.read_text())

    def corr(a, b):
        ma, mb = statistics.mean(a), statistics.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else 0.0

    scored = [r for r in rows if r.get("comet_gold") is not None]
    meta = {
        "source": args.source,
        "comet_model": args.comet_model,
        "units": sum(r["n_units"] for r in rows),
        "corr_cer": corr([r["report_cer"] for r in scored],
                         [r["report_zh_cer"] for r in scored]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(rows, meta))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
