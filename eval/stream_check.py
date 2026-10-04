"""逐条流式跑分 + 自检：跑一条 → 立刻出一条 → 自动检查这条有没有问题。

和 infer.py 不同点：infer 是"批量跑完再 score"，这个是"边跑边出边查"——
每条样本 generate 完，当场打印 原文/译文/AL/TTS/tts_err，并跑一串自检规则，
标 ✅(没问题) 或 ⚠️(列出问题)。可选 --asr-bleu 对译后语音即时回环再识别。

用法:
  python eval/stream_check.py --model plat-simult --manifest manifests/acl_en2zh_n60.jsonl \
         --limit 5 --save-audio --asr-bleu
"""
import argparse
import json
import os
import re
import time

from adapters import ADAPTERS
from config import abs_path
from metrics import char_cer, word_wer
from normalize import normalize, normalize_en

_CJK = re.compile(r"[一-鿿]")
_LAT = re.compile(r"[A-Za-z]")


def _cjk_ratio(s):
    chars = [c for c in s if not c.isspace()]
    return (sum(bool(_CJK.match(c)) for c in chars) / len(chars)) if chars else 0.0


def _lat_ratio(s):
    chars = [c for c in s if not c.isspace()]
    return (sum(bool(_LAT.match(c)) for c in chars) / len(chars)) if chars else 0.0


def check(item, res, tts_asr=None):
    """对单条结果跑自检，返回 (ok, [问题/提示...])。⚠️=问题，ℹ️=提示。"""
    warns = []
    tgt = item.get("target_lang", "")
    is_zh = "中" in tgt or tgt.lower() in ("zh", "chinese")
    hyp = (res.text or "").strip()
    ex = res.extra or {}

    if not res.ok:
        return False, [f"⚠️ 失败: {res.error}"]
    if not hyp:
        warns.append("⚠️ 空译文")
    else:  # 译文语言串味检测（en→zh 应主要中文；zh→en 应主要拉丁）
        if is_zh and _cjk_ratio(hyp) < 0.5:
            warns.append(f"⚠️ 译文非中文(中字占比 {_cjk_ratio(hyp):.0%})")
        if not is_zh and _lat_ratio(hyp) < 0.5:
            warns.append(f"⚠️ 译文非英文(拉丁占比 {_lat_ratio(hyp):.0%})")

    al, ttfb, sd = ex.get("al_s"), ex.get("ttfb_s"), ex.get("src_dur_s")
    if al is None:
        warns.append("⚠️ 无 AL")
    elif al < 0:
        warns.append(f"⚠️ AL 为负({al})")
    elif sd and al > sd * 2 + 6:
        warns.append(f"⚠️ AL 偏大({al}s vs 源 {sd}s)")

    if ex.get("text_from") == "si_token":
        warns.append("ℹ️ 无终稿(token 兜底)")

    # TTS：启用了存档却没拿到音频
    save_audio = bool(getattr(check, "_save_audio", False))
    if save_audio and not ex.get("tts_audio"):
        warns.append("⚠️ 无 TTS 音频")

    if tts_asr is not None and hyp:   # 译后语音再识别 vs 模型自己的译文(同 score 口径：先归一化)
        norm_fn = normalize if is_zh else normalize_en
        c = (char_cer if is_zh else word_wer)(norm_fn(hyp), norm_fn(tts_asr))
        res.extra["tts_err"] = round(c.cer, 4)
        res.extra["tts_err_metric"] = "CER" if is_zh else "WER"
        if c.cer > 0.6:
            warns.append(f"⚠️ 译后语音保真差(tts_err {c.cer:.0%})")

    return (not any(w.startswith("⚠️") for w in warns)), warns


def run(manifest, model, limit=0, save_audio=False, asr_bleu=False):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    for r in rows:
        if r.get("audio_path"):
            r["audio_path"] = abs_path(r["audio_path"])
    if limit:
        rows = rows[:limit]

    out_dir = os.path.join("audio_out", f"streamchk_{model}") if save_audio else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    adapter = ADAPTERS[model](audio_out_dir=out_dir) if save_audio else ADAPTERS[model]()
    check._save_audio = save_audio
    reasr = ADAPTERS["normal"]() if asr_bleu else None   # 译后语音回环再识别走产品 normal，不污染被测端点

    n_ok = 0
    print(f"== 流式自检 {model} · {os.path.basename(manifest)} · {len(rows)} 条 "
          f"{'(存TTS+回环再识别)' if asr_bleu else ''} ==\n", flush=True)
    for i, item in enumerate(rows, 1):
        t0 = time.perf_counter()
        res = adapter.generate(item)
        dt = time.perf_counter() - t0
        tts_asr = None
        if reasr and res.ok and (res.extra or {}).get("tts_audio"):
            try:
                tts_asr = reasr.transcribe(res.extra["tts_audio"], language="auto").text
            except Exception as e:
                tts_asr = None
                res.extra["tts_asr_err"] = str(e)[:60]
        ok, warns = check(item, res, tts_asr)
        n_ok += ok
        ex = res.extra or {}
        src = (item.get("source_text") or item.get("ref_text") or "")[:42]
        print(f"[{i}/{len(rows)}] {item['id']}  {'✅' if ok else '⚠️'}  ({dt:.1f}s)", flush=True)
        print(f"   原文: {src}", flush=True)
        print(f"   译文: {(res.text or '')[:50]}", flush=True)
        line = f"   AL={ex.get('al_s')}s ttfb={ex.get('ttfb_s')}s src={ex.get('src_dur_s')}s"
        if ex.get("tts_audio"):
            line += f"  TTS={os.path.basename(ex['tts_audio'])}"
        if ex.get("tts_err") is not None:
            line += f"  tts_err={ex['tts_err']}"
        print(line, flush=True)
        if tts_asr is not None:
            print(f"   译后语音再识别: {tts_asr[:50]}", flush=True)
        if warns:
            print("   " + " · ".join(warns), flush=True)
        print(flush=True)
    print(f"== 完成: {n_ok}/{len(rows)} 条无问题 ==", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--asr-bleu", action="store_true", help="对译后语音即时回环再识别算 tts_err")
    args = ap.parse_args()
    run(args.manifest, args.model, args.limit, args.save_audio, args.asr_bleu)
