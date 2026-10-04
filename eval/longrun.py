"""长音频同传整段跑 —— 把任意时长音频(会议/讲座)整段喂给某个同传 adapter,边收边落盘。

为什么单独一个入口(不走 infer.py):infer 把一条音频当一个 item、断了只返回错误丢空;
长音频(数小时)单会话端点大多撑不住,实测单会话上限(1x 裸跑 20260417 会议 2.97h):
  豆包 142min/80% · 讯飞 46min/26% · v2 ~4min · qwen 0.1s(秒拒)。
本脚本:整段 1x 喂 + **断点捕获**(断了保留已收译文和喂到的秒数)+ **增量落盘**(进程被杀也不丢)。
⚠️ 掉线后"从断点续喂"需 adapter 支持 offset(跳过已喂音频)——当前 adapter 尚不支持,故为单会话整段跑;
   offset-resume 是后续增强。

用法:
  python eval/longrun.py --model doubao-simult --audio 会议.mp3 --target-lang 英文 --out result/<name>.json
产物:
  <out>               汇总(ok/error/喂到秒数/coverage/译文全文/段数)
  <out>.stream.jsonl  每段译文即时追加(audio_t, text)——durable,进程被杀也留得住(仅 v2 adapter 自带)
"""
import argparse
import json
import os
import time

from adapters import ADAPTERS, load_pcm_bytes
from config import abs_path
from util import atomic_write_json


def run(model, audio, target_lang="英文", lang="zh", out=None):
    audio = abs_path(audio)
    total_dur = len(load_pcm_bytes(audio)) / 32000.0
    out = out or f"result/longrun_{model}.json"
    slog = out + ".stream.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(slog, "w").close()

    # plat-simult 自带 stream_log(每段即时落盘);其余 adapter 靠返回时的 partial 捕获
    try:
        ad = ADAPTERS[model](stream_log=slog)
    except TypeError:
        ad = ADAPTERS[model]()

    print(f"[longrun] {model} 整段 1x 跑 {audio} ({total_dur:.0f}s={total_dur/60:.0f}min) target={target_lang}", flush=True)
    t0 = time.time()
    item = {"id": "longrun", "task": "translate", "lang": lang,
            "audio_path": audio, "target_lang": target_lang}
    res = ad.generate(item)
    ex = res.extra or {}
    fed = ex.get("audio_fed_s", total_dur if res.ok else 0.0)

    rec = {"model": model, "audio": audio, "target_lang": target_lang,
           "ok": res.ok, "error": res.error,
           "total_dur_s": round(total_dur, 1), "fed_s": round(fed, 1),
           "coverage": round(fed / total_dur, 3) if total_dur else 0,
           "n_seg": ex.get("n_seg"), "wall_s": round(time.time() - t0, 1),
           "text_len": len(res.text or ""), "text": res.text}
    atomic_write_json(out, rec)
    print(f"[longrun] 完: ok={res.ok} 喂到 {fed:.0f}/{total_dur:.0f}s ({rec['coverage']:.0%}) "
          f"译文 {rec['text_len']} 字 err={res.error} → {out}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--audio", required=True)
    ap.add_argument("--target-lang", default="英文")
    ap.add_argument("--lang", default="zh", help="源语种(zh/en/...)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.model, args.audio, args.target_lang, args.lang, args.out)
