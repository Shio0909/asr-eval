"""同传记录导出：score 结果 → 按语向分文件夹，每条音频 3 个文件。

result/chuanyi/<语向>/            （如 zh2en / en2zh / my_data）
  <id>_source.<ext>    原音频（从 manifest 的 audio_path 拷贝）
  <id>_tts.<ext>       传译后音频（adapter 存的 tts_audio，mp3/wav）
  <id>_metrics.json    该条全部信息：原文 / 译文 / 译后语音转写 / 各项指标

无参考译文的数据集（用户自带）chrF 自动为 null，不记翻译质量，其余照记。

用法: python eval/export_records.py --result results/xxx.json --label zh2en
"""
import argparse
import json
import os
import shutil

from config import abs_path
from util import atomic_write_json

OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result", "chuanyi")


def export(result_json, label, out_root=OUT_ROOT):
    res = json.load(open(result_json, encoding="utf-8"))
    summary = res.get("summary", {})
    samples = res.get("samples", [])
    manifest = summary.get("manifest")
    infer_file = summary.get("infer_file")
    # id → manifest 行(原文/原音频/参考译文)
    mani = {}
    if manifest and os.path.exists(manifest):
        for ln in open(manifest, encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                mani[d["id"]] = d
    # id → infer extra(al/laal/ttfb/revision/e2e/tts_audio 等完整指标)
    inf = {}
    if infer_file and os.path.exists(infer_file):
        for ln in open(infer_file, encoding="utf-8"):
            d = json.loads(ln)
            if d.get("id") != "__meta__":
                inf[d["id"]] = d

    model = (summary.get("meta") or {}).get("model", "model")
    out_dir = os.path.join(out_root, label, model)   # 语向/模型/，多模型不互相覆盖
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for s in samples:
        if "error" in s:
            continue
        sid = s["id"]
        m = mani.get(sid, {})
        rec = inf.get(sid, {})
        ex = rec.get("extra") or {}
        files = {}
        # 原音频
        sp = abs_path(m["audio_path"]) if m.get("audio_path") else None
        if sp and os.path.exists(sp):
            ext = os.path.splitext(sp)[1] or ".wav"
            shutil.copy(sp, os.path.join(out_dir, f"{sid}_source{ext}"))
            files["source_audio"] = f"{sid}_source{ext}"
        # 传译后音频
        ta = ex.get("tts_audio")
        if ta and os.path.exists(ta):
            ext = os.path.splitext(ta)[1] or ".mp3"
            shutil.copy(ta, os.path.join(out_dir, f"{sid}_tts{ext}"))
            files["tts_audio"] = f"{sid}_tts{ext}"
        # 指标 json（chrF：无参考译文时为 null，不记翻译质量）
        chrf = (round((1 - s["err_rate"]) * 100, 2)
                if m.get("ref_text") and s.get("err_rate") is not None else None)
        metric_record = {
            "id": sid, "label": label, "lang": m.get("lang"),
            "source_text": m.get("source_text") or m.get("ref_text"),   # 原文
            "reference_translation": m.get("ref_text") if m.get("source_text") else None,  # 参考译文(有才填)
            "translation": rec.get("hyp") or s.get("hyp_norm"),         # 模型译文
            "tts_asr": s.get("tts_asr"),                                # 译后语音转写(ASR-BLEU 回环)
            "metrics": {
                "chrF": chrf,                          # 译文质量 vs 参考(无参考则 null)
                "al_s": ex.get("al_s"), "laal_s": ex.get("laal_s"),
                "ttfb_s": ex.get("ttfb_s"), "stable_ttfb_s": ex.get("stable_ttfb_s"),
                "revision_rate": ex.get("revision_rate"), "churn_rate": ex.get("churn_rate"),
                "e2e_ms": ex.get("e2e_ms_mean"), "src_dur_s": ex.get("src_dur_s"),
                "tts_err": s.get("tts_err"),           # TTS 保真度 CER/WER vs 模型译文(越低越好)
                "tts_err_metric": s.get("tts_err_metric"),
                # 官方 OmniSTEval 长音频同传指标为本次语料级分数，逐条导出时同步带上便于溯源。
                "corpus_long_yaal_cu_ms": summary.get("long_yaal_cu_ms"),
                "corpus_long_yaal_ca_ms": summary.get("long_yaal_ca_ms"),
            },
            **files,
        }
        atomic_write_json(os.path.join(out_dir, f"{sid}_metrics.json"), metric_record, indent=2)
        n += 1
    print(f"导出 {n} 条 → {out_dir}/  （每条 _source / _tts / _metrics.json）")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--label", required=True, help="语向文件夹名，如 zh2en / en2zh / my_data")
    args = ap.parse_args()
    export(args.result, args.label)
