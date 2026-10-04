"""便捷入口 — 一条命令跑完 infer + score（精度 + 性能）。

底层是分离的两步（infer.py 调模型存 hyp、score.py 算指标），需要单独跑/复用时直接用那两个。
- 断点续跑：默认开；崩了重跑同命令自动跳过已完成。
- 并发：--workers N 提速（accuracy-only；延迟会标 invalid）。

  uv run --with jiwer --with opencc --with soundfile --with numpy --with requests \
      --with websocket-client \
      python eval/runner.py --manifest manifests/aishell_lite.jsonl \
      --model ext-pro --language zh --out results/aishell_ext-pro.json
"""

import argparse
import json
import os

import infer as infer_mod
import score as score_mod
from adapters import ADAPTERS
from logconf import attach_job_log, get_logger

log = get_logger("runner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="ext-pro", choices=list(ADAPTERS))
    ap.add_argument("--language", default="auto")
    ap.add_argument("--target-lang", default="",
                    help="覆盖 manifest 的 target_lang（翻译/同传实验）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--hotwords", action="store_true",
                    help="把 manifest keywords 作为热词传给模型(场景 b)")
    ap.add_argument("--hotwords-text", default="",
                    help="每条样本都追加的自定义热词；与 manifest keywords 合并")
    ap.add_argument("--domain", default="",
                    help="领域提示词(legal/medical/finance/government_emergency)，仅被测主体支持")
    ap.add_argument("--embedding", action="store_true",
                    help="ASR 加 embedding 语义相似度(可选增强,需 EMB_KEY;端点挂则降级不影响主指标)")
    ap.add_argument("--comet", action="store_true",
                    help="翻译加 COMET 神经质量(可选增强,需 unbabel-comet+模型;下不动则降级不影响主指标)")
    ap.add_argument("--save-audio", action="store_true",
                    help="同传：存译后语音到 audio_out/（仅 simult 系 adapter）")
    ap.add_argument("--asr-bleu", action="store_true",
                    help="同传：用 ASR 回转译后语音→算 ASR-BLEU(语音保真，需 --save-audio)")
    ap.add_argument("--note", default="", help="纯展示用任务备注；不影响结果/排名/文件名")
    ap.add_argument("--request-params-json", default="{}",
                    help="已由接口 request_schema 校验的运行时请求参数 JSON")
    ap.add_argument("--config-hash", default="",
                    help="接口配置+运行参数指纹；用于隔离 infer/results")
    args = ap.parse_args()
    request_params = json.loads(args.request_params_json or "{}")

    import jobstore   # CLI 起的评测也登记进 jobs.json,看板可见(看板自起的设 DASH_JOB→跳过)
    manifest_base = os.path.splitext(os.path.basename(args.manifest))[0]
    dataset = manifest_base.split("_lite")[0].split("_full")[0]
    tier = "lite" if "_lite" in manifest_base else "full" if "_full" in manifest_base else "cli"
    dashboard_job = os.environ.get("DASH_JOB")
    effective_note = args.note
    if not dashboard_job and not effective_note:
        effective_note = "CLI·全量精度·断点续跑" if args.workers > 1 else "CLI·延迟串行"
    _jid = jobstore.start(
        args.model, dataset, tier=tier, result_file=os.path.basename(args.out),
        run_mode="accuracy" if args.workers > 1 else "latency", workers=args.workers,
        language=args.language, target_lang=args.target_lang,
        hotwords=bool(args.hotwords or args.hotwords_text), hotwords_text=args.hotwords_text,
        domain=args.domain, embedding=args.embedding, comet=args.comet,
        request_params=request_params, config_hash=args.config_hash, note=effective_note,
    )
    if _jid:
        os.environ["DASH_JOB"] = _jid
        os.environ["JOBSTORE_MANAGED"] = "1"
        attach_job_log(_jid)
        jobstore.update(_jid, log_file=_jid)
    log.info("评测开始: model=%s manifest=%s out=%s workers=%d",
             args.model, os.path.basename(args.manifest), os.path.basename(args.out), args.workers)
    try:
        infer_path, concurrent = infer_mod.run(
            args.manifest, args.model, args.language, args.workers,
            args.retries, None, not args.no_resume, args.limit, args.hotwords,
            args.domain, args.save_audio, note=effective_note, target_lang=args.target_lang,
            hotwords_text=args.hotwords_text, request_params=request_params,
            config_hash=args.config_hash,
        )
        log.info("=== 打分 ===")
        score_mod.run(args.manifest, infer_path, args.out, concurrent, args.embedding,
                      args.comet, args.asr_bleu)
        with open(args.out, encoding="utf-8") as result_src:
            summary = json.load(result_src).get("summary", {})
        jobstore.finish(_jid, "done", result_file=os.path.basename(args.out), summary=summary)
        log.info("评测完成: %s", os.path.basename(args.out))
    except BaseException as e:
        jobstore.finish(_jid, "failed", stage=f"失败: {str(e)[:60]}")
        log.exception("评测失败: model=%s manifest=%s", args.model, os.path.basename(args.manifest))
        raise


if __name__ == "__main__":
    main()
