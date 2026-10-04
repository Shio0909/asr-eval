"""打分阶段 — 读 infer 文件 + manifest 算指标。不调模型，可反复跑。

改口径/加指标(KRR/Embedding/COMET)只需重跑这步，不必重调模型。
中文→CER、英文(lang=en)→WER，自动选。
"""

import argparse
import json
import os
import re
import statistics
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation

from metrics import (char_cer, comet_score, corpus_cer, embedding_cosine, keyword_hits,
                     rouge_char, sent_chrf, translation_corpus, word_wer)
from normalize import normalize, normalize_basic, normalize_en, normalize_ja
from logconf import get_logger
from data_quality import exclusions_for_rows, reviewed_asr_metrics
from scorer_client import (speaker_score as remote_speaker_score,
                           utmos_score as remote_utmos_score,
                           whisper_transcribe as remote_whisper_transcribe,
                           xcomet_score as remote_xcomet_score)

log = get_logger("score")

# 多语言 ASR 口径路由（对齐 Whisper 官方）：有空格语种→词级 WER，无空格语种→字级 CER；
# 二者都用多语言通用 BasicTextNormalizer；日语再去空格，对齐 Kotoba/Reazon 公开基准。
# 中文/粤/方言/英文维持原口径（结果字节不变）。
_WER_BASIC = {"ar", "de", "fr", "es", "ru", "it", "pt", "nl", "pl", "tr", "vi", "id"}  # 有空格 → WER
_CER_BASIC = {"ja", "ko", "th"}                                                  # 无空格 → CER


def _asr_norm_metric(lang):
    """按样本 lang 返回 (归一化函数, 是否词级WER)。新增多语种走 BasicTextNormalizer，
    其余(中文/粤/方言/默认)保持中文 normalize + 字级 CER、英文 normalize_en + WER。"""
    if lang == "en":
        return normalize_en, True
    if lang in _WER_BASIC:
        return normalize_basic, True
    if lang == "ja":
        return normalize_ja, False
    if lang in _CER_BASIC:
        return normalize_basic, False
    return normalize, False
from util import (atomic_write_json, code_revision, ensure_unique_ids, manifest_sha,
                  manifest_sha_v2)

# 覆盖率门槛：成功样本占比低于此值的结果不可信（失败样本不计入精度，
# 模型"在难样本上报错"反而提分——榜单对 low_coverage 标灰且不参与排名）
COVERAGE_MIN = 0.95


def load_infer(path: str) -> tuple[dict, list]:
    """读 infer 文件，按 id 去重（优先成功、后写覆盖先写）。

    返回 (记录 dict, __meta__ 会话列表)。meta 行是 infer 每次运行写的
    provenance（模型/端点/参数/时间），用于结果溯源。
    """
    best, metas = {}, []

    def consume(lineno, line, *, final=False):
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            if final:
                log.warning("%s 最后一个非空行是不完整 JSON，按进程中断残行忽略(line %d)", path, lineno)
                return
            raise ValueError(f"{path}:{lineno} JSON 损坏（仅允许忽略文件尾部残行）") from exc
        if d.get("id") == "__meta__":
            metas.append(d)
            return
        if d["id"] not in best or d.get("ok"):
            best[d["id"]] = d

    # 保留一个非空行作为 look-ahead：只对真正的最后一行开放残缺容错，避免隐藏中间损坏。
    pending = None
    with open(path, encoding="utf-8") as src:
        for lineno, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            if pending is not None:
                consume(*pending)
            pending = (lineno, line)
    if pending is not None:
        consume(*pending, final=True)
    return best, metas


def _pct(values, p):
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    return round(statistics.quantiles(values, n=100)[p - 1], 3)


def _translation_target_norm(text, lang):
    """翻译长度比和术语命中使用目标语自己的稳定规一化口径。"""
    code = str(lang or "").strip().lower()
    if code in {"en", "english", "英文"} or "english" in code:
        return normalize_en(text), True
    if code in {"zh", "zho", "chinese", "中文", "简体中文"} or "chinese" in code:
        return normalize(text), False
    return normalize_basic(text), True


def _segment_reason(segment):
    for key in ("segment_reason", "cut_reason", "vad_reason", "reason"):
        value = segment.get(key)
        if value:
            return str(value)
    return None


def _numeric_literals(text):
    """抽取可无歧义核对的阿拉伯数字；不猜测中文口语数字或单位换算。"""
    out = []
    for raw in re.findall(r"(?<![\d.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\d.])", text or ""):
        percent = raw.endswith("%")
        value = raw.rstrip("%").replace(",", "")
        try:
            normalized = format(Decimal(value).normalize(), "f")
        except InvalidOperation:
            continue
        out.append(normalized + ("%" if percent else ""))
    return out


def _date_literals(text):
    """抽取 YYYY-MM-DD / YYYY年M月D日 这类显式日期，不从自然语言猜日期。"""
    found = []
    pattern = r"(?<!\d)(\d{4})(?:年|[-/.])(\d{1,2})(?:月|[-/.])(\d{1,2})(?:日)?(?!\d)"
    for year, month, day in re.findall(pattern, text or ""):
        month_i, day_i = int(month), int(day)
        if 1 <= month_i <= 12 and 1 <= day_i <= 31:
            found.append(f"{int(year):04d}-{month_i:02d}-{day_i:02d}")
    return found


def _multiset_hits(reference, hypothesis):
    ref, hyp = Counter(reference), Counter(hypothesis)
    return sum((ref & hyp).values()), sum(ref.values())


def _bootstrap_ci(items, stat, b=1000, seed=42):
    """err_rate 的 95% bootstrap 置信区间（固定种子可复现）。

    items: 样本级数据列表；stat: 重抽样列表 → 标量。两份结果 CI 重叠 →
    排名差异可能是抽样噪声，加样本再下结论。
    """
    import random
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    n = len(items)
    vals = sorted(stat(rng.choices(items, k=n)) for _ in range(b))
    return [round(vals[int(0.025 * b)], 4), round(vals[int(0.975 * b)] , 4)]


def _metric_signature(summary, output_variant):
    """人可读且稳定的评分口径签名；只记录，不参与兼容性判定。"""
    task = summary.get("task") or "asr"
    if task == "translate":
        if summary.get("quality_basis") == "omnisteval_softsegmenter":
            base = "chrF+BLEU|OmniSTEval-SoftSegmenter|longform-resegmented|ci=none"
        else:
            base = "chrF+BLEU|sacrebleu|sentence-ci-bootstrap1000-seed42"
    elif task == "summarize":
        base = "ROUGE-L+ROUGE-1|char|sentence-ci-bootstrap1000-seed42"
    elif task == "diarization":
        base = "DER|pyannote.metrics|sentence-ci-bootstrap1000-seed42"
    else:
        base = f"{summary.get('metric', 'CER')}|lang-router-v3-en-token-boundaries|corpus|bootstrap1000-seed42"
    extras = [name for name in ("emb_sim", "xcomet", "comet", "tts_err", "utmos",
                                "speaker_similarity", "long_yaal_cu_ms")
              if name in summary]
    return f"{base}|output={output_variant}" + (f"|extras={'+'.join(extras)}" if extras else "")


def _diar_der(ref_segs, hyp_json):
    """pyannote DER：参考分段(秒) vs 假设分段(JSON 编码的 [[s,e,spk],...])。"""
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
        hyp_segs = json.loads(hyp_json)
        ref_a, hyp_a = Annotation(), Annotation()
        for i, (s, e, spk) in enumerate(ref_segs):
            if e > s:
                ref_a[Segment(s, e), i] = str(spk)
        for i, (s, e, spk) in enumerate(hyp_segs):
            if e > s:
                hyp_a[Segment(s, e), i] = str(spk)
        return round(float(DiarizationErrorRate()(ref_a, hyp_a)), 4)
    except Exception:
        return None


def _add_embedding_metric(summary, samples, pairs=None):
    """可选增强：embedding 语义相似度。ASR=改写型字面 CER 高估其错误；翻译=意译被 chrF/BLEU 低估。

    pairs: [(ref,hyp),...] 完整文本，与 samples 成功样本顺序对齐（翻译必传——sample 里 ref_norm 截断了）。
    不传则用 samples 的 ref_norm/hyp_norm（ASR 是完整规一化文本）。
    **全程容错**：端点挂/限流/无 key/异常 → emb_sim=None + 提示，主指标照常。
    """
    try:
        succ = [s for s in samples if "ref_norm" in s and "hyp_norm" in s]  # 成功样本
        ps = pairs if pairs is not None else [(s["ref_norm"], s["hyp_norm"]) for s in succ]
        if not ps:
            return
        sims = embedding_cosine(ps)
        if not sims:  # None=未配 key 或端点不可用 → 降级，不污染主结果
            summary["emb_sim"] = None
            log.info("embedding 指标未计算（未配 EMB_KEY 或端点不可用），其余指标不受影响")
            return
        valid = [x for x in sims if x is not None]
        summary["emb_sim"] = round(sum(valid) / len(valid), 4) if valid else None
        summary["emb_coverage"] = round(len(valid) / len(sims), 3)
        for s, sim in zip(succ, sims):  # 回填到每条样本（前端样本详情显示）
            s["emb_sim"] = round(sim, 4) if sim is not None else None
    except Exception as e:  # 兜底：任何意外都不能让打分失败
        summary["emb_sim"] = None
        log.info("embedding 指标跳过（%s），其余指标不受影响", type(e).__name__)


def _add_tts_fidelity(summary, samples, lang_by_id, model="whisper-local", hyp_by_id=None):
    """TTS 保真度：GPU scorer Whisper 优先，本地 Whisper 降级。

    回转译后语音后与**模型自己的译文**比 CER(中)/WER(英)，隔离 TTS 环节
    （合成语音是否忠实念出了模型译文）。lang_by_id: id→译后语音语种(逐样本，决定 WER/CER + 强制回转 ASR 语种)。
    hyp_by_id: id→完整译文(samples 里 hyp_norm 被截断 [:200]，比对必须用完整译文)。
    ⚠️ 使用 Whisper：多语模型可能把英文 TTS 翻成中文，导致 tts_err 失真。
    全程容错：re-ASR 挂/无 tts_audio → 跳过，主指标不受影响。
    """
    hyp_by_id = hyp_by_id or {}
    candidates = [s for s in samples if s.get("tts_audio") and "error" not in s]
    if not candidates:
        return

    transcripts = {}
    remote = remote_whisper_transcribe([
        (str(s["id"]), s["tts_audio"], lang_by_id.get(s["id"], "en"))
        for s in candidates
    ])
    if remote:
        transcripts = {
            str(item.get("id")): item.get("text")
            for item in remote.get("items") or [] if item.get("text")
        }
        summary["tts_asr_model"] = remote.get("model") or "whisper-gpu"

    ad = None
    if len(transcripts) < len(candidates):
        try:
            from adapters import ADAPTERS
            ad = ADAPTERS[model]()
            summary.setdefault("tts_asr_model", model)
        except Exception as e:
            if not transcripts:
                log.info("TTS 保真度跳过（re-ASR 不可用：%s: %s）", type(e).__name__, e)

    refs, edits, n, metrics = [], [], 0, set()
    for s in candidates:
        p = s["tts_audio"]
        lang = lang_by_id.get(s["id"], "en")
        norm_fn, is_word = _asr_norm_metric(lang)   # en→normalize_en+WER；zh→去空格+CER
        try:
            transcript = transcripts.get(str(s["id"]))
            if not transcript and ad is not None:
                r = ad.transcribe(p, language=lang if lang in ("en", "zh", "ja", "ko", "yue") else "auto")
                transcript = r.text if r and r.ok and r.text else None
            if transcript:
                s["tts_asr"] = transcript[:300]
                hyp_full = hyp_by_id.get(s["id"]) or s.get("hyp_norm", "")   # 完整译文优先(截断会虚高)
                hyp_n, asr_n = norm_fn(hyp_full), norm_fn(transcript)  # 对比模型译文
                c = word_wer(hyp_n, asr_n) if is_word else char_cer(hyp_n, asr_n)
                s["tts_err"] = round(c.cer, 4)
                s["tts_err_metric"] = "WER" if is_word else "CER"
                refs.append(c.ref_len)
                edits.append(c.edits)
                metrics.add("WER" if is_word else "CER")
                n += 1
        except Exception:
            pass
    summary["tts_metric"] = "/".join(sorted(metrics)) if metrics else None
    summary["tts_err"] = round(corpus_cer(refs, edits), 4) if refs else None
    log.info("TTS 保真度：回转 %d 条，语料 %s=%s", n, summary["tts_metric"], summary["tts_err"])


def _add_speech_metrics(summary, samples, source_audio_by_id):
    """译后音频自然度与声线保持；仅远端 GPU scorer 可用时补充。"""
    generated = [(str(s["id"]), s["tts_audio"])
                 for s in samples if s.get("tts_audio") and "error" not in s]
    if not generated:
        return

    utmos = remote_utmos_score(generated)
    if utmos:
        by_id = {str(item.get("id")): item for item in utmos.get("items") or []}
        values = []
        for sample in samples:
            item = by_id.get(str(sample.get("id")))
            try:
                value = round(float(item["score"]), 4) if item else None
            except (KeyError, TypeError, ValueError):
                value = None
            if value is not None:
                sample["utmos"] = value
                values.append(value)
        summary["utmos"] = round(statistics.mean(values), 4) if values else None
        summary["utmos_coverage"] = round(len(values) / len(generated), 3)
        summary["utmos_model"] = utmos.get("model")

    pairs = [
        (item_id, source_audio_by_id[item_id], generated_path)
        for item_id, generated_path in generated if source_audio_by_id.get(item_id)
    ]
    speaker = remote_speaker_score(pairs)
    if speaker:
        by_id = {str(item.get("id")): item for item in speaker.get("items") or []}
        cosines, normalized = [], []
        for sample in samples:
            item = by_id.get(str(sample.get("id")))
            if not item or item.get("cosine") is None:
                continue
            cosine = round(float(item["cosine"]), 4)
            similarity = item.get("normalized_similarity")
            similarity = round(float(similarity), 4) if similarity is not None else round((cosine + 1) / 2, 4)
            sample["speaker_cosine"] = cosine
            sample["speaker_similarity"] = similarity
            cosines.append(cosine)
            normalized.append(similarity)
        summary["speaker_cosine"] = round(statistics.mean(cosines), 4) if cosines else None
        summary["speaker_similarity"] = round(statistics.mean(normalized), 4) if normalized else None
        summary["speaker_coverage"] = round(len(cosines) / len(pairs), 3) if pairs else 0.0
        summary["speaker_model"] = speaker.get("model")


def _has_simult_trace(infer_rows):
    return any(
        row.get("ok") and ((row.get("extra") or {}).get("simult_trace")
                           or (row.get("extra") or {}).get("translation_segments"))
        for row in infer_rows.values()
    )


def _add_omnisteval_metrics(summary, manifest, infer, out):
    """带真实流式事件的翻译自动补官方 LongYAAL；失败只记状态，不阻断主评分。"""
    try:
        from omnisteval_eval import evaluate
        root, _ = os.path.splitext(out)
        result = evaluate(manifest, infer, root + "__omnisteval", timing="stable")
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not result.get("ok"):
        summary["omnisteval"] = {"ok": False, "error": result.get("error", "unknown")}
        log.info("OmniSTEval 未计算（%s），现有质量/延迟指标不受影响", result.get("error"))
        return
    metric_keys = (
        "omnisteval_bleu", "omnisteval_chrf", "omnisteval_comet",
        "long_yaal_cu_ms", "long_yaal_ca_ms", "long_al_cu_ms", "long_al_ca_ms",
        "long_laal_cu_ms", "long_laal_ca_ms", "long_dal_cu_ms", "long_dal_ca_ms",
    )
    summary.update({key: result[key] for key in metric_keys if key in result})
    assets = result.get("assets") or {}
    # 有真实句级时间段的长音频，以 SoftSegmenter 重分段后的官方质量为榜单主口径；
    # 旧的整场直接拼接分数保留，避免隐藏口径变化。单段 document fallback 两者等价，不替换。
    segmented_longform = (assets.get("n_segments") or 0) > (assets.get("n_recordings") or 0)
    if segmented_longform and result.get("omnisteval_chrf") is not None:
        summary["chrF_document"] = summary.get("chrF")
        summary["BLEU_document"] = summary.get("BLEU")
        summary["chrF"] = round(result["omnisteval_chrf"], 2)
        if result.get("omnisteval_bleu") is not None:
            summary["BLEU"] = round(result["omnisteval_bleu"], 2)
        summary["err_rate"] = round(1 - result["omnisteval_chrf"] / 100, 4)
        summary["err_ci95"] = None  # 旧 CI 基于整场直拼样本，不能冒充重分段指标的 CI
        summary["quality_basis"] = "omnisteval_softsegmenter"
    summary["omnisteval"] = {
        "ok": True, "version": result.get("version"),
        "timing_basis": result.get("timing_basis"),
        "segmentation_sources": assets.get("segmentation_sources"),
        "n_recordings": assets.get("n_recordings"), "n_segments": assets.get("n_segments"),
        "output_dir": result.get("output_dir"),
    }


def run(manifest, infer, out=None, concurrent=False, embedding=False, comet=False, asr_bleu=False, use_edited=False):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    ensure_unique_ids(rows, source=manifest)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exclusions = exclusions_for_rows(rows, project_root)
    for r in rows:  # minutes(纪要)→summarize 口径(ROUGE 锚点)；formula(公式朗读)→ASR 口径(字级 CER)
        if r.get("task") == "minutes":
            r["task"] = "summarize"
        elif r.get("task") == "formula":
            r["task"] = None
    inf, metas = load_infer(infer)
    infer_meta = metas[-1] if metas else {}
    request_params = (infer_meta.get("request_params")
                      or (infer_meta.get("run_spec") or {}).get("request_params") or {})
    vad_soft_max_s = ((request_params.get("vad_soft_max_duration_ms") or 0) / 1000) or None
    vad_hard_max_s = ((request_params.get("vad_hard_max_duration_ms") or 0) / 1000) or None
    out = out or f"results/{os.path.splitext(os.path.basename(infer))[0]}.json"
    task = (rows[0].get("task") if rows else None) or "asr"
    # 有 meta 的新格式文件：延迟有效性按"记录级 workers==1"判定（续跑混采也准确）；
    # 旧格式只能信本次 --concurrent 标记
    per_record_workers = bool(metas)

    def is_serial(d):
        return d.get("workers", 1) == 1 if per_record_workers else not concurrent

    samples, latencies = [], []
    concurrent_latencies, concurrent_workers = [], set()
    output_variant_counts = {"raw": 0, "edited": 0}
    n_ok = n_fail = 0
    # ASR 累加
    ref_lens, edit_counts, rtfs = [], [], []
    sub_counts, del_counts, ins_counts = [], [], []
    sent_total = sent_err = 0
    hyp_lens, length_ratios = [], []
    raw_ref_lens, raw_edit_counts = [], []  # sse 修饰口径为主时，保留原始 ASR 汇总
    ttfbs = []  # 流式接口首包延迟(extra.ttfb_s)，串行口径
    stable_ttfbs = []  # 首个明确 commit 的译文；临时首包快但频繁回改时能揭示真实等待
    als, laals = [], []  # 同传延迟 AL/LAAL(秒，extra.al_s/laal_s)，需 1x 实时节奏采集，串行口径
    finish_tails = []  # 音频发送结束→task_finished；揭示末段定稿/TTS/收尾阻塞
    incremental_asr_ttfbs, incremental_translation_ttfbs = [], []
    incremental_asr_churns, incremental_translation_churns = [], []
    incremental_update_counts = []
    asr_stage_ms, translation_stage_ms, e2e_stage_ms = [], [], []
    revs = []  # 同传改写率(extra.revision_rate)：流式累积输出里已吐译文被回改的比例，与并发无关
    churns = []  # 被撤回/替换字符数 ÷ 最终译文字符数；比仅数 revision 事件更接近闪烁强度
    stability_flags = []  # 只有接口给 partial/revision 才能声称“0 改写”；仅终稿时是不可观测
    ws_control_examples, ws_control_signature_counts = [], {}
    kw_hit_total = kw_total = 0
    # 翻译累加
    tr_refs, tr_hyps, tr_srcs = [], [], []  # tr_srcs 供 COMET(参考式需 source 文本)
    source_asr_ref_lens, source_asr_edit_counts = [], []
    source_asr_metric_names = set()
    translation_length_ratios = []
    translation_term_hits = translation_term_total = 0
    translation_literal_ref_len = translation_literal_del = translation_literal_ins = 0
    numeric_hit = numeric_total = numeric_hyp_total = 0
    date_hit = date_total = 0
    # 同传/VAD：段长和成本可由 infer 重算；切段原因仅在服务端显式返回时统计，
    # 不用时长阈值反推，避免把自然静音误标成 hard_max/probability_valley。
    segment_durations_s = []
    segment_queue_ms, segment_ttft_ms = [], []
    segment_e2e_ms, segment_total_ms = [], []
    segment_reason_counts = {}
    segment_reason_observed = segment_count = 0
    simult_request_count = simult_audio_seconds = 0
    empty_segment_count = segment_error_count = 0
    simult_anomaly_counts = {}
    attempt_counts = []
    # 总结累加
    sum_rl, sum_r1 = [], []
    # 说话人分离累加(DER)
    der_list = []

    fail_reasons = {}   # 错误归并：只统计真失败(接口异常/无 infer)，空转写不算失败
    empty_reasons = {}   # 空转写：ok=True 但主结果为空，按正常样本计分，只做单独说明
    for r in rows:
        d = inf.get(r["id"])
        if not d or not d.get("ok"):
            n_fail += 1
            err = (d or {}).get("error", "no infer/empty")
            key = (err or "unknown")[:120]
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
            samples.append({"id": r["id"], "error": err})
            continue
        n_ok += 1
        ex = d.get("extra") or {}
        attempts = d.get("attempts", 1)
        if isinstance(attempts, int) and attempts > 0:
            attempt_counts.append(attempts)
        ws_control = ex.get("ws_control")
        if isinstance(ws_control, dict):
            ack_signature = (ws_control.get("task_started_stable_sha256")
                             or ws_control.get("task_started_sha256") or "")
            signature = f"{ws_control.get('task_start_sha256') or ''}:{ack_signature}"
            ws_control_signature_counts[signature] = ws_control_signature_counts.get(signature, 0) + 1
            if len(ws_control_examples) < 3 and all(
                    item.get("task_start_sha256") != ws_control.get("task_start_sha256")
                    or item.get("task_started_sha256") != ws_control.get("task_started_sha256")
                    for item in ws_control_examples):
                ws_control_examples.append(ws_control)
        trace_summary = ((ex.get("simult_trace") or {}).get("summary") or {})
        if "stability_observable" in trace_summary:
            stability_flags.append(bool(trace_summary["stability_observable"]))
        _rr = ex.get("revision_rate")
        if _rr is not None:
            revs.append(_rr)
        if ex.get("churn_rate") is not None:
            churns.append(ex["churn_rate"])
        if ex.get("incremental_asr_churn_rate") is not None:
            incremental_asr_churns.append(ex["incremental_asr_churn_rate"])
        if ex.get("incremental_translation_churn_rate") is not None:
            incremental_translation_churns.append(ex["incremental_translation_churn_rate"])
        if ex.get("incremental_update_count") is not None:
            incremental_update_counts.append(ex["incremental_update_count"])
        serial_record = is_serial(d)
        if serial_record:  # 串行值保留为正式延迟口径
            latencies.append(d["elapsed_s"])
            if ex.get("ttfb_s"):
                ttfbs.append(ex["ttfb_s"])
            if ex.get("stable_ttfb_s") is not None:
                stable_ttfbs.append(ex["stable_ttfb_s"])
            if ex.get("al_s") is not None:
                als.append(ex["al_s"])
            if ex.get("laal_s") is not None:
                laals.append(ex["laal_s"])
            if ex.get("finish_tail_s") is not None:
                finish_tails.append(ex["finish_tail_s"])
            if ex.get("incremental_asr_ttfb_s") is not None:
                incremental_asr_ttfbs.append(ex["incremental_asr_ttfb_s"])
            if ex.get("incremental_translation_ttfb_s") is not None:
                incremental_translation_ttfbs.append(ex["incremental_translation_ttfb_s"])
            if ex.get("asr_ms_mean") is not None:
                asr_stage_ms.append(ex["asr_ms_mean"])
            if ex.get("text_translate_ms_mean") is not None:
                translation_stage_ms.append(ex["text_translate_ms_mean"])
            if ex.get("e2e_ms_mean") is not None:
                e2e_stage_ms.append(ex["e2e_ms_mean"])
        else:  # 并发仍保留观测耗时，但不得进入正式延迟排名
            concurrent_latencies.append(d["elapsed_s"])
            if isinstance(d.get("workers"), int) and d["workers"] > 1:
                concurrent_workers.add(d["workers"])
        ref = r.get("ref_text", "")
        extra = d.get("extra") or {}
        has_edited_output = "edited_text" in extra
        edited_text = extra.get("edited_text", "") or ""
        # sse 排名默认走 LLM 修饰稿；没有修饰稿的模型保持原始 hyp 口径。
        # --use-edited 兼容旧脚本：现在与默认行为等价，不再单独产出 __edited 文件。
        hyp = edited_text or d["hyp"]
        output_variant_counts["edited" if edited_text else "raw"] += 1
        if not hyp:
            key = "edited_text" if edited_text else "hyp"
            empty_reasons[key] = empty_reasons.get(key, 0) + 1

        simult_diag = None
        translation_segments = ex.get("translation_segments")
        if isinstance(translation_segments, list) and translation_segments:
            record_durations = []
            record_anomalies = []
            previous_text = None
            for index, segment in enumerate(translation_segments):
                if not isinstance(segment, dict):
                    continue
                segment_count += 1
                reason = _segment_reason(segment)
                if reason:
                    segment_reason_observed += 1
                    segment_reason_counts[reason] = segment_reason_counts.get(reason, 0) + 1
                start_s, end_s = segment.get("start_s"), segment.get("end_s")
                duration_s = None
                if isinstance(start_s, (int, float)) and isinstance(end_s, (int, float)):
                    duration_s = round(end_s - start_s, 3)
                    if duration_s >= 0:
                        record_durations.append(duration_s)
                        segment_durations_s.append(duration_s)
                    else:
                        record_anomalies.append({"type": "invalid_segment_timestamps",
                                                 "segment_id": segment.get("segment_id"),
                                                 "start_s": start_s, "end_s": end_s})
                text = str(segment.get("text") or "").strip()
                compact_text = "".join(text.split())
                if duration_s is not None and 0 <= duration_s < 2 and len(compact_text) <= 12:
                    record_anomalies.append({"type": "short_phrase_island",
                                             "segment_id": segment.get("segment_id"),
                                             "duration_s": duration_s, "text": text[:80]})
                if (duration_s is not None and vad_hard_max_s is not None
                        and duration_s > vad_hard_max_s + 0.5):
                    record_anomalies.append({"type": "hard_max_exceeded",
                                             "segment_id": segment.get("segment_id"),
                                             "duration_s": duration_s,
                                             "hard_max_s": vad_hard_max_s})
                norm_text = " ".join(text.lower().split())
                if norm_text and previous_text == norm_text:
                    record_anomalies.append({"type": "adjacent_duplicate_translation",
                                             "segment_id": segment.get("segment_id"),
                                             "text": text[:80]})
                if norm_text:
                    previous_text = norm_text
                timing = segment.get("timing") or {}
                if isinstance(timing, dict) and serial_record:
                    for key, target in (("queue_ms", segment_queue_ms),
                                        ("first_token_ms", segment_ttft_ms),
                                        ("e2e_ms", segment_e2e_ms),
                                        ("total_ms", segment_total_ms)):
                        value = timing.get(key)
                        if isinstance(value, (int, float)):
                            target.append(value)
                    if isinstance(timing.get("queue_ms"), (int, float)) and timing["queue_ms"] > 1000:
                        record_anomalies.append({"type": "queue_spike",
                                                 "segment_id": segment.get("segment_id"),
                                                 "queue_ms": timing["queue_ms"]})
                    if isinstance(timing.get("e2e_ms"), (int, float)) and timing["e2e_ms"] > 5000:
                        record_anomalies.append({"type": "e2e_spike",
                                                 "segment_id": segment.get("segment_id"),
                                                 "e2e_ms": timing["e2e_ms"]})
            src_dur_s = ex.get("src_dur_s") or d.get("audio_s")
            if isinstance(src_dur_s, (int, float)) and src_dur_s > 0:
                simult_audio_seconds += src_dur_s
            record_requests = ex.get("segment_done_count")
            if not isinstance(record_requests, int):
                record_requests = ex.get("n_seg")
            if not isinstance(record_requests, int):
                record_requests = len(translation_segments)
            simult_request_count += max(record_requests, 0)
            empty_segment_count += int(ex.get("empty_segment_count") or 0)
            segment_error_count += int(ex.get("segment_error_count") or 0)
            for anomaly in record_anomalies:
                name = anomaly["type"]
                simult_anomaly_counts[name] = simult_anomaly_counts.get(name, 0) + 1
            simult_diag = {
                "n_segments": len(translation_segments),
                "n_requests": record_requests,
                "segments_per_minute": round(
                    len(translation_segments) * 60 / src_dur_s, 3
                ) if isinstance(src_dur_s, (int, float)) and src_dur_s > 0 else None,
                "segment_duration_p50_s": _pct(record_durations, 50),
                "segment_duration_p95_s": _pct(record_durations, 95),
                "segment_duration_max_s": round(max(record_durations), 3) if record_durations else None,
                "anomalies": record_anomalies,
            }

        if task == "diarization":
            der = _diar_der(r.get("ref") or [], hyp)
            if der is not None:
                der_list.append(der)
            samples.append({"id": r["id"],
                            "ref_norm": f"{r.get('n_spk_ref','?')} 个说话人(参考)",
                            "hyp_norm": f"{(d.get('extra') or {}).get('n_spk_hyp','?')} 个说话人(识别) · DER {der}" if der is not None else "DER 计算失败",
                            "err_rate": der, "elapsed_s": d["elapsed_s"], "rtf": 0})
        elif task == "translate":
            tr_refs.append(ref)
            tr_hyps.append(hyp)
            tr_srcs.append(r.get("source_text", "") or "")
            sample = {"id": r["id"], "src": r.get("source_text", ""),
                      "ref_norm": ref, "hyp_norm": hyp,
                      "err_rate": round(1 - sent_chrf(ref, hyp) / 100, 4),
                      "elapsed_s": d["elapsed_s"], "rtf": 0,
                      **({"al_s": ex.get("al_s")} if ex.get("al_s") is not None else {}),
                      **({"tts_audio": ex.get("tts_audio")} if ex.get("tts_audio") else {})}
            if simult_diag:
                sample["simult"] = simult_diag
            ref_target_norm, target_is_word = _translation_target_norm(ref, r.get("target_lang"))
            hyp_target_norm, _ = _translation_target_norm(hyp, r.get("target_lang"))
            ref_target_len = (len(ref_target_norm.split()) if target_is_word
                              else len(ref_target_norm))
            hyp_target_len = (len(hyp_target_norm.split()) if target_is_word
                              else len(hyp_target_norm))
            if ref_target_len:
                length_ratio = hyp_target_len / ref_target_len
                translation_length_ratios.append(length_ratio)
                sample["length_ratio"] = round(length_ratio, 3)
            literal_error = (word_wer(ref_target_norm, hyp_target_norm) if target_is_word
                             else char_cer(ref_target_norm, hyp_target_norm))
            translation_literal_ref_len += literal_error.ref_len
            translation_literal_del += literal_error.dele
            translation_literal_ins += literal_error.ins
            if literal_error.ref_len:
                sample["literal_omission_rate"] = round(
                    literal_error.dele / literal_error.ref_len, 4
                )
                sample["literal_addition_rate"] = round(
                    literal_error.ins / literal_error.ref_len, 4
                )
            ref_numbers, hyp_numbers = _numeric_literals(ref), _numeric_literals(hyp)
            if ref_numbers:
                hits, total = _multiset_hits(ref_numbers, hyp_numbers)
                numeric_hit += hits
                numeric_total += total
                numeric_hyp_total += len(hyp_numbers)
                sample["numeric_literal_hit"] = hits
                sample["numeric_literal_total"] = total
                sample["numeric_literal_recall"] = round(hits / total, 4)
            ref_dates, hyp_dates = _date_literals(ref), _date_literals(hyp)
            if ref_dates:
                hits, total = _multiset_hits(ref_dates, hyp_dates)
                date_hit += hits
                date_total += total
                sample["date_literal_hit"] = hits
                sample["date_literal_total"] = total
                sample["date_literal_recall"] = round(hits / total, 4)
            terms = [term for term in (r.get("terms") or [])
                     if isinstance(term, dict) and term.get("target")]
            if terms:
                hits = 0
                padded_hyp = f" {hyp_target_norm} "
                for term in terms:
                    term_norm, _ = _translation_target_norm(term["target"], r.get("target_lang"))
                    if term_norm and ((f" {term_norm} " in padded_hyp) if target_is_word
                                      else (term_norm in hyp_target_norm)):
                        hits += 1
                translation_term_hits += hits
                translation_term_total += len(terms)
                sample["term_hit"] = hits
                sample["term_total"] = len(terms)
                sample["term_recall"] = round(hits / len(terms), 4)
            source_ref = str(r.get("source_text") or "").strip()
            source_hyp = str(ex.get("asr_text") or ex.get("source_text") or "").strip()
            if source_ref and source_hyp:
                source_norm, source_is_word = _asr_norm_metric(
                    r.get("source_lang") or r.get("lang")
                )
                source_ref_norm = source_norm(source_ref)
                source_hyp_norm = source_norm(source_hyp)
                source_error = (word_wer(source_ref_norm, source_hyp_norm)
                                if source_is_word else char_cer(source_ref_norm, source_hyp_norm))
                source_asr_ref_lens.append(source_error.ref_len)
                source_asr_edit_counts.append(source_error.edits)
                source_asr_metric_names.add("WER" if source_is_word else "CER")
                sample.update({
                    "source_asr_hyp": source_hyp,
                    "source_asr_err_rate": round(source_error.cer, 4),
                })
            samples.append(sample)
        elif task == "summarize":
            ro = rouge_char(ref, hyp)
            sum_rl.append(ro["rl"])
            sum_r1.append(ro["r1"])
            samples.append({"id": r["id"], "src": r.get("source_text", ""),
                            "ref_norm": ref, "hyp_norm": hyp,
                            "err_rate": round(1 - ro["rl"], 4),
                            "elapsed_s": d["elapsed_s"], "rtf": 0})
        else:  # asr
            norm_fn, is_word = _asr_norm_metric(r.get("lang"))
            ref_n = norm_fn(ref)
            hyp_raw_n = norm_fn(d["hyp"])
            hyp_n = norm_fn(hyp)
            c = word_wer(ref_n, hyp_n) if is_word else char_cer(ref_n, hyp_n)
            ref_lens.append(c.ref_len)
            edit_counts.append(c.edits)
            sub_counts.append(c.sub)
            del_counts.append(c.dele)
            ins_counts.append(c.ins)
            sent_total += 1
            if c.edits > 0:
                sent_err += 1
            hyp_len = len(hyp_n.split()) if is_word else len(hyp_n)
            hyp_lens.append(hyp_len)
            if c.ref_len > 0:
                length_ratios.append(hyp_len / c.ref_len)
            # sse 有修饰稿时：主口径=edited，额外保留原始 ASR raw.* 口径。
            raw_c = None
            if has_edited_output:
                raw_c = word_wer(ref_n, hyp_raw_n) if is_word else char_cer(ref_n, hyp_raw_n)
                raw_ref_lens.append(raw_c.ref_len)
                raw_edit_counts.append(raw_c.edits)
            kws = r.get("keywords") or []
            # 别名与 hyp 必须同一口径（英文 normalize_en、其它各按 _asr_norm_metric；中文 normalize 会把
            # 数字转成中文数字，导致带数字的英文关键词永远匹配不上）
            kw_norm = norm_fn
            kws_norm = [{"aliases": [kw_norm(a) for a in ([k] if isinstance(k, str)
                         else (k.get("aliases") or [k.get("surface", "")]))]} for k in kws]
            hit, tot = keyword_hits(hyp_n, kws_norm)
            kw_hit_total += hit
            kw_total += tot
            dur = d.get("audio_s", 0.0)
            rtf = (d["elapsed_s"] / dur) if dur else 0.0
            if serial_record:
                rtfs.append(rtf)
            s = {"id": r["id"], "ref_norm": ref_n, "hyp_norm": hyp_n,
                 "err_rate": round(c.cer, 4), "kw_hit": hit, "kw_total": tot,
                 "sub": c.sub, "del": c.dele, "ins": c.ins,
                 "ref_len": c.ref_len, "hyp_len": hyp_len,
                 "length_ratio": round(hyp_len / c.ref_len, 3) if c.ref_len else None,
                 "audio_s": dur, "elapsed_s": d["elapsed_s"], "rtf": round(rtf, 3)}
            if has_edited_output:
                s["raw"] = {"hyp_norm": hyp_raw_n, "err_rate": round(raw_c.cer, 4)}
            samples.append(s)

    for sample in samples:
        exclusion = exclusions.get(str(sample.get("id")))
        if exclusion:
            sample["excluded_from_reviewed"] = True
            sample["exclusion_reason"] = exclusion.get("reason")

    # 溯源元信息：哪个模型/端点、哪份样本(指纹)、何时跑/何时打分
    cur_sha = manifest_sha(manifest)
    cur_sha_v2 = manifest_sha_v2(manifest)
    last = infer_meta
    active_variants = [name for name, count in output_variant_counts.items() if count]
    output_variant = active_variants[0] if len(active_variants) == 1 else ("mixed" if active_variants else "raw")
    meta = {"schema_version": 2,
            "run_id": last.get("run_id"), "run_spec": last.get("run_spec"),
            "code_revision": code_revision(),
            "infer_code_revision": last.get("code_revision"),
            "metric_signature": None,
            "output_variant_used": output_variant,
            "output_variant_counts": output_variant_counts,
            "model": last.get("model"), "endpoint": last.get("endpoint"),
            "language": last.get("language"),
            "target_lang": last.get("target_lang")
                           or (last.get("run_spec") or {}).get("target_lang"),
            "domain": last.get("domain"), "hotwords": last.get("hotwords") or None,
            "dataset_hotwords": last.get("dataset_hotwords") or None,
            "hotwords_text": last.get("hotwords_text")
                             or (last.get("run_spec") or {}).get("hotwords_text"),
            "request_params": last.get("request_params")
                              or (last.get("run_spec") or {}).get("request_params")
                              or {},
            "config_hash": last.get("config_hash")
                           or (last.get("run_spec") or {}).get("config_hash"),
            "client_host": last.get("client_host"),
            "note": last.get("note"),
            "manifest_sha": cur_sha,
            "manifest_sha_v2": cur_sha_v2,
            "infer_runs": len(metas) or None,
            "infer_started": last.get("started"),
            "scored_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if ws_control_examples:
        meta["ws_control"] = ws_control_examples[0]
        meta["ws_control_signature_counts"] = [
            {"signature": signature, "n": count}
            for signature, count in sorted(ws_control_signature_counts.items())
        ]
    if metas and last.get("manifest_sha") and last["manifest_sha"] != cur_sha:
        # manifest 在 infer 之后被重新生成过 → hyp 和样本可能对不上，结果不可信
        meta["manifest_sha_mismatch"] = last["manifest_sha"]
        log.warning("manifest 指纹与 infer 时不一致(%s → %s)，清单可能已重建，本结果不可信！",
                    last["manifest_sha"], cur_sha)
    if metas and last.get("manifest_sha_v2") and last["manifest_sha_v2"] != cur_sha_v2:
        meta["manifest_sha_v2_mismatch"] = last["manifest_sha_v2"]
        log.warning("manifest_sha_v2 与 infer 时不一致(%s → %s)；仅记录观察，暂不改变分组或排名",
                    last["manifest_sha_v2"], cur_sha_v2)

    coverage = n_ok / len(rows) if rows else 0.0
    summary = {"schema_version": 2,
               "task": task, "infer_file": infer, "manifest": manifest,
               "meta": meta,
               "output_variant_used": output_variant,
               "n_total": len(rows), "n_ok": n_ok, "n_fail": n_fail,
               "fail_rate": round(n_fail / len(rows), 4) if rows else None,
               "low_coverage": coverage < COVERAGE_MIN,  # 真 → 精度只基于部分样本，不可信
               "latency_valid": bool(latencies),
               "latency_p50_s": _pct(latencies, 50),
               "latency_p95_s": _pct(latencies, 95),
               **({"concurrent_latency_p50_s": _pct(concurrent_latencies, 50),
                   "concurrent_latency_p95_s": _pct(concurrent_latencies, 95),
                   "concurrent_workers": sorted(concurrent_workers) or None}
                  if concurrent_latencies else {}),
               **({"ttfb_p50_s": _pct(ttfbs, 50)} if ttfbs else {}),
               **({"ttfb_p95_s": _pct(ttfbs, 95)} if ttfbs else {}),
               **({"stable_ttfb_p50_s": _pct(stable_ttfbs, 50),
                   "stable_ttfb_p95_s": _pct(stable_ttfbs, 95)} if stable_ttfbs else {}),
               **({"al_p50_s": _pct(als, 50), "al_p95_s": _pct(als, 95),
                   "laal_p50_s": _pct(laals, 50), "laal_p95_s": _pct(laals, 95)} if als else {}),
               **({"finish_tail_p50_s": _pct(finish_tails, 50),
                   "finish_tail_p95_s": _pct(finish_tails, 95)} if finish_tails else {}),
               **({"incremental_asr_ttfb_p50_s": _pct(incremental_asr_ttfbs, 50)}
                  if incremental_asr_ttfbs else {}),
               **({"incremental_translation_ttfb_p50_s": _pct(
                       incremental_translation_ttfbs, 50)}
                  if incremental_translation_ttfbs else {}),
               **({"incremental_update_count_mean": round(statistics.mean(
                       incremental_update_counts), 2)}
                  if incremental_update_counts else {}),
               **({"incremental_asr_churn_rate": round(statistics.mean(
                       incremental_asr_churns), 3)}
                  if incremental_asr_churns else {}),
               **({"incremental_translation_churn_rate": round(statistics.mean(
                       incremental_translation_churns), 3)}
                  if incremental_translation_churns else {}),
               **({"asr_stage_p50_ms": _pct(asr_stage_ms, 50)} if asr_stage_ms else {}),
               **({"translation_stage_p50_ms": _pct(translation_stage_ms, 50)}
                  if translation_stage_ms else {}),
               **({"e2e_stage_p50_ms": _pct(e2e_stage_ms, 50),
                   "e2e_stage_p95_ms": _pct(e2e_stage_ms, 95)} if e2e_stage_ms else {}),
               **({"revision_rate": round(sum(revs) / len(revs), 3)} if revs else {}),
               **({"churn_rate": round(sum(churns) / len(churns), 3)} if churns else {}),
               **({"stability_coverage": round(sum(stability_flags) / len(stability_flags), 3)}
                  if stability_flags else {}),
               **({"use_edited": True} if use_edited else {}),
               **({"ranked_by": "edited"} if raw_ref_lens else {}),
               **({"empty_reasons": [{"reason": k, "n": v} for k, v in
                                    sorted(empty_reasons.items(), key=lambda kv: -kv[1])[:5]]}
                  if empty_reasons else {}),
               **({"fail_reasons": [{"reason": k, "n": v} for k, v in
                                    sorted(fail_reasons.items(), key=lambda kv: -kv[1])[:5]]}
                  if fail_reasons else {})}
    if attempt_counts:
        retry_total = sum(max(0, value - 1) for value in attempt_counts)
        summary.update({
            "retry_total": retry_total,
            "retry_sample_rate": round(
                sum(value > 1 for value in attempt_counts) / len(attempt_counts), 4
            ),
        })
    if segment_count:
        summary["simult_vad"] = {
            "n_segments": segment_count,
            "n_requests": simult_request_count,
            "audio_minutes": round(simult_audio_seconds / 60, 3),
            "segments_per_minute": round(
                segment_count * 60 / simult_audio_seconds, 3
            ) if simult_audio_seconds else None,
            "requests_per_minute": round(
                simult_request_count * 60 / simult_audio_seconds, 3
            ) if simult_audio_seconds else None,
            "avg_audio_s_per_request": round(
                statistics.mean(segment_durations_s), 3
            ) if segment_durations_s else None,
            "segment_duration_p50_s": _pct(segment_durations_s, 50),
            "segment_duration_p90_s": _pct(segment_durations_s, 90),
            "segment_duration_p95_s": _pct(segment_durations_s, 95),
            "segment_duration_max_s": round(max(segment_durations_s), 3)
                                      if segment_durations_s else None,
            "segment_lt_2s_rate": round(
                sum(value < 2 for value in segment_durations_s) / len(segment_durations_s), 4
            ) if segment_durations_s else None,
            "segment_gt_15s_rate": round(
                sum(value > 15 for value in segment_durations_s) / len(segment_durations_s), 4
            ) if segment_durations_s else None,
            "segment_gt_20s_rate": round(
                sum(value > 20 for value in segment_durations_s) / len(segment_durations_s), 4
            ) if segment_durations_s else None,
            "segment_gt_soft_max_rate": round(
                sum(value > vad_soft_max_s for value in segment_durations_s)
                / len(segment_durations_s), 4
            ) if segment_durations_s and vad_soft_max_s is not None else None,
            "segment_gt_hard_max_rate": round(
                sum(value > vad_hard_max_s for value in segment_durations_s)
                / len(segment_durations_s), 4
            ) if segment_durations_s and vad_hard_max_s is not None else None,
            "segment_reason_coverage": round(
                segment_reason_observed / segment_count, 4
            ),
            "segment_reason_counts": segment_reason_counts,
            "empty_segment_count": empty_segment_count,
            "segment_error_count": segment_error_count,
            "queue_p50_ms": _pct(segment_queue_ms, 50),
            "queue_p95_ms": _pct(segment_queue_ms, 95),
            "model_ttft_p50_ms": _pct(segment_ttft_ms, 50),
            "model_ttft_p95_ms": _pct(segment_ttft_ms, 95),
            "e2e_p50_ms": _pct(segment_e2e_ms, 50),
            "e2e_p95_ms": _pct(segment_e2e_ms, 95),
            "inference_p50_ms": _pct(segment_total_ms, 50),
            "inference_p95_ms": _pct(segment_total_ms, 95),
            "anomaly_counts": simult_anomaly_counts,
        }
    if coverage < COVERAGE_MIN:
        log.warning("覆盖率 %.0f%% < %.0f%%：失败样本不计入精度，本结果分数偏乐观，榜单将标灰不参与排名",
                    coverage * 100, COVERAGE_MIN * 100)

    if task == "diarization":
        summary.update({"metric": "DER",
                        "DER": round(statistics.mean(der_list), 4) if der_list else None,
                        "err_rate": round(statistics.mean(der_list), 4) if der_list else None,
                        "err_ci95": _bootstrap_ci(der_list, statistics.mean) if der_list else None})
    elif task == "translate":
        sc = translation_corpus(tr_refs, tr_hyps) if tr_refs else {"BLEU": None, "chrF": None}
        errs = [s["err_rate"] for s in samples if "err_rate" in s]
        summary.update({"metric": "chrF", "chrF": sc["chrF"], "BLEU": sc["BLEU"],
                        "err_rate": round(1 - sc["chrF"] / 100, 4) if sc["chrF"] is not None else None,
                        # CI 基于句级 chrF 均值口径（语料级 chrF 重算太贵），看波动够用
                        "err_ci95": _bootstrap_ci(errs, statistics.mean)})
        if translation_length_ratios:
            summary.update({
                "translation_length_ratio_mean": round(
                    statistics.mean(translation_length_ratios), 3
                ),
                "translation_length_ratio_p50": _pct(translation_length_ratios, 50),
                "translation_length_ratio_p95": _pct(translation_length_ratios, 95),
            })
        if translation_term_total:
            summary.update({
                "term_recall": round(translation_term_hits / translation_term_total, 4),
                "term_hit": translation_term_hits,
                "term_total": translation_term_total,
            })
        if translation_literal_ref_len:
            summary.update({
                # 译文允许意译，因此只作为字面漏/增译代理，不进入综合选优硬门槛。
                "literal_omission_proxy_rate": round(
                    translation_literal_del / translation_literal_ref_len, 4
                ),
                "literal_addition_proxy_rate": round(
                    translation_literal_ins / translation_literal_ref_len, 4
                ),
                "literal_proxy_basis": "target-reference edit deletions/insertions; paraphrase-sensitive",
            })
        if numeric_total:
            summary.update({
                "numeric_literal_recall": round(numeric_hit / numeric_total, 4),
                "numeric_literal_precision": round(
                    numeric_hit / numeric_hyp_total, 4
                ) if numeric_hyp_total else 0.0,
                "numeric_literal_hit": numeric_hit,
                "numeric_literal_total": numeric_total,
                "numeric_literal_scope": "explicit Arabic numerals only; no spoken-number guessing",
            })
        if date_total:
            summary.update({
                "date_literal_recall": round(date_hit / date_total, 4),
                "date_literal_hit": date_hit,
                "date_literal_total": date_total,
                "date_literal_scope": "explicit YYYY-MM-DD or YYYY年M月D日 only",
            })
        if source_asr_ref_lens:
            summary.update({
                "source_asr_metric": (
                    next(iter(source_asr_metric_names))
                    if len(source_asr_metric_names) == 1 else "mixed"
                ),
                "source_asr_error_rate": round(corpus_cer(
                    source_asr_ref_lens, source_asr_edit_counts
                ), 4),
                "source_asr_coverage": round(len(source_asr_ref_lens) / max(n_ok, 1), 4),
            })
    elif task == "summarize":
        errs = [round(1 - v, 4) for v in sum_rl]
        summary.update({"metric": "ROUGE-L",
                        "ROUGE-L": round(statistics.mean(sum_rl), 4) if sum_rl else None,
                        "ROUGE-1": round(statistics.mean(sum_r1), 4) if sum_r1 else None,
                        "err_rate": round(1 - statistics.mean(sum_rl), 4) if sum_rl else None,
                        "err_ci95": _bootstrap_ci(errs, statistics.mean)})
    else:
        mn = "WER" if (rows and _asr_norm_metric(rows[0].get("lang"))[1]) else "CER"
        pairs = list(zip(ref_lens, edit_counts))
        summary.update({"metric": mn,
                        mn: round(corpus_cer(ref_lens, edit_counts), 4) if ref_lens else None,
                        "err_rate": round(corpus_cer(ref_lens, edit_counts), 4) if ref_lens else None,
                        # 语料级 CER 的精确 bootstrap：重抽样后 Σ编辑/Σ参考字数
                        "err_ci95": _bootstrap_ci(
                            pairs, lambda s: sum(e for _, e in s) / max(sum(r for r, _ in s), 1)),
                        "SER": round(sent_err / sent_total, 4) if sent_total else None,
                        "sub_rate": round(sum(sub_counts) / max(sum(ref_lens), 1), 4) if ref_lens else None,
                        "del_rate": round(sum(del_counts) / max(sum(ref_lens), 1), 4) if ref_lens else None,
                        "ins_rate": round(sum(ins_counts) / max(sum(ref_lens), 1), 4) if ref_lens else None,
                        "extra_text_rate": round(sum(ins_counts) / max(sum(ref_lens), 1), 4) if ref_lens else None,
                        "avg_length_ratio": round(statistics.mean(length_ratios), 3) if length_ratios else None,
                        "total_sub": sum(sub_counts),
                        "total_del": sum(del_counts),
                        "total_ins": sum(ins_counts),
                        "total_ref_len": sum(ref_lens),
                        "total_hyp_len": sum(hyp_lens),
                        "KRR": round(kw_hit_total / kw_total, 4) if kw_total else None,
                        "rtf_mean": round(statistics.mean(rtfs), 3) if rtfs else None})
        if raw_ref_lens:
            raw_metric = f"{mn}_raw"
            raw_pairs = list(zip(raw_ref_lens, raw_edit_counts))
            summary.update({
                raw_metric: round(corpus_cer(raw_ref_lens, raw_edit_counts), 4) if raw_ref_lens else None,
                "err_rate_raw": round(corpus_cer(raw_ref_lens, raw_edit_counts), 4) if raw_ref_lens else None,
                "err_ci95_raw": _bootstrap_ci(
                    raw_pairs, lambda s: sum(e for _, e in s) / max(sum(r for r, _ in s), 1)),
                "n_raw": len(raw_ref_lens),
            })
        reviewed = reviewed_asr_metrics(summary, samples, exclusions)
        if reviewed:
            summary["reviewed"] = reviewed

    # 可选增强：embedding 语义相似度。翻译用完整译文(tr_refs/tr_hyps)，ASR 用规一化文本。容错降级。
    if embedding:
        if task == "translate":
            _add_embedding_metric(summary, samples, list(zip(tr_refs, tr_hyps)))
        elif task not in ("diarization", "summarize"):
            _add_embedding_metric(summary, samples)

    # 可选增强：远端 XCOMET-XL 优先；GPU scorer 未配置/不可用时回退本地 COMET。
    if comet and task == "translate" and tr_refs:
        triples = list(zip(tr_srcs, tr_refs, tr_hyps))
        remote = remote_xcomet_score(triples)
        if remote:
            scores = remote["scores"]
            system_score = remote.get("system_score")
            summary["xcomet"] = round(
                float(system_score) if system_score is not None else statistics.mean(scores), 4
            ) if scores else None
            summary["xcomet_model"] = remote.get("model") or "Unbabel/XCOMET-XL"
            spans = remote.get("error_spans") or []
            severity_counts = {}
            si = 0
            for s in samples:
                if "error" not in s and si < len(scores):
                    s["xcomet"] = scores[si]
                    s["xcomet_error_spans"] = spans[si] if si < len(spans) else []
                    for span in s["xcomet_error_spans"]:
                        severity = str((span or {}).get("severity") or "unknown").lower()
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    si += 1
            summary["xcomet_error_severity_counts"] = severity_counts
        else:
            scores = comet_score(triples)
            if scores is None:
                summary["comet"] = None
                log.info("XCOMET/COMET 未计算（GPU scorer 或本地 COMET 不可用），其余指标不受影响")
            else:
                summary["comet"] = round(sum(scores) / len(scores), 4) if scores else None
                si = 0
                for s in samples:
                    if "error" not in s and si < len(scores):
                        s["comet"] = scores[si]
                        si += 1

    # 官方同传口径：只对真实流式事件/旧版翻译事件自动执行；普通离线翻译不触发。
    if task == "translate" and _has_simult_trace(inf):
        _add_omnisteval_metrics(summary, manifest, infer, out)

    # TTS 保真度：ASR 回转译后语音 → 比模型译文 CER/WER。需先 --save-audio 存了 tts_audio。
    if asr_bleu and task == "translate":
        _L2 = {
            "英文": "en", "English": "en", "english": "en",
            "中文": "zh", "简体中文": "zh", "Chinese": "zh", "chinese": "zh",
            "日文": "ja", "Japanese": "ja", "japanese": "ja",
            "韩文": "ko", "Korean": "ko", "korean": "ko",
            "en": "en", "zh": "zh", "ja": "ja", "ko": "ko",
        }
        lang_by_id = {r["id"]: _L2.get(r.get("target_lang", ""), "en") for r in rows}
        hyp_by_id = {r["id"]: r["hyp"] for r in inf.values() if r.get("ok") and r.get("hyp")}   # 完整译文(samples 里被截断)
        _add_tts_fidelity(summary, samples, lang_by_id, hyp_by_id=hyp_by_id)
        source_audio_by_id = {str(r["id"]): r.get("audio_path") for r in rows if r.get("audio_path")}
        _add_speech_metrics(summary, samples, source_audio_by_id)

    metric_signature = _metric_signature(summary, output_variant)
    summary["metric_signature"] = metric_signature
    meta["metric_signature"] = metric_signature

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    atomic_write_json(out, {"summary": summary, "samples": samples})
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n明细 → {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--infer", required=True, help="infer 阶段产出的 jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrent", action="store_true", help="标记延迟不可信")
    ap.add_argument("--embedding", action="store_true",
                    help="ASR 加 embedding 语义相似度(可选增强,需 EMB_KEY;端点挂则降级不影响主指标)")
    ap.add_argument("--comet", action="store_true",
                    help="翻译加 COMET 神经质量(可选增强,需 unbabel-comet+模型;下不动则降级不影响主指标)")
    ap.add_argument("--asr-bleu", action="store_true",
                    help="同传：ASR 回转译后语音算 ASR-BLEU 语音保真(需 infer --save-audio)")
    ap.add_argument("--use-edited", action="store_true",
                    help="用 extra.edited_text(Pass 2 LLM 修饰稿)替代 hyp(Pass 1 原始 ASR)打分，"
                         "供 sse 两路对比")
    args = ap.parse_args()
    run(args.manifest, args.infer, args.out, args.concurrent, args.embedding, args.comet,
        args.asr_bleu, args.use_edited)


if __name__ == "__main__":
    main()
