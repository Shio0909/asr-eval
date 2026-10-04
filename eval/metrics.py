"""ASR 精度指标：CER（主）+ KRR（关键词召回，辅）。

- CER: 规一化后字级编辑距离 / 参考字数。语料级 = Σ编辑距离 / Σ参考字数。
- KRR: 在标了 keyword 的样本上，统计 Hyp 命中的固定表达占比；支持别名匹配。
"""

from dataclasses import dataclass

import jiwer


@dataclass
class CERResult:
    cer: float            # 该句 CER
    ref_len: int          # 参考字数
    edits: int            # 编辑距离(S+D+I)
    sub: int
    dele: int
    ins: int


def char_cer(ref_norm: str, hyp_norm: str) -> CERResult:
    """单句字级 CER。输入须为已规一化文本。"""
    # 字级：把字符串拆成以空格分隔的"字"，用 jiwer 词级口径算 = 字级
    r = " ".join(list(ref_norm))
    h = " ".join(list(hyp_norm))
    if not ref_norm:
        # 参考为空：有 hyp 则全是插入错误，CER 记为 hyp 长度(或 0/0→0)
        ins = len(hyp_norm)
        return CERResult(cer=(1.0 if ins else 0.0), ref_len=0, edits=ins, sub=0, dele=0, ins=ins)
    out = jiwer.process_words([r], [h])
    edits = out.substitutions + out.deletions + out.insertions
    return CERResult(
        cer=edits / len(ref_norm),
        ref_len=len(ref_norm),
        edits=edits,
        sub=out.substitutions,
        dele=out.deletions,
        ins=out.insertions,
    )


def word_wer(ref_norm: str, hyp_norm: str) -> CERResult:
    """词级 WER（英文）。输入须为已规一化(小写、保留词间空格)文本。"""
    ref_words = ref_norm.split()
    if not ref_words:
        ins = len(hyp_norm.split())
        return CERResult(cer=(1.0 if ins else 0.0), ref_len=0, edits=ins, sub=0, dele=0, ins=ins)
    out = jiwer.process_words([ref_norm], [hyp_norm])
    edits = out.substitutions + out.deletions + out.insertions
    return CERResult(cer=edits / len(ref_words), ref_len=len(ref_words),
                     edits=edits, sub=out.substitutions, dele=out.deletions, ins=out.insertions)


def corpus_cer(ref_lens, edit_counts) -> float:
    """语料级 CER = Σ编辑距离 / Σ参考字数（不是逐句 CER 取平均）。"""
    total_ref = sum(ref_lens)
    return (sum(edit_counts) / total_ref) if total_ref else 0.0


def translation_corpus(refs: list, hyps: list) -> dict:
    """翻译语料级 chrF + BLEU（sacrebleu，本地算无需 LLM）。

    chrF 字符级语言无关；BLEU 按参考文本 CJK 占比自动选分词器——
    中文译文（en→zh 等方向）必须 tokenize='zh'，默认 13a 按空格切会严重失真。
    """
    import sacrebleu
    sample = "".join(refs)[:2000]
    cjk = sum(1 for ch in sample if "一" <= ch <= "鿿")
    tok = "zh" if sample and cjk / len(sample) > 0.3 else "13a"
    return {
        "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok).score, 2),
        "chrF": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
    }


def sent_chrf(ref: str, hyp: str) -> float:
    import sacrebleu
    return round(sacrebleu.sentence_chrf(hyp, [ref]).score, 1)


def rouge_char(ref: str, hyp: str) -> dict:
    """中文字级 ROUGE-1(unigram F1) + ROUGE-L(LCS F1)。本地算，无需 LLM。"""
    a, b = list(ref), list(hyp)
    if not a or not b:
        return {"r1": 0.0, "rl": 0.0}
    # ROUGE-1: 字级 unigram 重叠
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    p = overlap / len(b)
    r = overlap / len(a)
    r1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    # ROUGE-L: LCS
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[n]
    lp, lr = lcs / len(b), lcs / len(a)
    rl = (2 * lp * lr / (lp + lr)) if (lp + lr) else 0.0
    return {"r1": round(r1, 4), "rl": round(rl, 4)}


def _embedding_key() -> str:
    """embedding 端点鉴权 key：env EMB_KEY 优先，否则 secrets_local.EMBEDDING_KEY。"""
    import os
    k = os.environ.get("EMB_KEY", "")
    if k:
        return k
    try:
        from secrets_local import EMBEDDING_KEY
        return EMBEDDING_KEY or ""
    except Exception:
        return ""


def embedding_cosine(pairs, batch=32, timeout=20.0, retries=2):
    """ref/hyp 文本对 → 每对的余弦相似度列表。**纯增强指标，任何失败返回 None（绝不抛）**。

    pairs: [(ref, hyp), ...]（建议传规一化后文本，与 CER 同口径）。
    调外部 OpenAI 兼容 /v1/embeddings（默认 siliconflow 免费 bge-m3，见 config）。
    设计目标=「挂了不影响主流程」：无 key / 网络 / 限流 / 超时 / 返回格式异常
    → 整体返回 None，调用方据此降级；单句文本缺失 embedding → 该位置为 None，其余照常。
    """
    import math
    import time
    key = _embedding_key()
    if not key:
        return None  # 未配 key → 静默跳过（不是错误，是没开启）
    try:
        import requests
        from config import EMBEDDING_MODEL, EMBEDDING_URL
    except Exception:
        return None
    texts = list({t for p in pairs for t in p if t})  # ref/hyp 去重，省调用
    emb = {}
    try:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            for attempt in range(retries + 1):
                try:
                    r = requests.post(EMBEDDING_URL,
                                      headers={"Authorization": f"Bearer {key}"},
                                      json={"model": EMBEDDING_MODEL, "input": chunk},
                                      timeout=timeout)
                    r.raise_for_status()
                    for t, item in zip(chunk, r.json()["data"]):
                        emb[t] = item["embedding"]
                    break
                except Exception:
                    if attempt < retries:
                        time.sleep(0.5 * (2 ** attempt))  # 退避（限流友好）
                    else:
                        raise
    except Exception:
        return None  # 重试用尽仍失败 → 整体降级，主指标不受影响

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    return [_cos(emb[ref], emb[hyp]) if ref in emb and hyp in emb else None
            for ref, hyp in pairs]


def _tgt_tokens(text: str, tgt_lang: str = "en") -> list:
    """目标语切词：CJK(中/日/韩) 按字、其余按空格。用于 AL 的目标长度计数。"""
    cjk = ("zh", "ja", "ko", "yue", "中文", "中", "cn", "日文", "韩文")
    if any(k in tgt_lang for k in cjk):
        return [c for c in text if not c.isspace()]
    return text.split()


def average_lagging(emissions, src_dur_s: float, ref_text: str = "",
                    tgt_lang: str = "en"):
    """同传延迟指标 AL / LAAL（秒），对齐 SimulEval/Ma et al. 口径。

    emissions: [(g_s, segment_text), ...] 按吐字顺序，g_s = 该段译文吐出时**已消耗的源音频秒数**
               （适配器在 1x 实时节奏下采集；非实时喂则失真）。
    AL  = (1/τ)·Σ_{i=1..τ}( g(i) − (i−1)·T/|Y| )，τ=首个 g(i)≥T 的目标 token 下标，T=源时长。
    LAAL 把分母 |Y| 换成 max(|Y|,|Y_ref|)，防短译文把 AL 刷低。
    信息不足（无 emission/源时长≤0/无目标 token）→ 返回 (None, None)，不抛。
    """
    if not emissions or not src_dur_s or src_dur_s <= 0:
        return None, None
    delays = []
    for g_s, seg in emissions:
        delays.extend([g_s] * len(_tgt_tokens(seg, tgt_lang)))  # 段内每个 token 同延迟
    if not delays:
        return None, None
    T = float(src_dur_s)
    Y = len(delays)
    Yref = len(_tgt_tokens(ref_text, tgt_lang)) or Y
    tau = next((i for i, g in enumerate(delays, 1) if g >= T - 1e-6), Y)  # 全源消耗的截断点

    def _al(denom):
        return sum(delays[i - 1] - (i - 1) * T / denom for i in range(1, tau + 1)) / tau

    return round(_al(Y), 3), round(_al(max(Y, Yref)), 3)


_COMET_CACHE = {}


def _external_comet_python():
    """返回隔离 COMET 环境的 Python；主环境无需承受其 NumPy/protobuf 版本约束。"""
    import os
    candidates = [
        os.environ.get("COMET_PYTHON", ""),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv-comet", "bin", "python"),
    ]
    return next((p for p in candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)


def comet_runtime_available():
    """主环境可导入 COMET，或已配置可用的隔离 COMET Python。"""
    import importlib.util
    import os
    import subprocess
    if importlib.util.find_spec("comet") is not None:
        return True
    python = _external_comet_python()
    if not python:
        return False
    worker = os.path.join(os.path.dirname(__file__), "comet_worker.py")
    try:
        return subprocess.run([python, worker, "--check"], timeout=10,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _external_comet_score(triples, model, batch):
    import json
    import os
    import subprocess
    python = _external_comet_python()
    if not python:
        return None
    worker = os.path.join(os.path.dirname(__file__), "comet_worker.py")
    payload = json.dumps({"model": model, "batch": batch, "triples": triples}, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [python, worker], input=payload, text=True, capture_output=True,
            timeout=float(os.environ.get("COMET_TIMEOUT_S", "3600")),
        )
        if proc.returncode:
            return None
        scores = json.loads(proc.stdout).get("scores")
        return [round(float(x), 4) for x in scores] if scores is not None else None
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        return None


def comet_score(triples, model="Unbabel/wmt22-comet-da", batch=8):
    """COMET 神经翻译质量(参考式，0~1 越高越好)。triples=[(src, ref, hyp), ...] → 每条分数列表。

    比 chrF/BLEU 更贴人工评分(WMT 元评测冠军级)，专治"换说法但意思对"被字面指标低估。
    **可选增强，全程容错返回 None(绝不抛)**：未装 unbabel-comet / 下不动模型 / 无算力 → 整体 None，
    主指标(chrF/BLEU)不受影响。模型 ~2.3G + 需 torch，故 opt-in(--comet)。模型按进程缓存避免重载。
    """
    try:
        from comet import download_model, load_from_checkpoint
    except Exception:
        return _external_comet_score(triples, model, batch)
    try:
        if model not in _COMET_CACHE:
            _COMET_CACHE[model] = load_from_checkpoint(download_model(model))
        data = [{"src": s or "", "ref": r or "", "mt": h or ""} for s, r, h in triples]
        out = _COMET_CACHE[model].predict(
            data, batch_size=batch, gpus=0, num_workers=1, progress_bar=False)
        scores = out.get("scores") if isinstance(out, dict) else getattr(out, "scores", None)
        return [round(float(x), 4) for x in scores] if scores is not None else None
    except Exception:
        return None  # 下模型/推理失败 → 整体降级，主指标不动


def keyword_hits(hyp_norm: str, keywords: list) -> tuple[int, int]:
    """KRR 单句：返回 (命中数, 关键词总数)。

    keywords: [{"surface": str, "aliases": [str,...]}] 或 [str,...]
    别名命中任一即算命中。匹配在规一化文本上做（调用方应传入已规一化的别名）。
    """
    if not keywords:
        return 0, 0
    hit = 0
    for kw in keywords:
        if isinstance(kw, str):
            aliases = [kw]
        else:
            aliases = kw.get("aliases") or [kw.get("surface", "")]
        if any(a and a in hyp_norm for a in aliases):
            hit += 1
    return hit, len(keywords)
