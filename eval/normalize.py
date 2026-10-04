"""统一文本规一化 — 复用 SpeechIO LeaderBoard 的 TextNorm。

口径对齐旧报告《ASR能力横评》:
全角转半角、英文统一小写、去语气词、去儿化、繁转简、数字规一化、去空格。
ref 与 hyp 用同一个 normalizer，保证口径一致。
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from textnorm_zh import TextNorm  # noqa: E402

# 默认配置 = LeaderBoard 口径
_NORMALIZER = TextNorm(
    to_banjiao=True,     # 全角→半角
    to_upper=False,
    to_lower=True,       # 英文统一小写
    remove_fillers=True, # 去语气词 呃 啊
    remove_erhua=True,   # 去儿化
    check_chars=False,
    remove_space=True,   # 去空格(LeaderBoard 标准; CER 字级)
    cc_mode="t2s",       # 繁→简
)


def normalize(text: str) -> str:
    """中文规一化(字级,去空格)。空/None 返回空串。"""
    if not text:
        return ""
    # KeSpeech transcript control token: it marks non-lexical spoken noise and
    # is not text an ASR model should be expected to emit.  Remove the exact
    # upstream marker before TextNorm turns it into the literal "spoken noise".
    text = text.replace("<SPOKEN_NOISE>", "")
    out = _NORMALIZER(text)
    return out or ""


_EN_NORMALIZER = None

_EN_NUMBER_BOUNDARY = "asrnumberboundary"
_EN_CARDINAL_WORD = (
    r"zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
_EN_LONG_DIGIT_CARDINAL_RE = re.compile(
    rf"\b(\d{{3,}}(?:[.,]\d+)?)\s*,?\s+(?=(?:{_EN_CARDINAL_WORD})\b)",
    re.IGNORECASE,
)
_EN_SPOKEN_YEAR_RE = (
    rf"(?:nineteen|twenty)[ -]+(?:"
    rf"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[ -]+(?:one|two|three|four|five|six|seven|eight|nine))?"
    rf"|oh[ -]+(?:one|two|three|four|five|six|seven|eight|nine)"
    rf"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|hundred)"
)
_EN_SPOKEN_YEAR_PERCENT_RE = re.compile(
    rf"\b({_EN_SPOKEN_YEAR_RE})\s*,?\s+(?=(?:{_EN_CARDINAL_WORD}|one\s+hundred)\s+percent\b)",
    re.IGNORECASE,
)
_EN_CLOCK_RANGE_RE = re.compile(
    r"(?<!\d)(\d{1,2}):([0-5]\d)\s*[-–—]\s*(\d{1,2}):([0-5]\d)(?!\d)"
)
_EN_CLOCK_RE = re.compile(r"(?<!\d)(\d{1,2}):([0-5]\d)(?!\d)")
_EN_SPACED_INITIALISM_RE = re.compile(r"(?<![a-z])(?:[a-z]\s+){1,}[a-z](?![a-z])")


def _clock_token(hour: str, minute: str) -> str:
    hour_value = int(hour)
    if hour_value > 23:
        return f"{hour}:{minute}"
    return str(hour_value) if minute == "00" else f"{hour_value}{minute}"


def _prepare_english_numbers(text: str) -> str:
    """Protect number boundaries and canonicalize explicit clock notation.

    Whisper's number normalizer otherwise merges independent adjacent values
    (``1976, thirty percent`` -> ``197630%``) and splits clock notation
    differently from spoken time (``06:30`` -> ``6 30`` vs ``six thirty`` ->
    ``630``).  A lexical sentinel forces a word boundary and is removed after
    the upstream normalizer has finished.
    """
    text = _EN_CLOCK_RANGE_RE.sub(
        lambda m: f"{_clock_token(m[1], m[2])} to {_clock_token(m[3], m[4])}",
        text,
    )
    text = _EN_CLOCK_RE.sub(lambda m: _clock_token(m[1], m[2]), text)
    text = _EN_SPOKEN_YEAR_PERCENT_RE.sub(rf"\1 {_EN_NUMBER_BOUNDARY} ", text)
    return _EN_LONG_DIGIT_CARDINAL_RE.sub(rf"\1 {_EN_NUMBER_BOUNDARY} ", text)


def _collapse_english_initialisms(text: str) -> str:
    """Make ``U.S.``, ``u s`` and ``US`` share one token."""
    return _EN_SPACED_INITIALISM_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def normalize_en(text: str) -> str:
    """英文规一化（Whisper-based + ASR token 边界修正）：缩写展开、拼写数字→阿拉伯、
    英美拼写统一、货币/去标点/小写，并统一相邻独立数字、时钟和首字母缩写口径。
    ref/hyp 同口径。空/None→空串。
    懒加载：中文路径不碰 whisper-normalizer 依赖。"""
    if not text:
        return ""
    global _EN_NORMALIZER
    if _EN_NORMALIZER is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _EN_NORMALIZER = EnglishTextNormalizer()
    normalized = _EN_NORMALIZER(_prepare_english_numbers(text)) or ""
    normalized = normalized.replace(_EN_NUMBER_BOUNDARY, " ")
    normalized = _collapse_english_initialisms(normalized)
    return " ".join(normalized.split())


_BASIC_NORMALIZER = None


def normalize_basic(text: str) -> str:
    """多语言通用规一化(Whisper BasicTextNormalizer 口径)：小写、去标点、Unicode 规范化、
    合并空白。Whisper 给非英文语种(德法西俄日韩等)用的就是这套——不做英文专属的缩写/数字展开，
    故不会糟蹋其它语言。德法西俄走它 + WER；日韩走它 + CER。ref/hyp 同口径。空/None→空串。
    懒加载：中英/中文路径不触发。"""
    if not text:
        return ""
    global _BASIC_NORMALIZER
    if _BASIC_NORMALIZER is None:
        from whisper_normalizer.basic import BasicTextNormalizer
        _BASIC_NORMALIZER = BasicTextNormalizer()
    return _BASIC_NORMALIZER(text) or ""


def normalize_ja(text: str) -> str:
    """日语公开基准口径：Whisper BasicTextNormalizer 后删除全部空格。

    对齐 kotoba-tech/kotoba-whisper 的 ``run_short_form_eval.py``。日语原文通常
    不以空格分词；BasicTextNormalizer 删除标点时留下的空格不应计入字符级 CER。
    """
    return normalize_basic(text).replace(" ", "").replace("。.", "。")
