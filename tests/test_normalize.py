"""normalize.py 黄金用例 — 口径对齐 SpeechIO LeaderBoard。

覆盖：繁→简、全半角、英文小写、去语气词、去儿化（含白名单）、
中文数字规一化、去标点、去空格（英文词间空格保留）。
期望值均已实跑验证；改口径必须同步改这里并在报告中注明。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from normalize import normalize, normalize_en


# ---------- 中文规一化 ----------

def test_traditional_to_simplified_and_punct():
    assert normalize("今天天氣很好。") == "今天天气很好"


def test_remove_fillers():
    # 呃/啊 按字符全部去除（SpeechIO 口径）
    assert normalize("呃这个啊就是说") == "这个就是说"


def test_remove_fillers_even_inside_words():
    # 注意：是按字符删，不做分词判断——"啊呀"的"啊"也会被删，属已知口径
    assert normalize("啊呀真好啊") == "呀真好"


def test_remove_erhua():
    assert normalize("他在那边儿玩儿") == "他在那边玩"
    assert normalize("便宜点儿吧") == "便宜点吧"


def test_erhua_whitelist_preserved():
    # 女儿/儿童 是实义词，白名单保护不删
    assert normalize("女儿在儿童医院") == "女儿在儿童医院"


def test_quanjiao_digits_to_chinese_number():
    # 全角→半角→中文数字（读数口径：123→一百二十三）
    assert normalize("１２３") == "一百二十三"


def test_digits_in_context():
    assert normalize("我有3个苹果") == "我有三个苹果"


def test_year_read_digit_by_digit():
    # 年份按位读：2023年→二零二三年
    assert normalize("2023年") == "二零二三年"


def test_percentage():
    assert normalize("100%") == "百分之一百"


def test_english_lowered_and_zh_space_removed():
    # 中英混排：英文小写；中文间空格删除，英文词间空格保留
    assert normalize("深度学习 用 GPT 模型") == "深度学习用gpt模型"
    assert normalize("Hello, World!  ＡＢＣ") == "hello world abc"


def test_empty_and_none():
    assert normalize("") == ""
    assert normalize(None) == ""


def test_remove_kespeech_spoken_noise_control_token():
    assert normalize("生产原<SPOKEN_NOISE>距离") == "生产原距离"
    assert normalize("<SPOKEN_NOISE>华人英雄") == "华人英雄"


def test_ref_hyp_same_caliber():
    # 口径一致性：阿拉伯数字与中文小写数字归一到同一表达
    assert normalize("100") == normalize("一百") == "一百"
    # 已知局限：大写中文数字(壹佰)不做归一，模型若输出大写会被记错——报告口径需注明
    assert normalize("壹佰") == "壹佰"


# ---------- 英文规一化（Whisper-based + ASR token 边界修正） ----------

def test_en_lower_punct_and_contraction():
    # 小写 + 去标点 + 缩写展开（don't → do not）
    assert normalize_en("Hello, World! don't stop") == "hello world do not stop"


def test_en_preserves_word_boundaries():
    assert normalize_en("THE CAT-SAT") == "the cat sat"


def test_en_whisper_caliber():
    # Whisper 口径关键差异：缩写展开 + 英美拼写统一(centre→center) + 货币/数字归一
    assert normalize_en("We won't go to the centre") == "we will not go to the center"
    assert normalize_en("I have 100 dollars and 5 cats") == "i have $100 and 5 cats"


def test_en_keeps_adjacent_year_and_percentage_as_separate_tokens():
    expected = "by 1976 30% of machu picchu"
    assert normalize_en("By 1976, thirty percent of Machu Picchu") == expected
    assert normalize_en("By 1976, 30 percent of Machu Picchu") == expected
    assert normalize_en("By nineteen seventy-six, thirty percent of Machu Picchu") == expected
    assert normalize_en("By nineteen seventy six thirty percent of Machu Picchu") == expected


def test_en_clock_notation_matches_spoken_time():
    assert normalize_en("06:30") == normalize_en("six thirty") == "630"
    assert normalize_en("11:35") == normalize_en("eleven thirty five") == "1135"
    assert normalize_en("12:00") == normalize_en("twelve") == "12"
    assert normalize_en("10:00-11:00 p.m.") == normalize_en("ten to eleven p m") == "10 to 11 pm"


def test_en_initialism_spelling_has_one_token_caliber():
    assert normalize_en("U.S. CIA p.m.") == normalize_en("US C.I.A. pm") == "us cia pm"


def test_en_empty():
    assert normalize_en("") == ""
    assert normalize_en(None) == ""
