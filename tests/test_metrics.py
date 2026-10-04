"""metrics.py 黄金用例 — 每个期望值都经过手算验证。

跑法（项目根目录）:
  uv run --quiet --with pytest --with jiwer --with opencc --with sacrebleu pytest tests/ -q

这些数字是整个评测的可信度地基：任何指标实现的改动若导致这里失败，
要么是引入了 bug，要么是口径变了——后者必须同步更新本文件并在报告中注明。
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from metrics import (char_cer, comet_score, corpus_cer, keyword_hits, rouge_char,
                     sent_chrf, translation_corpus, word_wer)


# ---------- CER（中文字级） ----------

def test_cer_identical():
    c = char_cer("今天天气好", "今天天气好")
    assert c.cer == 0 and c.edits == 0 and c.ref_len == 5


def test_cer_one_substitution():
    # 5 字参考，1 个替换 → CER 0.2
    c = char_cer("今天天气好", "今天天气坏")
    assert c.cer == 0.2 and c.sub == 1 and c.dele == 0 and c.ins == 0


def test_cer_deletion():
    # "今天天气"→"今天气" 丢 1 字 → 1/4
    c = char_cer("今天天气", "今天气")
    assert c.cer == 0.25 and c.dele == 1


def test_cer_insertion():
    # 多出 1 字 → 1/2
    c = char_cer("今天", "今天啊")
    assert c.cer == 0.5 and c.ins == 1


def test_cer_mixed_zh_en_chars():
    # 字级拆分：ref 8 字符(我用iphone)，"iphone"(6) vs "安卓"(2) = 2 替换 + 4 删除
    c = char_cer("我用iphone", "我用安卓")
    assert c.ref_len == 8 and c.sub == 2 and c.dele == 4 and c.cer == 0.75


def test_cer_empty_ref_nonempty_hyp():
    # 参考为空：CER 封顶 1.0，编辑数 = hyp 长度（计入语料级分子）
    c = char_cer("", "今天")
    assert c.cer == 1.0 and c.edits == 2 and c.ref_len == 0 and c.ins == 2


def test_cer_empty_both():
    c = char_cer("", "")
    assert c.cer == 0 and c.edits == 0


def test_cer_empty_hyp():
    # hyp 为空：全删除，CER 1.0
    c = char_cer("今天", "")
    assert c.cer == 1.0 and c.dele == 2


def test_cer_capped_not_above_ref_len_ratio():
    # hyp 比 ref 长很多时 CER 可以 >1（编辑距离口径），不做截断
    c = char_cer("好", "今天天气好")
    assert c.cer == 4.0 and c.ins == 4


# ---------- WER（英文词级） ----------

def test_wer_identical():
    c = word_wer("the cat sat", "the cat sat")
    assert c.cer == 0


def test_wer_deletion():
    c = word_wer("the cat sat", "the cat")
    assert c.cer == 1 / 3 and c.dele == 1


def test_wer_substitution():
    c = word_wer("i like cats", "i like dogs")
    assert c.cer == 1 / 3 and c.sub == 1


def test_wer_insertion():
    c = word_wer("hello world", "hello there world")
    assert c.cer == 0.5 and c.ins == 1


def test_wer_empty_ref():
    c = word_wer("", "hello world")
    assert c.cer == 1.0 and c.edits == 2 and c.ref_len == 0


def test_wer_numeric_boundary_regression_matches_human_word_units():
    from normalize import normalize_en
    ref = normalize_en(
        "by 1976 thirty percent of machu picchu had been restored and restoration continues till today"
    )
    hyp = normalize_en(
        "by 1976 30 percent of machu picchu has been restored and restoration continues to today"
    )
    c = word_wer(ref, hyp)
    assert c.ref_len == 14
    assert c.edits == c.sub == 2
    assert c.dele == c.ins == 0
    assert c.cer == 2 / 14


# ---------- 语料级 CER ----------

def test_corpus_cer_is_edit_sum_over_ref_sum():
    # Σ编辑/Σ参考字数 = (1+4)/(10+20)，不是逐句 CER 取平均
    assert corpus_cer([10, 20], [1, 4]) == 5 / 30


def test_corpus_cer_empty():
    assert corpus_cer([], []) == 0.0


# ---------- KRR（关键词召回） ----------

def test_krr_alias_any_hit_counts():
    # 别名命中任一即算；第二个词未出现
    hit, tot = keyword_hits("阿司匹林对头痛有效",
                            [{"aliases": ["阿斯匹林", "阿司匹林"]},
                             {"aliases": ["布洛芬"]}])
    assert (hit, tot) == (1, 2)


def test_krr_plain_string_keywords():
    hit, tot = keyword_hits("上海到北京的高铁", ["上海", "北京", "广州"])
    assert (hit, tot) == (2, 3)


def test_krr_no_keywords():
    assert keyword_hits("随便说点什么", []) == (0, 0)


def test_krr_surface_fallback():
    hit, tot = keyword_hits("打开空调", [{"surface": "空调"}])
    assert (hit, tot) == (1, 1)


# ---------- ROUGE（字级，总结场景） ----------

def test_rouge_identical():
    r = rouge_char("今天天气好", "今天天气好")
    assert r["r1"] == 1.0 and r["rl"] == 1.0


def test_rouge_hand_computed():
    # ref=今天天气好(5字) hyp=天气好(3字)
    # R1: 重叠 3 字 → P=1, R=0.6, F1=0.75；RL: LCS=3 → 同样 0.75
    r = rouge_char("今天天气好", "天气好")
    assert r["r1"] == 0.75 and r["rl"] == 0.75


def test_rouge_disjoint():
    r = rouge_char("今天天气", "明日降雨")
    assert r["r1"] == 0.0 and r["rl"] == 0.0


def test_rouge_empty():
    assert rouge_char("", "什么") == {"r1": 0.0, "rl": 0.0}


# ---------- 翻译（sacrebleu 封装，验证接线而非 sacrebleu 本身） ----------

def test_translation_perfect_match():
    sc = translation_corpus(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert sc["BLEU"] == 100.0 and sc["chrF"] == 100.0


def test_translation_zh_tokenizer():
    """中文参考自动切 tokenize='zh'（en→zh 方向）：完美命中应得 BLEU=100；
    若误用默认 13a，整句无空格只算 1 个 token、凑不出 4-gram → BLEU 失真为 0。"""
    refs = ["猫坐在垫子上面，外面正在下雨。"]
    sc = translation_corpus(refs, refs)
    assert sc["BLEU"] == 100.0 and sc["chrF"] == 100.0


def test_sent_chrf_range():
    assert sent_chrf("the cat", "the cat") == 100.0
    assert sent_chrf("the cat", "完全无关") < 5


# ---------- embedding 可选增强：容错契约（拿不到 key/端点 → None，绝不抛） ----------

def test_embedding_degrades_without_key(monkeypatch):
    """可选增强的核心契约：无 key 时静默降级返回 None（不抛、不联网）。

    锁住「端点挂/限流/无配置都不影响主指标」这一行为，防未来改动破坏容错。
    """
    import metrics
    monkeypatch.setattr(metrics, "_embedding_key", lambda: "")
    assert metrics.embedding_cosine([("今天天气不错", "今天天气很好")]) is None


def test_comet_accepts_prediction_object_scores(monkeypatch):
    """unbabel-comet 常见版本返回 Prediction.scores，而不是 dict。"""
    import metrics

    class FakeModel:
        def predict(self, data, batch_size, gpus, num_workers, progress_bar):
            assert data == [{"src": "src", "ref": "ref", "mt": "hyp"}]
            assert num_workers == 1
            return types.SimpleNamespace(scores=[0.87654])

    fake_comet = types.SimpleNamespace(
        download_model=lambda model: model,
        load_from_checkpoint=lambda path: FakeModel(),
    )
    monkeypatch.setitem(sys.modules, "comet", fake_comet)
    metrics._COMET_CACHE.clear()
    assert comet_score([("src", "ref", "hyp")]) == [0.8765]


def test_external_comet_score_uses_isolated_python(monkeypatch):
    import metrics

    monkeypatch.setattr(metrics, "_external_comet_python", lambda: "/isolated/bin/python")
    completed = types.SimpleNamespace(returncode=0, stdout=json.dumps({"scores": [0.81234]}))
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)

    assert metrics._external_comet_score([("src", "ref", "hyp")], "model", 8) == [0.8123]
