"""把各数据集转成统一 manifest（JSONL）。

每行: {"id","audio_path","ref_text","lang","keywords":[...]}
runner 只认这个格式，从而抹平 wav/Lhotse/tar/parquet 等不同封装。

用法:
  python eval/build_manifest.py wsyue --limit 50 --out manifests/wsyue_lite.jsonl
"""

import argparse
import json
import os

from config import rel_path
from data_quality import exclusion_for_row, load_exclusion_policy
from util import atomic_text_writer, ensure_unique_ids, validate_jsonl_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wsyue_long_text(path):
    """WSYue Long TextGrid 的“文本”层 → 整条音频参考转写。"""
    import re
    txt = open(path, encoding="utf-8").read()
    for block in re.split(r"(?=^\s*item \[\d+\]:)", txt, flags=re.MULTILINE):
        if re.search(r'^\s*name = "文本"\s*$', block, flags=re.MULTILINE):
            parts = re.findall(r'^\s*text = "(.*)"\s*$', block, flags=re.MULTILINE)
            return "".join(p.replace('""', '"') for p in parts if p)
    return ""


def build_wsyue(limit=0, subset=""):
    """WSYue-ASR-eval：Short(content.txt + wav_/) 或 Long(TextGrid + wav/)。"""
    subset = (subset or "short").strip().lower()
    if subset == "long":
        import glob
        base = os.path.join(ROOT, "datasets/asr/wsyue/Long")
        out = []
        for tg in sorted(glob.glob(os.path.join(base, "TextGrid", "*.TextGrid"))):
            uid = os.path.splitext(os.path.basename(tg))[0]
            path = os.path.join(base, "wav", uid + ".wav")
            text = _wsyue_long_text(tg)
            if not text or not os.path.exists(path):
                continue
            out.append({"id": uid, "audio_path": path, "ref_text": text,
                        "lang": "yue", "keywords": []})
            if limit and len(out) >= limit:
                break
        return out
    if subset != "short":
        raise ValueError("wsyue --subset 只支持 short/long")

    base = os.path.join(ROOT, "datasets/asr/wsyue/Short")
    out = []
    for line in open(os.path.join(base, "content.txt"), encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        wav, _age, _emo, _gender, text = parts[0], parts[1], parts[2], parts[3], parts[4]
        path = os.path.join(base, "wav_", wav)
        if not os.path.exists(path):
            continue
        out.append({"id": wav.replace(".wav", ""), "audio_path": path,
                    "ref_text": text, "lang": "yue", "keywords": []})
        if limit and len(out) >= limit:
            break
    return out


def build_aishell(limit=0):
    """AISHELL-1 test (Lhotse cuts): jsonl.gz → id/text/wav 路径。"""
    import glob
    import gzip
    base = os.path.join(ROOT, "datasets/asr/aishell1_test")
    wav_root = os.path.join(base, "wav_extracted")
    out = []
    for p in glob.glob(os.path.join(base, "**/*.jsonl.gz"), recursive=True):
        with gzip.open(p, "rt") as f:
            for line in f:
                d = json.loads(line)
                sup = d.get("supervisions", [{}])[0]
                text = sup.get("text", "")
                src = d.get("recording", {}).get("sources", [{}])[0].get("source", "")
                # Lhotse source 形如 "data/aishell_cuts_test.../BAC/x.wav"，解压目录无 data/ 前缀
                if src.startswith("data/"):
                    src = src[len("data/"):]
                path = os.path.join(wav_root, src)
                if not text or not os.path.exists(path):
                    continue
                out.append({"id": d.get("id"), "audio_path": path,
                            "ref_text": text, "lang": "zh", "keywords": []})
                if limit and len(out) >= limit:
                    return out
    return out


def build_aishell2(limit=0):
    """AISHELL-2 iOS 抽样子集（scripts/aishell2_subset.py 从官方 82G zip 产出，说话人分层抽样）。"""
    base = os.path.join(ROOT, "datasets/asr/aishell2_ios")
    tsv = os.path.join(base, "trans_subset.tsv")
    policy_path = os.path.join(ROOT, "datasets/exclusions/aishell2_ios.json")
    if not os.path.exists(policy_path):  # 兼容本地未迁移的旧 sidecar。
        policy_path = os.path.join(base, "exclusions.json")
    policy = load_exclusion_policy(policy_path)
    out = []
    if not os.path.exists(tsv):
        return out
    for line in open(tsv, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        utt, spk, text = parts
        path = os.path.join(base, "wav", spk, utt + ".wav")
        if not text or not os.path.exists(path):
            continue
        row = {"id": utt, "audio_path": path, "ref_text": text, "lang": "zh", "keywords": []}
        if exclusion_for_row(row, policy):
            continue
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def build_aishell2_eval(limit=0, subset="ios"):
    """AISHELL2-2018A-EVAL 官方 test，按 iOS/Android/Mic 通道分别构建。"""
    channels = {"ios": "iOS", "android": "Android", "mic": "Mic"}
    key = (subset or "ios").strip().lower()
    if key not in channels:
        raise ValueError("aishell2_eval --subset 只支持 ios/android/mic")

    channel = channels[key]
    base = os.path.join(
        ROOT, "datasets/asr/aishell2_eval/AISHELL-DEV-TEST-SET", channel, "test")
    trans_path = os.path.join(base, "trans.txt")
    wav_scp_path = os.path.join(base, "wav.scp")
    if not os.path.exists(trans_path) or not os.path.exists(wav_scp_path):
        return []

    trans = {}
    for line in open(trans_path, encoding="utf-8"):
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            trans[parts[0]] = parts[1]

    out = []
    for line in open(wav_scp_path, encoding="utf-8"):
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or parts[0] not in trans:
            continue
        utt, rel = parts
        path = rel if os.path.isabs(rel) else os.path.join(base, rel)
        if not os.path.exists(path):
            continue
        out.append({
            "id": utt, "audio_path": path, "ref_text": trans[utt],
            "lang": "zh", "keywords": [], "channel": key,
            "speaker_id": os.path.basename(os.path.dirname(rel)),
        })
        if limit and len(out) >= limit:
            break
    return out


def build_librispeech(limit=0, subset="clean"):
    """LibriSpeech test-clean/test-other (HF parquet, flac 内嵌): 抽 flac 到磁盘 + 建 manifest。

    英文 ASR。首次调用materialize音频到 audio/<id>.flac(幂等)。
    兼容两种下载布局：clean/test/0000.parquet（预期）或 all/test.clean/0000.parquet（实际 HF 下载）。
    """
    import pyarrow.parquet as pq
    base = os.path.join(ROOT, "datasets/asr/librispeech")
    split = (subset or "clean").strip().lower()
    if split not in ("clean", "other"):
        raise ValueError("librispeech --subset 只支持 clean/other")
    # 优先尝试预期路径，fallback 到 HF 实际下载布局
    pqf = os.path.join(base, f"{split}/test/0000.parquet")
    if not os.path.exists(pqf):
        pqf = os.path.join(base, f"all/test.{split}/0000.parquet")
    audio_dir = os.path.join(base, "audio" if split == "clean" else "audio_other")
    os.makedirs(audio_dir, exist_ok=True)
    t = pq.read_table(pqf, columns=["audio", "text", "id"])
    out = []
    for row in t.to_pylist():
        uid = row["id"]
        text = row.get("text", "")
        b = row["audio"].get("bytes")
        if not text or not b:
            continue
        fp = os.path.join(audio_dir, uid + ".flac")
        if not os.path.exists(fp):
            with open(fp, "wb") as f:
                f.write(b)
        out.append({"id": uid, "audio_path": fp,
                    "ref_text": text, "lang": "en", "keywords": []})
        if limit and len(out) >= limit:
            break
    return out


_OPEN_ASR_SUBSETS = {
    "ami": "ami/test-*.parquet",
    "earnings22": "earnings22/test-*.parquet",
    "gigaspeech": "gigaspeech/test-*.parquet",
    "spgispeech": "spgispeech/test-*.parquet",
    "voxpopuli": "voxpopuli/test-*.parquet",
    # 统一仓的 tedlium 配置目前没有可下载 Parquet；复用相同 TED-LIUM 3 test 音频/参考。
    "tedlium": "tedlium_source/release3/test-*.parquet",
}


def build_open_asr(limit=0, subset="ami"):
    """Open ASR Leaderboard 英文测试集：内嵌音频 Parquet → manifest。

    subset: ami/earnings22/gigaspeech/spgispeech/voxpopuli/tedlium。
    前五项来自 hf-audio/open-asr-leaderboard 的原始配置；TEDLIUM 来自
    distil-whisper/tedlium-prompted 的 release3 test，只读取原始 audio/text/id。
    """
    import glob
    import hashlib
    import re
    import pyarrow.parquet as pq

    key = (subset or "ami").strip().lower()
    if key not in _OPEN_ASR_SUBSETS:
        raise ValueError("open_asr --subset 只支持: " + "/".join(_OPEN_ASR_SUBSETS))
    base = os.path.join(ROOT, "datasets/asr/open_asr")
    parquet_files = sorted(glob.glob(os.path.join(base, _OPEN_ASR_SUBSETS[key])))
    if not parquet_files:
        raise FileNotFoundError(f"Open ASR {key} Parquet 未下载: {_OPEN_ASR_SUBSETS[key]}")

    audio_dir = os.path.join(base, "wav", key)
    os.makedirs(audio_dir, exist_ok=True)
    out = []
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        required = {"audio", "text", "id"}
        missing = required.difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{parquet_path} 缺少字段: {sorted(missing)}")
        for batch in parquet.iter_batches(columns=["audio", "text", "id"], batch_size=256):
            for row in batch.to_pylist():
                source_id = str(row.get("id") or "").strip()
                text = str(row.get("text") or "").strip()
                audio = row.get("audio") or {}
                audio_bytes = audio.get("bytes")
                if not source_id or not text or not audio_bytes:
                    continue
                digest = hashlib.sha1(f"{key}\0{source_id}".encode()).hexdigest()[:16]
                stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(source_id))[:80]
                source_ext = os.path.splitext(str(audio.get("path") or source_id))[1].lower()
                ext = source_ext if source_ext in {".wav", ".flac", ".mp3", ".ogg", ".m4a"} else ".wav"
                filename = f"{stem or 'audio'}_{digest}{ext}"
                audio_path = os.path.join(audio_dir, filename)
                if not os.path.exists(audio_path):
                    partial = audio_path + ".part"
                    with open(partial, "wb") as dst:
                        dst.write(audio_bytes)
                    os.replace(partial, audio_path)
                out.append({
                    "id": f"openasr_{key}_{digest}", "audio_path": audio_path,
                    "ref_text": text, "lang": "en", "keywords": [],
                    "source_id": source_id, "source_dataset": key,
                })
                if limit and len(out) >= limit:
                    return out
    return out


# FLORES-200 热门语种：短码 → (盘上文件名前缀, gemma 提示词用的中文名)。
# 同一批 1012 句已译成 200 语，扩语向=换文件，零下载。短码不含连字符，便于 subset 拆分。
_FLORES_LANG = {
    "en": ("eng_Latn", "英文"), "zh": ("zho_Hans", "中文"), "zht": ("zho_Hant", "繁体中文"),
    "ja": ("jpn_Jpan", "日文"), "ko": ("kor_Hang", "韩文"), "de": ("deu_Latn", "德文"),
    "fr": ("fra_Latn", "法文"), "es": ("spa_Latn", "西班牙文"), "ru": ("rus_Cyrl", "俄文"),
    "ar": ("arb_Arab", "阿拉伯文"), "pt": ("por_Latn", "葡萄牙文"), "it": ("ita_Latn", "意大利文"),
    "hi": ("hin_Deva", "印地文"), "vi": ("vie_Latn", "越南文"), "th": ("tha_Thai", "泰文"),
    "id": ("ind_Latn", "印尼文"),
}


def build_flores(limit=0, subset=""):
    """FLORES-200 任意语向（行对齐）。subset 形如 'en-de'（源-目标），缺省 zh→en。
    语种短码见 _FLORES_LANG（热门 16 语）。翻译场景：source_text + ref_text + target_lang。"""
    base = os.path.join(ROOT, "datasets/translation/flores_plus/flores200_dataset/devtest")
    src_k, tgt_k = subset.split("-", 1) if subset else ("zh", "en")
    src_file = _FLORES_LANG[src_k][0]
    tgt_file, tgt_name = _FLORES_LANG[tgt_k]
    src = open(os.path.join(base, f"{src_file}.devtest"), encoding="utf-8").read().splitlines()
    ref = open(os.path.join(base, f"{tgt_file}.devtest"), encoding="utf-8").read().splitlines()
    out = []
    for i, (s, r) in enumerate(zip(src, ref)):
        if not s.strip() or not r.strip():
            continue
        # 默认 zh→en 保持原 id/lang 口径，已有结果可续比；带 subset 才用语向化命名
        uid = f"flores_{i}" if not subset else f"flores_{src_k}{tgt_k}_{i}"
        lang = "zh-en" if not subset else f"{src_k}-{tgt_k}"
        out.append({"id": uid, "task": "translate", "lang": lang,
                    "source_text": s, "ref_text": r, "target_lang": tgt_name})
        if limit and len(out) >= limit:
            break
    return out


def build_wmt(limit=0, subset="wmt22-zh-en"):
    """WMT22/23 zh↔en（src/ref 行对齐）；每个 manifest 固定单一语向。"""
    base = os.path.join(ROOT, "datasets/translation/wmt")
    key = subset or "wmt22-zh-en"
    configs = {
        "wmt22-zh-en": ("wmt22", "zh-en", "英文"),
        "wmt22-en-zh": ("wmt22", "en-zh", "中文"),
        "wmt23-zh-en": ("wmt23", "zh-en", "英文"),
        "wmt23-en-zh": ("wmt23", "en-zh", "中文"),
    }
    if key not in configs:
        raise ValueError("wmt --subset 只支持 " + "/".join(configs))
    year, lang, target = configs[key]
    src = open(os.path.join(base, f"{year}.{lang}.src.txt"), encoding="utf-8").read().splitlines()
    ref = open(os.path.join(base, f"{year}.{lang}.ref.txt"), encoding="utf-8").read().splitlines()
    out = []
    for i, (s, r) in enumerate(zip(src, ref)):
        if not s.strip() or not r.strip():
            continue
        # 默认清单沿用历史 id，已有 WMT22 zh→en infer 可直接断点续跑。
        uid = f"wmt22_{i}" if key == "wmt22-zh-en" else f"{year}_{lang.replace('-', '')}_{i}"
        out.append({"id": uid, "task": "translate", "lang": lang,
                    "source_text": s, "ref_text": r, "target_lang": target})
        if limit and len(out) >= limit:
            break
    return out


def build_cnewsum(limit=0):
    """CNewSum 新闻摘要（article→summary）。总结场景：source_text + ref_text。"""
    import pyarrow.parquet as pq
    f = glob_first(os.path.join(ROOT, "datasets/summary/cnewsum/data/test*.parquet"))
    t = pq.read_table(f, columns=["id", "article", "summary"])
    out = []
    for r in t.to_pylist():
        if not r.get("article") or not r.get("summary"):
            continue
        out.append({"id": f"cnewsum_{r['id']}", "task": "summarize", "lang": "zh",
                    "source_text": r["article"], "ref_text": r["summary"]})
        if limit and len(out) >= limit:
            break
    return out


def build_ascend(limit=0):
    """ASCEND 中英 code-switch（parquet 内嵌 wav）→ 抽 wav 到磁盘 + manifest。中英混读用 CER(lang=zh)。"""
    import pyarrow.parquet as pq
    base = os.path.join(ROOT, "datasets/asr/ascend")
    audio_dir = os.path.join(base, "wav")
    os.makedirs(audio_dir, exist_ok=True)
    t = pq.read_table(glob_first(os.path.join(base, "**/test*.parquet")), columns=["id", "audio", "transcription"])
    out = []
    for row in t.to_pylist():
        text = row.get("transcription", "")
        b = row["audio"].get("bytes")
        if not text or not b:
            continue
        uid = str(row["id"])
        fp = os.path.join(audio_dir, uid + ".wav")
        if not os.path.exists(fp):
            open(fp, "wb").write(b)
        out.append({"id": f"ascend_{uid}", "audio_path": fp,
                    "ref_text": text, "lang": "zh", "keywords": []})
        if limit and len(out) >= limit:
            break
    return out


def glob_first(pat):
    import glob as _g
    return _g.glob(pat, recursive=True)[0]


def _extract_once(tar_path, dest_root):
    """解压 tar/tar.gz 到 dest_root，幂等（.done 标记，崩溃重跑会重解压保完整）。"""
    import tarfile
    os.makedirs(dest_root, exist_ok=True)
    marker = os.path.join(dest_root, "." + os.path.basename(tar_path) + ".done")
    if os.path.exists(marker):
        return
    with tarfile.open(tar_path) as t:
        t.extractall(dest_root)
    open(marker, "w").close()


def build_seaco(limit=0):
    """SeACo 热词测试集（AISHELL-1 test 子集 + 热词表）。ASR 场景 b 主力。

    repo 只有元数据：uttid(kaldi 双列) + text + hotword.txt。音频复用本仓
    aishell1_test 解压产物（cut 文件名带 -NNNN 后缀，去掉即 uttid）。
    keywords = 该句 ref 中出现的热词 → KRR 直接可算。
    """
    import glob
    base = os.path.join(ROOT, "datasets/asr/seaco_hotword/repo/data/test")
    hotwords = [w.strip() for w in open(os.path.join(base, "hotword.txt"), encoding="utf-8") if w.strip()]
    texts = {}
    for line in open(os.path.join(base, "text"), encoding="utf-8"):
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            texts[parts[0]] = parts[1].strip()
    wav = {}
    for p in glob.glob(os.path.join(ROOT, "datasets/asr/aishell1_test/wav_extracted/**/*.wav"),
                       recursive=True):
        wav[os.path.basename(p)[:-4].rsplit("-", 1)[0]] = p
    out = []
    for line in open(os.path.join(base, "uttid"), encoding="utf-8"):
        uid = line.split()[0] if line.strip() else ""
        if not uid or uid not in texts or uid not in wav:
            continue
        ref = texts[uid]
        out.append({"id": uid, "audio_path": wav[uid], "ref_text": ref, "lang": "zh",
                    "keywords": [w for w in hotwords if w in ref]})
        if limit and len(out) >= limit:
            break
    return out


def build_speechio(limit=0, subset=""):
    """SpeechIO ZH00000-26（Lhotse cuts jsonl.gz + tar.gz 音频，按需解压）。

    subset: 逗号分隔，如 "1,5" 或 "ZH00001"；空 = 全部 27 个子集（首次全解压 ~7G）。
    """
    import glob
    import gzip
    base = os.path.join(ROOT, "datasets/asr/speechio/yuekai_mirror/data")
    subs = sorted(glob.glob(os.path.join(base, "speechio_cuts_*.jsonl.gz")))
    if subset:
        want = {f"ZH{int(s):05d}" if s.strip().isdigit() else s.strip().upper()
                for s in subset.split(",")}
        subs = [p for p in subs if any(w in p for w in want)]
    ex_root = os.path.join(base, "extracted")
    out = []
    for jp in subs:
        _extract_once(jp.replace(".jsonl.gz", ".tar.gz"), ex_root)
        with gzip.open(jp, "rt") as f:
            for line in f:
                d = json.loads(line)
                sup = d.get("supervisions", [{}])[0]
                text = sup.get("text", "")
                src = d.get("recording", {}).get("sources", [{}])[0].get("source", "")
                if src.startswith("data/"):  # jsonl 里带 data/ 前缀，tar 内没有
                    src = src[len("data/"):]
                path = os.path.join(ex_root, src)
                if not text or not os.path.exists(path):
                    continue
                out.append({"id": d.get("id"), "audio_path": path,
                            "ref_text": text, "lang": "zh", "keywords": []})
                if limit and len(out) >= limit:
                    return out
    return out


def build_med_it(limit=0):
    """MED-IT 医疗对话 ASR test split（英文！tar 按说话人打包）。lang=en → WER。"""
    import glob
    base = os.path.join(ROOT, "datasets/asr/med_it")
    ex = os.path.join(base, "test_extracted")
    for tp in sorted(glob.glob(os.path.join(base, "test", "*.tar"))):
        _extract_once(tp, ex)
    out = []
    for line in open(os.path.join(base, "test.txt"), encoding="utf-8"):
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        uid, text = parts[0], parts[1].strip()
        p = os.path.join(ex, uid.split("-")[0], uid + ".wav")
        if not os.path.exists(p):
            continue
        out.append({"id": f"medit_{uid}", "audio_path": p,
                    "ref_text": text, "lang": "en", "keywords": []})
        if limit and len(out) >= limit:
            break
    return out


# FLEURS 语种短码 → (语种目录名, 中文目标语名)。FLEURS 102 语句级平行，跨语种按 sentence_id 对齐。
_FLEURS_LANG = {
    "en": ("en_us", "英文"), "zh": ("cmn_hans_cn", "中文"),
    "yue": ("yue_hant_hk", "粤语"),
    "de": ("de_de", "德文"), "fr": ("fr_fr", "法文"), "es": ("es_419", "西班牙文"),
    "ru": ("ru_ru", "俄文"), "ja": ("ja_jp", "日文"), "ko": ("ko_kr", "韩文"),
    "it": ("it_it", "意大利文"), "pt": ("pt_br", "葡萄牙文"), "ar": ("ar_eg", "阿拉伯文"),
    "th": ("th_th", "泰文"), "id": ("id_id", "印尼文"), "vi": ("vi_vn", "越南文"),
}
# 多语项默认覆盖的源语种(与 Gummy/讯飞支持语种取交集)；数据未下载的在 build 时自动跳过
_FLEURS_MULTI = ["zh", "yue", "de", "fr", "es", "ru", "ja", "ko", "it", "pt", "ar", "th", "id"]
# Qwen3-ASR 官方公开表中的 FLEURS 12 语言口径。
_FLEURS_ASR = {k: _FLEURS_LANG[k] for k in
               ("en", "zh", "yue", "ar", "de", "es", "fr", "it", "ja", "ko", "pt", "ru")}


def _fleurs_text(tsv_path):
    """FLEURS test.tsv → {sentence_id: 转写文本}（第 1 列 id、第 3 列原文）。"""
    m = {}
    for line in open(tsv_path, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 3 and c[0] not in m:
            m[c[0]] = c[2]
    return m


def build_fleurs(limit=0, subset="zh-en"):
    """FLEURS 语音翻译，任意 X→Y：源语种音频(tar.gz 按需解压) + 同 sentence_id 的目标语转写作译文参考。

    subset:
      - "src-tgt"  单语向，如 "de-en"（德语音频→英文）；缺省 "zh-en"（保持原 id/lang 口径，旧结果可续比）
      - "multi-en"/"multi-zh"  多语→目标语（源取 _FLEURS_MULTI 中数据就位者，未下载的跳过并 warn）
    跑法：gemma-text 用 source_text 直翻=翻译上限；级联/同传(plat-simult/gummy/xf)从音频起步。
    多语项配 `--shuffle --seed -n` 跨语种均匀抽样（main 里先全量再随机截断）。
    """
    base = os.path.join(ROOT, "datasets/translation/fleurs/data")
    subset = subset or "zh-en"
    if subset.startswith("multi"):
        tgt = subset.split("-", 1)[1] if "-" in subset else "en"
        srcs = [s for s in _FLEURS_MULTI if s != tgt]
    else:
        src1, tgt = subset.split("-", 1)
        srcs = [src1]
    legacy = subset == "zh-en"   # 默认语向保留旧 id/lang，已有 zh→en 结果可续比
    tgt_dir, tgt_name = _FLEURS_LANG[tgt]
    tgt_tsv = os.path.join(base, tgt_dir, "test.tsv")
    if not os.path.exists(tgt_tsv):
        raise FileNotFoundError(f"FLEURS 目标语 {tgt} 未下载: {tgt_tsv}")
    refs = _fleurs_text(tgt_tsv)
    out, skipped = [], []
    for src in srcs:
        sdir = os.path.join(base, _FLEURS_LANG[src][0])
        if not os.path.exists(os.path.join(sdir, "test.tsv")):
            skipped.append(src)
            continue
        audio_root = os.path.join(sdir, "audio/extracted")
        tar = os.path.join(sdir, "audio/test.tar.gz")
        if os.path.exists(tar):
            _extract_once(tar, audio_root)
        for line in open(os.path.join(sdir, "test.tsv"), encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if len(c) < 3 or c[0] not in refs:
                continue
            p = os.path.join(audio_root, "test", c[1])
            if not os.path.exists(p):
                continue
            uid = f"fleurs_{c[0]}_{c[1][:-4]}" if legacy else f"fleurs_{src}{tgt}_{c[1][:-4]}"
            out.append({"id": uid, "task": "translate", "lang": "zh-en" if legacy else f"{src}-{tgt}",
                        "audio_path": p, "source_text": c[2], "ref_text": refs[c[0]],
                        "target_lang": tgt_name})
            if limit and len(out) >= limit:
                break
        if limit and len(out) >= limit:
            break
    if skipped:
        print(f"[fleurs] 跳过未下载源语种: {','.join(skipped)}（datasets/download.sh 扩语种后自动纳入）")
    return out


def build_fleurs_asr(limit=0, subset="de"):
    """FLEURS 多语言 ASR：单语种 audio + 转写(tsv 第 3 列原文)。subset 为语种短码（见 _FLEURS_ASR）。
    lang 写短码 → score.py 的 _asr_norm_metric 据此选 WER/CER 与多语言归一化。"""
    lang_dir = _FLEURS_ASR[subset][0]
    base = os.path.join(ROOT, "datasets/translation/fleurs/data", lang_dir)
    audio_root = os.path.join(base, "audio/extracted")
    _extract_once(os.path.join(base, "audio/test.tar.gz"), audio_root)
    out = []
    for line in open(os.path.join(base, "test.tsv"), encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) < 3:
            continue
        p = os.path.join(audio_root, "test", c[1])
        if not os.path.exists(p):
            continue
        out.append({"id": f"fleurs_{subset}_{c[1][:-4]}", "audio_path": p,
                    "ref_text": c[2], "lang": subset, "keywords": []})
        if limit and len(out) >= limit:
            break
    return out


_COMMON_VOICE_LANG = {
    # subset(盘上目录) → score.py 使用的语言短码。zh-TW 仍按中文 CER，normalize 会繁转简。
    "en": "en", "zh-CN": "zh", "yue": "yue", "zh-TW": "zh",
    "ar": "ar", "de": "de", "es": "es", "fr": "fr", "it": "it",
    "ja": "ja", "ko": "ko", "pt": "pt", "ru": "ru",
}


def build_commonvoice(limit=0, subset="zh-CN"):
    """Common Voice 17 test（fsicoli 原始分片镜像），只读 test.tsv + test 音频 tar。

    subset 为 Qwen3-ASR 公布的 13 语言之一。镜像保留 CV17 原始 test 划分；只下载
    test，避免 Mozilla 现行整语言包把 train/dev 一并拉下。
    """
    import csv
    import glob

    locale = subset or "zh-CN"
    if locale not in _COMMON_VOICE_LANG:
        raise ValueError("commonvoice --subset 不支持: " + locale)
    base = os.path.join(ROOT, "datasets/asr/common_voice17")
    transcript = os.path.join(base, "transcript", locale, "test.tsv")
    tar_dir = os.path.join(base, "audio", locale, "test")
    audio_root = os.path.join(tar_dir, "extracted")
    for tar_path in sorted(glob.glob(os.path.join(tar_dir, "*.tar"))):
        _extract_once(tar_path, audio_root)

    audio_by_name = {
        os.path.basename(path): path
        for path in glob.glob(os.path.join(audio_root, "**", "*.mp3"), recursive=True)
    }
    out = []
    with open(transcript, encoding="utf-8", newline="") as src:
        for row in csv.DictReader(src, delimiter="\t"):
            rel = (row.get("path") or "").strip()
            text = (row.get("sentence") or "").strip()
            path = audio_by_name.get(os.path.basename(rel))
            if not rel or not text or not path:
                continue
            stem = os.path.splitext(os.path.basename(rel))[0]
            item = {
                "id": f"cv17_{locale}_{stem}", "audio_path": path,
                "ref_text": text, "lang": _COMMON_VOICE_LANG[locale], "keywords": [],
            }
            if row.get("client_id"):
                item["speaker_id"] = row["client_id"]
            out.append(item)
            if limit and len(out) >= limit:
                break
    return out


_JA_BENCHMARKS = {
    # 与 kotoba-tech/kotoba-whisper 公开 CER 表完全相同的数据仓库与 test split。
    "cv8": ("ja_cv8", "japanese-asr/ja_asr.common_voice_8_0", 4483),
    "jsut": ("jsut_basic5000", "japanese-asr/ja_asr.jsut_basic5000", 5000),
    "reazon": ("reazonspeech_test", "japanese-asr/ja_asr.reazonspeech_test", 5263),
}


def _materialize_ja_benchmark(base, source, expected):
    """把 HF Dataset Viewer 的 audio parquet 幂等物化为音频文件 + metadata.tsv。"""
    import glob
    import pyarrow.parquet as pq

    parquet_files = sorted(glob.glob(os.path.join(base, "parquet", "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"{source} parquet 未下载: {os.path.join(base, 'parquet')}")
    audio_dir = os.path.join(base, "audio")
    metadata = os.path.join(base, "metadata.tsv")
    if os.path.exists(metadata):
        with open(metadata, encoding="utf-8") as src:
            rows = [line for line in src if line.strip()]
        if len(rows) == expected:
            return metadata

    os.makedirs(audio_dir, exist_ok=True)
    tmp_metadata = metadata + ".tmp"
    count = 0
    with open(tmp_metadata, "w", encoding="utf-8") as dst:
        for parquet_path in parquet_files:
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(batch_size=64, columns=["audio", "transcription"]):
                for item in batch.to_pylist():
                    audio = item.get("audio") or {}
                    raw = audio.get("bytes")
                    text = str(item.get("transcription") or "").replace("\t", " ").replace("\n", " ").strip()
                    if not raw or not text:
                        continue
                    original = str(audio.get("path") or "")
                    ext = os.path.splitext(original)[1].lower()
                    if not ext or len(ext) > 6:
                        ext = ".wav"
                    uid = f"ja_{os.path.basename(base)}_{count:05d}"
                    filename = uid + ext
                    path = os.path.join(audio_dir, filename)
                    if not os.path.exists(path) or os.path.getsize(path) != len(raw):
                        partial = path + ".part"
                        with open(partial, "wb") as audio_dst:
                            audio_dst.write(raw)
                        os.replace(partial, path)
                    dst.write(f"{uid}\t{filename}\t{text}\n")
                    count += 1
    if count != expected:
        raise RuntimeError(f"{source} 行数不符：期望 {expected}，物化 {count}")
    os.replace(tmp_metadata, metadata)
    return metadata


def build_ja_benchmark(limit=0, subset="reazon"):
    """日语公开基准：CV8 Japanese / JSUT Basic5000 / ReazonSpeech held-out。"""
    key = (subset or "reazon").strip().lower()
    if key not in _JA_BENCHMARKS:
        raise ValueError("ja_benchmark --subset 只支持 cv8/jsut/reazon")
    dirname, source, expected = _JA_BENCHMARKS[key]
    base = os.path.join(ROOT, "datasets", "asr", dirname)
    metadata = _materialize_ja_benchmark(base, source, expected)
    out = []
    with open(metadata, encoding="utf-8") as src:
        for line in src:
            uid, filename, text = line.rstrip("\n").split("\t", 2)
            path = os.path.join(base, "audio", filename)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            out.append({"id": uid, "audio_path": path, "ref_text": text, "lang": "ja",
                        "keywords": [], "source_dataset": source, "split": "test"})
            if limit and len(out) >= limit:
                break
    return out


def build_acl6060(limit=0, subset="eval"):
    """ACL 60/60 英文演讲 → 中文 语音翻译（IWSLT 官方同传评测集）。
    源英文音频 + text_zh 中文翻译参考。subset: eval(默认,416 测试集)/dev(468)/all。
    数据一次性物化见 eval/fetch_acl6060.py（HF ymoslem/acl-6060，datasets/ 不入仓）。"""
    base = os.path.join(ROOT, "datasets/translation/acl6060")
    tsv = os.path.join(base, "acl6060.tsv")
    if not os.path.exists(tsv):
        raise FileNotFoundError(
            "ACL 60/60 未物化：先跑 `uv run --with datasets --with soundfile python eval/fetch_acl6060.py`")
    want = None if subset in ("", "all") else subset
    out = []
    for line in open(tsv, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) < 4:
            continue
        split, idx, en, zh = c[0], c[1], c[2], c[3]
        if want and split != want:
            continue
        p = os.path.join(base, "audio", split, f"{idx}.wav")
        if not os.path.exists(p) or not zh:
            continue
        out.append({"id": f"acl6060_{split}_{idx}", "task": "translate", "lang": "en-zh",
                    "audio_path": p, "source_text": en, "ref_text": zh, "target_lang": "中文"})
        if limit and len(out) >= limit:
            break
    return out


_LONGFORM_TARGETS = {
    "ar": "Arabic", "de": "German", "fa": "Persian", "fr": "French",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pt": "Portuguese",
    "ru": "Russian", "tr": "Turkish", "zh": "Chinese", "en": "English",
}


def _acl6060_docs(path):
    """读取 ACL 60/60 的文档和分段。

    官方 XML 的 abstract 中含未转义的 ``V&L``，标准 XML parser 会在 eval split
    直接报错；这里只读取结构稳定的 doc/seg 标签，绕开无关的 abstract。
    """
    import html
    import re

    raw = open(path, encoding="utf-8").read()
    docs = {}
    for doc_id, body in re.findall(r'<doc\s+docid="([^"]+)"[^>]*>(.*?)</doc>', raw,
                                   flags=re.DOTALL):
        segments = []
        for seg_id, text in re.findall(r'<seg\s+id="([^"]+)"[^>]*>(.*?)</seg>', body,
                                       flags=re.DOTALL):
            text = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
            if text:
                segments.append((seg_id, text))
        docs[doc_id] = segments
    return docs


def build_acl6060_long(limit=0, subset="eval-zh"):
    """ACL 60/60 完整演讲：每条是一段 9–12 分钟音频，而非原来的句级切片。

    subset 为 ``eval-zh`` / ``dev-fr`` / ``all-de``；目标语支持官方十语。
    参考仍是书面翻译，不伪装成人类同传输出。
    """
    subset = subset or "eval-zh"
    split, sep, target = subset.rpartition("-")
    if not sep or split not in {"dev", "eval", "all"} or target not in _LONGFORM_TARGETS:
        raise ValueError("acl6060_long --subset 形如 eval-zh/dev-fr/all-de，目标语需为 ar/de/fa/fr/ja/nl/pt/ru/tr/zh")
    base = os.path.join(ROOT, "datasets/translation/longform_raw/acl6060_full/extracted/2/acl_6060")
    out = []
    for current in (("dev", "eval") if split == "all" else (split,)):
        text_dir = os.path.join(base, current, "text/xml")
        source_docs = _acl6060_docs(os.path.join(text_dir, f"ACL.6060.{current}.en-xx.en.xml"))
        target_docs = _acl6060_docs(os.path.join(text_dir, f"ACL.6060.{current}.en-xx.{target}.xml"))
        for doc_id, source_segments in source_docs.items():
            target_by_id = dict(target_docs.get(doc_id, []))
            segments = [{"id": seg_id, "source_text": source,
                         "ref_text": target_by_id.get(seg_id, "")}
                        for seg_id, source in source_segments if target_by_id.get(seg_id)]
            audio = os.path.join(base, current, "full_wavs", doc_id + ".wav")
            if not segments or not os.path.exists(audio):
                continue
            out.append({
                "id": f"acl6060_long_{current}_en_{target}_{doc_id}",
                "task": "translate", "lang": f"en-{target}", "source_lang": "en",
                "target_lang": _LONGFORM_TARGETS[target], "audio_path": audio,
                "source_text": "\n".join(s["source_text"] for s in segments),
                "ref_text": "\n".join(s["ref_text"] for s in segments),
                "segments": segments, "segment_count": len(segments), "longform": True,
                "reference_type": "written_translation", "keywords": [],
            })
            if limit and len(out) >= limit:
                return out
    return out


def _xml_text(element):
    """XML 长文本压成单个稳定字符串，保留词间边界。"""
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def build_mcif_long(limit=0, subset="zh"):
    """MCIF Long 英文 ACL 演讲 → 中/德/意文档级书面翻译（21 条/语向）。"""
    import gzip
    import xml.etree.ElementTree as ET

    target = subset or "zh"
    if target not in {"zh", "de", "it"}:
        raise ValueError("mcif_long --subset 只支持 zh/de/it")
    base = os.path.join(ROOT, "datasets/translation/longform_raw/mcif")
    ref_file = os.path.join(base, f"MCIF.long.{target}.ref.xml.gz")
    with gzip.open(ref_file, "rb") as src:
        root = ET.parse(src).getroot()
    out = []
    for sample in root.findall(".//sample"):
        if sample.get("task") != "TRANS":
            continue
        audio_name = (sample.findtext("audio_path") or "").strip()
        audio = os.path.join(base, "MCIF_DATA/LONG_AUDIOS", audio_name)
        source = _xml_text(sample.find("metadata/transcript"))
        ref = _xml_text(sample.find("reference"))
        if not source or not ref or not os.path.exists(audio):
            continue
        iid = sample.get("iid") or sample.get("id") or os.path.splitext(audio_name)[0]
        out.append({
            "id": f"mcif_long_en_{target}_{iid}", "task": "translate",
            "lang": f"en-{target}", "source_lang": "en",
            "target_lang": _LONGFORM_TARGETS[target], "audio_path": audio,
            "source_text": source, "ref_text": ref, "longform": True,
            "reference_type": "written_translation", "keywords": [],
        })
        if limit and len(out) >= limit:
            break
    return out


def build_realsi(limit=0, subset="en-zh"):
    """RealSI 真实长音频同传：en→zh / zh→en，各 10 场、每场约 3–7 分钟。

    除整场参考外保留人工时间段和术语对，后续可直接计算分段稳定性或术语指标。
    """
    import glob

    pair = subset or "en-zh"
    if pair not in {"en-zh", "zh-en"}:
        raise ValueError("realsi --subset 只支持 en-zh/zh-en")
    source_lang, target = pair.split("-")
    folder = pair.replace("-", "2")
    base = os.path.join(ROOT, "datasets/translation/longform_raw/realsi/data", folder)
    out = []
    for meta_path in sorted(glob.glob(os.path.join(base, "json/*.json"))):
        with open(meta_path, encoding="utf-8") as src:
            data = json.load(src)
        vid = data.get("vid") or os.path.splitext(os.path.basename(meta_path))[0]
        audio = os.path.join(base, "wav", vid + ".wav")
        segments, terms, seen_terms = [], [], set()
        for segment in data.get("segment") or []:
            source = (segment.get("src_text") or "").strip()
            ref = (segment.get("trg_text") or "").strip()
            if source and ref:
                segments.append({"start_ms": segment.get("start_time"),
                                 "end_ms": segment.get("end_time"),
                                 "source_text": source, "ref_text": ref})
            for utterance in segment.get("utterance") or []:
                for term in utterance.get("term") or []:
                    key = ((term.get("src") or "").strip(), (term.get("trg") or "").strip())
                    if all(key) and key not in seen_terms:
                        seen_terms.add(key)
                        terms.append({"source": key[0], "target": key[1]})
        if not segments or not os.path.exists(audio):
            continue
        out.append({
            "id": vid, "task": "translate", "lang": pair,
            "source_lang": source_lang, "target_lang": _LONGFORM_TARGETS[target],
            "audio_path": audio,
            "source_text": "\n".join(s["source_text"] for s in segments),
            "ref_text": "\n".join(s["ref_text"] for s in segments),
            "segments": segments, "segment_count": len(segments),
            "duration_ms": data.get("duration"), "terms": terms,
            "keywords": [term["source"] for term in terms], "longform": True,
            "reference_type": "human_simultaneous_interpretation",
        })
        if limit and len(out) >= limit:
            break
    return out


def _materialize_bstc_development(base):
    """按需解出 BSTC 开发集；逐文件原子替换，重复构建不会重复解压。"""
    import shutil
    import zipfile

    archive = os.path.join(base, "CCMT_2019_BSTC/data/development_data.zip")
    if not os.path.exists(archive):
        raise FileNotFoundError(archive)
    extracted = os.path.join(base, "development")
    os.makedirs(extracted, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = os.path.basename(info.filename)
            if not name or not (name.endswith(".wav") or name.endswith(".asr.json")):
                continue
            target = os.path.join(extracted, name)
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                continue
            partial = target + ".partial"
            with bundle.open(info) as src, open(partial, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(partial, target)
    return extracted


def build_bstc_long(limit=0):
    """BSTC 2019 中文演讲→英文，开发集 16 场，保留官方句级时间戳。"""
    import glob

    base = os.path.join(ROOT, "datasets/translation/longform_raw/bstc_ccmt2019")
    extracted = _materialize_bstc_development(base)
    out = []
    for annotation in sorted(glob.glob(os.path.join(extracted, "*.asr.json"))):
        segments = []
        with open(annotation, encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                item = json.loads(line)
                source = (item.get("transcript") or item.get("asr") or "").strip()
                ref = (item.get("translation") or "").strip()
                if not source or not ref:
                    continue
                start_ms = round(float(item.get("offset") or 0) * 1000)
                end_ms = round((float(item.get("offset") or 0)
                                + float(item.get("duration") or 0)) * 1000)
                segments.append({"start_ms": start_ms, "end_ms": end_ms,
                                 "source_text": source, "ref_text": ref})
        talk_id = os.path.basename(annotation).removesuffix(".asr.json")
        audio = os.path.join(extracted, talk_id + ".wav")
        if not segments or not os.path.exists(audio):
            continue
        out.append({
            "id": f"bstc_dev_zh_en_{talk_id}", "task": "translate", "lang": "zh-en",
            "source_lang": "zh", "target_lang": "English", "audio_path": audio,
            "source_text": "\n".join(s["source_text"] for s in segments),
            "ref_text": "\n".join(s["ref_text"] for s in segments),
            "segments": segments, "segment_count": len(segments),
            "duration_ms": max(s["end_ms"] for s in segments), "longform": True,
            "reference_type": "written_translation", "keywords": [],
        })
        if limit and len(out) >= limit:
            break
    return out


def build_vcsum(limit=0):
    """VCSUM 会议总结 test split。注意：该 parquet 镜像 summary 字段全空，
    用人工标注的 discussion(讨论要点列表)拼接作参考摘要——报告口径需注明。"""
    import pyarrow.parquet as pq
    f = os.path.join(ROOT, "datasets/summary/vcsum/data/test-00000-of-00001.parquet")
    t = pq.read_table(f, columns=["id", "context", "discussion"])
    out = []
    for r in t.to_pylist():
        disc = [d.strip() for d in (r.get("discussion") or []) if d and d.strip()]
        if not r.get("context") or not disc:
            continue
        out.append({"id": f"vcsum_{r['id']}", "task": "summarize", "lang": "zh",
                    "source_text": r["context"], "ref_text": "\n".join(disc)})
        if limit and len(out) >= limit:
            break
    return out


def build_vwb(limit=0, subset="zh"):
    """Voices-in-the-Wild-Bench 鲁棒性 ASR（parquet 内嵌 wav，8 类扰动 × 真人/合成）。

    ⚠️ 只用真实样本(real-*，1500 条)：合成样本(sim-*/random_synthetic，3500 条)的音频
    是"随机干扰前缀 + 目标句"，但参考只标目标句 → 整段 CER 必爆(190%+)，是上游数据集
    标注与 CER 口径不兼容，非模型/本框架问题。详见 docs/plat-data-gaps.md。
    真实样本覆盖全部 8 类扰动(中英各半)，鲁棒性维度不损失。

    subset: real-* 内的子串过滤，逗号分隔。默认 "zh"=中文真实全扰动；"en" 英文。
    中英分清单建（score 按首行 lang 定 CER/WER），勿混。
    """
    import glob
    import pyarrow.parquet as pq
    base = os.path.join(ROOT, "datasets/asr/voices_wild_bench")
    audio_dir = os.path.join(base, "wav")
    os.makedirs(audio_dir, exist_ok=True)
    want = [s.strip() for s in (subset or "zh").split(",") if s.strip()]
    out = []
    for f in sorted(glob.glob(os.path.join(base, "data/*.parquet"))):
        t = pq.read_table(f, columns=["audio", "text", "subset", "name"])
        for row in t.to_pylist():
            sub = row.get("subset") or ""
            if not sub.startswith("real-"):  # 排除合成样本(参考不可信)
                continue
            if want and not any(w in sub for w in want):
                continue
            b = (row.get("audio") or {}).get("bytes")
            text = row.get("text", "")
            if not b or not text:
                continue
            uid = row["name"]
            fp = os.path.join(audio_dir, uid + ".wav")
            if not os.path.exists(fp):
                open(fp, "wb").write(b)
            out.append({"id": f"vwb_{uid}", "audio_path": fp, "ref_text": text,
                        "lang": "zh" if "-zh-" in f"-{sub}-" else "en", "keywords": []})
            if limit and len(out) >= limit:
                return out
    return out


def build_wildasr(limit=0, subset=""):
    """WildASR (Boson AI) 英文鲁棒性 ASR，全真人语音（parquet 内嵌 wav）。

    subset: 按 subset 字段子串过滤（clean/clipping/far_field/noise_gap/
    phone_codec/reverberation/demographic_accent），逗号分隔，空=全部 7 split。
    全英文 → lang=en → WER。
    """
    import glob
    import hashlib
    import pyarrow.parquet as pq
    base = os.path.join(ROOT, "datasets/asr/wildasr")
    audio_dir = os.path.join(base, "wav")
    os.makedirs(audio_dir, exist_ok=True)
    want = [s.strip() for s in subset.split(",") if s.strip()]
    out = []
    seen_ids = {}
    for f in sorted(glob.glob(os.path.join(base, "**/*.parquet"), recursive=True)):
        t = pq.read_table(f, columns=["audio", "transcript", "subset", "audio_hash_id"])
        for row in t.to_pylist():
            sub = row.get("subset") or ""
            if want and not any(w in sub for w in want):
                continue
            b = (row.get("audio") or {}).get("bytes")
            text = row.get("transcript", "")
            if not b or not text:
                continue
            uid = f"{sub}_{row['audio_hash_id'][:12]}"
            sample_id = f"wildasr_{uid}"
            if sample_id in seen_ids:
                fingerprint = (text, hashlib.sha256(b).digest())
                if seen_ids[sample_id] != fingerprint:
                    raise ValueError(f"WildASR 重复 ID 对应不同内容: {sample_id}")
                continue
            seen_ids[sample_id] = (text, hashlib.sha256(b).digest())
            fp = os.path.join(audio_dir, uid + ".wav")
            if not os.path.exists(fp):
                open(fp, "wb").write(b)
            out.append({"id": sample_id, "audio_path": fp, "ref_text": text,
                        "lang": "en", "keywords": []})
            if limit and len(out) >= limit:
                return out
    return out


def build_formula(limit=0):
    """TTS 公式转写 golden 集（8 条，LaTeX/MathML → 口语化朗读，自建示例）。"""
    out = []
    for line in open(os.path.join(ROOT, "datasets/golden/tts_formula.jsonl"), encoding="utf-8"):
        out.append(json.loads(line))
        if limit and len(out) >= limit:
            break
    return out


def build_minutes(limit=0):
    """会议纪要：AliMeeting 远场音频 × 4MUG 人工抽取式摘要（24 场配对，见 plat-data-gaps）。

    ref_text = 会议级 key_sentence 并集按句序拼接（多标注员取并集），作 ROUGE 锚点。
    打分走 summarize 口径；G-Eval 裁判二期。单场音频 15-30min，infer 很慢属正常。
    """
    import csv
    csv.field_size_limit(10 ** 9)
    base = os.path.join(ROOT, "datasets/summary/alimeeting4mug")
    ann = {}
    for fn in ("except_TS_test1.csv", "dev.csv", "train.csv"):
        p = os.path.join(base, "data", fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                try:
                    d = json.loads(row[1])
                    ann[d["meeting_key"]] = d
                except Exception:
                    continue
    out = []
    for line in open(os.path.join(base, "audio_paired_meetings.jsonl"), encoding="utf-8"):
        p = json.loads(line)
        d = ann.get(p["meeting_key"])
        if not d:
            continue
        sent = {str(s_["id"]): s_["s"] for s_ in d.get("sentences", [])}  # id 两侧类型不一(int vs str)，统一 str
        ids = set()
        for c in d.get("candidate") or []:           # 会议级摘要(多标注员并集)
            ids.update(c.get("key_sentence") or [])
        if not ids:                                   # 兜底：话题级
            for seg in d.get("topic_segment_ids") or []:
                for c in seg.get("candidate") or []:
                    ids.update(c.get("key_sentence") or [])
        ids = {str(i) for i in ids}
        ref = "\n".join(sent[i] for i in sorted(ids, key=lambda x: int(x)) if i in sent)
        if not ref:
            continue
        out.append({"id": f"minutes_{p['meeting_key']}", "task": "minutes",
                    "audio_path": p["audio_path"], "ref_text": ref, "lang": "zh"})
        if limit and len(out) >= limit:
            break
    return out


def _parse_textgrid_speakers(path):
    """AliMeeting far TextGrid(多 tier=多说话人) → 参考分段 [[start,end,spk],...]。
    每个 IntervalTier 名 N_SPKxxxx 是一个说话人，其非空 interval 是该人发言区间。"""
    import re
    txt = open(path, encoding="utf-8").read()
    segs, cur_spk = [], None
    # 逐 item 块解析：name="N_SPKxxxx" 起一个说话人，后续 xmin/xmax/text 是其区间
    for m in re.finditer(r'name = "([^"]+)"|xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', txt):
        if m.group(1):
            cur_spk = m.group(1)
        elif cur_spk and m.group(4).strip():  # 有文字=该说话人在说(空 interval 是静音)
            segs.append([float(m.group(2)), float(m.group(3)), cur_spk])
    return segs


def build_diar(limit=0):
    """说话人分离：AliMeeting far 会议(Test 20+Eval 8) 音频 + TextGrid 多说话人参考分段。
    task=diarization,ref 存参考分段,score 用 pyannote DER。单会议一条(整段几十分钟)。"""
    import glob
    out = []
    for split in ("Test", "Eval"):
        far = os.path.join(ROOT, f"datasets/asr/alimeeting/{split}_Ali/{split}_Ali_far")
        for au in sorted(glob.glob(os.path.join(far, "audio_dir", "*.wav"))):
            mid = os.path.basename(au).split("_MS")[0]  # R8003_M8001_MS801 → R8003_M8001
            tg = os.path.join(far, "textgrid_dir", mid + ".TextGrid")
            if not os.path.exists(tg):
                continue
            segs = _parse_textgrid_speakers(tg)
            if not segs:
                continue
            out.append({"id": f"diar_{mid}", "task": "diarization",
                        "audio_path": rel_path(au), "ref": segs,
                        "n_spk_ref": len(set(s[2] for s in segs)), "lang": "zh"})
            if limit and len(out) >= limit:
                return out
    return out


def build_wsc(limit=0, subset=""):
    """WSC-Eval（WenetSpeech-Chuan）川渝方言腔普通话，kaldi 风格 text+wav/。

    subset: "easy"/"hard" 逗号分隔，空=两者全要（Easy 安静朗读 6981 + Hard 直播短视频带噪 1392）。
    Short/Long 是同一批音频按时长的另一种切分，勿与 Easy/Hard 混用。
    文本按论文口径清洗：* 号（听不清）去掉；（）内容保留、括号去掉。
    """
    base = os.path.join(ROOT, "datasets/asr/wsc_eval/WSC-Eval-ASR")
    want = [s.strip().capitalize() for s in (subset or "easy,hard").split(",") if s.strip()]
    out = []
    for sub in want:
        for line in open(os.path.join(base, sub, "text"), encoding="utf-8"):
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) < 2:
                continue
            uid, text = parts
            text = text.replace("*", "").replace("（", "").replace("）", "").strip()
            path = os.path.join(base, sub, "wav", uid + ".wav")
            if not text or not os.path.exists(path):
                continue
            out.append({"id": f"wsc_{uid}", "audio_path": path, "ref_text": text,
                        "lang": "zh", "subset": sub.lower(), "keywords": []})
            if limit and len(out) >= limit:
                return out
    return out


_CHUAN_YU_CITIES = (
    ("成都", "chengdu"),
    ("重庆", "chongqing"),
    ("乐山", "leshan"),
    ("宜宾", "yibin"),
    ("泸州", "luzhou"),
    ("自贡", "zigong"),
    ("内江", "neijiang"),
    ("雅安", "yaan"),
    ("西昌", "xichang"),
    ("南充", "nanchong"),
    ("达州", "dazhou"),
    ("广安", "guangan"),
)


def build_chuan_yu(limit=0, subset=""):
    """MagicData 川渝 12 城市子方言集（UTTERANCEINFO.tsv + WAV/说话人/音频）。

    subset: 中文城市名或英文 slug，逗号分隔；空=全部 13,068 条。
    """
    import csv

    base = os.path.join(ROOT, "datasets/asr/chuan_yu_12city")
    aliases = {name.casefold(): name for name, _slug in _CHUAN_YU_CITIES}
    aliases.update({slug.casefold(): name for name, slug in _CHUAN_YU_CITIES})
    requested = []
    for token in (s.strip() for s in subset.split(",")):
        if not token:
            continue
        city = aliases.get(token.casefold())
        if not city:
            raise ValueError(f"chuan_yu --subset 不支持城市: {token}")
        requested.append(city)
    wanted = set(requested)

    out = []
    for city, slug in _CHUAN_YU_CITIES:
        if wanted and city not in wanted:
            continue
        metadata = os.path.join(base, city, "UTTERANCEINFO.txt")
        with open(metadata, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                wav_name = (row.get("UTTRANS_ID") or "").strip()
                speaker = (row.get("SPEAKER_ID") or "").strip()
                text = (row.get("TRANSCRIPTION") or "").strip()
                path = os.path.join(base, city, "WAV", speaker, wav_name)
                if not wav_name or not speaker or not text or not os.path.exists(path):
                    continue
                uid = os.path.splitext(wav_name)[0]
                out.append({
                    "id": f"chuan_yu_{slug}_{uid}",
                    "audio_path": path,
                    "ref_text": text,
                    "lang": "zh",
                    "dialect": city,
                    "city": city,
                    "speaker_id": speaker,
                    "keywords": [],
                })
                if limit and len(out) >= limit:
                    return out
    return out


def build_kespeech(limit=0, subset=""):
    """KeSpeech test（8 子方言腔普通话，parquet 内嵌 wav）→ 抽 wav + manifest。

    subset: Dialect 子串过滤（Northeastern/Southwestern/Zhongyuan/JiangHuai/
    JiLu/JiaoLiao/LanYin/Beijing），逗号分隔，空=全部 19,723 条。
    manifest 带 dialect 字段，供分方言 CER 透视。
    """
    import glob
    import re
    import pyarrow.parquet as pq
    base = os.path.join(ROOT, "datasets/asr/kespeech_test")
    audio_dir = os.path.join(base, "wav")
    os.makedirs(audio_dir, exist_ok=True)
    def norm_dialect(s):
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())
    want = [norm_dialect(s) for s in subset.split(",") if s.strip()]
    out = []
    for f in sorted(glob.glob(os.path.join(base, "data/*.parquet"))):
        t = pq.read_table(f, columns=["ID", "Text", "Dialect", "audio"])
        for row in t.to_pylist():
            dia = row.get("Dialect") or ""
            dia_norm = norm_dialect(dia)
            if want and not any(w in dia_norm for w in want):
                continue
            text = row.get("Text", "")
            b = (row.get("audio") or {}).get("bytes")
            if not text or not b:
                continue
            uid = row["ID"]
            fp = os.path.join(audio_dir, uid + ".wav")
            if not os.path.exists(fp):
                open(fp, "wb").write(b)
            out.append({"id": f"kespeech_{uid}", "audio_path": fp, "ref_text": text,
                        "lang": "zh", "dialect": dia, "keywords": []})
            if limit and len(out) >= limit:
                return out
    return out


def build_wenetspeech(limit=0, subset=""):
    """WenetSpeech test_net/test_meeting（Lhotse cuts，与 speechio 完全同构）。

    中文 ASR 业界最通用的对数基准——Qwen3-ASR/Fun-ASR/FireRedASR/Seed-ASR 等
    几乎人人用。net=网络多场景真实，meeting=会议(与 diar 的 AliMeeting 不同源)。
    subset: 'net'/'meeting' 逗号分隔，空=两者。⚠️ test_net 上游有少量标注错误(官方 issue #63)。
    """
    import glob
    import gzip
    base = os.path.join(ROOT, "datasets/asr/wenetspeech/data")
    pat = {"net": "cuts_TEST_NET*.jsonl.gz", "meeting": "cuts_TEST_MEETING*.jsonl.gz"}
    want = [s.strip().lower() for s in (subset or "net,meeting").split(",") if s.strip() in pat]
    subs = []
    for w in want:
        subs += sorted(glob.glob(os.path.join(base, pat[w])))
    ex_root = os.path.join(base, "extracted")
    out = []
    for jp in subs:
        _extract_once(jp.replace(".jsonl.gz", ".tar.gz"), ex_root)
        with gzip.open(jp, "rt") as f:
            for line in f:
                d = json.loads(line)
                sup = d.get("supervisions", [{}])[0]
                text = sup.get("text", "")
                src = d.get("recording", {}).get("sources", [{}])[0].get("source", "")
                if src.startswith("data/"):  # jsonl 里带 data/ 前缀，tar 内没有
                    src = src[len("data/"):]
                path = os.path.join(ex_root, src)
                if not text or not os.path.exists(path):
                    continue
                out.append({"id": d.get("id"), "audio_path": path,
                            "ref_text": text, "lang": "zh", "keywords": []})
                if limit and len(out) >= limit:
                    return out
    return out


def build_wubench(limit=0):
    """WenetSpeech-Wu-Bench 吴语/上海话 ASR（understanding/asr.parquet）。

    字段：utt_id / label(转写) / audio(直接是 wav bytes，非 {bytes,path} 结构) / task(恒为 asr)。
    """
    import pyarrow.parquet as pq
    base = os.path.join(ROOT, "datasets/asr/wu_bench")
    audio_dir = os.path.join(base, "wav")
    os.makedirs(audio_dir, exist_ok=True)
    t = pq.read_table(os.path.join(base, "understanding/asr.parquet"))
    cols = t.column_names
    out = []
    for i, row in enumerate(t.to_pylist()):
        text = row.get("label") or ""
        audio = row.get("audio")
        b = audio if isinstance(audio, (bytes, bytearray)) else (audio or {}).get("bytes")
        if not text or not b:
            continue
        uid = os.path.basename(str(row.get("utt_id") or i)).replace(".wav", "")
        fp = os.path.join(audio_dir, uid + ".wav")
        if not os.path.exists(fp):
            open(fp, "wb").write(b)
        out.append({"id": f"wu_{uid}", "audio_path": fp, "ref_text": text,
                    "lang": "zh", "keywords": []})
        if limit and len(out) >= limit:
            break
    if not out:
        raise RuntimeError(f"wu_bench parquet 字段不符预期: {cols}")
    return out


def build_covost2(limit=0, subset="zh-en"):
    """CoVoST2 语音翻译（fixie-ai 镜像 parquet，CommonVoice mp3 内嵌）。

    subset: "zh-en"（中文音频→英文，默认）或 "en-zh"。
    与 fleurs 同跑法：gemma-text 用 source_text 直翻=上限，级联从音频起步。
    """
    import glob
    import pyarrow.parquet as pq
    direction = (subset or "zh-en").strip()
    cfg = {"zh-en": ("zh-CN_en", "zh-en", "英文"), "en-zh": ("en_zh-CN", "en-zh", "中文")}[direction]
    base = os.path.join(ROOT, "datasets/translation/covost2", cfg[0])
    audio_dir = os.path.join(base, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    out = []
    for f in sorted(glob.glob(os.path.join(base, "test-*.parquet"))):
        t = pq.read_table(f, columns=["id", "audio", "sentence", "translation"])
        for row in t.to_pylist():
            b = (row.get("audio") or {}).get("bytes")
            src, ref = row.get("sentence", ""), row.get("translation", "")
            if not b or not src or not ref:
                continue
            uid = row["id"]
            fp = os.path.join(audio_dir, uid + ".mp3")  # CommonVoice 源是 mp3，load_wav_bytes 能读
            if not os.path.exists(fp):
                open(fp, "wb").write(b)
            out.append({"id": f"covost_{uid}", "task": "translate", "lang": cfg[1],
                        "audio_path": fp, "source_text": src, "ref_text": ref,
                        "target_lang": cfg[2]})
            if limit and len(out) >= limit:
                return out
    return out


def build_hardmt(limit=0, subset="zh-en"):
    """HardMTBench 术语/领域翻译（12 领域含金融/法律/医疗，逐条带术语对+难度）。

    subset: 方向 "zh-en"/"en-zh" + 可叠 domain 子串（如 "zh-en,finance"），逗号分隔。
    keywords 灌目标侧术语（alias 命中即算），供术语命中率透视。
    """
    toks = [s.strip().lower() for s in (subset or "zh-en").split(",") if s.strip()]
    direction = "en2zh" if "en-zh" in toks else "zh2en"
    domains = [t for t in toks if t not in ("zh-en", "en-zh")]
    lang, tgt = ("zh-en", "英文") if direction == "zh2en" else ("en-zh", "中文")
    out = []
    p = os.path.join(ROOT, "datasets/translation/hardmtbench/repo/HardMTBench.jsonl")
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        if not d["id"].endswith("_" + direction):
            continue
        if domains and not any(w in (d.get("domain") or "").lower() for w in domains):
            continue
        kws = [{"surface": t["source"], "aliases": [t["target"]]}
               for t in (d.get("terminology") or []) if t.get("target")]
        out.append({"id": f"hardmt_{d['id']}", "task": "translate", "lang": lang,
                    "source_text": d["source_text"], "ref_text": d["reference"],
                    "target_lang": tgt, "domain": d.get("domain", ""), "keywords": kws})
        if limit and len(out) >= limit:
            break
    return out


def build_wmt25(limit=0, subset=""):
    """WMT25 General en→zh（文档级，news/speech/social/literary 四域；speech 域贴演讲转译）。

    subset: domain 子串过滤（如 "speech"），逗号分隔，空=全部 87 篇。
    条目是整篇文档（数百到数千字），单条 infer 偏慢属正常。
    """
    want = [s.strip().lower() for s in subset.split(",") if s.strip()]
    out = []
    p = os.path.join(ROOT, "datasets/translation/wmt25/wmt25-genmt.jsonl")
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        if d.get("src_lang") != "en" or not d.get("tgt_lang", "").startswith("zh"):
            continue
        dom = (d.get("domain") or "").lower()
        if want and not any(w in dom for w in want):
            continue
        ref = ((d.get("refs") or {}).get("refA") or {}).get("ref", "")
        if not d.get("src_text") or not ref:
            continue
        out.append({"id": f"wmt25_{d['doc_id']}", "task": "translate", "lang": "en-zh",
                    "source_text": d["src_text"], "ref_text": ref,
                    "target_lang": "中文", "domain": dom})
        if limit and len(out) >= limit:
            break
    return out


def build_wmt25term(limit=0):
    """WMT25 Terminology Track2 金融（HKMA 年报 13 篇/年 × 10 年，文档级 zh-Hant↔en）。

    建 zh→en 方向：source=繁中原文，ref=英文，proper 术语表灌 keywords
    （繁中术语→英文别名列表，alias 命中任一即算）。⚠️ CC BY-NC 禁商用，仅限非商业用途。
    """
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "datasets/translation/wmt25_term/track2/full_data_*.jsonl"))):
        year = os.path.basename(p)[len("full_data_"):-len(".jsonl")]
        for i, line in enumerate(open(p, encoding="utf-8")):
            d = json.loads(line)
            if not d.get("zh") or not d.get("en"):
                continue
            kws = [{"surface": k, "aliases": v} for k, v in (d.get("proper") or {}).items() if v]
            out.append({"id": f"wmt25term_{year}_{i}", "task": "translate", "lang": "zh-en",
                        "source_text": d["zh"], "ref_text": d["en"],
                        "target_lang": "英文", "keywords": kws})
            if limit and len(out) >= limit:
                return out
    return out


BUILDERS = {"wsyue": build_wsyue, "aishell": build_aishell, "aishell2": build_aishell2,
            "aishell2_eval": build_aishell2_eval,
            "librispeech": build_librispeech, "open_asr": build_open_asr, "ascend": build_ascend,
            "flores": build_flores, "wmt": build_wmt, "cnewsum": build_cnewsum,
            "seaco": build_seaco, "speechio": build_speechio, "med_it": build_med_it,
            "fleurs": build_fleurs, "fleurs_asr": build_fleurs_asr,
            "commonvoice": build_commonvoice, "ja_benchmark": build_ja_benchmark,
            "vcsum": build_vcsum, "vwb": build_vwb,
            "wildasr": build_wildasr, "minutes": build_minutes, "formula": build_formula, "diar": build_diar,
            "wsc": build_wsc, "chuan_yu": build_chuan_yu,
            "kespeech": build_kespeech, "wubench": build_wubench,
            "wenetspeech": build_wenetspeech,
            "covost2": build_covost2, "hardmt": build_hardmt,
            "wmt25": build_wmt25, "wmt25term": build_wmt25term, "acl6060": build_acl6060,
            "acl6060_long": build_acl6060_long, "mcif_long": build_mcif_long,
            "realsi": build_realsi, "bstc_long": build_bstc_long}


def main():
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(BUILDERS))
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--shuffle", action="store_true",
                    help="随机抽样(Lite 用,比取前 N 更有代表性),配 --seed 可复现")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", default="", help="wsyue: 'short'/'long'；aishell2_eval: ios/android/mic；librispeech: clean/other；open_asr: ami/earnings22/gigaspeech/spgispeech/voxpopuli/tedlium；wmt: wmt22-zh-en 等；speechio: '1,5'/'ZH00001'；vwb: 'zh'/'en'；wsc: 'easy'/'hard'；"
                    "chuan_yu: 中文城市名或英文 slug；kespeech: 方言名；"
                    "covost2/hardmt: 'zh-en'/'en-zh'(hardmt 可叠 domain)；wmt25: domain；"
                    "fleurs_asr/commonvoice: 语言短码；ja_benchmark: cv8/jsut/reazon；acl6060_long: 'eval-zh'；mcif_long: zh/de/it；realsi: en-zh/zh-en")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _SUBSET_OK = ("wsyue", "aishell2_eval", "librispeech", "open_asr", "wmt", "speechio", "vwb", "wildasr", "wsc", "chuan_yu", "kespeech", "wenetspeech", "covost2", "hardmt", "wmt25", "flores", "fleurs", "fleurs_asr", "commonvoice", "ja_benchmark", "acl6060", "acl6060_long", "mcif_long", "realsi")
    kw = {"subset": args.subset} if args.dataset in _SUBSET_OK and args.subset else {}
    if args.shuffle:
        rows = BUILDERS[args.dataset](limit=0, **kw)  # 先全量再随机抽
        random.Random(args.seed).shuffle(rows)
        if args.limit:
            rows = rows[: args.limit]
    else:
        rows = BUILDERS[args.dataset](limit=args.limit, **kw)
    ensure_unique_ids(rows, source=f"{args.dataset} manifest")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with atomic_text_writer(args.out, validator=validate_jsonl_file) as f:
        for r in rows:
            if r.get("audio_path"):  # 存相对 ROOT 路径 → 跨机器可移植，不入指纹
                r = {**r, "audio_path": rel_path(r["audio_path"])}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.dataset}: 写入 {len(rows)} 条 → {args.out}")


if __name__ == "__main__":
    main()
