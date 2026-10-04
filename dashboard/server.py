"""测评 Dashboard 后端 — 包住 eval/ 流水线，给前端提供：
  GET  /                看板页面
  GET  /api/overview    测评集 + 模型 + 已有结果
  POST /api/run         触发评测(model×dataset×tier) → 后台跑 build_manifest + runner
  GET  /api/jobs        运行中/完成的任务状态
  GET  /api/job_request 单个任务第一条真实调用的脱敏请求详情
  GET  /api/job_response 单个任务第一条真实返回的脱敏限长示例
  GET  /api/result      单个结果详情(每个点)

启动: uv sync --locked && uv run python dashboard/server.py
"""

import base64
import difflib
import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime

import requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


sys.path.insert(0, os.path.join(ROOT, "eval"))   # 复用 eval/logconf 的中央日志配置
from config import load_dotenv  # noqa: E402

load_dotenv()

from infer import infer_file_path, language_tag  # noqa: E402
from adapters import (  # noqa: E402
    LONGFORM_PROGRESS_MARKER,
    capability_contract_for,
    request_contract_for,
)
import logconf  # noqa: E402
from data_quality import exclusions_for_rows, reviewed_asr_metrics  # noqa: E402
from metrics import comet_runtime_available  # noqa: E402
from omnisteval_eval import runtime_available as omnisteval_runtime_available  # noqa: E402
from scorer_client import (scorer_health, scorer_last_error, scorer_ping,  # noqa: E402
                           scorer_runtime_available,
                           scorer_warmup,
                           speaker_score as remote_speaker_score,
                           utmos_score as remote_utmos_score,
                           whisper_transcribe as remote_whisper_transcribe,
                           xcomet_score as remote_xcomet_score)
from util import atomic_write_json, manifest_sha_v2  # noqa: E402

LOG = logconf.get_logger("dashboard", file="dashboard.log")
JOBS_LOG_DIR = os.path.join(ROOT, "logs", "jobs")   # 每任务全量日志(学 OpenCompass 每任务日志)
JOBS_LOG_KEEP = int(os.environ.get("DASH_JOBLOG_KEEP", "500"))   # 保留最近 N 个,防无限累积
REVIEW_DRAFT_DIR = os.path.join(ROOT, "reviews", "drafts")
REVIEW_LOCK = threading.RLock()

def _prune_job_logs(keep=None):
    """按 mtime 裁掉最旧的每任务日志,只留最近 keep 个(裸 open 不归轮转管,故手动收口)。"""
    keep = JOBS_LOG_KEEP if keep is None else keep
    try:
        fs = sorted(glob.glob(os.path.join(JOBS_LOG_DIR, "*.log")), key=os.path.getmtime)
        for f in fs[:-keep] if keep > 0 else fs:
            try:
                os.remove(f)
            except OSError:
                pass
    except Exception:
        pass

_CHUAN_YU_CITIES = [
    ("chengdu", "成都", 1993),
    ("chongqing", "重庆", 2034),
    ("leshan", "乐山", 1308),
    ("yibin", "宜宾", 1190),
    ("luzhou", "泸州", 1330),
    ("zigong", "自贡", 885),
    ("neijiang", "内江", 889),
    ("yaan", "雅安", 727),
    ("xichang", "西昌", 1222),
    ("nanchong", "南充", 476),
    ("dazhou", "达州", 478),
    ("guangan", "广安", 536),
]
_CHUAN_YU_SRC = "MagicData 川渝 12 城市子方言集（CC BY-NC-ND 4.0，限非商业）"

# 测评集登记表（count/size 为实测值；runnable=有 build_manifest builder）
DATASETS = [
    # ── ASR 普通话·通用 ──
    {"id": "aishell", "name": "AISHELL-1 test", "scenario": "ASR·普通话", "count": 7176, "size": "0.9G", "lang": "zh", "metric": "CER", "runnable": True},
    {"id": "aishell2_ios", "name": "AISHELL-2 train · iOS抽样", "scenario": "ASR·普通话", "count": 1995, "size": "~0.3G", "lang": "zh", "metric": "CER", "runnable": True, "builder": "aishell2"},
    {"id": "aishell2_test_ios", "name": "AISHELL-2 test · iOS", "scenario": "ASR·普通话", "count": 5000, "size": "共享1.35G", "lang": "zh", "metric": "CER", "runnable": True, "builder": "aishell2_eval", "subset": "ios",
     "required_paths": ["asr/aishell2_eval/AISHELL-DEV-TEST-SET/iOS/test/wav.scp", "asr/aishell2_eval/AISHELL-DEV-TEST-SET/iOS/test/trans.txt"],
     "src": "AISHELL2-2018A-EVAL 官方 test（Apache-2.0）", "trait": "10 位说话人、5000 条室内朗读；iOS 近场通道", "probe": "AISHELL-2 行业主榜口径，与 MiMo/Qwen/Fun-ASR 等公开结果对标"},
    {"id": "aishell2_test_android", "name": "AISHELL-2 test · Android", "scenario": "ASR·普通话", "count": 5000, "size": "共享1.35G", "lang": "zh", "metric": "CER", "runnable": True, "builder": "aishell2_eval", "subset": "android",
     "required_paths": ["asr/aishell2_eval/AISHELL-DEV-TEST-SET/Android/test/wav.scp", "asr/aishell2_eval/AISHELL-DEV-TEST-SET/Android/test/trans.txt"],
     "src": "AISHELL2-2018A-EVAL 官方 test（Apache-2.0）", "trait": "与 iOS/Mic 同文本同说话人；Android 手机通道", "probe": "跨移动设备信道稳定性"},
    {"id": "aishell2_test_mic", "name": "AISHELL-2 test · Mic", "scenario": "ASR·普通话", "count": 5000, "size": "共享1.35G", "lang": "zh", "metric": "CER", "runnable": True, "builder": "aishell2_eval", "subset": "mic",
     "required_paths": ["asr/aishell2_eval/AISHELL-DEV-TEST-SET/Mic/test/wav.scp", "asr/aishell2_eval/AISHELL-DEV-TEST-SET/Mic/test/trans.txt"],
     "src": "AISHELL2-2018A-EVAL 官方 test（Apache-2.0）", "trait": "与 iOS/Android 同文本同说话人；高保真麦克风通道", "probe": "高保真麦克风信道与手机信道差异"},
    {"id": "speechio", "name": "SpeechIO ZH00-26", "scenario": "ASR·普通话", "count": 43178, "size": "6.1G", "lang": "zh", "metric": "CER", "runnable": True},
    # WenetSpeech 本体(2021，10000h+ 普通话通用大集)：中文 ASR 最通用对数基准(Qwen3-ASR/FireRedASR 等人人用)。gated 需登录
    {"id": "wenetspeech_net", "name": "WenetSpeech 网络", "scenario": "ASR·普通话", "count": 24774, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "wenetspeech", "subset": "net"},
    {"id": "wenetspeech_meeting", "name": "WenetSpeech 会议", "scenario": "ASR·会议", "count": 8370, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "wenetspeech", "subset": "meeting"},
    # ── ASR 普通话·领域(speechio 子集单切，演讲/播报/讲课体，测领域术语，非对话场景) ──
    {"id": "fin_zh00", "name": "ZH00 金融·会议演讲", "scenario": "ASR·金融", "count": 879, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "speechio", "subset": "ZH00000", "domain": "finance"},
    {"id": "gov_zh01", "name": "ZH01 时政·新闻联播", "scenario": "ASR·时政", "count": 5069, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "speechio", "subset": "ZH00001", "domain": "government_emergency"},
    {"id": "law_zh11", "name": "ZH11 法律·罗翔法考", "scenario": "ASR·法律", "count": 1053, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "speechio", "subset": "ZH00011", "domain": "legal"},
    # ── ASR 方言/口音(发音偏差维度，phase_c 补位已不可获取的自建集 sb2) ──
    {"id": "wsyue", "name": "WSYue 粤语·Short", "scenario": "ASR·粤语", "count": 7060, "size": "1.0G", "lang": "auto", "metric": "CER", "runnable": True, "subset": "short"},
    {"id": "wsyue_long", "name": "WSYue 粤语·Long", "scenario": "ASR·粤语", "count": 370, "size": "0.2G", "lang": "auto", "metric": "CER", "runnable": True, "builder": "wsyue", "subset": "long"},
    {"id": "wsc_easy", "name": "WSC-Eval 川渝·Easy", "scenario": "ASR·方言腔", "count": 6981, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "wsc", "subset": "easy"},
    {"id": "wsc_hard", "name": "WSC-Eval 川渝·Hard", "scenario": "ASR·方言腔", "count": 1392, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "wsc", "subset": "hard"},
    {"id": "chuan_yu_12city", "name": "MagicData 川渝12城·总集", "scenario": "ASR·川渝子方言", "count": 13068, "size": "4.2G", "lang": "zh", "metric": "CER", "runnable": True, "builder": "chuan_yu",
     "src": _CHUAN_YU_SRC, "trait": "官方标称 33h（WAV 总时长实测 38.47h），12 城市、38 位本地说话人", "probe": "城市级川渝子方言综合 CER"},
    *[
        {"id": f"chuan_yu_{slug}", "name": f"川渝12城·{city}", "scenario": "ASR·川渝子方言",
         "count": count, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True,
         "builder": "chuan_yu", "subset": city, "src": _CHUAN_YU_SRC,
         "trait": f"{city}本地说话人录制，官方普通话标准转写", "probe": f"{city}城市级子方言 CER"}
        for slug, city, count in _CHUAN_YU_CITIES
    ],
    {"id": "wubench", "name": "Wu-Bench 吴语", "scenario": "ASR·方言腔", "count": 4851, "size": "1.0G", "lang": "zh", "metric": "CER", "runnable": True},
    {"id": "kespeech_beijing", "name": "KeSpeech 北京", "scenario": "ASR·方言腔", "count": 265, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "Beijing"},
    {"id": "kespeech_northeastern", "name": "KeSpeech 东北", "scenario": "ASR·方言腔", "count": 351, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "Northeastern"},
    {"id": "kespeech_southwestern", "name": "KeSpeech 西南", "scenario": "ASR·方言腔", "count": 2693, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "Southwestern"},
    {"id": "kespeech_zhongyuan", "name": "KeSpeech 中原", "scenario": "ASR·方言腔", "count": 3239, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "Zhongyuan"},
    {"id": "kespeech_jianghuai", "name": "KeSpeech 江淮", "scenario": "ASR·方言腔", "count": 2271, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "JiangHuai"},
    {"id": "kespeech_jilu", "name": "KeSpeech 冀鲁", "scenario": "ASR·方言腔", "count": 2809, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "JiLu"},
    {"id": "kespeech_jiaoliao", "name": "KeSpeech 胶辽", "scenario": "ASR·方言腔", "count": 1443, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "JiaoLiao"},
    {"id": "kespeech_lanyin", "name": "KeSpeech 兰银", "scenario": "ASR·方言腔", "count": 1652, "size": "共享", "lang": "zh", "metric": "CER", "runnable": True, "builder": "kespeech", "subset": "LanYin"},
    # ── ASR 中英混 / 英文 / 领域 ──
    {"id": "ascend", "name": "ASCEND 中英混读", "scenario": "ASR·中英混读", "count": 1315, "size": "1.1G", "lang": "zh", "metric": "CER", "runnable": True},  # 注:36 条参考含 [UNK](标注听不清),这些样本 CER 天然偏高,横评各模型同伤
    {"id": "librispeech", "name": "LibriSpeech test-clean", "scenario": "ASR·英文", "count": 2620, "size": "0.3G", "lang": "auto", "metric": "WER", "runnable": True},
    {"id": "librispeech_other", "name": "LibriSpeech test-other", "scenario": "ASR·英文", "count": 2939, "size": "0.3G", "lang": "auto", "metric": "WER", "runnable": True, "builder": "librispeech", "subset": "other",
     "required_paths": ["asr/librispeech/other/test/0000.parquet"],
     "src": "LibriSpeech test-other · OpenSLR（CC BY 4.0）", "trait": "较困难英文有声书朗读，声学条件弱于 test-clean", "probe": "英文困难集识别与 clean→other 退化幅度（WER）"},
    {"id": "open_asr_ami", "name": "Open ASR · AMI", "scenario": "ASR·英文会议", "count": 12643, "size": "1.3G", "lang": "en", "metric": "WER", "runnable": True,
     "builder": "open_asr", "subset": "ami", "required_paths": ["asr/open_asr/ami/test-00000-of-00015.parquet"]},
    {"id": "open_asr_earnings22", "name": "Open ASR · Earnings22", "scenario": "ASR·英文金融", "count": 2741, "size": "1.1G", "lang": "en", "metric": "WER", "runnable": True,
     "builder": "open_asr", "subset": "earnings22", "required_paths": ["asr/open_asr/earnings22/test-00000-of-00005.parquet"]},
    {"id": "open_asr_gigaspeech", "name": "Open ASR · GigaSpeech", "scenario": "ASR·英文通用", "count": 19931, "size": "4.0G", "lang": "en", "metric": "WER", "runnable": True,
     "builder": "open_asr", "subset": "gigaspeech", "required_paths": ["asr/open_asr/gigaspeech/test-00000-of-00019.parquet"]},
    {"id": "open_asr_spgispeech", "name": "Open ASR · SPGISpeech", "scenario": "ASR·英文金融", "count": 39341, "size": "11.4G", "lang": "en", "metric": "WER", "runnable": True,
     "builder": "open_asr", "subset": "spgispeech", "required_paths": ["asr/open_asr/spgispeech/test-00000-of-00038.parquet"]},
    {"id": "open_asr_voxpopuli", "name": "Open ASR · VoxPopuli", "scenario": "ASR·英文演讲", "count": 1842, "size": "0.9G", "lang": "en", "metric": "WER", "runnable": True,
     "builder": "open_asr", "subset": "voxpopuli", "required_paths": ["asr/open_asr/voxpopuli/test-00000-of-00004.parquet"]},
    {"id": "open_asr_tedlium", "name": "Open ASR · TEDLIUM", "scenario": "ASR·英文演讲", "count": 1155, "size": "0.3G", "lang": "en", "metric": "WER", "runnable": True,
     "builder": "open_asr", "subset": "tedlium", "required_paths": ["asr/open_asr/tedlium_source/release3/test-00000-of-00001-d0fc02ea7a691c08.parquet"]},
    {"id": "med_it", "name": "MED-IT 医疗(英)", "scenario": "ASR·医疗(英)", "count": 2369, "size": "4.3G", "lang": "en", "metric": "WER", "runnable": True},
    # ── ASR 鲁棒性 ──
    {"id": "vwb", "name": "Voices-in-the-Wild(真实)", "scenario": "ASR·鲁棒性", "count": 750, "size": "1.7G", "lang": "zh", "metric": "CER", "runnable": True},
    {"id": "wildasr", "name": "WildASR (Boson)", "scenario": "ASR·鲁棒性(英)", "count": 7979, "size": "2.0G", "lang": "en", "metric": "WER", "runnable": True},
    # ── ASR 热词 ──
    {"id": "seaco", "name": "SeACo 热词", "scenario": "ASR·热词", "count": 808, "size": "<2M", "lang": "zh", "metric": "KRR", "runnable": True, "hotwords": True},
    {"id": "flores", "name": "FLORES-200", "scenario": "翻译·文本", "count": 1012, "size": "77M", "lang": "zh-en", "metric": "chrF", "runnable": True},
    {"id": "wmt", "name": "WMT22 zh→en", "scenario": "翻译·文本", "count": 1875, "size": "共享", "lang": "zh-en", "metric": "chrF", "runnable": True},
    {"id": "wmt22_en_zh", "name": "WMT22 en→zh", "scenario": "翻译·文本", "count": 2037, "size": "共享", "lang": "en-zh", "metric": "chrF", "runnable": True, "builder": "wmt", "subset": "wmt22-en-zh", "required_paths": ["translation/wmt/wmt22.en-zh.src.txt", "translation/wmt/wmt22.en-zh.ref.txt"]},
    {"id": "wmt23_zh_en", "name": "WMT23 zh→en", "scenario": "翻译·文本", "count": 1976, "size": "共享", "lang": "zh-en", "metric": "chrF", "runnable": True, "builder": "wmt", "subset": "wmt23-zh-en", "required_paths": ["translation/wmt/wmt23.zh-en.src.txt", "translation/wmt/wmt23.zh-en.ref.txt"]},
    {"id": "wmt23_en_zh", "name": "WMT23 en→zh", "scenario": "翻译·文本", "count": 2074, "size": "共享", "lang": "en-zh", "metric": "chrF", "runnable": True, "builder": "wmt", "subset": "wmt23-en-zh", "required_paths": ["translation/wmt/wmt23.en-zh.src.txt", "translation/wmt/wmt23.en-zh.ref.txt"]},
    {"id": "wmt25", "name": "WMT25 en→zh", "scenario": "翻译·文本(文档级)", "count": 87, "size": "15M", "lang": "en-zh", "metric": "chrF", "runnable": True},
    {"id": "hardmt", "name": "HardMTBench 术语", "scenario": "翻译·术语", "count": 10000, "size": "49M", "lang": "zh-en", "metric": "chrF", "runnable": True},
    {"id": "wmt25term", "name": "WMT25-Term 金融", "scenario": "翻译·术语(金融)", "count": 111, "size": "16M", "lang": "zh-en", "metric": "chrF", "runnable": True},
    {"id": "fleurs", "name": "FLEURS zh→en", "scenario": "翻译·语音", "count": 945, "size": "0.8G", "lang": "zh-en", "metric": "chrF", "runnable": True, "simul": True},
    {"id": "fleurs_multi", "name": "FLEURS 多语→en", "scenario": "翻译·语音(多语)", "count": 5253, "size": "共享", "lang": "multi-en", "source_langs": ["zh", "de", "fr", "es", "ru", "ja", "ko"], "metric": "chrF", "runnable": True, "simul": True, "builder": "fleurs", "subset": "multi-en",
     "src": "FLEURS · Google 多语语音翻译（CC-BY-4.0，102 语句级平行）", "trait": "多源语种音频→英文（现 7 语就位:zh/de/fr/es/ru/ja/ko，余随下载扩充）；配 --shuffle 跨语种均匀抽样", "probe": "多语言语音翻译/同传横评（chrF），对打 plat-simult/qwen/xf"},
    {"id": "covost2", "name": "CoVoST2 zh→en", "scenario": "翻译·语音", "count": 4898, "size": "951M", "lang": "zh-en", "metric": "chrF", "runnable": True, "simul": True},
    {"id": "acl6060", "name": "ACL 60/60 en→zh", "scenario": "翻译·语音", "count": 416, "size": "190M", "lang": "en-zh", "metric": "chrF", "runnable": True, "simul": True, "builder": "acl6060", "subset": "eval",
     "src": "ACL 60/60 · IWSLT 学术演讲多语翻译评测集（en→10 语，HF ymoslem/acl-6060）", "trait": "英文演讲音频→中文翻译参考；eval 416 / dev 468（subset 切）", "probe": "en→zh 语音翻译/同传（chrF），IWSLT 官方同传评测集，对打 plat-simult/qwen-simult"},
    {"id": "vcsum", "name": "VCSUM", "scenario": "总结·会议", "count": 136, "size": "6M", "lang": "zh", "metric": "ROUGE-L", "runnable": True},
    {"id": "cnewsum", "name": "CNewSum", "scenario": "总结·新闻", "count": 14355, "size": "422M", "lang": "zh", "metric": "ROUGE-L", "runnable": True},
    {"id": "minutes", "name": "AliMeeting×4MUG 会议纪要", "scenario": "总结·会议纪要(音频)", "count": 24, "size": "共享13G", "lang": "zh", "metric": "ROUGE-L", "runnable": True},
    {"id": "formula", "name": "TTS 公式转写 golden", "scenario": "其他·TTS 公式转写", "count": 8, "size": "<1M", "lang": "zh", "metric": "CER", "runnable": True},
    {"id": "diar", "name": "AliMeeting 说话人分离", "scenario": "其他·说话人分离", "count": 28, "size": "共享18G", "lang": "zh", "metric": "DER", "runnable": True},
]

# 长音频同传集按固定语向登记：同一音频在不同目标语下是不同评测任务，不能混在
# 一个 manifest 里随机抽样。required_paths 让可用性跟数据声明放在一起，新增数据集
# 不必继续扩写 _dataset_available 的中央 if/else。
_ACL6060_LONG_TARGETS = [
    ("zh", "中"), ("fr", "法"), ("de", "德"), ("ja", "日"), ("ru", "俄"),
    ("ar", "阿"), ("fa", "波斯"), ("nl", "荷"), ("pt", "葡"), ("tr", "土耳其"),
]
for _target, _label in _ACL6060_LONG_TARGETS:
    DATASETS.append({
        "id": f"acl6060_long_en_{_target}", "name": f"ACL 60/60 Long en→{_label}",
        "scenario": "翻译·语音(长音频)", "count": 5, "size": "共享2.0G",
        "lang": f"en-{_target}", "metric": "chrF", "runnable": True,
        "simul": True, "longform": True, "builder": "acl6060_long", "subset": f"eval-{_target}",
        "required_paths": [
            f"translation/longform_raw/acl6060_full/extracted/2/acl_6060/eval/text/xml/ACL.6060.eval.en-xx.{_target}.xml",
            "translation/longform_raw/acl6060_full/extracted/2/acl_6060/eval/full_wavs",
        ],
        "src": "ACL 60/60 · IWSLT 2023（CC BY 4.0）",
        "trait": "5 场完整英文 ACL 演讲，每场约 9–12 分钟；人工书面译文参考",
        "probe": f"长会话上下文保持、错误累积与 en→{_label} 实时延迟",
    })

for _target, _label in (("zh", "中"), ("de", "德"), ("it", "意")):
    DATASETS.append({
        "id": f"mcif_long_en_{_target}", "name": f"MCIF Long en→{_label}",
        "scenario": "翻译·语音(长音频)", "count": 21, "size": "共享1.2G",
        "lang": f"en-{_target}", "metric": "chrF", "runnable": True,
        "simul": True, "longform": True, "builder": "mcif_long", "subset": _target,
        "required_paths": [f"translation/longform_raw/mcif/MCIF.long.{_target}.ref.xml.gz",
                           "translation/longform_raw/mcif/MCIF_DATA/LONG_AUDIOS"],
        "src": "MCIF Long · FBK-MT（CC BY 4.0）",
        "trait": "21 场完整英文 ACL 演讲，约 2 小时/语向；文档级书面译文参考",
        "probe": f"5–6 分钟长语音的 en→{_label} 上下文保持与实时延迟",
    })

for _source, _target, _label, _minutes in (("en", "zh", "英→中", 44.4),
                                            ("zh", "en", "中→英", 51.1)):
    _folder = f"{_source}2{_target}"
    DATASETS.append({
        "id": f"realsi_{_source}_{_target}", "name": f"RealSI {_label}",
        "scenario": "翻译·语音(长音频)", "count": 10, "size": "共享576M",
        "lang": f"{_source}-{_target}", "metric": "chrF", "runnable": True,
        "simul": True, "longform": True, "builder": "realsi", "subset": f"{_source}-{_target}",
        "required_paths": [f"translation/longform_raw/realsi/data/{_folder}/json",
                           f"translation/longform_raw/realsi/data/{_folder}/wav"],
        "src": "RealSI · ByteDance Research（标注 CC BY 4.0；原视频版权需另行遵守）",
        "trait": f"10 场真实长音频，共约 {_minutes} 分钟；人工同传译文、时间段和术语标注",
        "probe": f"{_label} 真同传质量、术语保持、分段稳定性与实时延迟",
    })

DATASETS.append({
    "id": "bstc_long_zh_en", "name": "BSTC 2019 Long 中→英",
    "scenario": "翻译·语音(长音频)", "count": 16, "size": "174M",
    "lang": "zh-en", "metric": "chrF", "runnable": True,
    "simul": True, "longform": True, "builder": "bstc_long",
    "required_paths": [
        "translation/longform_raw/bstc_ccmt2019/CCMT_2019_BSTC/data/development_data.zip",
    ],
    "src": "BSTC 2019 · Baidu Speech Translation Corpus（CCMT 2019）",
    "trait": "16 场中文完整演讲，官方句级 offset/duration 与人工英文翻译；单场约 1–8 分钟",
    "probe": "中→英长会话同传质量、LongYAAL CU/CA、错误累积与分段稳定性",
})

# FLORES-200 热门语向（文本翻译，零下载：同 1012 句已译成 200 语，扩语向=换文件）。
# 翻译为次要场景，先铺主流 7 语；只 gemma-text 等文本翻译模型可跑（FLORES 无音频）。
_FLORES_HOT = [("en", "zh", "中"), ("en", "de", "德"), ("en", "fr", "法"), ("en", "es", "西"),
               ("en", "ru", "俄"), ("en", "ja", "日"), ("en", "ko", "韩"), ("en", "ar", "阿")]
for _s, _t, _cn in _FLORES_HOT:
    DATASETS.append({"id": f"flores_{_s}_{_t}", "name": f"FLORES en→{_cn}",
                     "scenario": "翻译·文本", "count": 1012, "size": "共享", "lang": f"{_s}-{_t}",
                     "metric": "chrF", "runnable": True, "builder": "flores", "subset": f"{_s}-{_t}",
                     "src": "FLORES-200 · Meta 专业人工翻译（1012 句 200 语对齐）",
                     "trait": f"英文 1012 句 → {_cn}文，零下载扩语向", "probe": f"en→{_cn}文本翻译（chrF）"})

# FLEURS 多语言 ASR（Qwen3-ASR 官方公开表的 12 语言 test 口径）。
_FLEURS_HOT = [
    ("en", "英语", "WER", 647), ("zh", "中文", "CER", 945), ("yue", "粤语", "CER", 819),
    ("ar", "阿拉伯语", "WER", 428), ("de", "德语", "WER", 862),
    ("es", "西班牙语", "WER", 908), ("fr", "法语", "WER", 676),
    ("it", "意大利语", "WER", 865), ("ja", "日语", "CER", 650),
    ("ko", "韩语", "CER", 382), ("pt", "葡萄牙语", "WER", 919),
    ("ru", "俄语", "WER", 775),
]
for _l, _cn, _m, _n in _FLEURS_HOT:
    DATASETS.append({"id": f"fleurs_asr_{_l}", "name": f"FLEURS {_cn} ASR", "scenario": f"ASR·{_cn}",
                     "count": _n, "size": "0.2-0.6G", "lang": _l, "metric": _m, "runnable": True,
                     "builder": "fleurs_asr", "subset": _l,
                     "src": "FLEURS · Google 多语语音（CC-BY-4.0）", "trait": f"{_cn}朗读语音 test split",
                     "probe": f"{_cn}多语言 ASR（{_m}）"})

# Common Voice 17：与 Qwen3-ASR 公布的 13 语言集合一致。
_COMMONVOICE = [
    ("en", "en", "英语", "WER", 16393), ("zh", "zh-CN", "中文", "CER", 10626),
    ("yue", "yue", "粤语", "CER", 2626), ("zht", "zh-TW", "繁体中文", "CER", 4982),
    ("ar", "ar", "阿拉伯语", "WER", 10480), ("de", "de", "德语", "WER", 16183),
    ("es", "es", "西班牙语", "WER", 15857), ("fr", "fr", "法语", "WER", 16159),
    ("it", "it", "意大利语", "WER", 15155), ("ja", "ja", "日语", "CER", 6261),
    ("ko", "ko", "韩语", "CER", 339), ("pt", "pt", "葡萄牙语", "WER", 9467),
    ("ru", "ru", "俄语", "WER", 10203),
]
for _id, _locale, _cn, _metric, _count in _COMMONVOICE:
    DATASETS.append({
        "id": f"commonvoice_{_id}", "name": f"Common Voice 17 {_cn}",
        "scenario": f"ASR·{_cn}", "count": _count, "size": "共享5.5G",
        "lang": _locale, "metric": _metric, "runnable": True,
        "builder": "commonvoice", "subset": _locale,
        "required_paths": [
            f"asr/common_voice17/transcript/{_locale}/test.tsv",
            f"asr/common_voice17/audio/{_locale}/test/{_locale}_test_0.tar",
        ],
        "src": "Mozilla Common Voice 17（CC0；fsicoli 原始 test 分片镜像）",
        "trait": f"{_cn}众包朗读 test split，真实说话人/设备/口音差异",
        "probe": f"{_cn}多语言与口音鲁棒性（{_metric}）",
    })

# 日语专项：与 Kotoba-Whisper 公布 CER 的三个 test split 完全同版本。
_JA_PUBLIC_BENCHMARKS = [
    ("ja_cv8_benchmark", "Common Voice 8 日语", "cv8", "ja_cv8", 4483, "0.15G",
     "japanese-asr/ja_asr.common_voice_8_0", "众包朗读；公开 Whisper/Reazon/Kotoba 基线"),
    ("ja_jsut_basic5000", "JSUT Basic5000", "jsut", "jsut_basic5000", 5000, "2.25G",
     "japanese-asr/ja_asr.jsut_basic5000", "单说话人干净朗读；日本语音研究常用基准"),
    ("ja_reazonspeech_test", "ReazonSpeech held-out", "reazon", "reazonspeech_test", 5263, "0.60G",
     "japanese-asr/ja_asr.reazonspeech_test", "日本电视/新闻/自然语音；独立留出测试集"),
]
for _id, _name, _subset, _dirname, _count, _size, _source, _trait in _JA_PUBLIC_BENCHMARKS:
    DATASETS.append({
        "id": _id, "name": _name, "scenario": "ASR·日语", "count": _count,
        "size": _size, "lang": "ja", "metric": "CER", "runnable": True,
        "builder": "ja_benchmark", "subset": _subset,
        "required_paths": [f"asr/{_dirname}/parquet"],
        "src": f"{_source}（Kotoba-Whisper 公开评测版本）",
        "trait": _trait, "probe": "日语短音频 ASR；Kotoba 公开规范化 CER 对数",
    })

# 数据集/测试集首次公开时间。能核到月份或日期的保留相应精度；组合集保留范围。
# family key 优先取 builder，使同一母集拆出的语言、领域和城市子集共享发布时间。
DS_PUBLISHED = {
    "aishell": "2017-09", "aishell2": "2018-08", "aishell2_eval": "2018-08", "speechio": "2022",
    "wenetspeech": "2021-10", "wsyue": "2025-09", "wsc": "2025-09",
    "chuan_yu": "2026-07", "wubench": "2026-01", "kespeech": "2021",
    "ascend": "2021-12", "librispeech": "2015-06", "open_asr": "2007–2022", "med_it": "2024-05",
    "commonvoice": "2024-03", "ja_benchmark": "2017–2024",
    "vwb": "2026-05", "wildasr": "2026-03", "seaco": "2023-08",
    "flores": "2022-07", "wmt": "2022–2023", "wmt25": "2025",
    "hardmt": "2026-05", "wmt25term": "2025", "fleurs": "2022-05",
    "fleurs_asr": "2022-05", "covost2": "2020-07", "acl6060": "2023-07",
    "acl6060_long": "2023-07", "mcif_long": "待核实", "realsi": "待核实",
    "bstc_long": "2019-04",
    "vcsum": "2023-05", "cnewsum": "2021-10", "minutes": "2023-03",
    "formula": "2026-06（自建）", "diar": "2022-02",
}

# 数据集档案（卡片悬停显示）：来源 / 数据特点 / 主要考察什么。事实依据 datasets/README.md + docs/dataset-plan.md
DS_DOSSIER = {
    "aishell": ("AISHELL-1 test · 希尔贝壳开源（yuekai HF 镜像）", "录音棚朗读普通话，安静近场", "普通话基础识别，业界对标基线"),
    "aishell2_ios": ("AISHELL-2 iOS 通道 · 官方授权 82G 全集的说话人分层抽样原始 2000 条；隔离 C0932 后有效 1995 条", "手机（iPhone）录制朗读普通话，近场", "普通话识别·移动端信道；exclusions.json 非破坏性隔离上游音文错位，Original CER 仍可追溯"),
    "wsyue": ("WSYue-ASR-eval Short · CC BY-NC 4.0", "0–10 秒粤语短句，含口语表达与中英混说", "粤语短句识别能力"),
    "wsyue_long": ("WSYue-ASR-eval Long · CC BY-NC 4.0", "10–30 秒真实粤语，TextGrid 人工转写", "长句上下文保持与错误累积"),
    "librispeech": ("LibriSpeech test-clean · OpenSLR 英文有声书", "干净英文朗读，无噪声", "纯英文识别基线（WER）"),
    "open_asr_ami": ("Open ASR Leaderboard · AMI", "多人英文会议自发语音", "会议识别 WER，与 MiMo/Qwen/FunASR 公开表对数"),
    "open_asr_earnings22": ("Open ASR Leaderboard · Earnings22", "英文财报电话会", "金融术语与口音 WER"),
    "open_asr_gigaspeech": ("Open ASR Leaderboard · GigaSpeech", "有声书、播客和网络视频", "英文多域识别 WER"),
    "open_asr_spgispeech": ("Open ASR Leaderboard · SPGISpeech", "英文金融会议与公司演讲", "金融专名与数字 WER"),
    "open_asr_voxpopuli": ("Open ASR Leaderboard · VoxPopuli", "欧洲议会英文演讲", "非美式口音与演讲 WER"),
    "open_asr_tedlium": ("Open ASR Leaderboard · TED-LIUM 3", "TED 英文演讲", "演讲识别 WER，与 MiMo 公开表对数"),
    "speechio": ("SpeechIO TIOBE 榜单 ZH00-26 全集（65.5h）", "新闻/演讲/讲课等 27 个真实场景混合", "中文 ASR 综合水平，业界公认横评基准"),
    "law_zh11": ("SpeechIO ZH11 子集 · 罗翔法考讲课（3.4h）", "讲课体，密集法律术语", "法律领域术语识别（非对话场景）"),
    "fin_zh00": ("SpeechIO ZH00 子集 · 金融会议演讲（1h）", "演讲体，金融术语", "金融领域术语识别"),
    "gov_zh01": ("SpeechIO ZH01 子集 · 新闻联播（9h）", "播报体，标准发音，时政词汇", "时政领域术语识别"),
    "med_it": ("MED-IT · 英文真实医患问诊", "真实诊室对话，自带医学术语热词", "医疗领域英文识别（WER）"),
    "ascend": ("ASCEND · 港科大 CAiRE 开源", "句内中英混杂的自发对话", "中英 code-switch 切换能力"),
    "seaco": ("SeACo 开放热词集 · 建在 AISHELL-1 test 上", "808 句 × 400 热词", "热词增益：开/关热词对比 KRR"),
    "vwb": ("Voices-in-the-Wild-Bench（MIT）", "真实采集噪声中文 750 条，8 类扰动（合成样本参考不可信，已弃用）", "中文真实环境鲁棒性"),
    "wildasr": ("WildASR · Boson AI（李沐团队）", "30h 全真人英文，削波/远场/混响/口音/电话编码等 7 类扰动", "英文真实环境鲁棒性（WER）"),
    "wsc": ("WSC-Eval · 西工大 ASLP（Apache-2.0），Qwen3-ASR 报告点名基准", "川渝方言腔普通话 9.7h，Easy 朗读 + Hard 直播短视频带噪", "发音偏差·方言腔（--subset hard 单测难例）"),
    "kespeech": ("KeSpeech test · HF 镜像（限科研）", "19,723 条带 8 子方言标签（中原/西南/冀鲁/胶辽/江淮/兰银/东北/北京）", "发音偏差·分方言 CER，可与主流模型报告对数"),
    "wubench": ("WenetSpeech-Wu-Bench · 西工大 ASLP（Apache-2.0，2026 新出）", "吴语/上海话 9.75h，人工精标", "吴语/上海腔识别（此前空白维度）"),
    "wenetspeech_net": ("WenetSpeech TEST_NET · 出门问问+西工大（CC BY 4.0，gated）", "网络多场景真实中文（视频/播客等），23h", "中文 ASR 最通用对数基准：与 Qwen3-ASR/FireRedASR/Seed-ASR 同口径(⚠️上游有少量标注错误 issue#63)"),
    "wenetspeech_meeting": ("WenetSpeech TEST_MEETING · 出门问问+西工大（CC BY 4.0，gated）", "真实会议录音中文，15h", "中文会议 ASR：业界通用对数基准(与 diar 的 AliMeeting 不同源)"),
    "flores": ("FLORES-200 · Meta 专业人工翻译", "多语平行语料，译文质量高", "zh↔en 文本翻译精度基线（chrF）"),
    "wmt": ("WMT22/23 通用赛道", "新闻/社交/电商等多领域", "文本翻译多域稳健性"),
    "wmt25": ("WMT25 General 官方测试集", "文档级 en→zh 87 篇，news/speech/social/literary 四域（speech 域贴演讲转译）", "最新文档级翻译；zh→en 官方最新仍是 WMT23"),
    "hardmt": ("HardMTBench（2026）· 12 难域 10k 句对", "金融/法律/医疗等领域，逐条带术语对+难度标注（⚠️repo 无 LICENSE）", "术语/领域翻译，keywords 自带术语命中口径"),
    "wmt25term": ("WMT25 Terminology Track2 · HKMA 年报（⚠️CC BY-NC 禁商用）", "金融繁中↔英 文档级，自带术语→别名词典", "金融术语翻译命中率（alias 命中即算）"),
    "fleurs": ("FLEURS · Google 多语语音翻译集", "朗读体语音 + 人工译文配对", "语音翻译：音频直接进翻译接口"),
    "covost2": ("CoVoST2 · Meta（CC0 可商用，fixie-ai 镜像）", "CommonVoice 众包朗读 zh→en test 4,898 条", "语音翻译标准集，体量大于 FLEURS"),
    "vcsum": ("VCSUM · 中文真实会议总结集", "长会议记录 + 人工摘要", "会议内容总结（ROUGE-L）"),
    "cnewsum": ("CNewSum · 字节开源中文新闻摘要", "新闻长文 + 人工摘要", "长文压缩与要点提取"),
    "minutes": ("AliMeeting 真实会议音频 × 4MUG 人工纪要（配对 24 场）", "27 分钟级长音频直进纪要接口", "端到端会议纪要（ROUGE 仅弱锚）"),
    "formula": ("自建 golden 集 · TTS 念数学公式", "8 条公式口语读法，多解弱锚", "公式转写专项"),
    "diar": ("AliMeeting Test+Eval · 28 场真实会议", "远场 8 通道，多人重叠说话", "说话人分离（DER，pyannote 口径）"),
}
for _d in DATASETS:
    _family = _d.get("builder", _d["id"])
    _d["published"] = DS_PUBLISHED.get(_family, DS_PUBLISHED.get(_d["id"], "待核实"))
    if _d["id"] in DS_DOSSIER:
        _d["src"], _d["trait"], _d["probe"] = DS_DOSSIER[_d["id"]]

# 模型条目有四个正交属性，分组按「角色」而非端点，否则榜单必然被误读：
#   group: iface=被测主体(远程平台) / upstream=上游 / baseline=外部基线 / custom=自定义
#   local: True = 本机进程，无网络往返 → 延迟/RTF 不可与远程端点比（前端打角标）
#   lineage: 指向同一条平台管线的接口 id（恒等回归关系，前端显示 ≡ 血缘）
#   unverified: True = 模型来源未核实
# id 沿用历史命名兼容已有结果文件，只改展示名。
_PLAT_BASE = os.environ.get("ASR_PLATFORM_URL", "").rstrip("/")
_PLAT_WS_BASE = os.environ.get("ASR_PLATFORM_WS_URL", _PLAT_BASE).rstrip("/")
_PLAT_DISPLAY = _PLAT_BASE.removeprefix("http://").removeprefix("https://")
_PLAT_WS_DISPLAY = _PLAT_WS_BASE.removeprefix("http://").removeprefix("https://")
_PLAT_KEY_ENV = "ASR_PLATFORM_TOKEN"
_RETIRED_BUILTIN_URLS = frozenset()


def _ep_display(env_key: str, path: str = "") -> str:
    """由环境变量派生看板展示地址（去协议头）；未配置返回空串。"""
    base = os.environ.get(env_key, "").strip().rstrip("/")
    base = base.removeprefix("http://").removeprefix("https://")
    return f"{base}{path}" if base else ""


def _is_platform_url(endpoint: str) -> bool:
    """endpoint 是否指向 ASR_PLATFORM_URL / ASR_PLATFORM_WS_URL 配置的内置平台。"""
    ep = (endpoint or "").removeprefix("http://").removeprefix("https://").removeprefix("ws://").removeprefix("wss://")
    return bool(ep) and any(ep.startswith(d) for d in (_PLAT_DISPLAY, _PLAT_WS_DISPLAY) if d)
MODELS = [
    # ── 被测主体 · 四档模型（id=档位名）──
    {"id": "light", "name": "Lite", "endpoint": "light", "group": "iface", "unverified": True, "url": _ep_display("LIGHT_URL", "/asr_lite")},
    {"id": "light-mlt", "name": "Lite MLT · 多语种", "endpoint": "light", "group": "iface", "url": _ep_display("LIGHT_URL", "/asr_mlt_nano")},
    {"id": "std", "name": "Standard", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/api/asr/std", "key_env": _PLAT_KEY_ENV},
    {"id": "adv", "name": "Advanced", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/api/asr/adv", "key_env": _PLAT_KEY_ENV},
    {"id": "adv-domain", "name": "Advanced Domain · 领域专业识别", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/api/asr/adv-domain", "key_env": _PLAT_KEY_ENV},
    {"id": "sse", "name": "SSE", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/api/asr/sse", "key_env": _PLAT_KEY_ENV, "dual_result": True},
    # compat(旧版)已从 UI 下架；adapter 保留可 CLI 跑。
    # ── 被测主体 · 挂在档位上的功能/场景接口（非独立档位）──
    {"id": "plat-realtime", "name": "流式识别(WS /v1/realtime)", "model": "realtime-transcribe", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_WS_DISPLAY}/v1/realtime?model=realtime-transcribe", "key_env": _PLAT_KEY_ENV},
    {"id": "plat-simult", "name": "同声传译(WS流式+TTS)", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_WS_DISPLAY}/ws/audio/simult-interpreting", "key_env": _PLAT_KEY_ENV},
    {"id": "plat-simult-ws", "name": "同声传译(WS真流式)", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/ws/audio/voice-input", "key_env": _PLAT_KEY_ENV},
    {"id": "plat-minutes", "name": "会议纪要(离线)", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/api/asr/meeting-minutes", "key_env": _PLAT_KEY_ENV},
    {"id": "plat-formula", "name": "TTS 公式转写", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/api/text/tts_formula_helper", "key_env": _PLAT_KEY_ENV},
    {"id": "plat-diar", "name": "说话人分离", "endpoint": "platform", "group": "iface", "url": f"{_PLAT_DISPLAY}/api/asr/adv+diarize", "key_env": _PLAT_KEY_ENV},
        # ext-adv 已从 UI 下架；adapter 保留可 CLI 跑。
    # ── 外部基线（竞品/开源对照）──
    # gemma-audio 已从 UI 下架；gemma-text/cascade-gemma(翻译/总结)保留。
    {"id": "sensevoice", "name": "SenseVoice (阿里)", "endpoint": "sensevoice", "group": "baseline", "url": _ep_display("SENSEVOICE_URL", "/api/asr_transcribe")},
    {"id": "xf-spark-slm-iat", "name": "讯飞方言大模型", "endpoint": "xfyun", "group": "baseline", "url": "wss://iat.cn-huabei-1.xf-yun.com/v1"},
    {"id": "qwen3-asr-ws", "name": "Qwen3-ASR 真流式(WS)", "model": "qwen3-asr", "endpoint": "qwen3-asr", "group": "baseline", "url": os.environ.get("QWEN3_ASR_WS_URL", "")},
    {"id": "gemma-text", "name": "Gemma-4-12B 文本", "endpoint": "gemma", "text": True, "group": "baseline", "url": _ep_display("GEMMA_URL", "/v1/chat/completions")},
    {"id": "cascade-gemma", "name": "级联 ASR→Gemma", "endpoint": "ext→gemma", "text": True, "group": "baseline"},
    # 第三方同传(WS 流式语音翻译；密钥走环境变量)
    {"id": "qwen-simult", "name": "Qwen3.5-LiveTranslate 同传", "model": "qwen3.5-livetranslate-flash-realtime", "endpoint": "dashscope", "group": "baseline", "url": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"},
    {"id": "doubao-simult", "name": "豆包同传 2.0 (Seed LiveInterpret)", "model": "ast-2.0", "endpoint": "bytedance", "group": "baseline", "url": "wss://openspeech.bytedance.com/api/v4/ast/v2/translate", "enabled": False},
    {"id": "qwen-simult-offline", "name": "Qwen3.5-LiveTranslate 离线(保留率分母)", "model": "qwen3.5-livetranslate-flash-realtime", "endpoint": "dashscope", "group": "baseline", "url": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"},
    {"id": "xf-simult", "name": "讯飞同声传译", "endpoint": "xfyun", "group": "baseline", "url": "wss://ws-api.xf-yun.com/v1/private/simult_interpretation"},
]

# 翻译场景模型的输入类型：audio=语音翻译(需音频,跑 fleurs)，text=文本翻译(需 source_text,跑 flores/wmt)
# 与数据集子类型(翻译·语音/翻译·文本)匹配 → 前端禁不兼容组合，杜绝"语音接口收到纯文本集"必然失败
_TRANS_INPUT = {"adv": "audio", "adv-domain": "audio", "sse": "audio",
                "plat-simult": "audio", "plat-simult-ws": "audio",
                "qwen-simult": "audio", "qwen-simult-offline": "audio", "xf-simult": "audio",
                "xf-spark-slm-iat": "audio",
                "doubao-simult": "audio", "cascade-gemma": "audio", "gemma-text": "text"}
for _m in MODELS:
    if _m["id"] in _TRANS_INPUT:
        _m["trans_input"] = _TRANS_INPUT[_m["id"]]

# 各场景可用模型（adv 是被测主体；翻译用其语音翻译，需带 audio_path 的清单如 FLEURS）
# 从 UI 下架但历史结果仍需归属的接口 id（文件名回退解析 + 前端 modelOf 共用）
RETIRED_MODELS = ["compat", "ext-adv", "gemma-audio"]

SCENARIO_MODELS = {
    "ASR": ["light", "std", "adv", "adv-domain", "sse", "sensevoice", "xf-spark-slm-iat", "qwen3-asr-ws"],
    # ↑ 下架接口的 id 收进 RETIRED_MODELS（见下），历史结果文件名解析仍可归属
    "翻译": ["adv", "adv-domain", "sse", "qwen-simult-offline", "gemma-text", "cascade-gemma"],   # 文本/离线语音翻译(只看 chrF)
    # 同传:语音→译文的"边说边译",独立场景,看延迟(AL/LAAL/ttfb)+质量(chrF)+译后语音保真(tts_err)。
    # 数据集走 simul 标记(语音翻译集,与「翻译·语音」共用集但口径不同)。
    "同传": ["plat-simult", "qwen-simult", "doubao-simult", "xf-simult", "plat-simult-ws"],
    "总结": ["plat-minutes", "gemma-text"],
    "其他": ["plat-formula", "plat-diar"],  # 非主指标专项: 公式/说话人分离(已接)/情感/声音标注(陆续)
}

# 接口覆盖看板数据（公开版默认为空；条目格式 {cat,name,ep,method,status,note}）
PLAT_COVERAGE = []


# Lite 抽样量按场景定：ASR 快(0.3s/条)抽多些；文本走 gemma(1-2s/条)抽少些；同传 1x 实时慢→抽少
LITE_N = {"ASR": 300, "翻译": 80, "同传": 80, "总结": 50}
JOBS = {}  # job_id -> dict
PROCS = {}  # job_id -> Popen(运行中子进程，用于停止)

# 任务持久化：JOBS 纯内存→重启丢、且看不到 CLI(infer/runner/longrun)起的任务。
# 写共享 jobs.json(gitignore),CLI 经 eval/jobstore.py 登记,/api/jobs 合并内存+磁盘。
JOBS_FILE = os.path.join(ROOT, "jobs.json")
_JOBS_LK = threading.Lock()


def _read_jobs_file():
    """共享锁读 jobs.json(与 jobstore/_flush_jobs 的 fcntl 排他写协调,避免读到半截)。"""
    import fcntl
    if not os.path.exists(JOBS_FILE):
        return {}
    try:
        f = open(JOBS_FILE, encoding="utf-8")
    except Exception:
        return {}
    try:
        fcntl.flock(f, fcntl.LOCK_SH)
        raw = f.read().strip()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN); f.close()
        except Exception:
            pass


def _flush_jobs():
    """内存 JOBS 合并进 jobs.json(保留外部 CLI 条目)。用 fcntl 排他锁,与 eval/jobstore.py 跨进程协调。"""
    import fcntl
    with _JOBS_LK:   # 进程内多线程(多 job 收尾)互斥
        try:
            f = open(JOBS_FILE, "a+", encoding="utf-8")
        except Exception:
            return
        try:
            fcntl.flock(f, fcntl.LOCK_EX)   # 跨进程:与 CLI 的 jobstore 写互斥
            f.seek(0); raw = f.read().strip()
            disk = json.loads(raw) if raw else {}
            for jid, j in JOBS.items():
                disk[jid] = {k: v for k, v in j.items() if k != "cancel"}
            f.seek(0); f.truncate(); f.write(json.dumps(disk, ensure_ascii=False))
        except Exception:
            pass
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN); f.close()
            except Exception:
                pass


def _purge_job_records(model):
    """删除 jobs.json 中该模型的历史任务，避免前端清单里残留。"""
    import fcntl
    with _JOBS_LK:
        try:
            f = open(JOBS_FILE, "a+", encoding="utf-8")
        except Exception:
            return
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read().strip()
            disk = json.loads(raw) if raw else {}
            for jid, rec in list(disk.items()):
                if isinstance(rec, dict) and rec.get("model") == model:
                    disk.pop(jid, None)
            f.seek(0); f.truncate(); f.write(json.dumps(disk, ensure_ascii=False))
        except Exception:
            pass
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN); f.close()
            except Exception:
                pass


def _mutate_jobs_file(mutator):
    """在 jobs.json 上执行一次原子读改写。mutator 接收 dict，返回响应用的值。"""
    import fcntl
    with _JOBS_LK:
        try:
            f = open(JOBS_FILE, "a+", encoding="utf-8")
        except Exception:
            return None
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read().strip()
            disk = json.loads(raw) if raw else {}
            ret = mutator(disk)
            f.seek(0); f.truncate(); f.write(json.dumps(disk, ensure_ascii=False))
            return ret
        except Exception:
            return None
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN); f.close()
            except Exception:
                pass


def _purge_job_records_for_result(result_file):
    """删除指向某个 results/*.json 的历史任务记录。"""
    base = os.path.basename(result_file or "")
    if not base:
        return 0

    def mut(disk):
        deleted = 0
        for jid, rec in list(disk.items()):
            if not isinstance(rec, dict):
                continue
            rf = os.path.basename(rec.get("result_file") or "")
            if not rf and isinstance(rec.get("summary"), dict):
                inf = rec["summary"].get("infer_file") or ""
                if inf:
                    rf = os.path.basename(inf.replace("infer/", "results/").replace(".jsonl", ".json"))
            if rf == base:
                disk.pop(jid, None)
                JOBS.pop(jid, None)
                deleted += 1
        return deleted

    return _mutate_jobs_file(mut) or 0


def lite_n_for(scenario):
    return LITE_N.get(scenario.split("·")[0], 100)


def _run_key(model, dataset, tier, limit, seed, hotwords, hotwords_text, domain, embedding, comet,
             speech_eval, language, config_hash, ds):
    n = limit or lite_n_for(ds["scenario"])
    return (model, dataset, tier,
            n if tier == "lite" else None,
            seed if tier == "lite" else None,
            bool(hotwords), hotwords_text or "", bool(domain), bool(embedding), bool(comet),
            bool(speech_eval),
            language or "", config_hash or "")


def _now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# 延迟模式按「端点」分锁，防止同一后端互相污染；精度模式按「模型」分锁，
# 允许 std/adv/sse 并行，但同一模型仍逐数据集串行，避免一次派发把端点打满。
_EP_LOCKS = {}
_EP_GLOBAL = threading.Lock()


def _ep_lock(model_id, run_mode="latency"):
    ep = next((m["endpoint"] for m in MODELS if m["id"] == model_id), model_id)
    key = f"model:{model_id}" if run_mode == "accuracy" else f"endpoint:{ep}"
    with _EP_GLOBAL:
        return _EP_LOCKS.setdefault(key, threading.Lock())

# 每个 manifest 路径一把锁：避免"全选"并发任务竞写同一文件；同参数复用
_MANI_LOCKS = {}
_MANI_GLOBAL = threading.Lock()


def _mani_lock(path):
    with _MANI_GLOBAL:
        return _MANI_LOCKS.setdefault(path, threading.Lock())
app = FastAPI()

# ── HTTP Basic 鉴权：一处中间件管住页面/API/音频，前端无需改动 ──
#   DASH_PASS 必须显式配置；设空串只用于隔离的本机开发环境。
AUTH_USER = os.environ.get("DASH_USER", "admin")
AUTH_CONFIGURED = "DASH_PASS" in os.environ
AUTH_PASS = os.environ.get("DASH_PASS", "")

# 防撞库：只把**真实凭据尝试**（带 Basic 头且解出用户名）记为撞库失败——无凭据的
# 探活/公网扫描(401)不计数，否则共享出口 IP 会被无关流量锁死。锁定键=IP+用户名：
# 同一出口 IP 的不同用户互不牵连。正常使用（口令正确）永远不触发；pod 重启清零
#（内存态，锁只是拖慢爆破，不是安全边界）。
AUTH_MAX_FAILURES = int(os.environ.get("AUTH_MAX_FAILURES", "5"))
AUTH_LOCKOUT_S = float(os.environ.get("AUTH_LOCKOUT_S", "900"))   # 15 分钟
_AUTH_FAILURES: dict[tuple[str, str], list[float]] = {}   # (ip, user) -> 最近失败时间戳
_AUTH_LOCKED_UNTIL: dict[tuple[str, str], float] = {}     # (ip, user) -> 锁定到期时间戳
_AUTH_GUARD_LK = threading.Lock()


def _auth_guard(ip: str, user: str) -> float:
    """返回该 (ip,user) 剩余锁定秒数（0=未锁）。同时记账一次失败。"""
    now = time.time()
    key = (ip, user)
    with _AUTH_GUARD_LK:
        locked_until = _AUTH_LOCKED_UNTIL.get(key, 0)
        if locked_until > now:
            return locked_until - now
        stamps = [t for t in _AUTH_FAILURES.get(key, []) if now - t < AUTH_LOCKOUT_S]
        stamps.append(now)
        _AUTH_FAILURES[key] = stamps
        if len(stamps) >= AUTH_MAX_FAILURES:
            _AUTH_LOCKED_UNTIL[key] = now + AUTH_LOCKOUT_S
            _AUTH_FAILURES.pop(key, None)
            return AUTH_LOCKOUT_S
        return 0.0


def _auth_clear(ip: str, user: str) -> None:
    with _AUTH_GUARD_LK:
        _AUTH_FAILURES.pop((ip, user), None)
        _AUTH_LOCKED_UNTIL.pop((ip, user), None)


@app.middleware("http")
async def basic_auth(request, call_next):
    if not AUTH_CONFIGURED:
        return PlainTextResponse("DASH_PASS 未配置；请通过环境变量或平台密钥显式设置", status_code=503)
    if AUTH_PASS:
        cli = request.client.host if request.client else "?"
        header = request.headers.get("authorization", "")
        # 先解出凭据里的用户名（解不出=探活/扫描，不参与撞库记账）
        attempted_user = ""
        if header.startswith("Basic "):
            try:
                attempted_user, _, _ = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            except Exception:
                attempted_user = ""
        # 锁定中的 (ip,user) 直接 429，不做账户密码校验（锁定优先，拖慢持续爆破）
        remaining = _AUTH_LOCKED_UNTIL.get((cli, attempted_user), 0) - time.time()
        if remaining > 0:
            return PlainTextResponse(
                f"too many failed logins; retry in {int(remaining)}s", status_code=429,
                headers={"Retry-After": str(int(remaining) + 1)},
            )
        ok = False
        if header.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                ok = secrets.compare_digest(user, AUTH_USER) and secrets.compare_digest(pw, AUTH_PASS)
            except Exception:
                ok = False
        if not ok:
            # 只有真实凭据尝试（解出用户名）才记撞库；无凭据/坏头的 401 不计数
            if attempted_user:
                locked_s = _auth_guard(cli, attempted_user)
            else:
                locked_s = 0.0
            LOG.warning("401 鉴权失败 %s %s", cli, request.url.path)   # 审计；不记凭据
            if locked_s > 0:
                return PlainTextResponse(
                    f"too many failed logins; retry in {int(locked_s)}s", status_code=429,
                    headers={"Retry-After": str(int(locked_s) + 1)},
                )
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="asr-dashboard"'})
        if attempted_user:
            _auth_clear(cli, attempted_user)   # 登录成功清该用户失败记账
    resp = await call_next(request)
    if request.url.path in ("/", "/review", "/lid") or request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


def _template_scenarios():
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    try:
        from adapters import TEMPLATE_SCENARIOS
        return TEMPLATE_SCENARIOS
    except Exception:
        return {}


def _providers():
    """厂商预设（OpenAI / 阿里 DashScope …）→ 前端「厂商」下拉，选一项即回填表单。"""
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    try:
        from adapters import PROVIDERS
        return PROVIDERS
    except Exception:
        return {}


def read_results():
    out = []
    res_dir = os.path.join(ROOT, "results")
    files = {os.path.basename(f): f for f in sorted(glob.glob(os.path.join(res_dir, "*.json")))}
    # rerun 还没落稳/中途挂掉时，canonical 结果会被暂存成 *.json.__rerun_*。
    # 这里把“canonical 不存在”的暂存件当作该结果的临时替身，避免旧记录在前端变成空壳。
    for f in sorted(glob.glob(os.path.join(res_dir, "*.json.__rerun_*"))):
        base = re.sub(r"\.\__rerun_.+$", "", os.path.basename(f))
        if base not in files:
            files[base] = f
    for base, f in sorted(files.items()):
        try:
            s = json.load(open(f, encoding="utf-8")).get("summary", {})
            s["_file"] = base
            s["_mtime"] = round(os.path.getmtime(f))  # 旧结果无 meta 时间，前端回退用
            if not s.get("model") or not s.get("dataset") or not s.get("tier"):
                base0 = os.path.splitext(base)[0]
                base0 = re.sub(r"__lang-[^.]+$", "", base0)
                for ds in sorted((d["id"] for d in DATASETS), key=len, reverse=True):
                    prefix = ds + "_"
                    if not base0.startswith(prefix):
                        continue
                    rest = base0[len(prefix):]
                    if "_lite_" in rest:
                        model, tail = rest.split("_lite_", 1)
                        s.setdefault("model", model)
                        s.setdefault("dataset", ds)
                        s.setdefault("tier", "lite")
                        m = re.search(r"(?:^|_)n(\d+)_s\d+", "_lite_" + tail)
                        if m:
                            s.setdefault("n", int(m.group(1)))
                        break
                    if "_full" in rest:
                        model = rest.split("_full", 1)[0]
                        s.setdefault("model", model)
                        s.setdefault("dataset", ds)
                        s.setdefault("tier", "full")
                        break
            meta = s.get("meta") or {}
            normal_route_polluted = (
                (s.get("model") == "std" or meta.get("model") == "std")
                and (bool(meta.get("hotwords") or meta.get("domain"))
                     or bool(re.search(r"(?:^|_)(?:hw|dom)(?:_|\.|$)", os.path.splitext(base)[0])))
            )
            if normal_route_polluted:
                s["invalid_identity"] = True
                s["invalid_reason"] = "历史 Standard 热词/领域任务实际路由到 adv-domain"
                s["low_coverage"] = True  # 沿用现有榜单排除通道；文件本身保留
            out.append(s)
        except Exception:
            pass
    return out


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/review")
def review_page():
    """独立样本复核页。它只读现有 manifest/results，草稿写入 reviews/，
    不会改动 infer、score 或默认榜单。
    """
    return FileResponse(os.path.join(HERE, "review.html"))


@app.get("/lid")
def lid_page():
    """独立 LID 结果页；只读 infer/*.jsonl，不混入 CER/WER 排名。"""
    return FileResponse(os.path.join(HERE, "lid.html"))


@app.get("/longform")
def longform_page():
    """长音频同传专用时间轴；不复用短样本的截断详情视图。"""
    return FileResponse(os.path.join(HERE, "longform.html"))


def _dataset_available(ds):
    """运行时检查数据集是否已下载到卷上。按关键文件级别判断，不只看目录。"""
    did = ds["id"]
    root = os.path.join(ROOT, "datasets")

    # 关键文件检查规则：每项至少列出 builder 必需的核心路径
    checks = {
        "aishell": ["asr/aishell1_test"],
        "aishell2_ios": ["asr/aishell2_ios/trans_subset.tsv"],
        "speechio": ["asr/speechio/yuekai_mirror"],
        "wenetspeech_net": [
            "asr/wenetspeech/data/cuts_TEST_NET.00000000.jsonl.gz",
            "asr/wenetspeech/data/cuts_TEST_NET.00000000.tar.gz",
            "asr/wenetspeech/data/cuts_TEST_NET.00000001.jsonl.gz",
            "asr/wenetspeech/data/cuts_TEST_NET.00000001.tar.gz",
            "asr/wenetspeech/data/cuts_TEST_NET.00000002.jsonl.gz",
            "asr/wenetspeech/data/cuts_TEST_NET.00000002.tar.gz",
        ],
        "wenetspeech_meeting": [
            "asr/wenetspeech/data/cuts_TEST_MEETING.00000000.jsonl.gz",
            "asr/wenetspeech/data/cuts_TEST_MEETING.00000000.tar.gz",
        ],
        "wsyue": ["asr/wsyue/Short/content.txt", "asr/wsyue/Short/wav_"],
        "wsyue_long": ["asr/wsyue/Long/TextGrid", "asr/wsyue/Long/wav"],
        "wsc": ["asr/wsc_eval/WSC-Eval-ASR"],
        "wubench": ["asr/wu_bench/understanding/asr.parquet"],
        "kespeech": ["asr/kespeech_test/data"],
        "ascend": ["asr/ascend/wav"],  # 收紧：检查 wav 目录而不只是根目录
        "librispeech": ["asr/librispeech/clean/test/0000.parquet"],
        "med_it": ["asr/med_it/test.txt"],
        "vwb": ["asr/voices_wild_bench/wav"],  # 收紧：检查 wav 目录
        "wildasr": ["asr/wildasr/wav"],  # 收紧：检查 wav 目录
        "seaco": ["asr/seaco_hotword/repo/data/test"],
        "flores": ["translation/flores_plus/flores200_dataset/devtest"],
        "wmt": [
            "translation/wmt/wmt22.zh-en.src.txt",
            "translation/wmt/wmt22.zh-en.ref.txt",
            "translation/wmt/wmt22.en-zh.src.txt",
            "translation/wmt/wmt22.en-zh.ref.txt",
            "translation/wmt/wmt23.zh-en.src.txt",
            "translation/wmt/wmt23.zh-en.ref.txt",
            "translation/wmt/wmt23.en-zh.src.txt",
            "translation/wmt/wmt23.en-zh.ref.txt",
        ],
        "hardmt": ["translation/hardmtbench/repo/HardMTBench.jsonl"],
        "wmt25": ["translation/wmt25/wmt25-genmt.jsonl"],
        "wmt25term": [
            "translation/wmt25_term/track2/full_data_2015.jsonl",
            "translation/wmt25_term/track2/full_data_2016.jsonl",
            "translation/wmt25_term/track2/full_data_2017.jsonl",
            "translation/wmt25_term/track2/full_data_2018.jsonl",
            "translation/wmt25_term/track2/full_data_2019.jsonl",
            "translation/wmt25_term/track2/full_data_2020.jsonl",
            "translation/wmt25_term/track2/full_data_2021.jsonl",
            "translation/wmt25_term/track2/full_data_2022.jsonl",
            "translation/wmt25_term/track2/full_data_2023.jsonl",
            "translation/wmt25_term/track2/full_data_2024.jsonl",
            "translation/wmt25_term/LICENSE.txt",
        ],
        "fleurs": [
            "translation/fleurs/data/cmn_hans_cn/test.tsv",
            "translation/fleurs/data/en_us/test.tsv",
        ],
        "fleurs_multi": [
            "translation/fleurs/data/cmn_hans_cn/test.tsv",
            "translation/fleurs/data/de_de/test.tsv",
            "translation/fleurs/data/fr_fr/test.tsv",
            "translation/fleurs/data/es_419/test.tsv",
            "translation/fleurs/data/ru_ru/test.tsv",
            "translation/fleurs/data/ja_jp/test.tsv",
            "translation/fleurs/data/ko_kr/test.tsv",
        ],
        "covost2": ["translation/covost2/zh-CN_en"],
        "acl6060": [
            "translation/acl6060/acl6060.tsv",
            "translation/acl6060/audio",
        ],
        "vcsum": ["summary/vcsum/data/test-00000-of-00001.parquet"],
        "cnewsum": ["summary/cnewsum/data"],
        "minutes": [
            "asr/alimeeting",
            "summary/alimeeting4mug/audio_paired_meetings.jsonl",
        ],
        "diar": ["asr/alimeeting"],
        "formula": ["golden/tts_formula.jsonl"],  # 与 build_formula 的唯一来源一致
    }

    # fleurs_asr_* 各语种检查对应目录
    if did.startswith("fleurs_asr_"):
        lang = did.split("_")[-1]
        lang_map = {"en": "en_us", "zh": "cmn_hans_cn", "yue": "yue_hant_hk", "ar": "ar_eg",
                    "de": "de_de", "fr": "fr_fr", "es": "es_419", "ru": "ru_ru",
                    "it": "it_it", "ja": "ja_jp", "ko": "ko_kr", "pt": "pt_br"}
        if lang in lang_map:
            checks[did] = [f"translation/fleurs/data/{lang_map[lang]}/test.tsv",
                           f"translation/fleurs/data/{lang_map[lang]}/audio/test.tar.gz"]

    if did.startswith("wsc_"):
        sub = did.split("_", 1)[1].capitalize()
        checks[did] = [f"asr/wsc_eval/WSC-Eval-ASR/{sub}/text"]

    if did == "chuan_yu_12city":
        checks[did] = [
            f"asr/chuan_yu_12city/{city}/UTTERANCEINFO.txt"
            for _slug, city, _count in _CHUAN_YU_CITIES
        ]
    elif did.startswith("chuan_yu_"):
        city = ds.get("subset", "")
        checks[did] = [
            f"asr/chuan_yu_12city/{city}/UTTERANCEINFO.txt",
            f"asr/chuan_yu_12city/{city}/WAV",
        ]

    if did.startswith("kespeech_"):
        checks[did] = ["asr/kespeech_test/data"]

    # flores_* 语向变体检查 flores 基础路径
    if did.startswith("flores_") and did != "flores":
        checks[did] = checks.get("flores", [])

    need = ds.get("required_paths") or checks.get(did)
    if not need:
        return True, ""  # 没有明确规则的，暂时认为可用（兼容未来新数据集）

    miss = [p for p in need if not os.path.exists(os.path.join(root, p))]
    return (not miss), ("缺少关键文件: " + ", ".join(os.path.basename(m) for m in miss[:3]) if miss else "")


@app.get("/api/overview")
def overview():
    total = sum(d["count"] for d in DATASETS)
    datasets = []
    for d in DATASETS:
        x = dict(d)
        ok, reason = _dataset_available(d)
        x.pop("required_paths", None)
        x["available"] = ok
        x["missing_reason"] = reason
        x["requires"] = _ds_requires(d)
        datasets.append(x)
    return {"datasets": datasets, "models": [_model_payload(m) for m in MODELS], "results": read_results(),
            "retired_models": RETIRED_MODELS,
            "total_samples": total, "lite_n": LITE_N,
            "scenarios": ["ASR", "翻译", "同传", "总结", "其他"], "scenario_models": SCENARIO_MODELS,
            "coverage": PLAT_COVERAGE, "template_scenarios": _template_scenarios(),
            "providers": _providers()}


class RunReq(BaseModel):
    model: str
    dataset: str
    tier: str  # lite | full
    limit: int | None = None  # 轻量条数(None=按场景默认)
    seed: int = 42            # 抽样种子(同种子→同样本，横评可复现)
    hotwords: bool = False    # 场景 b：manifest keywords 作热词传给模型
    hotwords_text: str = ""   # 每条样本都追加的自定义热词；与 manifest keywords 合并
    domain: bool = False      # 领域提示词 A/B：仅独立 adv-domain 接口接受
    embedding: bool = False   # ASR 可选增强：embedding 语义相似度(score-only，不改 infer/文件名)
    comet: bool = False       # 翻译/同传可选增强：优先 GPU XCOMET-XL，本地 COMET 降级
    speech_eval: bool = False # 同传语音口径：开启 TTS + 回转/UTMOS/声线；默认文本口径关闭 TTS
    rerun: bool = False       # True=清除旧 hyp 重新调模型(接口已改时)；False=断点续跑(复用 hyp)
    run_mode: str = "accuracy"  # accuracy=按模型并行、延迟无效；latency=按端点串行
    workers: int = Field(default=4, ge=1, le=16)  # accuracy 模式的样本级并发
    language: str = ""        # 独立实验参数，按任务透传；空值按 auto 处理
    target_lang: str = ""     # 翻译/同传目标语；空值沿用 manifest 默认
    request_params: dict = Field(default_factory=dict)  # 按接口 request_schema 校验/透传
    note: str = ""            # 纯展示用任务备注(如"方言热词优化版")；不影响结果/排名/文件名


def _put_preview_value(target, dotted_key, value):
    """能力契约的 wire_name 可用点号表示嵌套 JSON 字段。"""
    parts = [part for part in str(dotted_key or "").split(".") if part]
    if not parts:
        return
    cur = target
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _planned_request_preview(model, model_cfg, ds, req, runtime):
    """任务派发时按登记契约生成请求投影，不等待首条网络调用或模型结果。"""
    contract = request_contract_for(model, model_cfg)
    protocol = str(contract.get("protocol") or "")
    endpoint = str(model_cfg.get("url") or model_cfg.get("base_url")
                   or model_cfg.get("endpoint") or contract.get("endpoint") or "").strip()
    is_ws = protocol.startswith("websocket") or endpoint.startswith(("ws://", "wss://"))
    if endpoint and "://" not in endpoint:
        if is_ws and model in ("plat-simult", "plat-simult-ws"):
            configured = _PLAT_WS_BASE if model == "plat-simult" else _PLAT_BASE
            scheme = "wss://" if configured.startswith("https://") else "ws://"
        else:
            scheme = "wss://" if is_ws else "http://"
        endpoint = scheme + endpoint

    runtime_values = runtime.get("request_params") or {}

    def value_for(field):
        name, managed = field.get("name", ""), field.get("managed_by")
        if name in runtime_values:
            return runtime_values[name]
        if managed == "language" or name == "language":
            return req.language or "auto"
        if managed == "target_lang" or name == "target_lang":
            return req.target_lang or f"<数据集默认目标语: {ds.get('lang', '未登记')}>"
        if managed == "hotwords" or name in ("hotwords", "hot_words"):
            parts = ["<当前样本 keywords>"] if req.hotwords else []
            if req.hotwords_text:
                parts.append(req.hotwords_text)
            return " ".join(parts)
        if managed == "audio_path":
            return "<当前样本音频>"
        if managed == "source_text":
            return "<当前样本 source_text>"
        if name == "domain" and req.domain and ds.get("domain"):
            return ds["domain"]
        return field.get("default") if "default" in field else None

    values = [(field, value_for(field)) for field in (contract.get("request_schema") or [])
              if isinstance(field, dict)]
    values = [(field, value) for field, value in values if value is not None]

    if is_ws:
        message = {"event": "task_start"}
        ws_model = {"plat-simult": "simult-interpreting",
                    "plat-simult-ws": "voice-input"}.get(model)
        if ws_model:
            message["model"] = ws_model
        for key, value in (contract.get("fixed_request") or {}).items():
            _put_preview_value(message, key, value)
        for field, value in values:
            if field.get("managed_by") == "audio_path":
                continue
            name = field.get("name", "")
            wire = field.get("wire_name") or name
            if name in ("language", "target_lang", "hotwords") and "." not in wire:
                wire = "voice_input_setting." + {
                    "language": "language", "target_lang": "target_language",
                    "hotwords": "hot_words",
                }[name]
            _put_preview_value(message, wire, value)
        rendered = (f"CONNECT {endpoint}\nSEND TEXT "
                    + json.dumps(message, ensure_ascii=False, indent=2)
                    + "\nSEND BINARY <16 kHz mono PCM chunks from current sample>"
                    + "\nSEND TEXT " + json.dumps({"event": "task_finish"}, ensure_ascii=False))
        return {"source": "planned_contract", "method": "WEBSOCKET", "url": endpoint,
                "protocol": protocol or "websocket", "timeout_s": None, "curl": rendered,
                "audio_hidden": True,
                "note": "任务派发时根据能力契约生成；展示握手和帧序列，不等待模型返回。"}

    kwargs = {"headers": {}, "params": {}, "data": {}, "json": {}, "files": {}}
    if model_cfg.get("key_env") or model_cfg.get("api_key"):
        kwargs["headers"]["Authorization"] = "<Bearer token hidden>"
    for field, value in values:
        name = field.get("wire_name") or field.get("name", "")
        location = field.get("in") or "form"
        if field.get("managed_by") == "audio_path":
            kwargs["files"][name] = ("audio.wav", b"", "audio/wav")
        elif location == "query":
            kwargs["params"][name] = value
        elif location == "header":
            kwargs["headers"][name] = value
        elif location == "json":
            _put_preview_value(kwargs["json"], name, value)
        else:
            kwargs["data"][name] = value
    kwargs = {key: value for key, value in kwargs.items() if value}
    preview = logconf.build_request_preview("POST", endpoint, kwargs)
    preview.update(source="planned_contract", protocol=protocol or "http",
                   note="任务派发时根据能力契约生成；首条真实 HTTP 请求发出后会自动替换。")
    return preview


def _run_eval(job_id, model, ds_id, tier, limit=None, seed=42, hotwords=False, hotwords_text="",
              domain=False, rerun=False, embedding=False, comet=False, speech_eval=False, language="",
              target_lang="", note="", request_params=None, config_hash="",
              run_mode="accuracy", workers=4):
    j = JOBS[job_id]
    ds = next((d for d in DATASETS if d["id"] == ds_id), None)
    j["stage"] = "排队中(同模型)" if run_mode == "accuracy" else "排队中(同端点)"
    with _ep_lock(model, run_mode):
        j["started_at"] = _now_iso()   # 拿到队列锁才算开始，排队不计耗时
        j["started_ts"] = time.time()
        _run_eval_locked(
            j, model, ds_id, ds, tier, limit, seed, hotwords, hotwords_text, domain, rerun,
            embedding, comet, speech_eval, language=language, target_lang=target_lang, note=note,
            request_params=request_params, config_hash=config_hash, workers=workers,
        )
    j["finished_at"] = _now_iso()
    j["took"] = round(time.time() - j["started_ts"])
    _flush_jobs()   # 收尾落盘(状态/耗时),前端跨进程可见


def _module_available(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _speex_available():
    try:
        from speex_decode import find_speex_library
        return bool(find_speex_library())
    except (ImportError, OSError):
        return False


def _missing_runtime_dependencies(model, ds, comet=False, speech_eval=False):
    """按本次任务真实会走到的能力检查依赖；已安装时不再误报。"""
    import shutil

    requirements = []
    is_simul = model in set(SCENARIO_MODELS.get("同传", []))
    if is_simul:
        requirements.append(("websocket-client", lambda: _module_available("websocket")))
    if is_simul and speech_eval:
        requirements += [
            ("GPU scorer / faster-whisper",
             lambda: scorer_runtime_available() or _module_available("faster_whisper")),
            ("ffmpeg", lambda: bool(shutil.which("ffmpeg"))),
        ]
    if model == "doubao-simult":
        requirements.append(("protobuf>=6.31.1", lambda: _module_available("google.protobuf")))
    if model == "xf-simult":
        requirements.append(("libspeex1", _speex_available))
    if ds.get("metric") == "DER":
        requirements.append(("pyannote.metrics", lambda: _module_available("pyannote.metrics")))
    if comet:
        requirements.append(("GPU XCOMET-XL scorer / unbabel-comet",
                             lambda: scorer_runtime_available() or comet_runtime_available()))

    missing = []
    for label, check in requirements:
        if not check() and label not in missing:
            missing.append(label)
    return missing


def _parse_longform_progress(line):
    """解析 adapter 打到 stdout 的长音频样本内进度；无关/损坏行直接忽略。"""
    pos = line.find(LONGFORM_PROGRESS_MARKER)
    if pos < 0:
        return None
    try:
        payload = json.loads(line[pos + len(LONGFORM_PROGRESS_MARKER):].strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        for key in ("sample_index", "sample_total", "segments"):
            if payload.get(key) is not None:
                payload[key] = max(0, int(payload[key]))
        for key in ("audio_s", "audio_total_s"):
            payload[key] = max(0.0, round(float(payload.get(key) or 0.0), 1))
    except (TypeError, ValueError):
        return None
    payload["sample_id"] = str(payload.get("sample_id") or "")[:200]
    payload["latest_text"] = str(payload.get("latest_text") or "").strip()[:240]
    payload["phase"] = str(payload.get("phase") or "streaming")[:40]
    return payload


def _stream_clock(seconds):
    seconds = max(0, int(round(float(seconds or 0))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _longform_progress_log_line(progress, now=None):
    """只把有状态变化的长音频事件写入任务日志；心跳/临时译文仅更新任务卡。"""
    phase = progress.get("phase") or "streaming"
    if phase in ("streaming", "partial", "revision"):
        return None
    sample_i = progress.get("sample_index") or "-"
    sample_n = progress.get("sample_total") or "-"
    audio = (f"{_stream_clock(progress.get('audio_s'))}"
             f"/{_stream_clock(progress.get('audio_total_s'))}")
    prefix = now or _now_iso()
    segments = progress.get("segments") or 0
    if phase == "segment":
        latest = re.sub(r"\s+", " ", progress.get("latest_text") or "")
        latest = f" · 译文 {latest}" if latest else ""
        return (f"{prefix} INFO [segment] 场次 {sample_i}/{sample_n}"
                f" · 音频 {audio} · 第 {segments} 段定稿{latest}\n")
    label = {"started": "开始", "completed": "完成", "failed": "连接失败"}.get(
        phase, phase
    )
    return (f"{prefix} INFO [stream] 场次 {sample_i}/{sample_n} · {label}"
            f" · 音频 {audio} · 已完成 {segments} 段\n")


def _run_eval_locked(j, model, ds_id, ds, tier, limit=None, seed=42, hotwords=False,
                     hotwords_text="",
                     domain=False, rerun=False, embedding=False, comet=False,
                     speech_eval=False,
                     language="", target_lang="", note="", request_params=None, config_hash="",
                     workers=1):
    if j.get("cancel"):  # 排队中被取消 → 拿到锁后不再起子进程
        j["status"] = "cancelled"; j["stage"] = "已取消(排队中)"
        return
    try:
        n = limit or lite_n_for(ds["scenario"])
        mani = f"manifests/{ds_id}_{tier}_n{n}_s{seed}.jsonl" if tier == "lite" else f"manifests/{ds_id}_full.jsonl"
        bm = [sys.executable, "eval/build_manifest.py", ds.get("builder", ds_id), "--out", mani]
        if ds.get("subset"):  # 子集型数据集(如 law_zh11)复用母 builder
            bm += ["--subset", ds["subset"]]
        if tier == "lite":
            bm += ["--shuffle", "--limit", str(n), "--seed", str(seed)]
        j["stage"] = "build_manifest"
        with _mani_lock(mani):  # 同参数复用、避免并发竞写
            if not os.path.exists(os.path.join(ROOT, mani)):
                subprocess.run(bm, cwd=ROOT, check=True, capture_output=True, text=True)

        # 文件名带 n/seed/language：不同抽样/语言提示不同文件，避免旧结果和新结果互相覆盖
        hotwords_enabled = bool(hotwords or hotwords_text)
        hw = ("_hw" if hotwords_enabled else "") + ("_dom" if (domain and ds.get("domain")) else "")
        ltag = language_tag(language)
        ctag = f"__cfg-{config_hash}" if config_hash else ""
        out = (f"results/{ds_id}_{model}_lite_n{n}_s{seed}{hw}{ltag}{ctag}.json"
               if tier == "lite"
               else f"results/{ds_id}_{model}_full{hw}{ltag}{ctag}.json")
        j["result_file"] = os.path.basename(out)  # 完成后前端据此点开样本详情
        dom_arg = ds["domain"] if (domain and ds.get("domain")) else ""
        inf = infer_file_path(model, mani, hotwords_enabled, dom_arg, language, config_hash)
        prev_inf = prev_out = None
        staged_inf = staged_out = None
        if rerun:
            # 只要选了重跑，就必须重新调接口；旧 infer/results 先整体挪走，runner 看不到旧文件，
            # 不能走断点续跑/秒回旧结果。若失败/取消，再把旧文件原样恢复。
            prev_inf = os.path.join(ROOT, inf)
            prev_out = os.path.join(ROOT, out)
            ts = time.strftime("%Y%m%dT%H%M%S")
            staged_inf = prev_inf + f".__rerun_{j['id']}_{ts}"
            staged_out = prev_out + f".__rerun_{j['id']}_{ts}"
            for src, dst in ((prev_inf, staged_inf), (prev_out, staged_out)):
                if os.path.exists(src):
                    os.replace(src, dst)
        is_simul = model in set(SCENARIO_MODELS.get("同传", []))
        missing = _missing_runtime_dependencies(model, ds, comet, speech_eval)
        if missing:
            raise RuntimeError("当前任务缺少运行时依赖: " + ", ".join(missing)
                               + "；请重建运行镜像后在 /api/doctor 复核")
        # language 是独立实验参数，不从数据集 lang 推导；空值退回 auto。
        # 数据集 lang 只留给 score 的规一化/指标口径。
        use_lang = language or "auto"
        rn = [sys.executable, "eval/runner.py", "--manifest", mani, "--model", model,
              "--language", use_lang, "--workers", str(workers), "--out", out]
        if target_lang:
            rn += ["--target-lang", target_lang]
        if config_hash:
            rn += ["--config-hash", config_hash]
        if request_params:
            rn += ["--request-params-json", json.dumps(request_params, ensure_ascii=False, sort_keys=True)]
        if rerun:
            rn.append("--no-resume")
        if note:
            rn += ["--note", note]
        if hotwords:
           rn.append("--hotwords")
        if hotwords_text:
            rn += ["--hotwords-text", hotwords_text]
        if domain and ds.get("domain"):
            rn += ["--domain", ds["domain"]]
        if embedding and ds.get("metric") in ("CER", "WER", "chrF"):  # ASR + 翻译；端点挂自动降级
            rn.append("--embedding")
        if is_simul and speech_eval:  # 语音同传口径才开 TTS；文本口径必须关闭，避免阻塞定稿。
            rn += ["--save-audio", "--asr-bleu"]
        if comet and ds.get("metric") == "chrF":   # COMET 仅翻译/同传
            rn.append("--comet")
        j["stage"] = "infer+score"
        LOG.info("任务起跑 %s %s×%s tier=%s mode=%s workers=%s",
                 j["id"], model, ds_id, tier, j.get("run_mode"), workers)
        # 每任务全量日志落盘(学 OpenCompass 每任务日志)：完整 stdout/stderr 存 logs/jobs/<id>.log，
        # 不再只留 40 行尾巴；前端「查看完整日志」按 id 取。
        os.makedirs(JOBS_LOG_DIR, exist_ok=True)
        _prune_job_logs()   # 保留上限：每任务日志走裸 open 不归 RotatingFileHandler 管，自裁防塞满盘
        j["log_file"] = j["id"]
        # 子进程 ASR_LOG_FILE=0：其 logger 只走 stderr(下方并入 stdout 管道捕获)，不写 logs/eval.log
        # → 杜绝 dashboard 与多个子进程并发轮转同一文件的竞争。
        # 写入前逐行过 redact：深层库的 traceback/print 可能带密钥，不能裸落盘。
        jlog = open(os.path.join(JOBS_LOG_DIR, j["id"] + ".log"), "w", encoding="utf-8")
        try:
            jlog.write(logconf.redact(f"# job {j['id']} {model}×{ds_id} tier={tier} @ {_now_iso()}\n"
                                      f"# cmd: {' '.join(rn)}\n\n"))
            jlog.flush()
            # 流式读 runner 输出：同传/长音频逐条、普通短音频每 10 条打印 "  n/N…"。
            p = subprocess.Popen(rn, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors="replace",
                                 env={**os.environ, **_iface_env_overrides(), "DASH_JOB": j["id"], "ASR_LOG_FILE": "0"},
                                 start_new_session=True)  # 独立进程组 → 可整组 kill(uv+python)
            PROCS[j["id"]] = p
            tail = []
            for line in p.stdout:
                stream_progress = _parse_longform_progress(line)
                display_line = line
                if stream_progress:
                    stream_progress["updated_at"] = _now_iso()
                    phase = stream_progress.get("phase")
                    previous_progress = j.get("stream_progress") or {}
                    if stream_progress.get("latest_text") and phase in ("partial", "revision", "segment"):
                        stream_progress["latest_phase"] = phase
                    elif previous_progress.get("latest_phase"):
                        stream_progress["latest_phase"] = previous_progress["latest_phase"]
                    j["stream_progress"] = stream_progress
                    j["_stream_heartbeat_ts"] = time.time()
                    j["stage"] = {
                        "started": "同传流式处理中",
                        "streaming": "同传流式处理中",
                        "partial": "同传临时译文生成中",
                        "revision": "同传临时译文修订中",
                        "segment": "同传流式处理中",
                        "completed": "同传样本收尾",
                        "failed": "同传连接失败，等待重试",
                    }.get(phase, "同传流式处理中")
                    sample_i = stream_progress.get("sample_index") or 0
                    sample_n = stream_progress.get("sample_total") or 0
                    if sample_i and sample_n:
                        completed = sample_i if phase == "completed" else sample_i - 1
                        j["progress"] = f"{max(0, completed)}/{sample_n}"
                    display_line = _longform_progress_log_line(stream_progress)
                    if display_line is None:
                        continue
                safe_line = logconf.redact(display_line)
                jlog.write(safe_line); jlog.flush()
                tail = (tail + [safe_line])[-40:]
                if stream_progress:
                    continue
                m = re.search(r"(\d+)/(\d+)…", line)
                if m:
                    j["progress"] = f"{m.group(1)}/{m.group(2)}"
                elif "待跑" in line:
                    m = re.search(r"待跑 (\d+)", line)
                    if m:
                        j["progress"] = f"0/{m.group(1)}"
            p.wait()
        finally:
            jlog.close()
            PROCS.pop(j["id"], None)   # stdout 解码异常/中途抛错也要回收，否则 cancel 误判仍在跑
        if j.get("cancel"):  # 运行中被 kill
            for p, bak in ((prev_inf, staged_inf), (prev_out, staged_out)):
                if p and bak and os.path.exists(bak) and not os.path.exists(p):
                    os.replace(bak, p)
            j["status"] = "cancelled"; j["stage"] = "已停止"
            LOG.info("任务取消 %s", j["id"])
            return
        j["log"] = logconf.redact("".join(tail))[-2300:]   # 内联尾巴(前端悬停显示)也脱敏
        if p.returncode != 0:
            for p0, bak in ((prev_inf, staged_inf), (prev_out, staged_out)):
                if p0 and bak and os.path.exists(bak) and not os.path.exists(p0):
                    os.replace(bak, p0)
            j["status"] = "error"
            LOG.warning("任务失败 %s rc=%s（详见 logs/jobs/%s.log）", j["id"], p.returncode, j["id"])
            return
        if rerun:
            # 新结果已产出，再把旧结果归档留史；若归档失败，也不影响当前结果可读。
            for cur, bak in ((prev_inf, staged_inf), (prev_out, staged_out)):
                if not (cur and bak and os.path.exists(bak)):
                    continue
                hist = os.path.join(os.path.dirname(cur), "history")
                os.makedirs(hist, exist_ok=True)
                ts0 = time.strftime("%Y%m%dT%H%M%S", time.localtime(os.path.getmtime(bak)))
                base, ext = os.path.splitext(os.path.basename(cur))
                try:
                    os.replace(bak, os.path.join(hist, f"{base}__{ts0}{ext}"))
                except OSError:
                    pass
        j["summary"] = json.load(open(os.path.join(ROOT, out), encoding="utf-8"))["summary"]
        j["status"] = "done"
        LOG.info("任务完成 %s → %s", j["id"], os.path.basename(out))
    except subprocess.CalledProcessError as e:
        for p0, bak in ((locals().get("prev_inf"), locals().get("staged_inf")),
                        (locals().get("prev_out"), locals().get("staged_out"))):
            if p0 and bak and os.path.exists(bak) and not os.path.exists(p0):
                os.replace(bak, p0)
        j["status"] = "error"
        j["log"] = logconf.redact(e.stderr or str(e))[-1500:]
        LOG.warning("任务异常(build) %s: %s", j["id"], logconf.redact(str(e)[:200]))
    except Exception as e:
        for p0, bak in ((locals().get("prev_inf"), locals().get("staged_inf")),
                        (locals().get("prev_out"), locals().get("staged_out"))):
            if p0 and bak and os.path.exists(bak) and not os.path.exists(p0):
                os.replace(bak, p0)
        j["status"] = "error"
        j["log"] = logconf.redact(str(e))[-1500:]
        LOG.exception("任务异常 %s", j["id"])


@app.post("/api/run")
def run(req: RunReq):
    ds = next((d for d in DATASETS if d["id"] == req.dataset), None)
    if not ds or not ds["runnable"]:
        return JSONResponse({"error": "该测评集暂无 builder，不可运行(只 ASR 三件套可跑)"}, status_code=400)
    run_mode = req.run_mode.strip().lower()
    if run_mode not in {"accuracy", "latency"}:
        return JSONResponse({"error": "run_mode 只支持 accuracy/latency"}, status_code=400)
    # 同传本身以实时节奏采 AL/LAAL/TTFB，强制保留串行延迟口径。
    if req.model in set(SCENARIO_MODELS.get("同传", [])):
        run_mode = "latency"
    workers = req.workers if run_mode == "accuracy" else 1
    compat_error = _capability_compat_error(req.model, ds, req.target_lang)
    if compat_error:
        return JSONResponse({"error": compat_error}, status_code=400)
    model_cfg = _find_model(req.model)
    if not model_cfg:
        custom_cfg = next((c for c in _load_ifaces() if c.get("id") == req.model), {})
        model_cfg = {**custom_cfg, "id": req.model, "group": "custom",
                     "tmpl": custom_cfg.get("template", ""),
                     "scenarios": list(custom_cfg.get("scenarios") or ["ASR"])}
    model_caps = _model_caps(model_cfg)
    if req.domain and "domain" not in model_caps:
        return JSONResponse({"error": f"{req.model} 不支持 domain；请使用独立的 adv-domain 接口"},
                            status_code=400)
    try:
        hotwords_text = req.hotwords_text.strip()
        if len(hotwords_text) > 8192:
            raise ValueError("hotwords 最多 8192 个字符")
        runtime = _runtime_config(
            req.model, req.request_params,
            {"target_lang": req.target_lang, "hotwords_text": hotwords_text,
             "speech_eval": req.speech_eval},
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if req.hotwords or hotwords_text:
        # 不支持热词的模型显式拒绝：infer 会静默降级，但结果会被命名成 _hw(场景 b)，得出"热词无效"的假结论
        from adapters import model_supports_hotwords
        if not model_supports_hotwords(req.model):
            return JSONResponse({"error": f"{req.model} 不支持热词——取消勾选热词后再跑（否则产出假的场景 b 结果）"},
                                status_code=400)
    # 同参数任务去重：避免两个进程同时 append 同一个 infer 文件；score-only 选项也必须同口径才复用。
    n = req.limit or lite_n_for(ds["scenario"])
    key = _run_key(req.model, req.dataset, req.tier, req.limit, req.seed,
                   req.hotwords, hotwords_text, req.domain, req.embedding, req.comet,
                   req.speech_eval, req.language,
                   runtime["hash"], ds)
    dup = next((j for j in JOBS.values() if j["status"] == "running"
                and j.get("run_key") == key
                and not req.rerun), None)  # rerun 不与排队中的复用任务去重
    if dup:
        return {"job_id": dup["id"], "dedup": True}
    job_id = uuid.uuid4().hex[:8]
    request_preview = _planned_request_preview(req.model, model_cfg, ds, req, runtime)
    JOBS[job_id] = {"id": job_id, "model": req.model, "dataset": req.dataset,
                    "tier": req.tier, "limit": req.limit, "n": n, "seed": req.seed,
                    "hotwords": bool(req.hotwords or hotwords_text),
                  "dataset_hotwords": req.hotwords, "hotwords_text": hotwords_text,
                  "domain": req.domain,
                  "embedding": req.embedding, "comet": req.comet,
                  "speech_eval": req.speech_eval, "rerun": req.rerun,
                  "run_mode": run_mode, "workers": workers,
                  "language": req.language,
                  "target_lang": req.target_lang,
                  "request_params": runtime["request_params"],
                  "config_hash": runtime["hash"],
                  "request_preview": request_preview,
                  "note": req.note,
                  "run_key": key, "status": "running", "stage": "排队中",
                 "created_at": _now_iso(), "created_ts": time.time(), "origin": "dashboard"}
    _flush_jobs()
    LOG.info("派发任务 %s %s×%s tier=%s mode=%s workers=%s hw=%s dom=%s rerun=%s",
            job_id, req.model, req.dataset, req.tier, run_mode, workers,
            req.hotwords, req.domain, req.rerun)
    threading.Thread(target=_run_eval,
                     args=(job_id, req.model, req.dataset, req.tier, req.limit,
                           req.seed, req.hotwords, hotwords_text, req.domain, req.rerun,
                           req.embedding, req.comet, req.speech_eval,
                           req.language, req.target_lang, req.note,
                           runtime["request_params"], runtime["hash"], run_mode, workers),
                    daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """停止运行中 / 取消排队中的任务。运行中 kill 子进程组，排队中设标志(拿锁后跳过)。"""
    j = JOBS.get(job_id)
    if not j or j["status"] != "running":
        return JSONResponse({"error": "任务不在运行/排队中"}, status_code=400)
    j["cancel"] = True
    p = PROCS.get(job_id)
    if p and p.poll() is None:  # 运行中：杀整个进程组
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
        j["status"] = "cancelled"; j["stage"] = "停止中"
    else:  # 排队中：只设标志，拿到锁后 _run_eval_locked 会跳过
        j["status"] = "cancelled"; j["stage"] = "已取消"
    _flush_jobs()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, request: Request):
    """删除任务记录；运行中任务先取消再移除。"""
    j = JOBS.get(job_id)
    payload = {}
    try:
        raw = await request.body()
        payload = json.loads(raw.decode()) if raw else {}
    except Exception:
        payload = {}
    if j and j.get("status") == "running":
        j["cancel"] = True
        p = PROCS.get(job_id)
        if p and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        j["status"] = "cancelled"
        j["stage"] = "已移除"
        _flush_jobs()
    JOBS.pop(job_id, None)
    def mut(disk):
        disk.pop(job_id, None)
        return True
    _mutate_jobs_file(mut)
    return {"ok": True}


def _pid_alive(pid):
    """Return whether a locally registered CLI runner process still exists."""
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


_CLI_RESULT_CACHE = {}


def _cli_result_record(filename):
    """Resolve a CLI job's legacy ``out`` field to a safe result summary."""
    base = os.path.basename(filename or "")
    if not base.endswith(".json"):
        return None, None
    path = os.path.realpath(os.path.join(ROOT, "results", base))
    allowed = os.path.realpath(os.path.join(ROOT, "results"))
    if not path.startswith(allowed + os.sep) or not os.path.isfile(path):
        return None, None
    try:
        mtime = os.path.getmtime(path)
        cached = _CLI_RESULT_CACHE.get(path)
        if cached and cached[0] == mtime:
            return base, dict(cached[1])
        summary = json.load(open(path, encoding="utf-8")).get("summary", {})
        if not isinstance(summary, dict):
            return None, None
        _CLI_RESULT_CACHE[path] = (mtime, dict(summary))
        return base, summary
    except (OSError, ValueError, TypeError):
        return None, None


def _cli_request_preview(summary):
    meta = summary.get("meta") or {}
    spec = meta.get("run_spec") or {}
    endpoint = spec.get("endpoint") or meta.get("endpoint") or ""
    if not endpoint:
        return None
    model = spec.get("model") or meta.get("model") or ""
    language = spec.get("language") or meta.get("language") or "auto"
    form = dict(spec.get("request_params") or meta.get("request_params") or {})
    if model != "sse" or language != "auto":
        form.setdefault("language", language)
    kwargs = {
        "data": form,
        "files": {"file": ("audio.wav", b"", "audio/wav")},
    }
    if _is_platform_url(endpoint):
        kwargs["headers"] = {"Authorization": "<Bearer token hidden>"}
    preview = logconf.build_request_preview("POST", endpoint, kwargs)
    preview.update(
        source="result_metadata",
        note="由结果中的实际运行端点与参数重建；历史 CLI 未保留独立原始请求日志。",
    )
    return preview


def _hydrate_cli_job(j, result_file, summary):
    meta = summary.get("meta") or {}
    spec = meta.get("run_spec") or {}
    j["result_file"] = result_file
    j["summary"] = summary
    if j.get("tier") in (None, "", "cli"):
        j["tier"] = "full" if "_full" in result_file else "lite" if "_lite" in result_file else "cli"
    workers = spec.get("workers") or meta.get("workers")
    if workers is not None:
        j.setdefault("workers", workers)
        j.setdefault("run_mode", "accuracy" if workers > 1 else "latency")
    for key in ("language", "target_lang", "hotwords", "dataset_hotwords", "hotwords_text",
                "domain", "request_params", "config_hash", "note"):
        value = spec.get(key) if key in spec else meta.get(key)
        if value not in (None, "", False, {}):
            j.setdefault(key, value)
    j.setdefault("started_at", meta.get("infer_started") or j.get("created_at"))
    preview = _cli_request_preview(summary)
    if preview:
        j.setdefault("request_preview", preview)
    try:
        if os.path.isfile(_job_log_path(j.get("id", ""))):
            j["log_file"] = j["id"]
    except (ValueError, PermissionError):
        pass


@app.get("/api/jobs")
def jobs():
    out = []
    now = time.time()
    # 外部(CLI/longrun 经 jobstore 登记)、不在内存的任务 → 也展示
    ext = {jid: r for jid, r in _read_jobs_file().items() if jid not in JOBS}
    seq = list(JOBS.values())[::-1] + sorted(ext.values(),
            key=lambda r: r.get("started_ts") or r.get("created_ts") or 0, reverse=True)
    for j in seq:
        j = dict(j)
        if j.get("origin") == "cli" and j.get("status") == "done":
            result_file, summary = _cli_result_record(j.get("result_file") or j.get("out"))
            if result_file:
                _hydrate_cli_job(j, result_file, summary)
        heartbeat_ts = j.pop("_stream_heartbeat_ts", None)
        if isinstance(j.get("stream_progress"), dict):
            stream_progress = dict(j["stream_progress"])
            stream_progress["heartbeat_age_s"] = (
                round(max(0, now - heartbeat_ts)) if heartbeat_ts else None
            )
            j["stream_progress"] = stream_progress
        if j.get("status") == "running":
            if j.get("origin") == "cli" and j.get("started_ts"):
                # CLI runner 从起跑即开始执行，不经过 dashboard 的端点锁队列，因此不会
                # 自带 started_at。补齐展示字段，避免前端误画成“排队中 0s”。
                j["started_at"] = j.get("started_at") or j.get("created_at")
            if j.get("started_ts"):      # 已开始：运行时长
                j["elapsed"] = round(now - j["started_ts"])
            elif j.get("created_ts"):    # 还在排队：等待时长（与运行时长区分）
                j["waiting"] = round(now - j["created_ts"])
            if j.get("origin") == "dashboard" and j.get("id") not in JOBS:
                j["status"] = "error"
                j["stage"] = "dashboard 服务重启，任务状态已丢失"
            elif j.get("origin") == "cli" and j.get("pid") and not _pid_alive(j["pid"]):
                j["status"] = "error"
                j["stage"] = "CLI 进程已退出，任务未正常收尾"
            # 外部任务无心跳 >30min 视为可能已死(进程被杀,jobstore 没收尾)
            elif (j.get("origin") == "cli" and not j.get("pid")
                  and j.get("started_ts") and now - j["started_ts"] > 1800):
                j["stage"] = (j.get("stage") or "") + " · 可能已结束(无更新)"
        j.pop("started_ts", None)
        j.pop("created_ts", None)
        j.pop("cancel", None)
        out.append(j)
    return {"jobs": out}


@app.get("/api/export")
def export(manifest: str):
    """同清单结果 → Markdown 对比表（manifest 传 basename，如 aishell_lite_n300_s42.jsonl）。"""
    rows = [s for s in read_results()
            if s.get("manifest") and os.path.basename(s["manifest"]) == os.path.basename(manifest)]
    if not rows:
        return PlainTextResponse("（该清单暂无结果）", status_code=404)
    rows.sort(key=lambda s: (s.get("low_coverage", False), s.get("err_rate") or 1))
    sha = (rows[0].get("meta") or {}).get("manifest_sha", "")
    lines = [f"## {os.path.basename(manifest)}" + (f"（指纹 {sha}）" if sha else ""),
             "", "| 结果 | 指标 | 值 | BLEU | 95%CI | KRR | 失败率 | p50延迟 | AL同传 | 改写率 | RTF | n | infer 时间 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        m = s.get("metric", "")
        v = s.get(m)
        ci = s.get("err_ci95")
        meta = s.get("meta") or {}
        warn = " ⚠️覆盖不足" if s.get("low_coverage") else ""
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} |".format(
            s["_file"].replace(".json", "") + warn, m,
            v if v is not None else "—",
            s.get("BLEU") if s.get("BLEU") is not None else "—",
            f"[{ci[0]}, {ci[1]}]" if ci else "—",
            s.get("KRR") if s.get("KRR") is not None else "—",
            s.get("fail_rate") if s.get("fail_rate") is not None else "—",
            f'{s["latency_p50_s"]}s' if s.get("latency_p50_s") is not None else "—",
            f'{s["al_p50_s"]}s' if s.get("al_p50_s") is not None else "—",
            s.get("revision_rate") if s.get("revision_rate") is not None else "—",
            s.get("rtf_mean") if s.get("rtf_mean") is not None else "—",
            s.get("n_ok"), s.get("n_total"), meta.get("infer_started", "—")))
    lines += ["", "> 口径：失败样本不计精度(覆盖<95%标⚠️不参与排名)；延迟为本地→端点端到端(含网络)，串行采集；"
              "AL=同传滞后(Average Lagging，秒)，改写率=流式已吐译文被回改比例(越低越稳)。"]
    return PlainTextResponse("\n".join(lines), media_type="text/markdown; charset=utf-8")


# ── 自定义接口注册（「＋添加接口」）：配置持久化到 interfaces.json，
#    eval/adapters.py 启动时自动读同一文件注册 ADAPTERS → runner 子进程直接可用 ──
IFACE_FILE = os.path.join(HERE, "interfaces.json")
_BUILTIN_MODELS = {m["id"]: dict(m) for m in MODELS}


def _custom_iface_base(cfg: dict, raw: str | None = None) -> str:
    """Return the service root even if an edit submitted the display endpoint."""
    base = (raw if raw is not None else cfg.get("base_url", "")).strip().rstrip("/")
    template = cfg.get("template", "")
    if template == "plat-multipart" and "/api/asr/" in base:
        base = base.split("/api/asr/", 1)[0]
    elif template in ("openai-audio", "openai-chat", "openai-chat-audio"):
        suffix = "/" + (cfg.get("path") or "/v1").strip("/")
        while suffix != "/" and base.endswith(suffix):
            base = base[:-len(suffix)].rstrip("/")
    return base


def _custom_iface_display_url(cfg: dict) -> str:
    base = _custom_iface_base(cfg)
    if cfg.get("template") == "custom-http-asr":
        return base
    display_base = base.removeprefix("http://").removeprefix("https://")
    path = (cfg.get("path") or "").strip()
    if cfg.get("template") == "plat-multipart":
        return f"{display_base}/api/asr/{path or 'adv'}"
    if cfg.get("template") in ("openai-audio", "openai-chat", "openai-chat-audio"):
        suffix = path if path.startswith("/") else "/" + path if path else "/v1"
        return display_base + suffix
    return display_base


def _normalize_custom_iface(cfg: dict) -> dict:
    if cfg.get("kind") == "builtin":
        # 旧持久卷若保存过已下线的内置地址覆盖，启动时自动回到当前默认值。
        if cfg.get("id") in _BUILTIN_MODELS and (cfg.get("url") or "") in _RETIRED_BUILTIN_URLS:
            default = _BUILTIN_MODELS[cfg["id"]]
            cfg["url"] = default.get("url", cfg["url"])
            cfg["endpoint"] = default.get("endpoint", "platform")
        if cfg.get("id") in _BUILTIN_ENV and not cfg.get("key_env"):
            if _BUILTIN_ENV[cfg["id"]] in ("ASR_PLATFORM_URL", "ASR_PLATFORM_WS_URL"):
                cfg["key_env"] = _PLAT_KEY_ENV
        return cfg
    if cfg.get("base_url"):
        cfg["base_url"] = _custom_iface_base(cfg)
    if cfg.get("template"):
        cfg["url"] = _custom_iface_display_url(cfg)
    return cfg


def _load_ifaces():
    try:
        return [_normalize_custom_iface(c) for c in json.load(open(IFACE_FILE, encoding="utf-8"))]
    except Exception:
        return []


def _save_ifaces(cfgs):
    atomic_write_json(IFACE_FILE, [_normalize_custom_iface(dict(c)) for c in cfgs])


def _find_model(iid: str):
    return next((m for m in MODELS if m["id"] == iid), None)


def _apply_builtin_overrides(cfgs):
    for c in cfgs:
        if c.get("kind") != "builtin":
            continue
        m = _find_model(c["id"])
        if not m:
            continue
        for k in ("name", "url", "endpoint", "group", "text", "trans_input", "key_env"):
            if k in c:
                m[k] = c[k]
        if "enabled" in c:
            m["enabled"] = bool(c["enabled"])


def _register_iface_models(cfgs):
    for c in cfgs:
        if c.get("kind") == "builtin":
            continue
        if any(m["id"] == c["id"] for m in MODELS):
            continue
        MODELS.append({"id": c["id"], "name": c["name"], "endpoint": c.get("base_url", ""),
                       "group": c.get("group", "custom"), "url": c.get("url", c.get("base_url", "")),
                       "base_url": c.get("base_url", ""), "model": c.get("model", ""),
                       "path": c.get("path", ""),
                       "scenarios": list(c.get("scenarios") or ["ASR"]),
                       "local": any(h in c.get("base_url", "") for h in ("localhost", "127.0.0.1")),
                       "text": "ASR" not in (c.get("scenarios") or ["ASR"]),
                       "trans_input": "text" if c.get("template") == "openai-chat" else "audio",
                       "tmpl": c.get("template", ""), "caps": c.get("caps"),
                       "language_spec": c.get("language_spec"),
                       "request_schema": c.get("request_schema") or [],
                       "dual_result": bool(c.get("dual_result")),
                       "enabled": bool(c.get("enabled", True))})
        for sc in (c.get("scenarios") or ["ASR"]):
            if sc in SCENARIO_MODELS and c["id"] not in SCENARIO_MODELS[sc]:
                SCENARIO_MODELS[sc].append(c["id"])


# ── 接口能力标签（caps）：hard=场景能力(缺了跑不了该类集)，soft=功能能力(缺了只能当对照组) ──
# 内置接口从场景注册和独立端点 adapter 派生；禁止通过运行时换路由推导能力。
# 自定义接口按模板给默认，interfaces.json 的 caps 字段可覆盖（接口管理 → 编辑）。
_TMPL_CAPS = {"openai-audio": ["asr"], "openai-chat-audio": ["asr"],
              "plat-multipart": ["asr"], "asr_lite": ["asr", "hotwords"],
              "sensevoice": ["asr"], "plat-sse": ["asr"], "openai-chat": ["translate_text", "summarize"],
              "custom-http-asr": ["asr", "translate_audio"]}
_DOMAIN_MODELS = {"adv-domain"}
_CAPS_CACHE = {}


def _model_caps(m):
    mid = m["id"]
    if m.get("caps"):                      # interfaces.json 显式声明优先（自定义接口编辑保存）
        return sorted(set(m["caps"]))
    if m.get("group") == "custom":
        caps = set(_TMPL_CAPS.get(m.get("tmpl", ""), ["asr"]))
        scenarios = set(m.get("scenarios") or [])
        scenario_caps = set()
        if "ASR" in scenarios:
            scenario_caps.add("asr")
        if "翻译" in scenarios:
            scenario_caps.add("translate_text" if m.get("tmpl") == "openai-chat" else "translate_audio")
        if "总结" in scenarios:
            scenario_caps.add("summarize")
        hard_caps = {"asr", "translate_text", "translate_audio", "summarize", "simult", "diar"}
        if scenario_caps:
            caps = (caps - hard_caps) | scenario_caps
        if m.get("dual_result"):
            caps.add("dual_result")
        contract = request_contract_for(mid, m)
        if any(field.get("managed_by") == "hotwords"
               for field in (contract.get("request_schema") or []) if isinstance(field, dict)):
            caps.add("hotwords")
        return sorted(caps)
    if mid in _CAPS_CACHE:
        return _CAPS_CACHE[mid]
    caps = set()
    if mid in SCENARIO_MODELS.get("ASR", []):
        caps.add("asr")
    if mid in SCENARIO_MODELS.get("总结", []):
        caps.add("summarize")
    if mid in SCENARIO_MODELS.get("同传", []):
        caps.add("simult")
    if mid in SCENARIO_MODELS.get("翻译", []):
        caps.add("translate_audio" if _TRANS_INPUT.get(mid, "audio") == "audio" else "translate_text")
    caps |= {"plat-diar": {"diar"}, "plat-minutes": {"minutes"}, "plat-formula": {"formula"}}.get(mid, set())
    if mid in _DOMAIN_MODELS:
        caps.add("domain")
    if m.get("dual_result"):
        caps.add("dual_result")
    try:
        from adapters import model_supports_hotwords
        if model_supports_hotwords(mid):
            caps.add("hotwords")
    except Exception:
        pass
    _CAPS_CACHE[mid] = sorted(caps)
    return _CAPS_CACHE[mid]


def _ds_requires(d):
    """数据集能力需求：hard 缺了置灰不可选，soft 缺了可跑但只当对照组（热词/领域 A/B 语义显式化）。"""
    sc = d.get("scenario", "")
    if d.get("metric") == "DER":
        hard = ["diar"]
    elif sc.startswith("翻译"):
        hard = ["translate_audio" if "语音" in sc else "translate_text"]
    elif sc.startswith("总结"):
        hard = ["summarize"]
    else:
        hard = ["asr"]
    soft = (["hotwords"] if d.get("hotwords") else []) + (["domain"] if d.get("domain") else [])
    return {"hard": hard, "soft": soft}


_LANG_CODE_ALIASES = {
    "英文": "en", "english": "en", "中文": "zh", "简体中文": "zh",
    "simplified chinese": "zh", "chinese": "zh", "粤语": "yue", "cantonese": "yue",
    "日文": "ja", "japanese": "ja", "韩文": "ko", "korean": "ko",
    "西班牙文": "es", "spanish": "es", "法文": "fr", "french": "fr",
    "德文": "de", "german": "de", "俄文": "ru", "russian": "ru",
    "arabic": "ar", "dutch": "nl", "indonesian": "id", "italian": "it",
    "malay": "ms", "portuguese": "pt", "thai": "th", "turkish": "tr",
    "urdu": "ur", "vietnamese": "vi",
}


def _language_code(value) -> str:
    raw = str(value or "").strip()
    return _LANG_CODE_ALIASES.get(raw, _LANG_CODE_ALIASES.get(raw.lower(), raw.lower()))


def _dataset_language_pair(ds: dict, target_override="") -> tuple[str, str]:
    raw = str(ds.get("lang") or "")
    src, _, target = raw.partition("-")
    return _language_code(src), _language_code(target_override or target)


def _capability_compat_error(model_id: str, ds: dict, target_lang="") -> str:
    model = _find_model(model_id)
    if not model:
        cfg = next((c for c in _load_ifaces() if c.get("id") == model_id), None)
        if cfg:
            model = {**cfg, "id": model_id, "group": "custom",
                     "tmpl": cfg.get("template", ""),
                     "scenarios": list(cfg.get("scenarios") or ["ASR"])}
        else:
            return f"接口 {model_id} 未登记"
    if model.get("enabled") is False:
        return f"接口 {model_id} 已停用"
    caps = _model_caps(model)
    def sat(required):
        return required in caps or (required == "translate_audio" and "simult" in caps)
    missing = [required for required in _ds_requires(ds)["hard"] if not sat(required)]
    if missing:
        return f"{model_id} 缺少数据集 {ds['id']} 所需能力: {', '.join(missing)}"

    contract = capability_contract_for(model_id, model)
    pairs = contract.get("language_pairs") or []
    if pairs:
        source, target = _dataset_language_pair(ds, target_lang)
        dataset_sources = [_language_code(x) for x in (ds.get("source_langs") or [source])]
        def pair_allows(pair, candidate):
            sources = {_language_code(x) for x in pair.get("source", [])}
            targets = {_language_code(x) for x in pair.get("target", [])}
            return ("*" in sources or candidate in sources) and ("*" in targets or target in targets)
        compatible = all(any(pair_allows(pair, candidate) for pair in pairs)
                         for candidate in dataset_sources)
        if not compatible:
            source_label = "/".join(dataset_sources) if len(dataset_sources) > 1 else (source or "?")
            return f"{model_id} 未登记语向 {source_label}→{target or '?'}，拒绝生成无效结果"
    return ""


_PLAT_LANGUAGE_SPEC = {
    "supported": True,
    "values": ["auto", "Chinese", "English", "Chinese,English"],
    "note": "平台 OpenAPI：auto 自动识别；可填单个语言名，或用英文逗号组成候选集（如 Chinese,English）。",
}

_MODEL_LANGUAGE_SPECS = {
    # 活体：该服务的 FastAPI 枚举为 auto/中文/英文/日文；zh 会 422。
    "light": {"supported": True, "values": ["auto", "中文", "英文", "日文"],
             "note": "/asr_lite 使用中文枚举；不要填 zh/en。"},
    "light-mlt": {"supported": True,
                 "values": ["auto", "中文", "英文", "粤语", "日文", "韩文", "越南语", "印尼语",
                            "泰语", "马来语", "菲律宾语", "阿拉伯语", "印地语", "保加利亚语",
                            "克罗地亚语", "捷克语", "丹麦语", "荷兰语", "爱沙尼亚语", "芬兰语",
                            "希腊语", "匈牙利语", "爱尔兰语", "拉脱维亚语", "立陶宛语", "马耳他语",
                            "波兰语", "葡萄牙语", "罗马尼亚语", "斯洛伐克语", "斯洛文尼亚语", "瑞典语"],
                 "note": "/asr_mlt_nano OpenAPI 枚举；使用中文语言名。"},
    "std": _PLAT_LANGUAGE_SPEC,
    "adv": _PLAT_LANGUAGE_SPEC,
    "adv-domain": _PLAT_LANGUAGE_SPEC,
    "sse": _PLAT_LANGUAGE_SPEC,
    "sensevoice": {"supported": True, "default": "zh",
                   "values": ["zh", "en", "yue", "ja", "ko", "nospeech"],
                   "note": "作为 /api/asr_transcribe 的 language query 参数发送；该服务默认 zh，不声明 auto。"},
    "xf-spark-slm-iat": {"supported": False, "values": [],
                         "note": "当前 adapter 固定 zh_cn + mulacc，不读取任务 language。"},
}
_TMPL_LANGUAGE_SPECS = {
    "asr_lite": {"supported": True, "values": [],
                 "note": "language 会作为 multipart form 字段原样发送；具体枚举以该服务 OpenAPI 为准。"},
    "sensevoice": _MODEL_LANGUAGE_SPECS["sensevoice"],
    "plat-multipart": _PLAT_LANGUAGE_SPEC,
    "plat-sse": _PLAT_LANGUAGE_SPEC,
    "plat-sse-dual": _PLAT_LANGUAGE_SPEC,
    "openai-audio": {"supported": True, "values": [],
                     "note": "auto 表示不发送 language；其他非空值作为 multipart language 原样发送，具体取值由服务决定。"},
    "openai-chat-audio": {"supported": True, "values": ["auto", "zh", "en"],
                          "note": "作为 asr_options.language 发送。"},
    "custom-http-asr": {"supported": True, "values": [],
                        "note": "非 auto 值默认作为 language query 参数发送；可由高级 HTTP 配置覆盖。"},
}


def _provider_language_spec(m):
    """旧 interfaces.json 没存 language_spec 时，按已知预设的 base_url 补帮助元数据。"""
    target = (m.get("endpoint") or m.get("url") or "").strip().rstrip("/")
    target = target.removeprefix("http://").removeprefix("https://")
    for p in _providers().values():
        base = (p.get("base_url") or "").strip().rstrip("/")
        base = base.removeprefix("http://").removeprefix("https://")
        if base and target == base and p.get("language_spec"):
            return p["language_spec"]
    return None


def _known_custom_language_spec(m):
    """给未保存 language_spec 的现有自定义连接补充已核对的服务说明。"""
    model = (m.get("model") or "").lower()
    endpoint = (m.get("endpoint") or m.get("base_url") or m.get("url") or "").lower()
    if "api.siliconflow.cn" in endpoint and "sensevoicesmall" in model:
        return {
            "supported": True, "values": [],
            "note": "SiliconFlow 公开转写接口未声明 language 参数；连接器可原样发送用于实测，但服务是否采用该值未获文档确认。",
        }
    if "qwen3-asr-1.7b" in model:
        return {
            "supported": True,
            "values": ["auto", "Chinese", "English", "Cantonese", "Japanese", "Korean"],
            "note": "8770 OpenAPI 接受任意字符串且不提供枚举；auto 不发送，其他值原样发送。Qwen3-ASR 示例使用语言名称。",
        }
    if "dolphin-cn-dialect" in model:
        return {
            "supported": True, "values": [],
            "note": "当前按 OpenAI Audio 协议原样发送非 auto 值；该服务未登记固定枚举，需以本机服务实际实现为准。",
        }
    return None


def _model_payload(m):
    b = _BUILTIN_MODELS.get(m["id"])
    out = dict(m)
    out["builtin"] = bool(b)
    out["editable"] = True
    out["enabled"] = bool(m.get("enabled", True))
    out["caps"] = _model_caps(m)
    contract = request_contract_for(m.get("id", ""), m)
    schema = _sanitize_request_schema(contract.get("request_schema") or [])
    out["request_schema"] = schema
    out["request_contract"] = {
        "status": contract.get("status") or ("verified" if schema else "unverified"),
        "source": contract.get("source") or "unverified",
        "field_count": len(schema),
        "editable_count": sum(1 for x in schema if x.get("editable") is not False and not x.get("managed_by")),
        "task_count": sum(1 for x in schema if x.get("managed_by") in ("language", "target_lang", "hotwords")),
        **{key: contract[key] for key in (
            "kind", "endpoint", "protocol", "profile_of", "fixed_request",
            "managed_request", "verified_version", "verified_at", "note",
        ) if key in contract},
    }
    out["capability_contract"] = capability_contract_for(m.get("id", ""), m)
    out["language_spec"] = (m.get("language_spec")
                            or _provider_language_spec(m)
                            or _MODEL_LANGUAGE_SPECS.get(m["id"])
                            or _known_custom_language_spec(m)
                            or _TMPL_LANGUAGE_SPECS.get(m.get("tmpl", ""))
                            or {"supported": False, "values": [],
                                "note": "该接口尚未登记 language 参数能力。"})
    if b:
        out["default_name"] = b.get("name", "")
        out["default_url"] = b.get("url", "")
    return out


_BUILTIN_ENV = {
    "adv": "ASR_PLATFORM_URL",
    "adv-domain": "ASR_PLATFORM_URL",
    "std": "ASR_PLATFORM_URL",
    "compat": "ASR_PLATFORM_URL",
    "sse": "ASR_PLATFORM_URL",
    "plat-simult": "ASR_PLATFORM_WS_URL",
    "plat-simult-ws": "ASR_PLATFORM_URL",
    "plat-minutes": "ASR_PLATFORM_URL",
    "plat-diar": "ASR_PLATFORM_URL",
    "plat-formula": "ASR_PLATFORM_URL",
    "ext-adv": "EXT_ADV_URL",
    "light": "LIGHT_URL",
    "sensevoice": "SENSEVOICE_URL",
    "xf-spark-slm-iat": "XF_IAT_URL",
    "gemma-audio": "GEMMA_URL",
    "gemma-text": "GEMMA_URL",
    "cascade-gemma": "GEMMA_URL",
}


def _service_base(url: str) -> str:
    """从内置接口展示 URL 还原服务根地址；WS adapter 自己再补协议和 path。"""
    raw = (url or "").strip()
    if not raw or "未配置" in raw:
        raise RuntimeError("接口地址未配置，请先在「接口管理 → 编辑」中填写 URL")
    if "://" not in raw:
        raw = "http://" + raw
    raw = raw.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
    for marker in ("/api/", "/ws/"):
        if marker in raw:
            raw = raw.split(marker, 1)[0]
            break
    return raw.rstrip("/")


_BUILTIN_SPECIAL_TESTS = {
    "plat-simult", "plat-simult-ws", "plat-minutes", "plat-formula", "plat-diar",
    "qwen-simult", "qwen-simult-offline", "doubao-simult", "xf-simult",
    "cascade-gemma",
}


def _builtin_test_cfg(iid: str, m: dict, override: dict | None = None):
    url = (m.get("url") or "").strip()
    cfg = None
    if iid in ("adv", "adv-domain", "std", "compat"):
        ep = iid
        base = url.split("/api/asr/", 1)[0] if "/api/asr/" in url else url.rsplit("/", 1)[0]
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": _norm_base(base), "template": "plat-multipart", "path": ep, "scenarios": ["ASR"]}
    elif iid == "sse":
        base = url.split("/api/asr/", 1)[0] if "/api/asr/" in url else url.rsplit("/", 1)[0]
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": _norm_base(base), "template": "plat-sse", "path": "", "scenarios": ["ASR"]}
    elif iid == "sensevoice":
        base = url.split("/api/asr_transcribe", 1)[0] if "/api/asr_transcribe" in url else url
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": _norm_base(base), "template": "sensevoice", "path": "", "scenarios": ["ASR"]}
    elif iid in ("ext-adv", "light", "light-mlt"):
        route = "/asr_mlt_nano" if iid == "light-mlt" else "/asr_lite"
        base = url.split(route, 1)[0] if route in url else url
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": _norm_base(base), "template": "asr_lite", "path": route, "scenarios": ["ASR"]}
    elif iid in ("gemma-audio",):
        base = url.split("/v1", 1)[0] if "/v1" in url else url
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": _norm_base(base), "template": "openai-audio", "model": "gemma-4-12B-it", "path": "/v1", "scenarios": ["ASR"]}
    elif iid == "gemma-text":
        base = url.split("/v1", 1)[0] if "/v1" in url else url
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": _norm_base(base), "template": "openai-chat", "model": "gemma-4-12B-it", "path": "/v1", "scenarios": ["翻译", "总结"]}
    elif iid == "xf-spark-slm-iat":
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": url, "template": "xf-spark-slm-iat",
               "scenarios": ["ASR"]}
    elif iid == "plat-realtime":
        # 流式 ASR WebSocket；模板走 adaper 内建连，非 _BUILTIN_SPECIAL_TESTS 的「同传」类
        base = url.split("/v1/realtime", 1)[0] if "/v1/realtime" in url else url
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": _norm_base(base),
               "template": "plat-realtime", "model": m.get("model", ""),
               "scenarios": ["ASR"]}
    elif iid == "qwen3-asr-ws":
        # 真流式 WS（基线）；与同传共用该端点 → 评测并发上限 2 路
        base = url.rstrip("/")
        if not base.endswith("/ws"):
            base = base + "/ws"
        cfg = {"id": iid, "name": m.get("name", iid), "base_url": base,
               "template": "qwen3-asr-ws", "model": m.get("model", ""),
               "max_workers": 2, "scenarios": ["ASR"]}
    elif iid in _BUILTIN_SPECIAL_TESTS:
        internal = iid.startswith("plat-")
        cfg = {"id": iid, "name": m.get("name", iid),
               "base_url": _service_base(url) if internal else url,
               "template": iid,
               "model": m.get("model", ""),
               "scenarios": (["翻译"] if iid == "cascade-gemma" else
                             ["总结"] if iid == "plat-minutes" else
                             ["其他"] if iid in ("plat-formula", "plat-diar") else ["同传"])}
    if cfg is None:
        raise RuntimeError(f"内置接口 {iid} 尚未登记测试策略")
    if m.get("key_env"):
        cfg["key_env"] = m["key_env"]
    for key in ("api_key", "key_env"):
        if override and override.get(key):
            cfg[key] = override[key]
    return cfg


def _with_scheme(u: str) -> str:
    """内置 MODELS 的 url 是免 scheme 的展示串；导出成 adapter base_url 前必须补上，否则 requests 报 MissingSchema。"""
    u = (u or "").strip()
    return u if (not u or "://" in u) else "http://" + u


def _iface_env_overrides():
    env = {}
    for c in _load_ifaces():
        if c.get("kind") != "builtin":
            continue
        key = _BUILTIN_ENV.get(c.get("id"))
        if key and c.get("url"):
            if key == "ASR_PLATFORM_URL":
                env[key] = _service_base(c["url"])
            elif key == "ASR_PLATFORM_WS_URL" and "/ws/audio/" in c["url"]:
                env[key] = c["url"].split("/ws/audio/", 1)[0]
            elif key == "SENSEVOICE_URL" and "/api/asr_transcribe" in c["url"]:
                env[key] = c["url"].split("/api/asr_transcribe", 1)[0]
            elif key in ("EXT_ADV_URL", "LIGHT_URL") and "/asr_lite" in c["url"]:
                env[key] = c["url"].split("/asr_lite", 1)[0]
            elif key == "GEMMA_URL" and "/v1/" in c["url"]:
                env[key] = c["url"].split("/v1/", 1)[0]
            else:
                env[key] = c["url"]
    return {k: _with_scheme(v) for k, v in env.items()}


_apply_builtin_overrides(_load_ifaces())
_register_iface_models(_load_ifaces())


class SniffReq(BaseModel):
    base_url: str
    api_key: str = ""


def _norm_base(u):
    u = u.strip().rstrip("/")
    return u if u.startswith(("http://", "https://")) else "http://" + u


_REQUEST_PARAM_LOCATIONS = {"form", "query", "json", "header"}
_REQUEST_PARAM_TYPES = {"string", "boolean", "integer", "number", "array", "object"}
_SENSITIVE_PARAM_RE = re.compile(
    r"(?:^|[_-])(authorization|auth|api[_-]?key|access[_-]?key|secret|token|password|cookie|credential|private[_-]?key|signature)(?:$|[_-])",
    re.I,
)
_TASK_MANAGED_PARAMS = {
    "language": "language",
    "target_lang": "target_lang",
    "target_language": "target_lang",
    "hot_words": "hotwords",
    "hotwords": "hotwords",
    "model": "interface",
    "messages": "task",
    "input": "task",
    "source_text": "task",
    "file": "audio",
    "audio": "audio",
    "stream": "interface",
    "response_format": "interface",
}


def _sensitive_param_name(name: str) -> bool:
    return bool(_SENSITIVE_PARAM_RE.search((name or "").strip()))


def _sanitize_request_schema(schema) -> list[dict]:
    """Validate the persisted request contract.

    Public input fields remain visible even when the manifest/profile manages
    them; only editable non-managed fields are accepted from request_params.
    Credentials remain interface-level secrets and are never contract fields.
    """
    if schema in (None, ""):
        return []
    if not isinstance(schema, list):
        raise ValueError("request_schema 必须是数组")
    if len(schema) > 64:
        raise ValueError("request_schema 最多登记 64 个参数")
    out, seen = [], set()
    for raw in schema:
        if not isinstance(raw, dict):
            raise ValueError("request_schema 每项必须是对象")
        name = str(raw.get("name") or "").strip()
        if not name or len(name) > 80 or not re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+", name):
            raise ValueError(f"非法参数名: {name or '(空)'}")
        if _sensitive_param_name(name):
            raise ValueError(f"敏感参数 {name} 不能登记为运行时参数；请使用接口密钥配置")
        if name in seen:
            raise ValueError(f"重复参数: {name}")
        seen.add(name)
        location = str(raw.get("in") or "form").lower()
        typ = str(raw.get("type") or "string").lower()
        if location not in _REQUEST_PARAM_LOCATIONS:
            raise ValueError(f"{name} 的位置 {location} 不受支持")
        if typ not in _REQUEST_PARAM_TYPES:
            typ = "string"
        item = {"name": name, "in": location, "type": typ}
        for key in ("title", "description", "format", "wire_name", "depends_on"):
            if raw.get(key) not in (None, ""):
                item[key] = str(raw[key])[:500]
        if "required" in raw:
            item["required"] = bool(raw["required"])
        if "default" in raw and raw["default"] is not None:
            item["default"] = raw["default"]
        for key in ("api_default", "eval_default"):
            if key in raw:
                item[key] = raw[key]
        if isinstance(raw.get("enum"), list):
            item["enum"] = raw["enum"][:100]
        for key in ("minimum", "maximum", "minLength", "maxLength"):
            if isinstance(raw.get(key), (int, float)):
                item[key] = raw[key]
        managed = _TASK_MANAGED_PARAMS.get(name) or raw.get("managed_by")
        if managed:
            item["managed_by"] = str(managed)
            item["editable"] = False
        elif raw.get("editable") is False:
            item["editable"] = False
        out.append(item)
    return out


def _resolve_openapi_schema(doc: dict, node) -> dict:
    if not isinstance(node, dict):
        return {}
    cur = dict(node)
    seen = set()
    while isinstance(cur.get("$ref"), str) and cur["$ref"].startswith("#/"):
        ref = cur.pop("$ref")
        if ref in seen:
            break
        seen.add(ref)
        target = doc
        try:
            for part in ref[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError):
            break
        cur = {**(target if isinstance(target, dict) else {}), **cur}
    if isinstance(cur.get("allOf"), list):
        merged = {k: v for k, v in cur.items() if k != "allOf"}
        props, required = {}, []
        for part in cur["allOf"]:
            resolved = _resolve_openapi_schema(doc, part)
            props.update(resolved.get("properties") or {})
            required.extend(resolved.get("required") or [])
            merged.update({k: v for k, v in resolved.items()
                           if k not in ("properties", "required")})
        if props:
            merged["properties"] = props
        if required:
            merged["required"] = list(dict.fromkeys(required))
        cur = merged
    for choice_key in ("anyOf", "oneOf"):
        choices = cur.get(choice_key)
        if isinstance(choices, list):
            choice = next(
                (_resolve_openapi_schema(doc, x) for x in choices
                 if _resolve_openapi_schema(doc, x).get("type") != "null"),
                {},
            )
            cur = {**choice, **{k: v for k, v in cur.items() if k != choice_key}}
            break
    return cur


def _openapi_field(doc: dict, name: str, location: str, schema: dict,
                   *, required=False, description="") -> dict | None:
    field_schema = _resolve_openapi_schema(doc, schema)
    if field_schema.get("format") == "binary":
        return None
    typ = field_schema.get("type")
    if isinstance(typ, list):
        typ = next((x for x in typ if x != "null"), "string")
    if not typ:
        typ = "string"
    field = {
        "name": name,
        "in": location,
        "type": typ,
        "required": bool(required),
    }
    for key in ("default", "enum", "minimum", "maximum", "minLength", "maxLength", "format"):
        if key in field_schema and field_schema[key] is not None:
            field[key] = field_schema[key]
    desc = description or field_schema.get("description") or field_schema.get("title")
    if desc:
        field["description"] = str(desc)
    if name in _TASK_MANAGED_PARAMS:
        field["managed_by"] = _TASK_MANAGED_PARAMS[name]
        field["editable"] = False
    return field


def _operation_template(path: str) -> tuple[str, str]:
    clean = path.rstrip("/")
    if "/api/asr/" in clean:
        endpoint = clean.rsplit("/", 1)[-1]
        if endpoint in ("adv", "std", "compat", "adv-domain"):
            return "plat-multipart", endpoint
        if endpoint == "sse":
            return "plat-sse", ""
        return "custom-http-asr", clean
    if clean.endswith("/asr_lite"):
        return "asr_lite", "/asr_lite"
    if clean.endswith("/asr_mlt_nano"):
        return "asr_lite", "/asr_mlt_nano"
    if clean.endswith("/api/asr_transcribe"):
        return "sensevoice", ""
    if clean.endswith("/audio/transcriptions"):
        return "openai-audio", clean[:-len("/audio/transcriptions")] or "/v1"
    if clean.endswith("/chat/completions"):
        return "openai-chat", clean[:-len("/chat/completions")] or "/v1"
    return "custom-http-asr", clean


def _openapi_operations(doc: dict) -> list[dict]:
    """Convert OpenAPI operations into the small schema understood by the UI."""
    operations = []
    for path, path_item in (doc.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method in ("post", "put"):
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue
            fields, audio_field, body_type = [], "", ""
            parameters = list(path_item.get("parameters") or []) + list(op.get("parameters") or [])
            for raw_param in parameters:
                param = _resolve_openapi_schema(doc, raw_param)
                name = str(param.get("name") or "")
                location = str(param.get("in") or "query").lower()
                if not name or location not in ("query", "header") or _sensitive_param_name(name):
                    continue
                field = _openapi_field(
                    doc, name, location, param.get("schema") or {},
                    required=param.get("required", False),
                    description=param.get("description") or "",
                )
                if field:
                    fields.append(field)
            request_body = _resolve_openapi_schema(doc, op.get("requestBody") or {})
            content = request_body.get("content") or {}
            media_type = next(
                (x for x in ("multipart/form-data", "application/x-www-form-urlencoded",
                             "application/json") if x in content),
                "",
            )
            if media_type:
                body_type = "multipart" if media_type == "multipart/form-data" else (
                    "json_base64" if media_type == "application/json" else "form"
                )
                location = "json" if media_type == "application/json" else "form"
                body_schema = _resolve_openapi_schema(doc, content[media_type].get("schema") or {})
                required = set(body_schema.get("required") or [])
                for name, raw_schema in (body_schema.get("properties") or {}).items():
                    resolved = _resolve_openapi_schema(doc, raw_schema)
                    if resolved.get("format") == "binary":
                        audio_field = audio_field or name
                        continue
                    if _sensitive_param_name(name):
                        continue
                    field = _openapi_field(
                        doc, name, location, resolved, required=name in required,
                    )
                    if field:
                        fields.append(field)
            template, template_path = _operation_template(path)
            deduped = {}
            for field in fields:
                deduped[field.get("name")] = field
            try:
                fields = _sanitize_request_schema(list(deduped.values()))
            except ValueError:
                fields = []
            operations.append({
                "path": path,
                "method": method.upper(),
                "summary": op.get("summary") or op.get("operationId") or "",
                "template": template,
                "template_path": template_path,
                "body_type": body_type or "multipart",
                "audio_field": audio_field,
                "request_schema": fields,
            })
    return operations


def _coerce_request_param(field: dict, value):
    typ, name = field.get("type"), field["name"]
    try:
        if typ == "boolean":
            if isinstance(value, bool):
                out = value
            elif str(value).strip().lower() in ("true", "1", "yes", "on"):
                out = True
            elif str(value).strip().lower() in ("false", "0", "no", "off"):
                out = False
            else:
                raise ValueError
        elif typ == "integer":
            out = int(value)
        elif typ == "number":
            out = float(value)
        elif typ in ("array", "object"):
            out = value if isinstance(value, (list, dict)) else json.loads(value)
            if typ == "array" and not isinstance(out, list):
                raise ValueError
            if typ == "object" and not isinstance(out, dict):
                raise ValueError
        else:
            out = str(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError(f"参数 {name} 不是有效的 {typ}") from None
    if field.get("enum") and out not in field["enum"]:
        raise ValueError(f"参数 {name} 只能取: {', '.join(map(str, field['enum']))}")
    if isinstance(out, (int, float)):
        if field.get("minimum") is not None and out < field["minimum"]:
            raise ValueError(f"参数 {name} 不能小于 {field['minimum']}")
        if field.get("maximum") is not None and out > field["maximum"]:
            raise ValueError(f"参数 {name} 不能大于 {field['maximum']}")
    return out


def _normalize_runtime_request_params(schema, values) -> dict:
    clean_schema = _sanitize_request_schema(schema)
    if values in (None, ""):
        values = {}
    if not isinstance(values, dict):
        raise ValueError("request_params 必须是对象")
    if len(json.dumps(values, ensure_ascii=False)) > 16384:
        raise ValueError("request_params 过大（最多 16 KiB）")
    by_name = {field["name"]: field for field in clean_schema}
    unknown = sorted(set(values) - set(by_name))
    if unknown:
        raise ValueError("包含未登记参数: " + ", ".join(unknown))
    out = {}
    for name, field in by_name.items():
        if field.get("editable") is False or field.get("managed_by"):
            if name in values:
                raise ValueError(f"参数 {name} 由 {field.get('managed_by', '接口')} 自动管理")
            continue
        if name in values:
            raw = values[name]
        elif "default" in field:
            raw = field["default"]
        elif field.get("required"):
            raise ValueError(f"缺少必填参数: {name}")
        else:
            continue
        out[name] = _coerce_request_param(field, raw)
    return out


def _runtime_config(model: str, request_params=None, task_params=None) -> dict:
    custom = next((c for c in _load_ifaces()
                   if c.get("kind") != "builtin" and c.get("id") == model), None)
    cfg = custom or _find_model(model)
    if not cfg:
        if request_params:
            raise ValueError(f"{model} 未登记可配置请求参数")
        return {"hash": "", "request_params": {}, "request_schema": []}
    contract = request_contract_for(model, cfg)
    schema = _sanitize_request_schema(contract.get("request_schema") or [])
    params = _normalize_runtime_request_params(schema, request_params or {})
    if model in {"plat-simult", "plat-simult-ws"}:
        soft_max = params.get("vad_soft_max_duration_ms", 15000)
        hard_max = params.get("vad_hard_max_duration_ms", 30000)
        if hard_max < soft_max:
            raise ValueError("vad_hard_max_duration_ms 不能小于 vad_soft_max_duration_ms")
    if model == "plat-simult":
        pipeline_mode = params.get("pipeline_mode", "single_stage")
        if pipeline_mode != "multi_stage" and params.get("return_asr_text"):
            raise ValueError("return_asr_text 仅在 pipeline_mode=multi_stage 时可启用")
    snapshot = {
        key: cfg.get(key)
        for key in (
            "id", "base_url", "template", "model", "path", "method", "body_type",
            "audio_field", "audio_filename", "audio_mime", "response_type",
            "text_path", "fallback_paths", "error_path", "dual_result",
        )
        if cfg.get(key) not in (None, "")
    }
    def without_secrets(value):
        if isinstance(value, dict):
            return {
                str(k): without_secrets(v)
                for k, v in value.items()
                if not _sensitive_param_name(str(k))
            }
        if isinstance(value, list):
            return [without_secrets(v) for v in value]
        return value

    for key in ("body_json", "query_json", "headers_json"):
        raw = cfg.get(key)
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            snapshot[key] = without_secrets(parsed)
    semantic_schema_keys = {
        "name", "in", "type", "required", "default", "enum", "minimum", "maximum",
        "minLength", "maxLength", "managed_by", "editable", "wire_name", "api_default",
        "eval_default", "depends_on",
    }
    snapshot["request_schema"] = [
        {k: field[k] for k in field if k in semantic_schema_keys}
        for field in schema
    ]
    snapshot["request_params"] = params
    clean_task_params = {
        str(k): v for k, v in (task_params or {}).items()
        if v not in (None, "") and k in ("target_lang", "hotwords_text", "speech_eval")
    }
    if clean_task_params:
        snapshot["task_params"] = clean_task_params
    snapshot["request_contract"] = {
        key: contract[key] for key in (
            "source", "kind", "endpoint", "protocol", "profile_of", "fixed_request",
            "verified_version", "verified_at",
        ) if key in contract
    }
    if not custom and not params and not clean_task_params:
        return {"hash": "", "request_params": {}, "request_schema": schema}
    digest = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    return {"hash": digest, "request_params": params, "request_schema": schema}


@app.post("/api/interfaces/sniff")
def sniff_iface(req: SniffReq):
    """探测目标服务的协议类型：OpenAI 兼容(/models) → openapi.json 特征 → 放弃。
    带上表单里的 api_key（DashScope 等 /models 也要鉴权）；401/403 视为'兼容但要 key'。"""
    base = _norm_base(req.base_url)
    headers = {"Authorization": f"Bearer {req.api_key}"} if req.api_key else {}
    host = base.split("//", 1)[-1].lower()
    if "api.xiaomimimo.com" in host:
        return {"template": "openai-chat-audio", "path": "/v1",
                "models": ["mimo-v2.5-asr"],
                "hint": "识别为小米 MiMo —— Chat 音频转写（/v1/chat/completions + input_audio）"}
    # 域名指纹：命中已知厂商直接提示走「厂商」下拉预设（比逐个嗅探更准、还带 key_env）
    for prov, label in (("dashscope.aliyuncs", "阿里 DashScope"), ("api.openai.com", "OpenAI")):
        if prov in host:
            return {"template": "openai-audio", "path": "/v1", "models": [],
                    "hint": f"识别为 {label} —— 建议直接用上方「厂商」下拉选预设（密钥走环境变量，免手填）"}
    for prefix in ("/v1", "/api/v3", "/compatible-mode/v1"):
        try:
            r = requests.get(base + prefix + "/models", headers=headers, timeout=6)
            if r.status_code in (401, 403):
                return {"template": None, "path": prefix, "models": [],
                        "hint": f"OpenAI 兼容（{prefix}）但要鉴权(HTTP {r.status_code})——"
                                "在 Key 框填密钥后重新嗅探（vllm/sglang 的 --api-key）"}
            data = r.json().get("data") if r.ok else None
            if isinstance(data, list) and data:
                ids = [m.get("id", "") for m in data]
                chat_audio = [i for i in ids if "mimo" in i.lower() and "asr" in i.lower()]
                if chat_audio:
                    return {"template": "openai-chat-audio", "path": prefix,
                            "models": (chat_audio + [i for i in ids if i not in chat_audio])[:30],
                            "hint": f"OpenAI Chat 音频兼容（{prefix}），检测到 MiMo ASR model"}
                kw = ("asr", "speech", "audio", "voice", "whisper", "tele")
                asr = [i for i in ids if any(k in i.lower() for k in kw)]
                rest = [i for i in ids if i not in asr]
                hint = f"OpenAI 兼容（{prefix}），共 {len(ids)} 个 model"
                if asr:
                    hint += f"，疑似语音 {len(asr)} 个排在前"
                else:
                    # 没有一个像语音模型 → 大概率是文本 LLM(翻译/总结)，模板给 Chat
                    hint += "——未见语音类 model，已按文本 LLM 选 Chat 模板（翻译/总结）；若确是语音转写请手改"
                    return {"template": "openai-chat", "path": prefix,
                            "models": rest[:30], "hint": hint}
                try:  # 同一服务可能也挂平台接口，提示可手动切
                    op = requests.get(base + "/openapi.json", headers=headers, timeout=4)
                    if op.ok and "/api/asr/" in " ".join(op.json().get("paths", {}).keys()):
                        hint += " ｜ ⚠ 同时检测到平台接口：请手动把模板改成 plat-multipart、path 填 adv/std/compat"
                except Exception:
                    pass
                return {"template": "openai-audio", "path": prefix,
                        "models": (asr + rest)[:30], "hint": hint}
        except Exception:
            pass
    try:
        r = requests.get(base + "/openapi.json", headers=headers, timeout=6)
        if r.ok:
            doc = r.json()
            operations = _openapi_operations(doc)
            paths = " ".join(doc.get("paths", {}).keys())
            if "/asr_lite" in paths:
                selected = next((x for x in operations if x["template"] == "asr_lite"), {})
                return {"template": "asr_lite", "path": "", "models": [],
                        "hint": "/asr_lite 协议",
                        "operations": operations, "request_schema": selected.get("request_schema", [])}
            if any(x["template"] == "plat-multipart" for x in operations):
                selected = next((x for x in operations
                                 if x["template"] == "plat-multipart"
                                 and x["template_path"] == "adv"), None)
                selected = selected or next(
                    (x for x in operations if x["template"] == "plat-multipart"), {}
                )
                return {"template": "plat-multipart",
                        "path": selected.get("template_path") or "adv", "models": [],
                        "hint": "平台风格（请选择本次实际调用的端点）",
                        "operations": operations, "request_schema": selected.get("request_schema", [])}
            if any(x["template"] == "plat-sse" for x in operations):
                selected = next(x for x in operations if x["template"] == "plat-sse")
                return {"template": "plat-sse", "path": "", "models": [],
                        "hint": "平台风格 SSE",
                        "operations": operations, "request_schema": selected.get("request_schema", [])}
            if "asr_transcribe" in paths:
                selected = next((x for x in operations if x["template"] == "sensevoice"), {})
                return {"template": "sensevoice", "path": "", "models": [],
                        "hint": "SenseVoice 风格",
                        "operations": operations, "request_schema": selected.get("request_schema", [])}
            selected = operations[0] if operations else {}
            return {"template": selected.get("template"), "path": selected.get("template_path", ""),
                    "models": [], "hint": "openapi 可读；请选择要注册的具体操作",
                    "operations": operations, "request_schema": selected.get("request_schema", [])}
    except Exception:
        pass
    return {"template": None, "path": "", "models": [], "hint": "探测失败（无 /models 和 /openapi.json），手选模板"}


class AddIfaceReq(BaseModel):
    name: str
    base_url: str
    template: str
    model: str = ""
    api_key: str = ""
    key_env: str = ""          # 选了厂商预设时：密钥从该环境变量读（容器化），api_key 留空
    path: str = ""
    scenarios: list[str] = ["ASR"]
    enabled: bool = True
    prompt_tpl: str = ""       # openai-chat 翻译提示词模板({src}/{tgt})：MT 专用模型按官方模板喂
    dual_result: bool = False    # sse 类：结果里同时保留 raw + edited，排名默认走 edited
    language_spec: dict | None = None  # 厂商预设的语言参数帮助元数据；不绑定实际实验值
    itn: bool = False           # asr_lite 评测口径默认关闭 ITN
    return_timestamps: bool = False
    method: str = "POST"
    headers_json: str = ""
    body_json: str = ""
    query_json: str = ""
    body_type: str = "multipart"
    audio_field: str = "file"
    audio_filename: str = "audio.wav"
    audio_mime: str = "audio/wav"
    response_type: str = "json"
    text_path: str = ""
    fallback_paths: str = ""
    error_path: str = ""
    request_schema: list[dict] = Field(default_factory=list)  # OpenAPI/手工登记；仅这些字段可在运行页修改


# 固定冒烟样本（AISHELL-1 test，文本中性不引误会）；缺了才随机退回任一 aishell1 音频(无参考)
_SMOKE_WAV = "datasets/asr/aishell1_test/wav_extracted/aishell_cuts_test.00000000/BAC/BAC009S0913W0355-3450.wav"
_SMOKE_REF = "服务广大跑步爱好者"


def _smoke_sample():
    p = os.path.join(ROOT, _SMOKE_WAV)
    if os.path.exists(p):
        return p, _SMOKE_REF
    wavs = glob.glob(os.path.join(ROOT, "datasets/asr/aishell1_test/wav_extracted/**/*.wav"), recursive=True)
    if wavs:
        return wavs[0], ""
    raise RuntimeError("找不到冒烟样本音频(aishell1_test)")


def _configured_key(cfg: dict, default_env: str) -> str:
    if cfg.get("api_key"):
        return cfg["api_key"]
    if cfg.get("key_env"):
        return os.environ.get(cfg["key_env"], "")
    return os.environ.get(default_env, "")


def _ws_parts(url: str) -> tuple[str, str, str]:
    from urllib.parse import urlsplit
    raw = (url or "").strip()
    if "://" not in raw:
        raw = "wss://" + raw
    p = urlsplit(raw)
    base = f"{p.scheme}://{p.netloc}{p.path}"
    return base, p.hostname or "", p.path or "/"


def _builtin_smoke_adapter(cfg: dict, audio_out_dir: str | None = None):
    """专项内置接口不能套通用 HTTP 模板，按真实 adapter 构造轻量冒烟实例。"""
    from adapters import (
        CascadeLightGemmaAdapter,
        DoubaoSimultAdapter,
        QwenLiveTranslateAdapter,
        PlatformDiarAdapter,
        PlatformFormulaAdapter,
        PlatformMinutesAdapter,
        PlatformWSSimultAdapter,
        PlatformWSVoiceInputAdapter,
        XfSimultAdapter,
        XfSparkSlmIatAdapter,
    )
    template = cfg["template"]
    base = cfg.get("base_url", "")
    plat_key = _configured_key(cfg, "ASR_PLATFORM_TOKEN")
    if template == "plat-simult":
        return PlatformWSSimultAdapter(base_url=base, api_key=plat_key,
                                  audio_out_dir=audio_out_dir)
    if template == "plat-simult-ws":
        return PlatformWSVoiceInputAdapter(base_url=base, api_key=plat_key)
    if template == "plat-minutes":
        return PlatformMinutesAdapter(base_url=base, api_key=plat_key)
    if template == "plat-formula":
        return PlatformFormulaAdapter(base_url=base, api_key=plat_key)
    if template == "plat-diar":
        return PlatformDiarAdapter(base_url=base, api_key=plat_key)
    if template in ("qwen-simult", "qwen-simult-offline"):
        ws_url, _, _ = _ws_parts(base)
        return QwenLiveTranslateAdapter(
            base_url=ws_url,
            api_key=_configured_key(cfg, "DASHSCOPE_API_KEY"),
            model=cfg.get("model") or "qwen3.5-livetranslate-flash-realtime",
            pace=template != "qwen-simult-offline",
            audio_out_dir=audio_out_dir,
        )
    if template == "doubao-simult":
        ws_url, _, _ = _ws_parts(base)
        return DoubaoSimultAdapter(base_url=ws_url, audio_out_dir=audio_out_dir)
    if template in ("xf-simult", "xf-spark-slm-iat"):
        _, host, path = _ws_parts(base)
        klass = XfSimultAdapter if template == "xf-simult" else XfSparkSlmIatAdapter
        kwargs = {"host": host, "path": path}
        if template == "xf-simult":
            kwargs["audio_out_dir"] = audio_out_dir
        return klass(**kwargs)
    if template == "cascade-gemma":
        return CascadeLightGemmaAdapter()
    raise RuntimeError(f"专项测试策略 {template} 未实现")


def _iface_smoke(cfg, audio_out_dir: str | None = None):
    """返回 (结果, 冒烟元信息)：元信息说明测试发了什么，弹窗才能讲明白'识别结果'是转写不是告警。"""
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    template = cfg.get("template")
    if template in _BUILTIN_SPECIAL_TESTS or template == "xf-spark-slm-iat":
        ad = _builtin_smoke_adapter(cfg, audio_out_dir=audio_out_dir)
    else:
        from adapters import make_custom_adapter
        ad = make_custom_adapter(cfg)
    scs = cfg.get("scenarios") or ["ASR"]
    if template == "openai-chat":
        item = ({"task": "summarize", "source_text": "人工智能正在改变世界，机器学习让计算机能从数据中自主学习并不断进步。"}
                if "总结" in scs else
                {"task": "translate", "source_text": "今天天气很好，我们一起去公园散步吧。", "target_lang": "英文"})
        res = ad.generate(item)
        meta = {"mode": "text", "input": item["source_text"],
                "test_action": "发送一段源文本请接口处理",
                "output_label": "摘要" if item["task"] == "summarize" else "译文"}
    elif template == "plat-formula":
        item = {"task": "asr", "source_text": r"已知 $x^2+y^2=1$，求 $y$。"}
        res = ad.generate(item)
        meta = {"mode": "text", "input": item["source_text"],
                "test_action": "发送一个含 LaTeX 的公式文本",
                "output_label": "公式口语化结果"}
    else:
        wav, ref = _smoke_sample()
        if template in {"plat-simult", "plat-simult-ws", "qwen-simult",
                        "qwen-simult-offline", "doubao-simult", "xf-simult",
                        "cascade-gemma"}:
            res = ad.generate({"task": "translate", "audio_path": wav, "lang": "zh-en",
                               "target_lang": "英文", "ref_text": ""})
            meta = {"mode": "audio", "sample": os.path.basename(wav),
                    "test_action": "按中译英发送一条测试音频",
                    "output_label": "译文"}
        elif template == "plat-minutes":
            res = ad.generate({"task": "summarize", "audio_path": wav})
            meta = {"mode": "audio", "sample": os.path.basename(wav),
                    "test_action": "发送一条短音频验证会议纪要协议",
                    "output_label": "纪要"}
        elif template == "plat-diar":
            res = ad.transcribe(wav, language="auto")
            meta = {"mode": "audio", "sample": os.path.basename(wav),
                    "test_action": "发送一条测试音频并开启说话人分离",
                    "output_label": "说话人分段 JSON"}
        else:
            res = ad.transcribe(wav, language="auto")  # auto=评测口径默认；lite 端点只收 auto(传 zh 422)
            meta = {"mode": "audio", "sample": os.path.basename(wav), "ref_text": ref,
                    "test_action": "发送一条测试音频请接口转写",
                    "output_label": "识别结果"}
    if not res.ok or not (res.text or "").strip():
        raise RuntimeError(res.error or "返回空文本")
    return res, meta


_SMOKE_TTS_TEMPLATES = {
    "plat-simult", "qwen-simult", "qwen-simult-offline", "doubao-simult", "xf-simult",
}


def _smoke_audio_dir(iid: str, cfg: dict) -> str | None:
    """接口测试音频使用稳定目录；同一接口重复测试会覆盖同名样本，不无限累积。"""
    if cfg.get("template") not in _SMOKE_TTS_TEMPLATES:
        return None
    digest = hashlib.sha256(iid.encode("utf-8")).hexdigest()[:12]
    path = os.path.join(ROOT, "audio_out", "smoke", digest)
    os.makedirs(path, exist_ok=True)
    return path


def _smoke_audio_payload(res) -> dict:
    path = (getattr(res, "extra", None) or {}).get("tts_audio")
    if not path:
        return {"smoke_audio": None}
    real = os.path.realpath(path)
    allowed = os.path.realpath(os.path.join(ROOT, "audio_out"))
    if real != allowed and not real.startswith(allowed + os.sep):
        return {"smoke_audio": None}
    return {"smoke_audio": real}


def _iface_cfg_from_req(req: AddIfaceReq, iid: str):
    _base = req.base_url.rstrip("/").replace("http://", "").replace("https://", "")
    _p = (req.path or "").strip()
    if req.template == "plat-multipart":  # path=adv/std/compat 变体
        disp_url = f"{_base}/api/asr/{_p or 'adv'}"
    elif req.template in ("openai-audio", "openai-chat", "openai-chat-audio"):  # path=前缀(默认 /v1)
        disp_url = _base + (_p if _p.startswith("/") else "/" + _p if _p else "/v1")
    elif req.template == "custom-http-asr":
        disp_url = req.base_url.strip()
    else:
        disp_url = _base
    cfg = {"id": iid, "name": req.name or iid, "base_url": _norm_base(req.base_url),
           "template": req.template, "model": req.model, "api_key": req.api_key,
           "key_env": req.key_env, "path": req.path, "scenarios": req.scenarios or ["ASR"],
           "enabled": req.enabled, "url": disp_url,
           **({"dual_result": True} if req.dual_result else {}),
           **({"language_spec": req.language_spec} if req.language_spec else {}),
           **({"prompt_tpl": req.prompt_tpl} if req.prompt_tpl else {})}
    request_schema = _sanitize_request_schema(req.request_schema)
    if request_schema:
        cfg["request_schema"] = request_schema
        if not cfg.get("language_spec"):
            language = next((x for x in request_schema if x["name"] == "language"), None)
            if language:
                cfg["language_spec"] = {
                    "supported": True,
                    "default": language.get("default", "auto"),
                    "values": language.get("enum") or [],
                    "note": language.get("description")
                            or "由该端点 OpenAPI 登记；任务页的 language 实验参数会原样传入。",
                }
    if req.template == "asr_lite":
        cfg.update({"itn": req.itn, "return_timestamps": req.return_timestamps})
    if req.template == "custom-http-asr":
        cfg.update({"method": req.method or "POST", "headers_json": req.headers_json,
                    "body_json": req.body_json, "query_json": req.query_json,
                    "body_type": req.body_type or "multipart", "audio_field": req.audio_field or "file",
                    "audio_filename": req.audio_filename or "audio.wav",
                    "audio_mime": req.audio_mime or "audio/wav",
                    "response_type": req.response_type or "json", "text_path": req.text_path,
                    "fallback_paths": req.fallback_paths, "error_path": req.error_path})
    return cfg


@app.post("/api/interfaces/smoke")
def smoke_iface(req: AddIfaceReq):
    """只冒烟不注册：给前端先展示返回文本，确认协议字段后再保存。"""
    if req.key_env and not req.api_key and not os.environ.get(req.key_env):
        return JSONResponse({"error": f"环境变量 {req.key_env} 未设置"}, status_code=400)
    try:
        cfg = _iface_cfg_from_req(req, "x-smoke")
        res, sm = _iface_smoke(cfg)
        return {"ok": True, "smoke_text": res.text[:500], "smoke_s": round(res.elapsed_s, 2), **sm}
    except Exception as e:
        return JSONResponse({"error": f"冒烟异常: {e}"}, status_code=400)


@app.post("/api/interfaces")
def add_iface(req: AddIfaceReq):
    """注册自定义接口：先用内置样本音频冒烟（解析出非空文本才算通过）→ 持久化。"""
    import re as _re
    iid = "x-" + _re.sub(r"[^a-z0-9]+", "-", req.name.lower()).strip("-")[:24]
    if any(m["id"] == iid for m in MODELS):
        return JSONResponse({"error": f"id {iid} 已存在，换个名称"}, status_code=400)
    try:
        cfg = _iface_cfg_from_req(req, iid)
    except ValueError as e:
        return JSONResponse({"error": f"参数能力登记失败: {e}"}, status_code=400)
    if req.key_env and not req.api_key and not os.environ.get(req.key_env):
        return JSONResponse({"error": f"环境变量 {req.key_env} 未设置——容器部署请配好该 env，"
                                      f"或在 Key 框临时填明文密钥后重试"}, status_code=400)
    # 冒烟：内置一条 aishell 样本
    try:
        res, sm = _iface_smoke(cfg)
    except Exception as e:
        return JSONResponse({"error": f"冒烟异常: {e}"}, status_code=400)
    cfgs = _load_ifaces()
    cfgs.append(cfg)
    _save_ifaces(cfgs)
    _register_iface_models([cfg])
    return {"id": iid, "smoke_text": res.text[:120], "smoke_s": round(res.elapsed_s, 2), **sm}


class EditIfaceReq(BaseModel):
    name: str = ""
    base_url: str = ""
    enabled: bool = True
    api_key: str = ""
    key_env: str = ""
    caps: list[str] | None = None   # 能力声明（仅自定义接口；内置由代码派生只读）
    path: str | None = None
    request_schema: list[dict] | None = None


@app.put("/api/interfaces/{iid}")
def edit_iface(iid: str, req: EditIfaceReq):
    cfgs = _load_ifaces()
    m = _find_model(iid)
    if not m:
        return JSONResponse({"error": "not found"}, status_code=404)
    builtin = iid in _BUILTIN_MODELS
    if builtin:
        cfg = next((c for c in cfgs if c.get("kind") == "builtin" and c.get("id") == iid), {"kind": "builtin", "id": iid})
        cfg["name"] = req.name or m.get("name", "")
        cfg["url"] = (req.base_url or m.get("url", "")).strip()
        cfg["enabled"] = bool(req.enabled)
        cfg["api_key"] = req.api_key or cfg.get("api_key", "")
        cfg["key_env"] = req.key_env or cfg.get("key_env", "")
        cfg["endpoint"] = cfg["url"]
        cfgs = [c for c in cfgs if not (c.get("kind") == "builtin" and c.get("id") == iid)] + [cfg]
        _save_ifaces(cfgs)
        m["name"], m["url"], m["endpoint"], m["enabled"] = cfg["name"], cfg["url"], cfg["url"], cfg["enabled"]
        return {"ok": True, "builtin": True}
    cfg = next((c for c in cfgs if c.get("id") == iid), None)
    if not cfg:
        return JSONResponse({"error": "not found"}, status_code=404)
    cfg["name"] = req.name or cfg.get("name", iid)
    raw_base = req.base_url or cfg.get("base_url", "")
    old_base = cfg.get("base_url", "")
    old_path = cfg.get("path", "")
    cfg["base_url"] = _norm_base(_custom_iface_base(cfg, raw_base)) if raw_base else ""
    if req.path is not None:
        cfg["path"] = req.path
    endpoint_changed = bool(
        (old_base and cfg["base_url"] != old_base)
        or (req.path is not None and cfg.get("path", "") != old_path)
    )
    schema_reset = bool(endpoint_changed and cfg.get("request_schema")
                        and req.request_schema is None)
    if req.request_schema is not None:
        try:
            cfg["request_schema"] = _sanitize_request_schema(req.request_schema)
        except ValueError as e:
            return JSONResponse({"error": f"参数能力登记失败: {e}"}, status_code=400)
    elif schema_reset:
        # 端点换了不能继续声称支持旧 OpenAPI 字段；重新嗅探/登记前不再生成旧表单。
        cfg["request_schema"] = []
    cfg["url"] = _custom_iface_display_url(cfg)
    cfg["enabled"] = bool(req.enabled)
    cfg["api_key"] = req.api_key or cfg.get("api_key", "")
    cfg["key_env"] = req.key_env or cfg.get("key_env", "")
    if req.caps is not None:   # 能力声明：勾了热词的 openai-audio 等接口，评测时才会真的传 hot_words
        cfg["caps"] = req.caps
        m["caps"] = req.caps
    _save_ifaces(cfgs)
    m["name"], m["url"], m["endpoint"], m["base_url"], m["enabled"] = (
        cfg["name"], cfg["url"], cfg.get("base_url", ""), cfg.get("base_url", ""), cfg["enabled"]
    )
    m["path"] = cfg.get("path", "")
    m["request_schema"] = cfg.get("request_schema") or []
    return {"ok": True, "builtin": False, "schema_reset": schema_reset}


@app.post("/api/interfaces/{iid}/reset")
def reset_iface(iid: str):
    if iid not in _BUILTIN_MODELS:
        return JSONResponse({"error": "only builtin supports reset"}, status_code=400)
    cfgs = [c for c in _load_ifaces() if not (c.get("kind") == "builtin" and c.get("id") == iid)]
    _save_ifaces(cfgs)
    src, cur = _BUILTIN_MODELS[iid], _find_model(iid)
    if cur:
        for k in ("name", "url", "endpoint"):
            cur[k] = src.get(k, cur.get(k))
        cur["enabled"] = True
    return {"ok": True}


@app.post("/api/interfaces/{iid}/test")
def test_iface(iid: str):
    m = _find_model(iid)
    if not m:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        if iid in _BUILTIN_MODELS:
            override = next((c for c in _load_ifaces()
                             if c.get("kind") == "builtin" and c.get("id") == iid), None)
            cfg = _builtin_test_cfg(iid, m, override)
            audio_out_dir = _smoke_audio_dir(iid, cfg)
            res, sm = _iface_smoke(cfg, audio_out_dir=audio_out_dir)
            return {"ok": True, "smoke_text": res.text[:120], "smoke_s": round(res.elapsed_s, 2),
                    "tts_requested": bool(audio_out_dir), **sm, **_smoke_audio_payload(res)}
        cfg = next((c for c in _load_ifaces() if c.get("id") == iid), None)
        if not cfg:
            return JSONResponse({"error": "not found"}, status_code=404)
        audio_out_dir = _smoke_audio_dir(iid, cfg)
        res, sm = _iface_smoke(cfg, audio_out_dir=audio_out_dir)
        return {"ok": True, "smoke_text": res.text[:120], "smoke_s": round(res.elapsed_s, 2),
                "tts_requested": bool(audio_out_dir), **sm, **_smoke_audio_payload(res)}
    except Exception as e:
        return JSONResponse({"error": f"测试失败: {e}"}, status_code=400)


@app.delete("/api/interfaces/{iid}")
def del_iface(iid: str):
    cfgs = [c for c in _load_ifaces() if c["id"] != iid]
    _save_ifaces(cfgs)
    MODELS[:] = [m for m in MODELS if m["id"] != iid]
    for sc in SCENARIO_MODELS:
        SCENARIO_MODELS[sc] = [x for x in SCENARIO_MODELS[sc] if x != iid]
    return {"ok": True}


# ── 模型评测记录的计数/清理（任意模型都可清理记录；删接口条目仅自定义见上）──
def _result_model(path, summary):
    """结果文件归属哪个模型：优先内部 meta.model；老格式按已知 dataset/model 解析。"""
    m = (summary.get("meta") or {}).get("model")
    if m:
        return m
    base = os.path.splitext(os.path.basename(path))[0]
    ds_ids = sorted((d["id"] for d in DATASETS), key=len, reverse=True)
    # 已下架接口仍要能认领各自的历史结果文件名
    model_ids = sorted((*(m["id"] for m in MODELS), *RETIRED_MODELS), key=len, reverse=True)
    for ds_id in ds_ids:
        prefix = ds_id + "_"
        if not base.startswith(prefix):
            continue
        rest = base[len(prefix):]
        for suffix in ("_full", "_lite"):
            idx = rest.find(suffix)
            if idx > 0:
                cand = rest[:idx]
                if cand in model_ids:
                    return cand
        for model_id in model_ids:
            if rest == model_id or rest.startswith(model_id + "_"):
                return model_id
    return base.rsplit("_", 1)[-1]


def _model_files(model):
    """某模型的 (results 文件列表, infer 文件列表)。"""
    res = []
    for p in glob.glob(os.path.join(ROOT, "results", "*.json")):
        try:
            s = json.load(open(p, encoding="utf-8")).get("summary", {})
        except Exception:
            s = {}
        if _result_model(p, s) == model:
            res.append(p)
    inf = glob.glob(os.path.join(ROOT, "infer", f"{model}__*.jsonl"))
    return res, inf


@app.get("/api/model_records")
def model_records(model: str):
    """某模型有多少评测记录（results + infer 中间产物），删除前给前端确认用。"""
    res, inf = _model_files(model)
    return {"model": model, "results": len(res), "infer": len(inf),
            "result_files": [os.path.basename(p) for p in res]}


class PurgeReq(BaseModel):
    model: str


@app.post("/api/model_records/purge")
def purge_model_records(req: PurgeReq):
    """删除某模型的全部 results + infer 文件（不可恢复）。不动 MODELS 注册（内置模型仍在）。"""
    res, inf = _model_files(req.model)
    deleted = 0
    for p in res + inf:
        try:
            os.remove(p)
            deleted += 1
        except Exception:
            pass
    _purge_job_records(req.model)
    return {"model": req.model, "deleted": deleted, "results": len(res), "infer": len(inf)}


# ── 数据集预览：复用 builder 取前 N 条；音频经 /api/audio 限定目录回放 ──
PREVIEW_CACHE = {}


def _dataset_preview_item(row: dict, *, longform: bool) -> dict:
    source_text = row.get("source_text") or ""
    ref_text = row.get("ref_text") or ""
    segments = row.get("segments")
    segment_count = row.get("segment_count")
    if segment_count is None and isinstance(segments, list):
        segment_count = len(segments)
    return {
        "id": row.get("id"),
        "audio_path": row.get("audio_path"),
        "ref_text": ref_text if longform else ref_text[:400],
        "source_text": source_text if longform else source_text[:400],
        "source_chars": len(source_text),
        "ref_chars": len(ref_text),
        "segment_count": segment_count,
        "duration_ms": row.get("duration_ms"),
        "term_count": len(row.get("terms") or []),
        "reference_type": row.get("reference_type"),
        "task": row.get("task") or "asr",
        "lang": row.get("lang"),
        "longform": longform,
    }


@app.get("/api/dataset_preview")
def dataset_preview(id: str, n: int = 8):
    ds = next((d for d in DATASETS if d["id"] == id), None)
    if not ds or not ds.get("runnable"):
        return JSONResponse({"error": "该数据集未接 builder，暂不可预览"}, status_code=400)
    key = (id, n)
    if key in PREVIEW_CACHE:
        return PREVIEW_CACHE[key]
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    try:
        import build_manifest as bm
        kw = {"subset": ds["subset"]} if ds.get("subset") else {}
        rows = bm.BUILDERS[ds.get("builder", id)](limit=n, **kw)
    except Exception as e:
        return JSONResponse({"error": f"builder 预览失败: {str(e)[:200]}"}, status_code=500)
    longform = bool(ds.get("longform"))
    items = [_dataset_preview_item(r, longform=longform) for r in rows[:n]]
    res = {
        "items": items,
        "longform": longform,
        "dataset": {
            "id": ds.get("id"), "name": ds.get("name"), "lang": ds.get("lang"),
            "scenario": ds.get("scenario"), "trait": ds.get("trait"),
        },
    }
    PREVIEW_CACHE[key] = res
    return res


@app.get("/api/manifest_audio")
def manifest_audio(manifest: str):
    """清单的 id→绝对音频路径映射（样本详情回放用；无音频的文本任务返回空）。"""
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from config import abs_path
    mp = os.path.join(ROOT, manifest)
    out = {}
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("audio_path"):
                out[d["id"]] = abs_path(d["audio_path"])
    return out


@app.get("/api/manifest_source")
def manifest_source(manifest: str):
    """清单的 id→原文映射（翻译/总结详情对照用：没有原文没法判断谁译得忠实）。截断口径与样本展示一致。"""
    mp = os.path.join(ROOT, manifest)
    out = {}
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("source_text"):
                out[d["id"]] = d["source_text"][:300]
    return out


@app.get("/api/manifest_longform_ids")
def manifest_longform_ids(manifest: str):
    """返回清单中的长音频样本 id，供结果详情显示专用时间轴入口。"""
    mp = os.path.realpath(os.path.join(ROOT, manifest))
    allowed = os.path.realpath(os.path.join(ROOT, "manifests"))
    if not mp.startswith(allowed + os.sep) or not os.path.exists(mp):
        return []
    out = []
    with open(mp, encoding="utf-8") as src:
        for line in src:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("longform"):
                out.append(str(row.get("id")))
    return out


@app.get("/api/audio")
def audio_file(path: str):
    """回放预览音频。路径白名单：只允许 datasets/ 和 audio_out/（防任意文件读取）。
    flac 转 wav 流出（FileResponse 按系统 mimetypes 猜出 audio/x-flac，浏览器 <audio> 不认）。"""
    real = os.path.realpath(path)
    allowed_roots = [os.path.realpath(os.path.join(ROOT, name))
                     for name in ("datasets", "audio_out")]
    if not any(real == allowed or real.startswith(allowed + os.sep)
               for allowed in allowed_roots):
        return JSONResponse({"error": "路径越界"}, status_code=403)
    if not os.path.exists(real):
        return JSONResponse({"error": "not found"}, status_code=404)
    if real.lower().endswith(".flac"):
        import io
        import soundfile as sf
        from fastapi.responses import Response
        data, sr = sf.read(real, dtype="int16")
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    return FileResponse(real)


def _lid_output_path(file: str):
    base = os.path.basename(file or "")
    if not base.endswith(".jsonl"):
        raise ValueError("LID 结果必须是 infer/*.jsonl")
    path = os.path.realpath(os.path.join(ROOT, "infer", base))
    allowed = os.path.realpath(os.path.join(ROOT, "infer"))
    if not path.startswith(allowed + os.sep) or not os.path.exists(path):
        raise FileNotFoundError("LID 结果不存在")
    return path


def _lid_dataset_payloads():
    datasets = []
    for ds in DATASETS:
        if not ds.get("runnable") or ds.get("metric") not in {"CER", "WER"}:
            continue
        available, reason = _dataset_available(ds)
        datasets.append({
            "id": ds["id"],
            "name": ds["name"],
            "scenario": ds["scenario"],
            "count": ds.get("count"),
            "available": available,
            "missing_reason": reason,
        })
    return datasets


def _normalize_lid_url(value: str):
    from urllib.parse import urlsplit

    url = (value or "").strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LID 服务地址必须是完整的 http:// 或 https:// URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LID 服务地址不能包含账号密码、query 或 fragment")
    return url


def _lid_experiment_slug(value: str):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "default").strip()).strip("-._")
    return (slug or "default")[:40]


def _lid_job_paths(ds_id: str, n: int, seed: int, endpoint: str, experiment: str):
    endpoint_hash = hashlib.sha256(endpoint.encode()).hexdigest()[:8]
    slug = _lid_experiment_slug(experiment)
    manifest = f"manifests/lid_{ds_id}_n{n}_s{seed}.jsonl"
    output = f"infer/fireredlid-{slug}-{endpoint_hash}__{ds_id}_n{n}_s{seed}.jsonl"
    return manifest, output


class LidRunReq(BaseModel):
    dataset: str
    limit: int = Field(default=100, ge=1, le=10000)
    seed: int = 42
    url: str = ""
    experiment: str = "default"


@app.get("/api/lid/config")
def lid_config():
    return {
        "datasets": _lid_dataset_payloads(),
        "default_url": os.environ.get("FIRERED_LID_URL", "").strip(),
        "defaults": {"limit": 100, "seed": 42, "experiment": "default"},
    }


def _run_lid_job(job_id: str, ds: dict, n: int, seed: int, endpoint: str, experiment: str):
    j = JOBS[job_id]
    j["stage"] = "排队中（同 LID 服务）"
    with _ep_lock("lid:" + endpoint):
        if j.get("cancel"):
            j["status"] = "cancelled"
            j["stage"] = "已取消（排队中）"
            j["finished_at"] = _now_iso()
            _flush_jobs()
            return
        j["started_at"] = _now_iso()
        j["started_ts"] = time.time()
        try:
            manifest, output = _lid_job_paths(ds["id"], n, seed, endpoint, experiment)
            j["result_file"] = os.path.basename(output)
            build = [sys.executable, "eval/build_manifest.py", ds.get("builder", ds["id"]),
                     "--out", manifest, "--shuffle", "--limit", str(n), "--seed", str(seed)]
            if ds.get("subset"):
                build += ["--subset", ds["subset"]]
            j["stage"] = "构建样本清单"
            with _mani_lock(manifest):
                if not os.path.exists(os.path.join(ROOT, manifest)):
                    subprocess.run(build, cwd=ROOT, check=True, capture_output=True, text=True)
            if j.get("cancel"):
                j["status"] = "cancelled"
                j["stage"] = "已取消"
                return

            command = [
                sys.executable, "eval/lid_infer.py",
                "--manifest", manifest,
                "--url", endpoint,
                "--out", output,
                "--timeout", "30",
                "--retries", "0",
                "--experiment", experiment,
            ]
            j["stage"] = "LID 推理"
            os.makedirs(JOBS_LOG_DIR, exist_ok=True)
            _prune_job_logs()
            j["log_file"] = j["id"]
            log_path = os.path.join(JOBS_LOG_DIR, j["id"] + ".log")
            tail = []
            with open(log_path, "w", encoding="utf-8") as jlog:
                jlog.write(logconf.redact(
                    f"# job {j['id']} FireRedLID×{ds['id']} experiment={experiment} @ {_now_iso()}\n"
                    f"# cmd: {' '.join(command)}\n\n"
                ))
                jlog.flush()
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors="replace", env={**os.environ, "ASR_LOG_FILE": "0"},
                    start_new_session=True,
                )
                PROCS[j["id"]] = process
                try:
                    for line in process.stdout:
                        safe_line = logconf.redact(line)
                        jlog.write(safe_line)
                        jlog.flush()
                        tail = (tail + [safe_line])[-40:]
                        match = re.search(r"(\d+)/(\d+)…", line)
                        if match:
                            j["progress"] = f"{match.group(1)}/{match.group(2)}"
                    process.wait()
                finally:
                    PROCS.pop(j["id"], None)

            j["log"] = "".join(tail)[-2300:]
            if j.get("cancel"):
                j["status"] = "cancelled"
                j["stage"] = "已停止"
            elif process.returncode != 0:
                j["status"] = "error"
                j["stage"] = "LID 运行失败"
            else:
                metas, samples = _read_lid_output(os.path.join(ROOT, output))
                j["summary"] = _lid_summary(metas, samples)
                j["progress"] = f"{len(samples)}/{n}"
                j["status"] = "done"
                j["stage"] = "完成"
        except subprocess.CalledProcessError as exc:
            j["status"] = "error"
            j["stage"] = "构建样本清单失败"
            j["log"] = logconf.redact(exc.stderr or str(exc))[-1500:]
        except Exception as exc:
            j["status"] = "error"
            j["stage"] = "LID 任务异常"
            j["log"] = logconf.redact(str(exc))[-1500:]
            LOG.exception("LID 任务异常 %s", job_id)
        finally:
            j["finished_at"] = _now_iso()
            if j.get("started_ts"):
                j["took"] = round(time.time() - j["started_ts"])
            _flush_jobs()


@app.post("/api/lid/run")
def start_lid_run(req: LidRunReq):
    ds = next((row for row in DATASETS if row["id"] == req.dataset), None)
    if not ds or not ds.get("runnable") or ds.get("metric") not in {"CER", "WER"}:
        return JSONResponse({"error": "该数据集不能用于 LID"}, status_code=400)
    available, reason = _dataset_available(ds)
    if not available:
        return JSONResponse({"error": reason or "该数据集尚未下载"}, status_code=400)
    try:
        endpoint = _normalize_lid_url(req.url or os.environ.get("FIRERED_LID_URL", ""))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    experiment = _lid_experiment_slug(req.experiment)
    key = ("lid", ds["id"], req.limit, req.seed, endpoint, experiment)
    duplicate = next((job for job in JOBS.values()
                      if job.get("status") == "running" and job.get("run_key") == key), None)
    if duplicate:
        return {"job_id": duplicate["id"], "dedup": True}

    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {
        "id": job_id,
        "task": "lid",
        "model": "FireRedLID",
        "dataset": ds["id"],
        "limit": req.limit,
        "n": req.limit,
        "seed": req.seed,
        "endpoint": endpoint,
        "experiment": experiment,
        "run_key": key,
        "status": "running",
        "stage": "排队中",
        "created_at": _now_iso(),
        "created_ts": time.time(),
        "origin": "dashboard",
    }
    _flush_jobs()
    threading.Thread(
        target=_run_lid_job,
        args=(job_id, ds, req.limit, req.seed, endpoint, experiment),
        daemon=True,
    ).start()
    return {"job_id": job_id}


def _read_lid_output(path: str):
    metas, records = [], {}
    with open(path, encoding="utf-8") as src:
        lines = [(lineno, line.strip()) for lineno, line in enumerate(src, 1) if line.strip()]
    for pos, (lineno, line) in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if pos == len(lines) - 1:
                LOG.warning("LID 读取 %s 时忽略尾行残缺 JSON(line %s)", path, lineno)
                break
            raise ValueError(f"{path}:{lineno} JSON 损坏") from exc
        sid = str(row.get("id") or "")
        if sid == "__meta__":
            metas.append(row)
        elif sid and (sid not in records or row.get("ok")):
            records[sid] = row
    return metas, list(records.values())


def _lid_label(row):
    if not row.get("ok"):
        return "__failed__"
    label = row.get("pred_label")
    if label:
        return str(label)
    parts = [row.get("pred_language"), row.get("pred_dialect")]
    return " ".join(str(x) for x in parts if x) or "(unknown)"


def _lid_summary(metas, samples):
    counts = {}
    confidences, rtfs = [], []
    for row in samples:
        if not row.get("ok"):
            continue
        label = _lid_label(row)
        counts[label] = counts.get(label, 0) + 1
        if isinstance(row.get("confidence"), (int, float)):
            confidences.append(float(row["confidence"]))
        if isinstance(row.get("service_rtf"), (int, float)):
            rtfs.append(float(row["service_rtf"]))
    n_ok = sum(counts.values())
    n_processed = len(samples)
    expected = (metas[-1].get("n_total") if metas else None) or n_processed
    labels = [{"label": label, "count": count,
               "rate": round(count / n_ok, 6) if n_ok else 0}
              for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return {
        "n_expected": expected,
        "n_processed": n_processed,
        "n_ok": n_ok,
        "n_fail": n_processed - n_ok,
        "coverage": round(n_ok / n_processed, 6) if n_processed else 0,
        "progress": round(n_processed / expected, 6) if expected else 0,
        "n_labels": len(labels),
        "confidence_mean": round(sum(confidences) / len(confidences), 6) if confidences else None,
        "rtf_mean": round(sum(rtfs) / len(rtfs), 6) if rtfs else None,
        "labels": labels,
    }


@app.get("/api/lid/results")
def lid_results():
    runs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "infer", "*.jsonl"))):
        try:
            metas, samples = _read_lid_output(path)
            if not metas or metas[-1].get("task") != "lid":
                continue
            runs.append({
                "file": os.path.basename(path),
                "manifest": metas[-1].get("manifest"),
                "endpoint": metas[-1].get("endpoint"),
                "experiment": metas[-1].get("experiment"),
                "started": metas[-1].get("started"),
                "mtime": round(os.path.getmtime(path)),
                "summary": _lid_summary(metas, samples),
            })
        except Exception as exc:
            LOG.warning("跳过不可读 LID 结果 %s: %s", path, str(exc)[:120])
    runs.sort(key=lambda row: row["mtime"], reverse=True)
    return {"runs": runs}


@app.get("/api/lid/result")
def lid_result(file: str, label: str = "", q: str = "", offset: int = 0, limit: int = 40):
    try:
        path = _lid_output_path(file)
        metas, samples = _read_lid_output(path)
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    if not metas or metas[-1].get("task") != "lid":
        return JSONResponse({"error": "不是 LID 输出"}, status_code=400)

    query = (q or "").strip().lower()
    filtered = []
    for row in samples:
        if label and _lid_label(row) != label:
            continue
        haystack = " ".join(str(row.get(key) or "") for key in
                            ("id", "pred_label", "pred_language", "pred_dialect",
                             "ref_language", "ref_dialect", "ref_text")).lower()
        if query and query not in haystack:
            continue
        item = dict(row)
        item["display_label"] = _lid_label(row)
        if item.get("audio_path"):
            item["audio_path"] = (item["audio_path"] if os.path.isabs(item["audio_path"])
                                  else os.path.join(ROOT, item["audio_path"]))
        filtered.append(item)

    offset = max(int(offset or 0), 0)
    limit = min(max(int(limit or 40), 1), 200)
    return {
        "file": os.path.basename(path),
        "meta": metas[-1],
        "summary": _lid_summary(metas, samples),
        "filter": {"label": label, "q": q, "total": len(filtered),
                   "offset": offset, "limit": limit},
        "samples": filtered[offset:offset + limit],
    }


@app.delete("/api/result")
def del_result(file: str):
    """删除单个结果文件（results/*.json）。infer 原始 hyp 保留，重跑可复用。"""
    p = os.path.join(ROOT, "results", os.path.basename(file))
    if os.path.exists(p):
        os.remove(p)
        _purge_job_records_for_result(os.path.basename(file))
        return {"ok": True}
    return JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/infer")
def del_infer(file: str):
    """删除一个明确命名的 infer JSONL；只允许 infer/ 直属文件。"""
    base = os.path.basename(file or "")
    if base != file or not base.endswith(".jsonl"):
        return JSONResponse({"error": "bad infer filename"}, status_code=400)
    path = os.path.realpath(os.path.join(ROOT, "infer", base))
    allowed = os.path.realpath(os.path.join(ROOT, "infer"))
    if not path.startswith(allowed + os.sep):
        return JSONResponse({"error": "路径越界"}, status_code=403)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    os.remove(path)
    return {"ok": True, "file": base}


@app.delete("/api/audio_artifacts")
def del_audio_artifacts(directory: str):
    """删除一个明确命名的 audio_out 子目录；不接受路径或通配符。"""
    base = os.path.basename(directory or "")
    if base != directory or not base:
        return JSONResponse({"error": "bad audio directory"}, status_code=400)
    path = os.path.realpath(os.path.join(ROOT, "audio_out", base))
    allowed = os.path.realpath(os.path.join(ROOT, "audio_out"))
    if not path.startswith(allowed + os.sep):
        return JSONResponse({"error": "路径越界"}, status_code=403)
    if not os.path.isdir(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    shutil.rmtree(path)
    return {"ok": True, "directory": base}


@app.get("/api/result")
def result(file: str):
    base = os.path.basename(file)
    p = os.path.join(ROOT, "results", base)
    if not os.path.exists(p):
        staged = sorted(glob.glob(os.path.join(ROOT, "results", base + ".__rerun_*")), key=os.path.getmtime, reverse=True)
        if staged:
            p = staged[0]
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.load(open(p, encoding="utf-8")))


# ── 单人样本复核：独立 JSON 草稿，完全旁路，不接入 score ──

_DATA_REVIEW_CHOICES = {
    "unreviewed", "ref_correct", "ref_incorrect", "audio_unscorable", "uncertain",
}
_OUTPUT_REVIEW_CHOICES = {
    "unreviewed", "content_correct", "recognition_error", "format_only",
    "hallucination", "empty_output", "uncertain",
}
_ISSUE_REVIEW_CHOICES = {
    "unreviewed", "recognition_error", "reference_error", "acceptable_variant",
    "normalization", "format_only", "uncertain",
}
_SCOPE_REVIEW_CHOICES = {
    "unreviewed", "all_recognition_error", "large_omission", "hallucination",
    "audio_mismatch", "empty_output", "uncertain",
}


class ReviewIssueReq(BaseModel):
    issue_id: str
    decision: str = "unreviewed"
    note: str = ""


class ReviewSaveReq(BaseModel):
    file: str
    sample_id: str
    data_decision: str = "unreviewed"
    corrected_ref: str | None = None
    # output_decision / format_tags 保留为 v1 API 兼容字段；新页面写 scope_decision + issues。
    output_decision: str = "unreviewed"
    format_tags: list[str] = Field(default_factory=list)
    scope_decision: str = "unreviewed"
    issues: list[ReviewIssueReq] = Field(default_factory=list)
    comment: str = ""


def _review_result(file: str):
    """只允许读 results/*.json；返回 (结果路径, 结果内容)。"""
    base = os.path.basename(file or "")
    if not base.endswith(".json"):
        raise ValueError("结果文件必须是 results/*.json")
    path = os.path.realpath(os.path.join(ROOT, "results", base))
    allowed = os.path.realpath(os.path.join(ROOT, "results"))
    if not path.startswith(allowed + os.sep) or not os.path.exists(path):
        raise FileNotFoundError("结果文件不存在")
    with open(path, encoding="utf-8") as src:
        return path, json.load(src)


def _review_manifest(result_data: dict):
    """解析结果绑定的 manifest，并防止清单改变后误审旧结果。"""
    summary = result_data.get("summary") or {}
    manifest = summary.get("manifest") or ""
    if not manifest:
        raise ValueError("旧结果缺少 manifest，无法安全复核")
    path = os.path.realpath(manifest if os.path.isabs(manifest) else os.path.join(ROOT, manifest))
    allowed = os.path.realpath(os.path.join(ROOT, "manifests"))
    if not path.startswith(allowed + os.sep) or not os.path.exists(path):
        raise FileNotFoundError("manifest 不存在或路径越界")
    current_sha = manifest_sha_v2(path)
    recorded_sha = ((summary.get("meta") or {}).get("manifest_sha_v2") or "").strip()
    if recorded_sha and recorded_sha != current_sha:
        raise RuntimeError(
            f"manifest_sha_v2 已改变（结果 {recorded_sha} / 当前 {current_sha}），拒绝误审"
        )
    return path, current_sha


def _review_infer_records(result_data: dict):
    """读取该结果绑定的 infer 原始 hyp，供格式审核；缺失时诚实降级。"""
    infer_file = ((result_data.get("summary") or {}).get("infer_file") or "").strip()
    if not infer_file:
        return {}
    path = os.path.realpath(infer_file if os.path.isabs(infer_file) else os.path.join(ROOT, infer_file))
    allowed = os.path.realpath(os.path.join(ROOT, "infer"))
    if not path.startswith(allowed + os.sep) or not os.path.exists(path):
        return {}
    records = {}
    with open(path, encoding="utf-8") as src:
        lines = [(lineno, line.strip()) for lineno, line in enumerate(src, 1) if line.strip()]
    for pos, (lineno, line) in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if pos == len(lines) - 1:
                LOG.warning("复核读取 %s 时忽略尾行残缺 JSON(line %s)", path, lineno)
                break
            LOG.warning("复核读取 %s 时遇到中间损坏 JSON(line %s)", path, lineno)
            return {}
        sid = str(row.get("id") or "")
        if not sid or sid == "__meta__":
            continue
        if sid not in records or row.get("ok"):
            records[sid] = row
    return records


def _longform_text_units(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"\n+|(?<=[。！？!?])|(?<=\.)(?=\s+[A-Z])", text)
    return [" ".join(part.split()) for part in parts if part and part.strip()]


def _longform_text_weight(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9]+|[\u3400-\u9fff]", str(text or ""))))


def _longform_balanced_groups(text: str, count: int) -> list[str]:
    """把无分段文档切成约 count 组；只用于展示，不回写评测数据。"""
    units = _longform_text_units(text)
    count = max(1, int(count or 1))
    if not units:
        return [""] * count
    groups, current, current_weight = [], [], 0
    remaining_weight = sum(_longform_text_weight(unit) for unit in units)
    for unit in units:
        slots = max(1, count - len(groups))
        target = remaining_weight / slots
        weight = _longform_text_weight(unit)
        if current and current_weight + weight > target and len(groups) < count - 1:
            groups.append(" ".join(current))
            remaining_weight -= current_weight
            current, current_weight = [], 0
        current.append(unit)
        current_weight += weight
    groups.append(" ".join(current))
    return groups + [""] * max(0, count - len(groups))


def _longform_source_timeline(row: dict, duration_s: float) -> tuple[list[dict], str]:
    raw_segments = row.get("segments") or []
    exact = bool(raw_segments) and all(
        segment.get("start_ms") is not None and segment.get("end_ms") is not None
        for segment in raw_segments
    )
    if raw_segments:
        timeline = []
        if exact:
            for index, segment in enumerate(raw_segments):
                timeline.append({
                    "index": index + 1,
                    "start_s": round(float(segment["start_ms"]) / 1000, 3),
                    "end_s": round(float(segment["end_ms"]) / 1000, 3),
                    "source_text": segment.get("source_text") or "",
                    "ref_text": segment.get("ref_text") or "",
                })
            return timeline, "exact"
        total_weight = sum(_longform_text_weight(s.get("source_text")) for s in raw_segments)
        cursor = 0
        for index, segment in enumerate(raw_segments):
            start_s = duration_s * cursor / total_weight
            cursor += _longform_text_weight(segment.get("source_text"))
            end_s = duration_s * cursor / total_weight
            timeline.append({
                "index": index + 1, "start_s": round(start_s, 3), "end_s": round(end_s, 3),
                "source_text": segment.get("source_text") or "",
                "ref_text": segment.get("ref_text") or "",
            })
        return timeline, "estimated"

    count = max(1, int((duration_s + 19.999) // 20))
    sources = _longform_balanced_groups(row.get("source_text") or "", count)
    refs = _longform_balanced_groups(row.get("ref_text") or "", count)
    return [{
        "index": index + 1,
        "start_s": round(index * duration_s / count, 3),
        "end_s": round((index + 1) * duration_s / count, 3),
        "source_text": sources[index], "ref_text": refs[index],
    } for index in range(count)], "estimated"


def _longform_translation_events(infer_row: dict, duration_s: float) -> tuple[list[dict], str]:
    saved = (infer_row.get("extra") or {}).get("translation_segments") or []
    events = []
    for segment in saved:
        try:
            end_s = float(segment.get("end_s"))
        except (TypeError, ValueError, AttributeError):
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            event = {"end_s": round(max(0.0, min(duration_s, end_s)), 3), "text": text}
            try:
                event["start_s"] = round(max(0.0, min(duration_s, float(segment.get("start_s")))), 3)
            except (TypeError, ValueError):
                pass
            try:
                event["emitted_at_s"] = round(max(0.0, float(segment.get("emitted_at_s"))), 3)
            except (TypeError, ValueError):
                pass
            for key in ("segment_id", "pipeline_mode"):
                if segment.get(key) is not None:
                    event[key] = segment[key]
            asr_text = str(segment.get("asr_text") or segment.get("original_text") or "").strip()
            if asr_text:
                event["asr_text"] = asr_text
            timing = segment.get("timing") or segment.get("timings")
            if isinstance(timing, dict):
                event["timing"] = timing
            events.append(event)
    if events:
        return sorted(events, key=lambda item: item["end_s"]), "event"

    units = _longform_text_units(infer_row.get("hyp") or "")
    total_weight = sum(_longform_text_weight(unit) for unit in units)
    cursor = 0
    for unit in units:
        cursor += _longform_text_weight(unit)
        events.append({"end_s": round(duration_s * cursor / max(1, total_weight), 3), "text": unit})
    return events, "estimated"


@app.get("/api/longform_sample")
def longform_sample(file: str, id: str):
    """组合 manifest 全文 + infer 原始输出，生成长音频专用时间轴；不使用 score 的 200 字摘要。"""
    try:
        _, result_data = _review_result(file)
        manifest_path, _ = _review_manifest(result_data)
        manifest_row = None
        with open(manifest_path, encoding="utf-8") as src:
            for line in src:
                row = json.loads(line)
                if str(row.get("id")) == str(id):
                    manifest_row = row
                    break
        if not manifest_row or not manifest_row.get("longform"):
            return JSONResponse({"error": "该样本不是长音频场次"}, status_code=404)
        infer_row = _review_infer_records(result_data).get(str(id)) or {}
        if not infer_row:
            return JSONResponse({"error": "未找到该场次的 infer 原始输出"}, status_code=404)
        duration_s = float(infer_row.get("audio_s")
                           or (infer_row.get("extra") or {}).get("src_dur_s")
                           or (manifest_row.get("duration_ms") or 0) / 1000 or 1)
        timeline, source_timing = _longform_source_timeline(manifest_row, duration_s)
        translation_events, translation_timing = _longform_translation_events(infer_row, duration_s)
        for segment in timeline:
            segment["hyp_parts"] = []
        for event in translation_events:
            at = event["end_s"]
            target = next((segment for segment in timeline
                           if segment["start_s"] <= at < segment["end_s"]), timeline[-1])
            target["hyp_parts"].append(event["text"])
        for segment in timeline:
            segment["hyp_text"] = " ".join(segment.pop("hyp_parts")).strip()

        result_sample = next((sample for sample in result_data.get("samples") or []
                              if str(sample.get("id")) == str(id)), {})
        audio_path = manifest_row.get("audio_path") or ""
        if audio_path and not os.path.isabs(audio_path):
            audio_path = os.path.join(ROOT, audio_path)
        summary = result_data.get("summary") or {}
        meta = summary.get("meta") or {}
        return {
            "id": str(id), "file": os.path.basename(file),
            "model": meta.get("model") or summary.get("model") or "",
            "lang": manifest_row.get("lang") or "",
            "duration_s": round(duration_s, 3), "audio_path": os.path.realpath(audio_path),
            "tts_audio": result_sample.get("tts_audio") or (infer_row.get("extra") or {}).get("tts_audio"),
            "reference_type": manifest_row.get("reference_type"),
            "source_timing": source_timing, "translation_timing": translation_timing,
            "manifest_segment_count": len(manifest_row.get("segments") or []),
            "model_segment_count": len(translation_events),
            "model_segments": translation_events if translation_timing == "event" else [],
            "metrics": {key: result_sample.get(key) for key in ("err_rate", "elapsed_s", "al_s")
                        if result_sample.get(key) is not None},
            "segments": timeline,
        }
    except (ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _review_draft_path(manifest_sha: str):
    if not re.fullmatch(r"[0-9a-f]{12}", manifest_sha or ""):
        raise ValueError("manifest_sha_v2 非法")
    return os.path.join(REVIEW_DRAFT_DIR, manifest_sha + ".json")


def _load_review_draft(manifest_path: str, manifest_sha: str):
    path = _review_draft_path(manifest_sha)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as src:
            data = json.load(src)
        if data.get("manifest_sha_v2") != manifest_sha or not isinstance(data.get("samples"), dict):
            raise ValueError("审核草稿结构损坏")
        return data
    return {
        "schema_version": 2,
        "manifest": os.path.relpath(manifest_path, ROOT),
        "manifest_sha_v2": manifest_sha,
        "reviewer": AUTH_USER,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "updated_at": None,
        "samples": {},
    }


def _diff_blocks(ref: str, hyp: str, basis: str):
    """把字符对齐的连续/相邻差异合并成「差异块」，避免每个字点一次。

    相隔不超过 1 个相同字符的编辑会自动合并；页面仍保留「整句批量判定」
    处理大段错误，不强迫审核者接受对齐算法的细碎切分。
    """
    ref, hyp = ref or "", hyp or ""
    opcodes = [op for op in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes()
               if op[0] != "equal"]
    merged = []
    for tag, i1, i2, j1, j2 in opcodes:
        if merged and i1 - merged[-1]["ref_end"] <= 1 and j1 - merged[-1]["hyp_end"] <= 1:
            prev = merged[-1]
            prev["ref_end"], prev["hyp_end"] = i2, j2
            prev["operation"] = prev["operation"] if prev["operation"] == tag else "mixed"
        else:
            merged.append({"operation": tag, "ref_start": i1, "ref_end": i2,
                           "hyp_start": j1, "hyp_end": j2})
    out = []
    for idx, block in enumerate(merged, 1):
        block = dict(block)
        block.update({
            "issue_id": f"{basis}:{block['ref_start']}:{block['ref_end']}:{block['hyp_start']}:{block['hyp_end']}",
            "basis": basis,
            "ref_text": ref[block["ref_start"]:block["ref_end"]],
            "hyp_text": hyp[block["hyp_start"]:block["hyp_end"]],
            "affects_original_cer": basis == "score",
            "large": max(block["ref_end"] - block["ref_start"],
                         block["hyp_end"] - block["hyp_start"]) >= 8,
            "index": idx,
        })
        out.append(block)
    return out


def _review_complete(record: dict, result_file: str, score_issue_ids=None):
    data_decision = (record.get("data_review") or {}).get("decision", "unreviewed")
    if data_decision == "unreviewed":
        return False
    output = ((record.get("output_reviews") or {}).get(result_file) or {})
    if output.get("scope_decision", "unreviewed") != "unreviewed":
        return True
    if score_issue_ids is None:  # 旧草稿/旧调用的兼容判定
        return output.get("decision", "unreviewed") != "unreviewed"
    if not score_issue_ids:  # 评分文本完全一致；raw 格式差异是可选诊断
        return True
    decisions = {issue.get("issue_id"): issue.get("decision", "unreviewed")
                 for issue in output.get("issues") or []}
    return all(decisions.get(issue_id, "unreviewed") != "unreviewed"
               for issue_id in score_issue_ids)


@app.get("/api/reviews/results")
def review_results():
    """按数据集→manifest 批次→模型返回当前可审核结果。

    默认隐藏 smoke、退役模型、缺 dataset/model/manifest 指纹的旧结果；
    同一 manifest + model 只留最新一份，避免文件名平铺刷屏。
    """
    active_models = {m.get("id") for m in MODELS}
    latest = {}
    sha_cache = {}
    for item in read_results():
        task = item.get("task") or ("asr" if item.get("metric") in ("CER", "WER") else "")
        meta = item.get("meta") or {}
        file = item.get("_file") or ""
        model, dataset, sha = item.get("model"), item.get("dataset"), meta.get("manifest_sha_v2")
        if (task != "asr" or not item.get("manifest") or not file or file.startswith("_")
                or not model or not dataset or model not in active_models
                or model in RETIRED_MODELS):
            continue
        if not sha:  # 旧结果只读计算当前 manifest 指纹，不回写历史 result。
            manifest_key = item["manifest"]
            if manifest_key not in sha_cache:
                try:
                    _, sha_cache[manifest_key] = _review_manifest({"summary": item})
                except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
                    sha_cache[manifest_key] = None
            sha = sha_cache[manifest_key]
        if not sha:
            continue
        row = {
            "file": file, "model": model, "dataset": dataset,
            "metric": item.get("metric"), "err_rate": item.get("err_rate"),
            "n_total": item.get("n_total"), "manifest": item.get("manifest"),
            "manifest_sha_v2": sha, "mtime": item.get("_mtime") or 0,
        }
        key = (sha, model)
        if key not in latest or row["mtime"] > latest[key]["mtime"]:
            latest[key] = row
    ds_names = {d["id"]: d.get("name", d["id"]) for d in DATASETS}
    groups = {}
    for row in latest.values():
        batch = groups.setdefault(row["manifest_sha_v2"], {
            "key": row["manifest_sha_v2"], "dataset": row["dataset"],
            "dataset_name": ds_names.get(row["dataset"], row["dataset"]),
            "manifest": row["manifest"], "n_total": row["n_total"], "models": [],
        })
        batch["models"].append(row)
    priority = {name: idx for idx, name in enumerate(("adv", "sse", "std", "light"))}
    batches = list(groups.values())
    for batch in batches:
        batch["models"].sort(key=lambda r: (priority.get(r["model"], 99), r["model"]))
    batches.sort(key=lambda b: (b["dataset_name"], os.path.basename(b["manifest"]), b["key"]))
    return {"batches": batches}


@app.get("/api/reviews/context")
def review_context(file: str):
    """结果样本 + 原始参考 + 音频 + 当前草稿。不修改任何评分文件。"""
    try:
        _, data = _review_result(file)
        manifest_path, sha = _review_manifest(data)
        with REVIEW_LOCK:
            draft = _load_review_draft(manifest_path, sha)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    manifest_rows = {}
    with open(manifest_path, encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            manifest_rows[str(row.get("id"))] = row
    infer_records = _review_infer_records(data)

    from config import abs_path
    result_file = os.path.basename(file)
    items = []
    exclusions = exclusions_for_rows(list(manifest_rows.values()), ROOT)
    for sample in data.get("samples") or []:
        sid = str(sample.get("id"))
        row = manifest_rows.get(sid, {})
        infer_record = infer_records.get(sid) or {}
        infer_extra = infer_record.get("extra") or {}
        audio_path = abs_path(row["audio_path"]) if row.get("audio_path") else None
        review = (draft.get("samples") or {}).get(sid) or {}
        score_diff = _diff_blocks(sample.get("ref_norm", ""), sample.get("hyp_norm", ""), "score")
        raw_diff = (_diff_blocks(row.get("ref_text", ""), infer_record.get("hyp") or "", "raw")
                    if infer_record.get("hyp") is not None else [])
        score_issue_ids = [block["issue_id"] for block in score_diff]
        large_error = bool((sample.get("err_rate") or 0) >= 0.5
                           or any(block["large"] for block in score_diff)
                           or len(score_diff) >= 6)
        items.append({
            "id": sid,
            "audio_path": audio_path,
            "ref_text": row.get("ref_text", ""),
            "ref_norm": sample.get("ref_norm", ""),
            "hyp_norm": sample.get("hyp_norm", ""),
            "hyp_raw": infer_record.get("hyp"),
            "edited_text": infer_extra.get("edited_text"),
            "raw": sample.get("raw"),
            "err_rate": sample.get("err_rate"),
            "sub": sample.get("sub"), "del": sample.get("del"), "ins": sample.get("ins"),
            "error": sample.get("error"),
            "score_diff": score_diff,
            "raw_diff": raw_diff,
            "large_error_suggested": large_error,
            "review": review,
            "review_complete": _review_complete(review, result_file, score_issue_ids),
            "excluded_from_reviewed": sid in exclusions,
            "exclusion_reason": (exclusions.get(sid) or {}).get("reason"),
        })
    items.sort(key=lambda x: (x.get("error") is None, x.get("err_rate") or 0), reverse=True)
    reviewed = sum(1 for item in items if item["review_complete"])
    summary = data.get("summary") or {}
    reviewed_metric = reviewed_asr_metrics(summary, data.get("samples") or [], exclusions)
    return {
        "result_file": result_file,
        "manifest": os.path.relpath(manifest_path, ROOT),
        "manifest_sha_v2": sha,
        "reviewer": draft.get("reviewer") or AUTH_USER,
        "original_metric": summary.get("metric"),
        "original_err_rate": summary.get("err_rate"),
        "reviewed": reviewed_metric,
        "counts": {"total": len(items), "reviewed": reviewed, "remaining": len(items) - reviewed},
        "samples": items,
    }


@app.put("/api/reviews/sample")
def save_review(req: ReviewSaveReq):
    """保存一条审核草稿。读→改→原子替换在同一把锁内，防止多标签页互相覆盖。"""
    if req.data_decision not in _DATA_REVIEW_CHOICES:
        return JSONResponse({"error": "data_decision 非法"}, status_code=400)
    if req.output_decision not in _OUTPUT_REVIEW_CHOICES:
        return JSONResponse({"error": "output_decision 非法"}, status_code=400)
    if req.scope_decision not in _SCOPE_REVIEW_CHOICES:
        return JSONResponse({"error": "scope_decision 非法"}, status_code=400)
    if any(issue.decision not in _ISSUE_REVIEW_CHOICES for issue in req.issues):
        return JSONResponse({"error": "issue decision 非法"}, status_code=400)
    if req.data_decision == "ref_incorrect" and not (req.corrected_ref or "").strip():
        return JSONResponse({"error": "标注错误时必须填写修订参考"}, status_code=400)
    try:
        _, data = _review_result(req.file)
        manifest_path, sha = _review_manifest(data)
        sample = next((s for s in data.get("samples") or [] if str(s.get("id")) == req.sample_id), None)
        if sample is None:
            raise ValueError("样本不属于该结果")
        manifest_row = None
        with open(manifest_path, encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    row = json.loads(line)
                    if str(row.get("id")) == req.sample_id:
                        manifest_row = row
                        break
        manifest_row = manifest_row or {}
        infer_record = _review_infer_records(data).get(req.sample_id) or {}
        blocks = _diff_blocks(sample.get("ref_norm", ""), sample.get("hyp_norm", ""), "score")
        if infer_record.get("hyp") is not None:
            blocks += _diff_blocks(manifest_row.get("ref_text", ""), infer_record.get("hyp") or "", "raw")
        block_by_id = {block["issue_id"]: block for block in blocks}
        if len({issue.issue_id for issue in req.issues}) != len(req.issues):
            raise ValueError("差异点 issue_id 重复")
        unknown = [issue.issue_id for issue in req.issues if issue.issue_id not in block_by_id]
        if unknown:
            raise ValueError("差异点已过期或不属于当前文本")
        stored_issues = []
        for issue in req.issues:
            stored = dict(block_by_id[issue.issue_id])
            stored["decision"] = issue.decision
            stored["note"] = issue.note.strip()[:500]
            stored_issues.append(stored)
        score_issue_ids = [block["issue_id"] for block in blocks if block["basis"] == "score"]
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        result_file = os.path.basename(req.file)
        with REVIEW_LOCK:
            draft = _load_review_draft(manifest_path, sha)
            draft["schema_version"] = 2
            record = draft["samples"].setdefault(req.sample_id, {"output_reviews": {}})
            record["data_review"] = {
                "decision": req.data_decision,
                "corrected_ref": ((req.corrected_ref or "").strip() or None)
                if req.data_decision == "ref_incorrect" else None,
            }
            previous_output = (record.setdefault("output_reviews", {}).get(result_file) or {})
            record["output_reviews"][result_file] = {
                "schema_version": 2,
                "scope_decision": req.scope_decision,
                "issues": stored_issues,
                # 保留旧整句结论供追溯，新完成判定不依赖它。
                "legacy_decision": previous_output.get("decision") or (
                    req.output_decision if req.output_decision != "unreviewed" else None),
                "format_tags": sorted({str(x).strip()[:80] for x in req.format_tags if str(x).strip()}),
            }
            record["comment"] = req.comment.strip()[:2000]
            record["reviewer"] = AUTH_USER
            record["updated_at"] = now
            record["status"] = ("reviewed" if _review_complete(record, result_file, score_issue_ids)
                                else "draft")
            draft["updated_at"] = now
            os.makedirs(REVIEW_DRAFT_DIR, exist_ok=True)
            atomic_write_json(_review_draft_path(sha), draft, indent=2)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"ok": True, "manifest_sha_v2": sha, "sample_id": req.sample_id,
            "status": record["status"], "review": record}


@app.get("/api/doctor")
def doctor():
    """环境自检：端点探活 + 可选依赖可导入 + 外部二进制 + 数据集可用性。跑评测前一键体检，端点悄悄死掉第一时间看见。"""
    import shutil
    import socket as _socket
    eps, seen = [], set()
    for m in MODELS:
        u = (m.get("url") or "").strip()
        hostport = u.split("://")[-1].split("/")[0]
        if ":" not in hostport or hostport in seen:
            continue
        seen.add(hostport)
        h, _, p = hostport.rpartition(":")
        try:
            s = _socket.create_connection((h, int(p)), timeout=3)
            s.close()
            ok, err = True, ""
        except Exception as e:
            ok, err = False, str(e)[:80]
        eps.append({"host": hostport, "ok": ok, "err": err,
                    "models": [x["id"] for x in MODELS if hostport in (x.get("url") or "")]})
    scorer = scorer_health(timeout=5.0)
    scorer_ok = bool(scorer and scorer.get("ok"))
    _DEPS = [
        ("websocket-client(同传)", "websocket", True),
        ("protobuf(豆包同传)", "google.protobuf", True),
        ("GPU scorer / faster-whisper(同传tts_err)", "faster_whisper", True),
        ("pyannote.metrics(DER)", "pyannote.metrics", True),
        ("GPU XCOMET-XL / unbabel-comet(可选重型)", "comet", False),
        ("OmniSTEval(LongYAAL 同传)", "omnisteval", False),
        ("jiwer", "jiwer", True),
        ("sacrebleu", "sacrebleu", True),
        ("opencc", "opencc", True),
        ("whisper-normalizer", "whisper_normalizer", True),
        ("soundfile", "soundfile", True),
        ("scipy(重采样)", "scipy", True),
    ]
    deps = []
    for label, mod, required in _DEPS:
        found = ((scorer_ok or comet_runtime_available()) if mod == "comet" else
                 (scorer_ok or _module_available(mod)) if mod == "faster_whisper" else
                 omnisteval_runtime_available() if mod == "omnisteval" else
                 _module_available(mod))
        deps.append({"name": label, "ok": found, "required": required,
                     "severity": "error" if required else "warning",
                     "reason": "" if found else f"无法导入 {mod}"})
    ffmpeg_ok = bool(shutil.which("ffmpeg"))
    speex_ok = _speex_available()
    bins = [{"name": "ffmpeg(mp3/m4a 解码，CoVoST2 等)", "ok": ffmpeg_ok,
             "required": True, "severity": "error",
             "reason": "" if ffmpeg_ok else "PATH 中未找到 ffmpeg"},
            {"name": "libspeex1(仅讯飞同传 --save-audio)", "ok": speex_ok,
             "required": False, "severity": "warning",
             "reason": "" if speex_ok else "未找到可加载的 libspeex"}]
    ds_missing = []
    for d in DATASETS:
        if not d.get("runnable"):
            continue
        try:
            ok, reason = _dataset_available(d)
        except Exception as e:
            ok, reason = False, str(e)[:60]
        if not ok:
            ds_missing.append({"id": d["id"], "reason": reason})
    scorer_public = None
    if scorer is not None:
        scorer_public = {key: scorer.get(key) for key in (
            "ok", "cuda", "gpu", "compute_capability", "gpu_memory_total_gib",
            "version", "torch", "packages", "busy_metric", "resident_model",
            "last_error"
        )}
    return {"endpoints": eps, "deps": deps, "bins": bins, "scorer": scorer_public,
            "datasets": {"runnable": sum(1 for d in DATASETS if d.get("runnable")), "missing": ds_missing}}


class ScorerSmokeReq(BaseModel):
    metric: str = "whisper"


@app.post("/api/scorer_warmup")
def scorer_warmup_api(req: ScorerSmokeReq):
    metric = (req.metric or "").strip().lower()
    if metric not in {"whisper", "utmos", "speaker", "xcomet"}:
        return JSONResponse({"error": "metric 仅支持 whisper/utmos/speaker/xcomet"}, status_code=422)
    result = scorer_warmup(metric)
    if result is None:
        return JSONResponse({"error": f"{metric} scorer 预热启动失败",
                             "detail": scorer_last_error()}, status_code=502)
    return result


def _scorer_smoke_audio():
    """从已构建清单选一条共享卷音频；不接受调用方传路径。"""
    manifests = sorted(
        glob.glob(os.path.join(ROOT, "manifests", "*.jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )
    for manifest in manifests:
        try:
            with open(manifest, encoding="utf-8") as src:
                for line in src:
                    row = json.loads(line)
                    audio_path = row.get("audio_path")
                    if not audio_path:
                        continue
                    path = audio_path if os.path.isabs(audio_path) else os.path.join(ROOT, audio_path)
                    if os.path.isfile(path):
                        return str(row.get("id") or "smoke"), path
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None, None


@app.post("/api/scorer_smoke")
def scorer_smoke(req: ScorerSmokeReq):
    """用共享卷中的一条现有音频验证 GPU scorer 的真实模型推理。"""
    metric = (req.metric or "").strip().lower()
    if metric not in {"ping", "whisper", "utmos", "speaker", "xcomet"}:
        return JSONResponse({"error": "metric 仅支持 ping/whisper/utmos/speaker/xcomet"}, status_code=422)
    started = time.time()
    sample_id, audio_path = _scorer_smoke_audio()
    if metric not in {"ping", "xcomet"} and not audio_path:
        return JSONResponse({"error": "共享清单中没有可用音频"}, status_code=409)

    if metric == "ping":
        raw = scorer_ping()
        result = raw if raw else None
    elif metric == "whisper":
        raw = remote_whisper_transcribe([("smoke", audio_path, None)])
        item = ((raw or {}).get("items") or [{}])[0]
        result = ({"model": (raw or {}).get("model"),
                   "text_chars": len(str(item.get("text") or "")),
                   "language": item.get("language")} if raw else None)
    elif metric == "utmos":
        raw = remote_utmos_score([("smoke", audio_path)])
        item = ((raw or {}).get("items") or [{}])[0]
        result = ({"model": (raw or {}).get("model"), "score": item.get("score")}
                  if raw else None)
    elif metric == "speaker":
        raw = remote_speaker_score([("smoke", audio_path, audio_path)])
        item = ((raw or {}).get("items") or [{}])[0]
        result = ({"model": (raw or {}).get("model"), "cosine": item.get("cosine"),
                   "normalized_similarity": item.get("normalized_similarity")}
                  if raw else None)
    else:
        raw = remote_xcomet_score([("Hello world.", "你好，世界。", "你好，世界。")])
        result = ({"model": (raw or {}).get("model"),
                   "system_score": (raw or {}).get("system_score")}
                  if raw else None)

    if result is None:
        return JSONResponse({"error": f"{metric} scorer 调用失败",
                             "detail": scorer_last_error()}, status_code=502)
    return {"ok": True, "metric": metric, "sample_id": sample_id,
            "elapsed_s": round(time.time() - started, 3), "result": result}


@app.get("/api/result_history")
def result_history(file: str):
    """同组合(模型×数据集×n×seed)的历史结果时间线——rerun 归档 + 当前，供漂移趋势图。"""
    base = os.path.splitext(os.path.basename(file))[0]
    base = re.sub(r"\.\__rerun_.+$", "", base)
    points = []

    def _point(path, ts):
        try:
            s = json.load(open(path, encoding="utf-8"))["summary"]
        except Exception:
            return
        m = s.get("metric", "")
        points.append({"ts": ts, "metric": m, "value": s.get(m),
                       "err_rate": s.get("err_rate"), "fail_rate": s.get("fail_rate"),
                       "n_ok": s.get("n_ok"), "n_total": s.get("n_total"),
                       "low_coverage": s.get("low_coverage")})

    for p in sorted(glob.glob(os.path.join(ROOT, "results", "history", base + "__*.json"))):
        ts = os.path.splitext(os.path.basename(p))[0].rsplit("__", 1)[-1]  # 归档名尾部 YYYYMMDDTHHMMSS
        _point(p, ts)
    cur = os.path.join(ROOT, "results", base + ".json")
    if not os.path.exists(cur):
        staged = sorted(glob.glob(os.path.join(ROOT, "results", base + ".json.__rerun_*")), key=os.path.getmtime, reverse=True)
        if staged:
            cur = staged[0]
    if os.path.exists(cur):
        _point(cur, time.strftime("%Y%m%dT%H%M%S", time.localtime(os.path.getmtime(cur))))
    return {"points": points}


def _job_log_path(id: str):
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", id or ""):
        raise ValueError("bad id")
    real = os.path.realpath(os.path.join(JOBS_LOG_DIR, os.path.basename(id) + ".log"))
    allowed = os.path.realpath(JOBS_LOG_DIR)
    if not real.startswith(allowed + os.sep):
        raise PermissionError("路径越界")
    return real


@app.get("/api/job_request")
def job_request(id: str):
    """优先返回首条真实 HTTP 请求；尚未发出或 WS 协议则返回派发时的契约投影。"""
    try:
        real = _job_log_path(id)
    except ValueError:
        return JSONResponse({"error": "bad id"}, status_code=400)
    except PermissionError:
        return JSONResponse({"error": "路径越界"}, status_code=403)
    record = JOBS.get(id) or _read_jobs_file().get(id) or {}
    planned = record.get("request_preview")
    if not planned and record.get("origin") == "cli":
        _, summary = _cli_result_record(record.get("result_file") or record.get("out"))
        planned = _cli_request_preview(summary or {})
    marker = re.compile(re.escape(logconf.REQUEST_PREVIEW_MARKER) + r"([A-Za-z0-9_\-]+)")
    if os.path.exists(real):
        with open(real, encoding="utf-8", errors="replace") as f:
            for line in f:
                match = marker.search(line)
                if not match:
                    continue
                token = match.group(1)
                try:
                    raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
                    preview = json.loads(raw.decode())
                    preview.setdefault("source", "actual_http")
                    return preview
                except Exception:
                    return JSONResponse({"error": "请求详情损坏"}, status_code=500)
    if isinstance(planned, dict):
        return planned
    return JSONResponse({"error": "任务尚未发出可记录的请求"}, status_code=404)


def _normalized_job_response(id: str):
    """Fallback for WS/streaming jobs: return one normalized infer row, never raw audio."""
    record = JOBS.get(id) or _read_jobs_file().get(id) or {}
    summary = record.get("summary") or {}
    result_file = os.path.basename(record.get("result_file") or record.get("out") or "")
    if not summary and result_file:
        path = os.path.realpath(os.path.join(ROOT, "results", result_file))
        allowed = os.path.realpath(os.path.join(ROOT, "results"))
        if path.startswith(allowed + os.sep) and os.path.isfile(path):
            try:
                summary = (json.load(open(path, encoding="utf-8")) or {}).get("summary") or {}
            except (OSError, ValueError):
                summary = {}
    infer_file = os.path.basename(summary.get("infer_file") or "")
    if not infer_file:
        return None
    path = os.path.realpath(os.path.join(ROOT, "infer", infer_file))
    allowed = os.path.realpath(os.path.join(ROOT, "infer"))
    if not path.startswith(allowed + os.sep) or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                row = json.loads(line)
                if row.get("id") == "__meta__":
                    continue
                sample = {k: row.get(k) for k in (
                    "id", "ok", "hyp", "error", "elapsed_s", "audio_s", "extra"
                ) if row.get(k) is not None}
                return {
                    "source": "normalized_infer",
                    "status": None,
                    "content_type": "application/json",
                    "response_bytes": None,
                    "body_type": "normalized-json",
                    "body": logconf.sanitize_response_value(sample),
                    "truncated": False,
                    "note": "该协议未记录可安全展示的原始响应，以下为第一条标准化评测输出。",
                }
    except (OSError, ValueError):
        return None
    return None


@app.get("/api/job_response")
def job_response(id: str):
    """Return one sanitized/bounded wire response, with normalized infer fallback."""
    try:
        real = _job_log_path(id)
    except ValueError:
        return JSONResponse({"error": "bad id"}, status_code=400)
    except PermissionError:
        return JSONResponse({"error": "路径越界"}, status_code=403)
    marker = re.compile(re.escape(logconf.RESPONSE_PREVIEW_MARKER) + r"([A-Za-z0-9_\-]+)")
    if os.path.exists(real):
        with open(real, encoding="utf-8", errors="replace") as f:
            for line in f:
                match = marker.search(line)
                if not match:
                    continue
                try:
                    token = match.group(1)
                    raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
                    preview = json.loads(raw.decode())
                    if preview.get("body_type") == "stream":
                        fallback = _normalized_job_response(id)
                        if fallback:
                            return fallback
                        preview["note"] = "流式响应尚未形成标准化样例；任务产出首条结果后会自动显示。"
                    return preview
                except Exception:
                    break
    fallback = _normalized_job_response(id)
    if fallback:
        return fallback
    return JSONResponse({"error": "任务尚未产生可展示的返回示例"}, status_code=404)


@app.get("/api/job_log")
def job_log(id: str):
    """单个任务的全量运行日志(logs/jobs/<id>.log)。
    防任意文件读取：①id 只许字母数字下划线连字符 ②basename 去路径 ③realpath 必须落在 JOBS_LOG_DIR 内。"""
    try:
        real = _job_log_path(id)
    except ValueError:
        return JSONResponse({"error": "bad id"}, status_code=400)
    except PermissionError:
        return JSONResponse({"error": "路径越界"}, status_code=403)
    if not os.path.exists(real):
        return PlainTextResponse("(暂无全量日志：该任务可能由 CLI 起，或尚未产生输出)", status_code=404)
    with open(real, encoding="utf-8") as f:
        return PlainTextResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    from config import DASH_HOST, DASH_PORT
    # 不接 uvicorn access log：看板每 2s 轮询 /api/jobs，接进来会刷爆 dashboard.log、
    # 淹掉任务生命周期日志(且 uvicorn.run 的 dictConfig 也会覆盖预设 handler)。
    # dashboard.log 只留我们自己的结构化应用日志：启动/派发/任务起跑完成失败/401 审计/异常。
    LOG.info("Dashboard 启动 → http://localhost:%s", DASH_PORT)
    uvicorn.run(app, host=DASH_HOST, port=DASH_PORT)
