#!/usr/bin/env bash
# 测评数据集一键下载脚本
# 用法: bash datasets/download.sh [phase_a|phase_b|phase_c|multilingual|open_asr|japanese|chuan_yu|all]
#   phase_a  核心可白嫖集 (~5-7GB, 无需登录)
#   phase_b  2026-06 调研新增: 接口补缺 (鲁棒性/会议/医疗/情感/声音事件/语音翻译, 直链已逐一核实)
#   phase_c  2026-06 二轮检索: 发音偏差/方言腔 (WSC-Eval/KeSpeech/Wu-Bench) + 翻译术语 (HardMTBench/WMT25) + CoVoST2 补拉
#   multilingual  Qwen3-ASR 对数口径: FLEURS 12 语 + Common Voice 17 13 语（只下 test）
#   open_asr  英文 Open ASR Leaderboard: AMI/Earnings22/GigaSpeech/SPGISpeech/VoxPopuli/TEDLIUM
#   chuan_yu MagicData 川渝 12 城市子方言集 (需 CHUAN_YU_12CITY_URL)
#   all      phase_a + phase_b + phase_c（chuan_yu 因需独立 URL 不自动包含）
# 依赖: hf (huggingface_hub CLI), git, git-lfs, curl, uv(用于 sacrebleu 拉 WMT)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=0
PHASE="${1:-phase_a}"

mkdir -p downloads datasets/asr datasets/translation datasets/summary

file_md5() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    echo "缺少 md5sum/md5，无法校验下载包" >&2
    return 1
  fi
}

download_chuan_yu() {
  local url="${CHUAN_YU_12CITY_URL:-}"
  local expected_md5="8a2f6c393f8e5dd21ba9283d3030966e"
  local archive="downloads/chuan_yu_12city.zip"
  local partial="${archive}.part"
  local target="datasets/asr/chuan_yu_12city"
  local wav_count metadata_count

  if [ -d "$target" ]; then
    wav_count="$(find "$target" -type f -name '*.wav' | wc -l | tr -d ' ')"
    metadata_count="$(find "$target" -type f -name 'UTTERANCEINFO.txt' | wc -l | tr -d ' ')"
    if [ "$wav_count" = "13068" ] && [ "$metadata_count" = "12" ]; then
      echo "  [川渝12城] 已解压完整（13068 WAV），跳过"
      return
    fi
  fi
  if [ -z "$url" ]; then
    echo "缺少 CHUAN_YU_12CITY_URL；请把 MagicHub 下载直链通过环境变量传入" >&2
    return 2
  fi

  if [ -f "$archive" ] && [ "$(file_md5 "$archive")" = "$expected_md5" ]; then
    echo "  [川渝12城] 复用已校验压缩包"
  elif [ -f "$partial" ] && [ "$(file_md5 "$partial")" = "$expected_md5" ]; then
    mv "$partial" "$archive"
    echo "  [川渝12城] 复用已完成的断点下载"
  else
    echo "  [川渝12城] 断点续传 3.2GB 压缩包..."
    curl -fL --retry 5 --retry-delay 5 -C - -o "$partial" "$url"
    if [ "$(file_md5 "$partial")" != "$expected_md5" ]; then
      echo "川渝12城压缩包 MD5 不匹配；保留 $partial 供排查" >&2
      return 1
    fi
    mv "$partial" "$archive"
  fi

  mkdir -p "$target"
  if command -v 7zz >/dev/null 2>&1; then
    7zz x -y "$archive" "-o$target"
  elif command -v 7z >/dev/null 2>&1; then
    7z x -y "$archive" "-o$target"
  else
    # 该 ZIP 的中文路径是 GBK 且未打 UTF-8 标志，普通 unzip 会产生乱码。
    python3 - "$archive" "$target" <<'PY'
import os
import sys
import zipfile

archive, target = sys.argv[1:3]
root = os.path.realpath(target)
with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        if not info.flag_bits & 0x800:
            try:
                info.filename = info.filename.encode("cp437").decode("gbk")
            except UnicodeEncodeError:
                pass  # 新版 Python 已从 ZIP Unicode Path extra field 还原中文。
        destination = os.path.realpath(os.path.join(root, info.filename))
        if os.path.commonpath((root, destination)) != root:
            raise RuntimeError(f"unsafe zip path: {info.filename}")
        zf.extract(info, root)
PY
  fi

  wav_count="$(find "$target" -type f -name '*.wav' | wc -l | tr -d ' ')"
  metadata_count="$(find "$target" -type f -name 'UTTERANCEINFO.txt' | wc -l | tr -d ' ')"
  if [ "$wav_count" != "13068" ] || [ "$metadata_count" != "12" ]; then
    echo "川渝12城解压不完整：期望 13068 WAV/12 标注，实际 $wav_count/$metadata_count" >&2
    return 1
  fi
  echo "  [川渝12城] 完成：$wav_count WAV → $target"
}

phase_a() {
  echo "### Phase A: 核心可白嫖集 ###"

  # --- ASR: 粤语 WSYue (~1.5GB, 非gated) ---
  hf download ASLP-lab/WSYue-ASR-eval --repo-type dataset --local-dir datasets/asr/wsyue
  for d in Short Long; do
    [ -f "datasets/asr/wsyue/$d/wav.tar.gz" ] && tar xzf "datasets/asr/wsyue/$d/wav.tar.gz" -C "datasets/asr/wsyue/$d"
  done
  [ -f "datasets/asr/wsyue/Long/TextGrid.tar.gz" ] && \
    tar xzf "datasets/asr/wsyue/Long/TextGrid.tar.gz" -C "datasets/asr/wsyue/Long"

  # --- ASR: SpeechIO 普通话测试集 (ZH00000-26 unlocked, 6.5GB) ---
  #   注意: 官方 OSS 桶 (oss://speechio-leaderboard) 已禁止匿名访问;
  #   改用 HF 镜像 yuekai/speechio (Lhotse cuts 格式, 含全部 27 个 unlocked 集)
  hf download yuekai/speechio --repo-type dataset --local-dir datasets/asr/speechio/yuekai_mirror
  #   规一化脚本 (中文 TN) 在官方仓库, 顺手 clone 备用:
  git clone --depth 1 https://github.com/SpeechColab/Leaderboard.git downloads/speechio_leaderboard || true
  echo "  [SpeechIO] 中文规一化脚本: downloads/speechio_leaderboard/utils/textnorm_zh.py"

  # --- 翻译: FLORES200 (官方 tarball, 含中英) ---
  mkdir -p datasets/translation/flores_plus
  curl -sL -o downloads/flores200.tar.gz "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
  tar xzf downloads/flores200.tar.gz -C datasets/translation/flores_plus

  # --- 翻译: WMT22/23 zh-en/en-zh 测试集 (经 sacrebleu) ---
  uv run --quiet --with sacrebleu python3 - <<'PY'
import shutil, os
from sacrebleu.utils import get_source_file, get_reference_files
os.makedirs("datasets/translation/wmt", exist_ok=True)
for ts in ["wmt22","wmt23"]:
    for lp in ["zh-en","en-zh"]:
        try:
            src=get_source_file(ts,lp); refs=get_reference_files(ts,lp)
            shutil.copy(src, f"datasets/translation/wmt/{ts}.{lp}.src.txt")
            shutil.copy(refs[0], f"datasets/translation/wmt/{ts}.{lp}.ref.txt")
            print(f"  WMT {ts} {lp} ok")
        except Exception as e:
            print(f"  WMT {ts} {lp} skip: {e}")
PY

  # --- 翻译: CoVoST2 (fixie-ai 镜像, 音频内嵌 parquet; facebook/covost2 靠 loading script 已废) ---
  #   只拉 zh↔en 两个语向的 test (zh-CN_en/* 会连 16 个 train 分片一起拖, 评测不需要)
  hf download fixie-ai/covost2 --repo-type dataset \
    --include "zh-CN_en/test*" --include "en_zh-CN/test*" \
    --local-dir datasets/translation/covost2 || \
    echo "  [CoVoST2] fixie-ai 镜像拉取失败, 检查网络/HF"

  # --- 总结: VCSUM / CNewSum (纯文本只作 gemma 基线; LCSTS/CSDS 与 CNewSum 定位重复, 2026-06 已裁) ---
  hf download renhehuang/vcsum-meeting-summary --repo-type dataset --local-dir datasets/summary/vcsum
  hf download ethanhao2077/cnewsum-processed   --repo-type dataset --local-dir datasets/summary/cnewsum

  # --- Phase A+: 二轮深挖新增 (替代原"必须自建") ---
  # 医疗 ASR (自带音频)
  hf download SandO114/Medical_Interview --repo-type dataset --local-dir datasets/asr/med_it
  # 中文热词测试集 + NER 标注 (音频复用 AISHELL-1)
  mkdir -p datasets/asr/seaco_hotword datasets/asr/aishell_ner
  git clone --depth 1 https://github.com/R1ckShi/SeACo-Paraformer.git datasets/asr/seaco_hotword/repo || true
  git clone --depth 1 https://github.com/Alibaba-NLP/AISHELL-NER.git datasets/asr/aishell_ner/repo || true
  # AISHELL-1 test 音频底座 (SeACo/AISHELL-NER 依赖)
  #   注意: 官方 AISHELL/AISHELL-1 HF 镜像残缺(仅训练说话人), 用 yuekai/aishell 的 test 分片
  hf download yuekai/aishell --repo-type dataset --include "*test*" --local-dir datasets/asr/aishell1_test

  echo "### Phase A 完成 ###"
}

phase_b() {
  echo "### Phase B: 接口补缺 (2026-06 调研, 来源逐一核实过) ###"

  # --- ASR 鲁棒性诊断: Voices-in-the-Wild-Bench (MIT, 5k 条 zh2.5k/en2.5k, 8 类扰动含丢包/回声) ---
  hf download zhifeixie/Voices-in-the-Wild-Bench --repo-type dataset \
    --local-dir datasets/asr/voices_wild_bench

  # --- ASR 鲁棒性(英,全真人): WildASR (Boson AI, Apache-2.0, 10k 条/30h, 7 split 含电话编码/混响/口音) ---
  #   与 VWB 互补: VWB 有中文但 70% 合成, WildASR 全真人但暂只有英文
  hf download bosonai/WildASR --repo-type dataset --local-dir datasets/asr/wildasr

  # --- 会议音频: AliMeeting test 10h (真实普通话会议, 说话人分离+会议纪要底座) ---
  #   OpenSLR 119 页面的真实直链在阿里 OSS(us.openslr.org 证书有问题); CC BY-SA 4.0 口径
  ALI_OSS="https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr"
  curl -L -C - -o downloads/alimeeting_test.tar.gz "$ALI_OSS/Test_Ali.tar.gz" && \
  curl -L -C - -o downloads/alimeeting_eval.tar.gz "$ALI_OSS/Eval_Ali.tar.gz" && \
    mkdir -p datasets/asr/alimeeting && \
    tar xzf downloads/alimeeting_test.tar.gz -C datasets/asr/alimeeting && \
    tar xzf downloads/alimeeting_eval.tar.gz -C datasets/asr/alimeeting
  #   纪要标注 AliMeeting4MUG(抽取式摘要, 524 场带 ES) 在 ModelScope 需登录+SDK token, 按会议 ID 与音频对齐:
  echo "  [AliMeeting4MUG] 摘要标注需 ModelScope 登录: modelscope.cn/datasets/modelscope/Alimeeting4MUG"

  # （裁撤说明: AISHELL-4 被 AliMeeting 全面替代——后者带 4MUG 摘要标注链路;
  #   唯一增量是 4-8 人 diarization 压测, 真需要时: us.openslr.org/resources/111/test.tar.gz ~5.2G）

  # --- 中文医疗 ASR: MMedFD 公开子集 (蚂蚁, 音频内嵌 parquet; 注意为脱敏 TTS 重合成, 当诊断集) ---
  hf download HanselZz/MMedFD --repo-type dataset --include "*User*" \
    --local-dir datasets/asr/mmedfd || echo "  [MMedFD] 拉取失败"
  # --- 中文医疗 ASR: MultiMed 中文子集 (MIT, test 225 条, 仅够冒烟) ---
  hf download leduckhai/MultiMed --repo-type dataset --include "Chinese/*" \
    --local-dir datasets/asr/multimed_zh || echo "  [MultiMed] 拉取失败"

  # --- 中文语音情感: CASIA 6 类 1200 条 (HF 第三方转传, 标 apache-2.0, 73MB) ---
  hf download BillyLin/CASIA_speech_emotion_recognition --repo-type dataset \
    --local-dir datasets/asr/casia_emotion

  # --- 声音事件标注: ESC-50 (2000 条 50 类, CC BY-NC 整体/ESC-10 子集 CC BY, ~600MB) ---
  curl -L -o downloads/esc50.zip "https://github.com/karolpiczak/ESC-50/archive/master.zip" && \
    mkdir -p datasets/asr/esc50 && unzip -oq downloads/esc50.zip -d datasets/asr/esc50

  # --- 真实通用 ASR 补强: WenetSpeech TEST_NET/TEST_MEETING → 已转正到 phase_c(gated, 需登录) ---

  echo "### Phase B 完成 ###"
}

phase_c() {
  echo "### Phase C: 2026-06 二轮检索 — 发音偏差/方言腔 + 翻译术语 (来源逐一核实过) ###"

  # --- ASR 川渝方言腔普通话: WSC-Eval (ASLP-lab, Apache-2.0, ~9.7h, Easy/Hard 难度分层) ---
  #   Qwen3-ASR 技术报告点名基准; 只拉 Easy/Hard (Short/Long 是同批音频按时长的另一种切分, 纯冗余 ~2.4G;
  #   Long 是切分前的原始长音频, 将来测长音频再单独拉), 跳过 WSC-Eval-TTS 子集
  hf download ASLP-lab/WSC-Eval --repo-type dataset \
    --include "WSC-Eval-ASR/Easy/*" --include "WSC-Eval-ASR/Hard/*" --include "WSC-Eval-ASR/readme.md" \
    --local-dir datasets/asr/wsc_eval

  # --- ASR 8 子方言腔普通话: KeSpeech test (HF 镜像免登录; 官方 GitHub 走百度云不便) ---
  #   19,723 条带子方言标签 (中原/西南/冀鲁/胶辽/江淮/兰银/东北/北京), 3.45GB parquet
  #   license 原仓 dataset_license.md 限科研用途
  hf download TwinkStart/KeSpeech --repo-type dataset --local-dir datasets/asr/kespeech_test

  # --- ASR 吴语/上海话: WenetSpeech-Wu-Bench (ASLP-lab, Apache-2.0, ~9.75h, 2026-02 文件已传齐) ---
  #   只拉 ASR; 同仓另有 ast.parquet(吴语→普通话翻译)/emotion 等, 需要再补
  hf download ASLP-lab/WenetSpeech-Wu-Bench --repo-type dataset \
    --include "understanding/asr.parquet" --include "README.md" \
    --local-dir datasets/asr/wu_bench

  # --- ASR 中文最通用对数基准: WenetSpeech test_net + test_meeting (Lhotse cuts, 与 speechio 同构) ---
  #   ⚠️ gated: 需先 `hf auth login` + 在 huggingface.co/datasets/wenet-e2e/wenetspeech 网页点同意条款
  #   只拉 test 分片(net 3片 + meeting 1片), 绝不碰训练集 cuts_L/M/S (全量 1TB!)
  #   net=网络多场景真实, meeting=会议; ⚠️ test_net 上游有少量标注错误(官方 issue #63)
  hf download wenet-e2e/wenetspeech --repo-type dataset \
    --include "data/cuts_TEST_NET*" --include "data/cuts_TEST_MEETING*" \
    --local-dir datasets/asr/wenetspeech || \
    echo "  [WenetSpeech] 拉取失败——需 hf auth login + 网页同意条款(gated)"

  # --- 语音翻译: CoVoST2 zh↔en test (fixie-ai 镜像, CC0 可商用, 音频内嵌 parquet) ---
  #   phase_a 当时未拉成 (目录为空), 此处补拉, 只要 test 分片
  hf download fixie-ai/covost2 --repo-type dataset \
    --include "zh-CN_en/test*" --include "en_zh-CN/test*" \
    --local-dir datasets/translation/covost2

  # --- 术语翻译: HardMTBench (10k 句对 × 12 领域含金融/法律/医疗, zh↔en 双向, 逐条带术语+难度标注) ---
  #   ⚠️ repo 无 LICENSE 文件 (论文自称 open-sourced), 对外报告引用前需确认
  mkdir -p datasets/translation/hardmtbench
  git clone --depth 1 https://github.com/jasonNLP/HardMTBench.git datasets/translation/hardmtbench/repo || true

  # --- 翻译: WMT25 General en→zh test (文档级, news/speech/social/literary 四域; speech 域贴演讲转译场景) ---
  #   只拉单个测试集文件, 不 clone 160MB 全仓 (其余是各系统输出); 无 zh→en, 该方向最新仍是 WMT23
  mkdir -p datasets/translation/wmt25
  curl -sL -o datasets/translation/wmt25/wmt25-genmt.jsonl \
    "https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/main/data/wmt25-genmt.jsonl"

  # --- 术语翻译·金融: WMT25 Terminology Track2 (en↔zh-Hant 繁中, 文档级带术语词典) ---
  #   ⚠️ CC BY-NC 4.0 禁商用, 仅限非商业用途; 繁中需 OpenCC 转简并标注口径
  mkdir -p datasets/translation/wmt25_term/track2
  for y in 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
    curl -sL -o "datasets/translation/wmt25_term/track2/full_data_${y}.jsonl" \
      "https://raw.githubusercontent.com/wmt-conference/wmt25-terminology/main/ranking/references/track2/full_data_${y}.jsonl"
  done
  curl -sL -o datasets/translation/wmt25_term/LICENSE.txt \
    "https://raw.githubusercontent.com/wmt-conference/wmt25-terminology/main/LICENSE.txt"

  echo "### Phase C 完成 ###"
}

multilingual_asr() {
  echo "### Multilingual ASR: FLEURS 12语 + Common Voice 17 13语 + LibriSpeech other ###"
  local lang
  local fleurs_args=()
  local cv_args=()

  # FLEURS 本地已有 en/zh/de/es/fr/ja/ko/ru；重复执行会由 hf 校验后跳过。
  for lang in en_us cmn_hans_cn yue_hant_hk ar_eg de_de es_419 fr_fr it_it ja_jp ko_kr pt_br ru_ru; do
    fleurs_args+=(--include "data/$lang/test.tsv" --include "data/$lang/audio/test.tar.gz")
  done
  hf download google/fleurs --repo-type dataset "${fleurs_args[@]}" \
    --local-dir datasets/translation/fleurs

  # Mozilla 2025-10 起从 HF 撤空官方仓；fsicoli 镜像保留 CV17 原始 test tar/tsv。
  # 只拉 Qwen3-ASR 公布的 13 语言 test，不拉 train/dev。
  for lang in en zh-CN yue zh-TW ar de es fr it ja ko pt ru; do
    cv_args+=(--include "audio/$lang/test/*" --include "transcript/$lang/test.tsv")
  done
  hf download fsicoli/common_voice_17_0 --repo-type dataset "${cv_args[@]}" \
    --local-dir datasets/asr/common_voice17

  hf download openslr/librispeech_asr --repo-type dataset \
    --include "other/test/*" --local-dir datasets/asr/librispeech
}

open_asr_english() {
  echo "### Open ASR English: AMI/Earnings22/GigaSpeech/SPGISpeech/VoxPopuli/TEDLIUM ###"
  mkdir -p datasets/asr/open_asr
  hf download hf-audio/open-asr-leaderboard --repo-type dataset \
    --include "ami/test-*.parquet" \
    --include "earnings22/test-*.parquet" \
    --include "gigaspeech/test-*.parquet" \
    --include "spgispeech/test-*.parquet" \
    --include "voxpopuli/test-*.parquet" \
    --local-dir datasets/asr/open_asr
  # 统一仓的 tedlium 配置当前只剩 metadata；此镜像保留同一 TED-LIUM 3 test 的 1,155 条音频与参考。
  hf download distil-whisper/tedlium-prompted --repo-type dataset \
    --include "release3/test-*.parquet" \
    --local-dir datasets/asr/open_asr/tedlium_source
  echo "### Open ASR English 完成 ###"
}

download_exact() {
  local url="$1" target="$2" expected="$3" partial="${2}.part" actual
  mkdir -p "$(dirname "$target")"
  actual="$(stat -f %z "$target" 2>/dev/null || stat -c %s "$target" 2>/dev/null || echo 0)"
  if [ "$actual" = "$expected" ]; then
    echo "  [日语] 已下载 $(basename "$target")"
    return
  fi
  curl -fL --retry 5 --retry-all-errors -C - -o "$partial" "$url"
  actual="$(stat -f %z "$partial" 2>/dev/null || stat -c %s "$partial" 2>/dev/null || echo 0)"
  if [ "$actual" != "$expected" ]; then
    echo "日语 parquet 大小不符: $target 期望=$expected 实际=$actual" >&2
    return 1
  fi
  mv "$partial" "$target"
}

japanese_asr() {
  echo "### Japanese ASR: CV8 / JSUT Basic5000 / ReazonSpeech held-out ###"
  local hf="https://huggingface.co/datasets/japanese-asr"
  download_exact "$hf/ja_asr.common_voice_8_0/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet" \
    datasets/asr/ja_cv8/parquet/0000.parquet 151322876
  download_exact "$hf/ja_asr.reazonspeech_test/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet" \
    datasets/asr/reazonspeech_test/parquet/0000.parquet 299352669
  download_exact "$hf/ja_asr.reazonspeech_test/resolve/refs%2Fconvert%2Fparquet/default/test/0001.parquet" \
    datasets/asr/reazonspeech_test/parquet/0001.parquet 301183933
  download_exact "$hf/ja_asr.jsut_basic5000/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet" \
    datasets/asr/jsut_basic5000/parquet/0000.parquet 461656723
  download_exact "$hf/ja_asr.jsut_basic5000/resolve/refs%2Fconvert%2Fparquet/default/test/0001.parquet" \
    datasets/asr/jsut_basic5000/parquet/0001.parquet 458729891
  download_exact "$hf/ja_asr.jsut_basic5000/resolve/refs%2Fconvert%2Fparquet/default/test/0002.parquet" \
    datasets/asr/jsut_basic5000/parquet/0002.parquet 663493701
  download_exact "$hf/ja_asr.jsut_basic5000/resolve/refs%2Fconvert%2Fparquet/default/test/0003.parquet" \
    datasets/asr/jsut_basic5000/parquet/0003.parquet 666749556
  echo "### Japanese ASR 完成；运行 ja_benchmark builder 会物化音频 ###"
}

# （AISHELL-4 见 phase_b 内裁撤说明）
# （phase_c 调研中确认未公开/按需: SlideASR-R 真实热词集论文有但 HF 仓只放了合成 S;
#   AISHELL-5 车载重叠 OpenSLR159 直链可下; SeniorTalk/ChildMandarin 老人儿童集 gated 留联系方式可批）

case "$PHASE" in
  phase_a) phase_a ;;
  phase_b) phase_b ;;
  phase_c) phase_c ;;
  multilingual) multilingual_asr ;;
  open_asr) open_asr_english ;;
  japanese) japanese_asr ;;
  chuan_yu) download_chuan_yu ;;
  all)     phase_a; phase_b; phase_c; multilingual_asr ;;
  *) echo "未知阶段: $PHASE"; exit 1 ;;
esac
