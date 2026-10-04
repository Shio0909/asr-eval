# 测评数据集（datasets/）

本目录存放语音模型测评集用到的**公开可白嫖数据集**。自建集（领域 ASR、结构化纪要、热词扩充）另行管理。
重新下载：`bash datasets/download.sh phase_a`（核心集）/ `phase_b`（2026-06 调研新增）/ `phase_c`（2026-06 二轮检索）/ `multilingual`（FLEURS 12 语、Common Voice 17 十三语、LibriSpeech test-other）/ `open_asr`（英文 Open ASR 六集）/ `chuan_yu`（川渝 12 城，需下载 URL）/ `all`。

> 注：`downloads/` 存放原始压缩包与 SpeechIO 官方仓库（含规一化脚本），不是数据本体。
>
> `all` 包含 `phase_a + phase_b + phase_c + multilingual`，不包含需登录链接的 `chuan_yu`，
> 也不包含 ACL6060 的物化步骤（见 `eval/fetch_acl6060.py`）。数据本体不入仓，本文件的“已有”仅表示该数据集有下载与 builder 支持。

## 长音频同传

| 路径 | 数据集 | 已接语向 | 整场样本 / 时长 | 参考类型 | builder |
|---|---|---|---|---|---|
| `translation/longform_raw/acl6060_full/` | ACL 60/60 Full | en→中/法/德/日/俄/阿/波斯/荷/葡/土 | eval 每语向 5 场、约 57 分钟，单场约 9–12 分钟（另有 dev 5 场） | 人工书面翻译 | `acl6060_long --subset eval-zh` 等 |
| `translation/longform_raw/mcif/` | MCIF Long | en→中/德/意 | 每语向 21 场，约 2 小时，单场约 5–6 分钟 | 文档级书面翻译 | `mcif_long --subset zh\|de\|it` |
| `translation/longform_raw/realsi/` | RealSI | en→zh / zh→en | 各 10 场，44.4 / 51.1 分钟，单场约 3–7 分钟 | 人工同传译文，含时间段与术语 | `realsi --subset en-zh\|zh-en` |
| `translation/longform_raw/bstc_ccmt2019/` | BSTC 2019 dev | zh→en | 16 场中文演讲，单场约 1–8 分钟 | 句级人工英文翻译，含 offset/duration | `bstc_long` |

这四套均以“一条 manifest = 一场完整音频”接入，原 ACL6060 416 条句级 eval
仍保留，二者不混榜。RealSI 的 `segments/terms/duration_ms` 会原样进入 manifest，
用于后续分段稳定性、术语和延迟分析。BSTC development 压缩包已完整校验并登记为
可运行数据集，builder 首次使用时幂等解压；EPIC 仍未登记，避免页面能选但实际读到半包。

> 状态：🟢 基础集 · 🆕 扩展集。
> 以下按评测维度列出可用的公开数据集：基础语音→SpeechIO/AISHELL，噪音干扰→VWB，发音偏差→**WSC-Eval/KeSpeech/Wu-Bench/MagicData 川渝 12 城**，固定表达→SeACo/AISHELL-NER，说话重叠→AliMeeting。

## 已落地（Phase A · 无需登录 · 实测可拉）

| 状态 | 路径 | 数据集 | 内容 | 体积 | 用途 | License |
|---|---|---|---|---|---|---|
| 🟢 | `asr/speechio/yuekai_mirror/` | SpeechIO ZH00000-26 | 27 个 unlocked 普通话测试集（Lhotse cuts）| ~6.5 GB | ASR 普通话主力 | 开源（HF 镜像 yuekai/speechio）|
| 🟢 | `asr/wsyue/` | WSYue-ASR-eval | 粤语 Short 7060 + Long 370 条 wav + 标注（builder `wsyue --subset short\|long`）| ~1.2 GB | ASR 粤语短句/长句 | CC BY-NC 4.0 |
| 🆕 | `translation/flores_plus/flores200_dataset/` | FLORES-200 | dev/devtest，含 zho_Hans/zho_Hant/eng_Latn | ~77 MB | 文本翻译精度基线 | CC-BY-SA 4.0 |
| 🆕 | `translation/wmt/` | WMT22/23 | zh-en/en-zh src+ref（1875/2037/1976/2074 句）| ~2 MB | 文本翻译·多域 | 学术开放 |
| 🆕 | `summary/vcsum/` | VCSUM | 中文会议总结 train/dev/test parquet | ~6 MB | 总结·会议 | 学术 |
| 🆕 | `summary/cnewsum/` | CNewSum | 中文新闻摘要 | ~422 MB | 总结·新闻长文 | 学术 |

> 2026-06-10 裁撤：LCSTS/CSDS 已删（与 CNewSum 定位重复，语音平台不需三套文本总结）。

**Phase A 合计 ≈ 9 GB**（音频 7.6 GB + 文本 1.4 GB）。SpeechIO/WSYue 虽是"已有"，仍在本地备一份便于统一跑分。

## Phase A+（二轮深挖新增 · 替代原"必须自建"项）

| 状态 | 路径 | 数据集 | 内容 | 体积 | 替代的自建工作 |
|---|---|---|---|---|---|
| 🆕 | `asr/med_it/` | MED-IT | **英文**真实医患问诊 ASR（指标 WER），含音频+转写+医学术语热词 | ~4.6 GB | 医疗领域 ASR 测评集 |
| 🆕 | `asr/seaco_hotword/repo/data/` | SeACo 热词测试集 | 热词列表+text+uttid（建在 AISHELL-1 test 上），自带 B-WER 评测 | <2 MB | 中文热词增益 benchmark |
| 🆕 | `asr/aishell_ner/repo/data/` | AISHELL-NER | AISHELL-1 上的实体/NER 标注（实体即热词）| ~37 MB | 中文热词/实体对照 |
| 🆕 | `asr/aishell1_test/` | AISHELL-1 test（yuekai/aishell，Lhotse）| 普通话朗读 test 集（S0764–S0916，~7176 条）| ~0.9 GB | SeACo/AISHELL-NER 的**音频底座** |
| 🆕 | `asr/aishell2_ios/` | AISHELL-2 iOS 子集（官方授权 82G zip 分层抽样，`scripts/aishell2_subset.py` 产出：400 spk × 5，seed 42）| iPhone 信道朗读普通话；原始 2000 条，有效 1995 条 | ~0.23 GB | 普通话·移动端信道；`datasets/exclusions/aishell2_ios.json` 隔离 C0932 上游音文错位；⚠️ 授权数据，只放内部存储，全量 zip 勿推公网 |
| 🆕 | `asr/aishell2_eval/` | **AISHELL2-2018A-EVAL 官方 test**（Apache-2.0）| 5000 条 × iOS / Android / Mic 三个平行通道，10 位说话人 | 压缩 1.35 GB；解出 test 约 1.35 GB | 推荐使用 `aishell2_eval --subset ios\|android\|mic`；iOS 为行业主榜，三通道分别出 CER；dev 2500 条不参与榜单 |

> ⚠️ 官方 `AISHELL/AISHELL-1` HF 镜像残缺（仅训练说话人 S0002–S0101，无 test），故改用 `yuekai/aishell` 的 test 分片（Lhotse cuts）。
> SeACo（test 808 uttid/400 热词）与 AISHELL-NER（test ~7176 句）只含标注，音频复用 `asr/aishell1_test/`。MED-IT 自带音频。
> 这批把医疗 ASR、中文热词评测从"自建"降级为"白嫖"，详见 ../docs/dataset-plan.md §4.1。

## Phase A++（二轮逛 HF/GitHub 新增 · 免登录直下 · 体积可控）

| 状态 | 路径 | 数据集 | 内容 | 体积 | 补的缺口 |
|---|---|---|---|---|---|
| 🆕 | `asr/ascend/` | CAiRE/ASCEND | 中英 code-switch 自发对话（train/dev/test parquet）| ~1.2 GB | **中英混合**维度 |
| 🆕 | `translation/fleurs/` | FLEURS 12 语 test | Qwen3-ASR 公开表同口径：en/zh/yue/ar/de/es/fr/it/ja/ko/pt/ru，共 8,876 条；cmn/en 仍复用作语音翻译 | ~10 GB（含解压） | **多语 ASR** + 语音翻译 |
| 🆕 | `asr/common_voice17/` | Common Voice 17 十三语 test | en/zh-CN/yue/zh-TW/ar/de/es/fr/it/ja/ko/pt/ru，共 134,731 条；Mozilla 原始数据的 `fsicoli` test 分片镜像 | ~5.5 GB（压缩） | **多语/口音 ASR**（CC0）|
| 🆕 | `asr/ja_cv8/`、`asr/jsut_basic5000/`、`asr/reazonspeech_test/` | 日语公开三基准 | Kotoba-Whisper 公开评测的精确 test 版本：CV8 日语 4,483、JSUT Basic5000 5,000、ReazonSpeech held-out 5,263；builder `ja_benchmark --subset cv8\|jsut\|reazon` | ~3.0 GB Parquet + 物化音频 | **日语公开 CER 对数**；规整为 Whisper BasicTextNormalizer 后去空格 |
| 🆕 | `asr/librispeech/` | LibriSpeech test-clean + test-other（HF `openslr/librispeech_asr` parquet）| 英文 ASR 2,620 + 2,939 条，flac 内嵌 parquet（builder 首次抽到 `audio/` / `audio_other/`）| ~650 MB | **纯英文 ASR**（指标 WER，clean/other 难度对照）|
| 🆕 | `asr/open_asr/` | Open ASR Leaderboard 英文六集 | AMI 12,643、Earnings22 2,741、GigaSpeech 19,931、SPGISpeech 39,341、VoxPopuli 1,842、TEDLIUM 1,155 条 | ~19 GB Parquet | **英文公开榜对数**（builder `open_asr`）|
| 🆕 | `summary/alimeeting4mug/data/` | Alimeeting4MUG (AMC) test（ModelScope，需 token）| **结构化纪要**：82 场中文会议，TSV `idx+content`(JSON)，含 sentences/话题分段+标题/关键句/action_ids（50 场有待办）| ~7 MB | **议题/待办/关键句**（填结构化纪要缺口）|

> **Alimeeting4MUG 获取**（需 ModelScope token）：`uv run --with modelscope --with datasets --with addict --with oss2 python` →
> `api=HubApi(); api.login("<TOKEN>"); MsDataset.load('Alimeeting4MUG', namespace='modelscope', subset_name='default', split='test')`。
> ⚠️ 下载会成功，但 `datasets` 自动解析 CSV 会因字段不一致报错 → 数据已落 `~/.cache/modelscope`，**手动用 `pandas.read_csv(sep='\t')` 读**（2 列 idx+content，content 是 JSON）。

> ⚠️ **磁盘**：以下大块体积较大，默认不下载，需要时再单独拉取。

## Phase B（2026-06 调研新增 · `download.sh phase_b`）

| 状态 | 路径 | 数据集 | 内容 | 体积 | 补的缺口 | License |
|---|---|---|---|---|---|---|
| 🆕 | `asr/voices_wild_bench/` | Voices-in-the-Wild-Bench | 鲁棒性 ASR，**只用真实样本 real-*(1500 条,中文 750)**；合成 sim-*(3500)因参考"只标目标句、音频含随机干扰前缀"弃用(详见 ../docs/plat-data-gaps) | 1.7 GB | **中文真实噪声鲁棒性**（builder `vwb`,8 类扰动）| MIT |
| 🆕 | `asr/wildasr/` | WildASR (Boson AI/李沐) | 鲁棒性 ASR：上游 10,058 行/29.65h，按 `subset + audio_hash_id` 去重后 7,979 条/23.31h **全真人英文**，7 split（削波/远场/噪声间隙/电话编码/混响/口音）| 2.0 GB | **英文鲁棒性**（builder `wildasr`）| Apache-2.0 |
| 🆕 | `asr/alimeeting/` | AliMeeting Test+Eval | 28 场真实中文会议（远场 8ch + 近场），TextGrid 逐句+说话人。阿里 OSS 直链 | ~13 GB | **会议音频底座 + 说话人分离** | CC BY-SA 4.0 |
| 🆕 | `summary/alimeeting4mug/data/` | 4MUG 全三 split | test1 82 + dev 65 + train 295 场标注；**与本地音频配对 24 场** → `audio_paired_meetings.jsonl` | ~30 MB | 会议纪要参考（抽取式，叠 G-Eval）| CC-BY-4.0 |
| 🟡 | （speechio 内共享）| 领域子集 ZH00000/01/11 | 金融(1h 会议演讲)/时政(9h 新闻联播)/法律(3.4h 罗翔讲课)，看板已登记 `fin_zh00/gov_zh01/law_zh11`。**演讲/播报体，测术语非对话场景** | 0 | **领域 ASR** | 同 SpeechIO |
| ⬜ 待下 | — | MMedFD 公开子集（⚠️脱敏 TTS 重合成）/ MultiMed 中文(test 225 条) / CASIA 情感(表演式 1200 条) / ESC-50(声音事件) | phase_b 脚本已备好 | <1 GB | 医疗中文·情感·声音标注 | 各异 |

## Phase C（2026-06 二轮检索 · `download.sh phase_c` · 发音偏差/方言腔 + 翻译术语）

| 状态 | 路径 | 数据集 | 内容 | 体积 | 补的缺口 | License |
|---|---|---|---|---|---|---|
| 🆕 | `asr/wsc_eval/` | WSC-Eval（WenetSpeech-Chuan）| 川渝方言腔普通话 8,373 条，**Easy 6981 + Hard 1392 难度分层**，人工校对，10 领域。Qwen3-ASR 技术报告点名基准（只下 Easy/Hard，Short/Long 是同批音频另一种切分，纯冗余未下）| 1.1 GB | **发音偏差·方言腔**（sb2 缺口）| Apache-2.0 |
| 🆕 | `asr/chuan_yu_12city/` | MagicData 川渝 12 城市子方言集 | 官方标称 33h，WAV 总时长实测 38.47h；13,068 句 / 38 人 / 12 城，builder `chuan_yu --subset 成都\|chengdu` | 3.3 GB zip / 4.2 GB 解压 | **城市级川渝子方言 CER** | CC BY-NC-ND 4.0，限非商业 |
| 🆕 | `asr/kespeech_test/` | KeSpeech test（HF 镜像 TwinkStart）| 19,723 条带 **8 子方言标签**（中原/西南/冀鲁/胶辽/江淮/兰银/东北/北京），可出分方言 CER；2025-26 主流模型报告通用口径 | 3.7 GB | **发音偏差·分方言**对外可对数 | 原仓限科研 |
| 🆕 | `asr/wu_bench/` | WenetSpeech-Wu-Bench | 吴语/上海话 ASR 4,851 条（`understanding/asr.parquet`，字段 utt_id/label/audio），同仓另有吴语→普通话 AST 任务可后补 | 1.0 GB | **吴语/上海腔**（此前空白）| Apache-2.0 |
| 🆕 | `asr/wenetspeech/` | WenetSpeech TEST_NET + TEST_MEETING | **中文 ASR 最通用对数基准**：net 24,774 条(网络多场景真实) + meeting 8,370 条(会议)，Lhotse cuts 格式(同 speechio)。⚠️ **gated 需 `hf auth login` + 网页同意**；只拉 test 分片(别碰 1TB 训练集)；test_net 上游有少量标注错误(issue #63) | 3.4 GB (net 23h + meeting 15h) | **业界对数**(Qwen3-ASR/FireRedASR/Seed-ASR/Fun-ASR 等人人用) | CC BY 4.0 |
| 🆕 | `translation/covost2/` | CoVoST2 zh↔en test（fixie-ai 镜像）| 语音翻译双向 test 各 4,898 条，CommonVoice mp3 内嵌 parquet，免登录（phase_a 当时未拉成，本次补齐）| 951 MB | 语音翻译升级（FLEURS 偏朗读）| **CC0 可商用** |
| 🆕 | `translation/hardmtbench/repo/` | HardMTBench | 10k 句对 × **12 领域含金融/法律/医疗**，zh↔en 双向，逐条带术语对+知识密度+难度标注，单文件 JSONL | 16 MB | **术语/领域翻译**（原标"待补"）| ⚠️ repo 无 LICENSE，对外引用前确认 |
| 🆕 | `translation/wmt25/` | WMT25 General en→zh | 文档级，news/**speech**/social/literary 四域（speech 域贴演讲转译场景）。⚠️ 无 zh→en——该方向官方最新仍是 WMT23 | 15 MB | 最新 en→zh 多域 | 研究用途 |
| 🆕 | `translation/wmt25_term/` | WMT25 Terminology Track2 | **金融 en↔zh-Hant（繁中）**文档级，带 doc 级术语词典（full_data_2015–2024，HKMA 年报）| 16 MB | 金融术语翻译 | **CC BY-NC 禁商用**，仅内部 |

> 调研中证伪/按需项：**SlideASR-R**（真实演讲+OCR 热词，EMNLP'25）论文声称开源但 HF 仓 `RUIH/SlideASR-Bench` 只放了合成 SlideASR-S，**真实子集未放出**；**ContextASR-Bench**（30 万实体，MIT）是 2025-26 contextual ASR 事实标准但音频为 **TTS 合成**，与"只用真实样本"口径冲突，如下载须与真实语音榜隔离。

MagicData 川渝 12 城下载链接由 MagicHub 登录后生成，不把带签名 URL 写进仓库。下载命令：

```bash
CHUAN_YU_12CITY_URL='<MagicHub ZIP 直链>' bash datasets/download.sh chuan_yu
```

脚本支持 `.part` 断点续传、MD5 校验（`8a2f6c393f8e5dd21ba9283d3030966e`）并正确处理 ZIP 内 GBK 城市名。

## 候选 · 已核实未下（按需启用）

| 数据集 | 来源 | 内容 | 启用条件 |
|---|---|---|---|
| **ASR-CTeleCSC** | MagicHub `mandarin-chinese-conversational-speech-corpus-telephony`（免费，需登录）| **真实 8kHz 电话信道中文对话**：5.2h / 16 段 / 192MB，带转写（WAV+TXT）。CC BY-NC-ND（**禁商用**，内部诊断可）| 唯一的真实中文电话信道材料（VWB/WildASR 均无此格），测通用 ASR 窄带掉点。**触发条件：产品确认接电话/呼叫中心音频**，否则不下 |
| MagicData-RAMC | OpenSLR 123 直链 | 180h 真实中文自发对话（663 人，16kHz 手机信道），通用话题。CC BY-NC-ND | 补"对话体/自发口语"维度 |
| Earnings-21/22 | GitHub `revdotcom/speech-datasets` 直 clone | 真实英文财报电话会 39h+119h（转写 CC BY-SA 4.0，含中式口音英语）| 需要英文金融基线时 |
| ~~WenetSpeech TEST_NET/TEST_MEETING~~ | **✅ 已转正到 Phase C** | — | 2026-06 调研确认是中文 ASR 最通用对数基准，已加 |
| AISHELL-5 Eval1/Eval2 | OpenSLR 159 直链（各 ~1.8G）| 2025 新出·车载多通道 2-4 人重叠真实对话（远场 4 麦+近场），人工转写 | AliMeeting 之外的第二个重叠/分离场景（声学条件完全不同）时再下 |
| **AISHELL-2** | 需官网申请签协议（非免费白嫖）| 中文 ASR 高频对数集(~12 份报告用)，但获取门槛高 | 要 AISHELL-2 维度可比性时（认可度高于 SpeechIO，仅次于 WenetSpeech）|
| ~~Common Voice zh/en/yue~~ | **✅ 已转正到 Phase A++，并扩到 Qwen3-ASR 同口径 13 语** | — | CV17 test-only 镜像已接入，原始数据 CC0 |
| SeniorTalk（BAAI）| HF `BAAI/SeniorTalk`（gated 留联系方式秒批）| 75-85 岁老人真实会话 55.5h，ASR test 5,869 条/3.77h，字级人工转写。CC BY-NC-SA | 系统测"老人发音偏差"（弱读/含混）时申请 |
| ChildMandarin（BAAI）| HF `BAAI/ChildMandarin`（gated 同上）| 3-5 岁儿童 41.25h，test 4.12h，字级人工转写。CC BY-NC-SA | 儿童语音（错读/模糊发音极端样本）时申请 |
| EmergentTTS-Eval | HF `bosonai/EmergentTTS-Eval`（NeurIPS 2025）| TTS 评测基准 | 将来测 TTS 端点时 |
| ~~BSTC（百度同传）~~ | **✅ development 16 场已转正** | zh→en 长音频同传 | `bstc_long`，官方 offset/duration 进入 OmniSTEval |

> ❌ **确认公开不存在**（2026-06 两轮 deep-research 定论，头部厂商 Fun-ASR/Seed-ASR/Qwen3-ASR 全用内部集）：中文**金融/政务对话体语音**、中文**法律庭审**、中文**儿童语音**（仅商用库）。路径＝TTS 可控合成兜底 + ZH00000/01 真实锚点校准。
> AISHELL-4 旧条目因 AliMeeting 已覆盖而作废；Common Voice 已因多语 ASR 展示需求重新接入。

## 必须自建（公开集空白，详见 ../docs/dataset-plan.md §4）

1. 领域 ASR：**金融 / 法律 / 政务**（医疗已被 MED-IT 覆盖，不在此列；法律最难）
2. 业务热词场景：机器人讲解 / 儿童陪伴（带 keyword）
3. 结构化会议纪要（议题/决策/待办）—— 可基于 AMC-A + AMI 方案补标
> 专业领域音频获取策略（业务流量脱敏 / 商用采购 / 公开音视频爬标 / **TTS 合成**）见 ../docs/dataset-plan.md §4。

## 配套工具（已就位）

- 中文文本规一化：SpeechIO 官方 `textnorm_zh.py`，已纳入项目 `eval/textnorm_zh.py`，来源见根目录 `NOTICE`
- 指标库（用时 pip）：jiwer / sacrebleu / unbabel-comet / deepeval / rouge-score / bert-score
