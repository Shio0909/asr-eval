# 语音模型测评集 · 数据集方案

> 状态：设计稿 v1 · 日期：2026-06-08
> 范围：ASR 识别 / 翻译 / 总结 三大场景的测评数据选型、自建清单、两档规模设计
>
> **2026-07-31 说明**：这是历史选型与规模设计稿，不是线上库存清单。现行注册数据、
> 许可和本地落地状态以 `datasets/README.md` 为准，生产卷是否可跑以看板
> `/api/overview` 为准。下文保留的候选研究记录不等于待下载任务。
> **2026-08-19 更新**：BSTC development 16 场已完整落地并注册 `bstc_long`；下文“未验证/候选”描述仅保留历史决策轨迹。

---

## 0. TL;DR

- **ASR 评测维度与公开集对应**：基础语音→SpeechIO，噪音干扰→VWB(真实样本)，**发音偏差→WSC-Eval/KeSpeech/Wu-Bench/MagicData 川渝 12 城**，固定表达→SeACo+AISHELL-NER，说话重叠→AliMeeting。
- **本项目覆盖 ASR 之外的两个新场景（翻译/总结）+ 三块 ASR 新能力**，对应的新增数据集见 §0.1。
- 新增里**大部分能公开获取**（翻译 FLORES/WMT/CoVoST2/HardMTBench、总结 VCSUM/CNewSum）；BSTC 获取链路未验证，CSDS/LCSTS 已从现行下载计划移除。业务特色仍需自建或补标：① 自有提示词/热词增益 ② 领域识别 ③ 结构化会议纪要。
- **快速验证（Lite）~1.8k 条**，10–15 分钟跑完；**全量（Full）约 6–7 万条**，用于对外报告。两档共用 manifest+runner，`--tier` 切换。

---

## 0.1 已有 vs 新增数据集清单（本节直接回答"要新增哪些"）

图例：🟢 基础集 · 🆕 新增·公开白嫖 · 🔧 新增·需自建 · ⏳ 下载中 · ✅ 已落地 `datasets/`

| 场景 | 子能力 | 数据集 | 状态 |
|---|---|---|---|
| **ASR 普通话** | 通用识别 | 公开集组合（见上） | 无单一自建集，维度由公开集覆盖 |
| **ASR 发音偏差** | 方言腔/口音普通话 | WSC-Eval（川渝）+ MagicData 川渝 12 城（城市级）+ KeSpeech test（8 子方言）+ Wu-Bench（吴语）| 🆕 新增公开 · ✅ |
| **ASR 普通话** | 通用识别 | SpeechIO ZH00000-26（65.5h/43,178）| 🟢 已有 · ✅ |
| **ASR 粤语** | 通用识别 | WSYue-ASR-eval | 🟢 已有 · ✅ |
| **ASR 中英混读** | code-switch | CAiRE/ASCEND（1,315）| 🆕 新增公开 · ✅ |
| **ASR 英文** | 英文 ASR（WER）| LibriSpeech test-clean（2,620）+ test-other（2,939）| 🆕 新增公开 · ✅ |
| **ASR 多语/口音** | 12/13 语公开对标 | FLEURS 12 语 test（8,876）+ Common Voice 17 十三语 test（134,731）| 🆕 新增公开 · ✅ |
| ASR | **自有提示词/热词增益·KRR** | SeACo 开放热词测试集 + AISHELL-NER（中文热词 benchmark，自带 B-WER）| 🆕 新增公开（替代大半自建）|
| ASR | 热词·业务专属场景 | 扩充自建集（机器人讲解/儿童陪伴 + keyword）| 🔧 仅这部分自建 |
| ASR | 领域识别·**医疗** | MED-IT（英文真实问诊+医学术语）；中文医疗集仍待核实 | 🆕 英文集已落地 |
| ASR | 领域识别·**金融/法律/政务** | 中文语音公开空白 → 自建（金融英文有 SPGISpeech/Earnings 做对照）| 🔧 自建 |
| ASR | 会议多说话人（可选对照）| AISHELL-4 / AliMeeting | 🆕 新增公开（大，需登录/申请）|
| **翻译·文本** | zh↔en 精度基线 | FLORES-200（含中英）| 🆕 新增公开 · ✅ |
| **翻译·文本** | zh↔en 多域 | WMT22/23 | 🆕 新增公开 · ✅ |
| **翻译·语音** | zh↔en | FLEURS cmn+en；CoVoST2 `fixie-ai/covost2` 镜像（音频内嵌、免登录）| 🆕 新增公开 · ✅ |
| **翻译·同传** | 中英同传 | BSTC | 🆕 新增公开（需百度账号）|
| 翻译 | 业务专名/术语（可选）| 自建术语小集 | 🔧 新增自建 |
| **总结·会议** | 会议总结 | VCSUM | 🆕 新增公开 · ✅ |
| **总结·对话** | 客服/对话总结 | CSDS | 历史候选，已从现行下载计划移除 |
| **总结·新闻** | 新闻长文 | CNewSum | 🆕 新增公开 · ✅ |
| **总结·短文** | 短文摘要 | LCSTS | 历史候选，已从现行下载计划移除 |
| **总结·结构化纪要·待办** | action item | AMC-A（中文 AliMeeting action item）+ AMI（英文 decision/action 四段，做标注标杆）| 🆕 新增公开 |
| **总结·结构化纪要·决策** | decision 字段 | 中文偏弱，借 AMI 方案在 AMC-A/VCSUM 上补标 | 🔧 少量自建（补标）|
| 翻译·术语 | 金融/法律/医疗术语翻译 | HardMTBench（中英+三领域+术语标注）+ WMT Terminology | 🆕 新增公开 |

**一句话**：ASR 三件套已有；本项目**新增公开集**＝翻译(FLORES/WMT/CoVoST2/BSTC/HardMTBench) + 总结(VCSUM/CSDS/CNewSum/LCSTS/AMC-A) + ASR(SeACo热词/AISHELL-NER/MED-IT医疗)；**真正只能自建的极少**——见 §4。公开集已大半下载到 `datasets/`。

---

## 0.2 翻译 / 总结：测什么 · 用什么 · 缺什么（一眼看全）

> 关键区分：**总结一定是纯文本输入**；**翻译看子类型**——文本翻译纯文本，语音翻译/同传才要音频。
> 测评是"考试卷"不是"训练料"——每类几百~两千条就够算稳指标，量上都不缺。

### 翻译场景（原文 + 翻译提示词 → 目标语言）

| 测哪种 | 输入 | 用的数据集（已下） | 量 | 够不够 |
|---|---|---|---|---|
| 文本翻译 中↔英（精度基线）| 纯文本 | FLORES-200 | 1,012/向 | ✅ 够 |
| 文本翻译 中↔英（多域贴业务）| 纯文本 | WMT22/23 zh-en+en-zh | ~1,900/向 | ✅ 够 |
| 语音翻译 中/英（听音频直接翻）| 音频 | FLEURS cmn+en | 944+646 | 🟡 通用够，偏朗读 |
| **业务/专业术语翻译**（金融/法律/医疗）| 文本 | HardMTBench（12 领域 10k 句对）+ WMT25-Term Track2（金融繁中，NC 禁商用）| 20k 条目 + 10 年 HKMA 年报 | ✅ **已下（phase_c）** |

### 总结场景（文字 + 提示词 → 总结）— 全部纯文本

| 测哪种 | 用的数据集（已下） | 量 | 够不够 |
|---|---|---|---|
| 会议总结 | VCSUM | 136 | ✅ 够 |
| 对话/客服总结 | CSDS | ~800 | ✅ 够 |
| 新闻长文总结 | CNewSum | 14,355 | ✅ 够 |
| 短文总结 | LCSTS | 725 | ✅ 够 |
| **结构化纪要（议题/待办/关键句）** | Alimeeting4MUG test（82 场，50 场带 action_ids）| 已下 | ✅ 议题/待办/关键句齐；**决策**字段仍弱（可借 AMI 补标）|

### 还要补的（仅这几项）
1. ~~**业务术语翻译集**~~ ✅ 已落地（2026-06-12 phase_c）：HardMTBench + WMT25-Term Track2 金融繁中
2. **结构化纪要集（议题/决策/待办）** → 接 AMC-A(中文 action item) + 借 AMI(英文 decision/action) 标注方案，在 VCSUM 上补标
3. ~~（可选）语音翻译若要更贴业务~~ ✅ CoVoST2 zh↔en test 已下（fixie-ai 镜像，CC0）；BSTC 仍候选（OpenDataLab 需注册）
4. （核实结论 2026-06-12）zh→en 文本翻译官方最新仍是 WMT23——WMT24/25 都只有 en→zh；WMT25 General 的 speech 域已下，贴演讲转译场景

---

## 1. 测评能力矩阵

测评对象不是"一个模型"，而是 **`{型号} × {场景} × {提示词} × {模式} × {精度+性能}`**：

| 场景 | 输入 | 提示词档 | 模式 | 精度指标 | 性能指标 |
|---|---|---|---|---|---|
| ASR 识别 | 音频 | a.默认 / b.自有(热词/领域) | 非流式 / 流式 | CER·WER·Emb·KRR·ΔPrompt | RTFx·P50/P95·TTFT·定稿延迟 |
| 翻译 | 音频或文本 + 目标语 | 翻译prompt(±热词) | 非流式 / 同传 | COMET·chrF·BLEU·judge | 端到端·同传首字延迟 |
| 总结 | 文字(或会议音频) + prompt | 总结prompt | 离线 / 流式 | G-Eval四维·ROUGE·BERTScore·字段抽取·Faithfulness | 总结耗时·长音频端到端 |

竞品（豆包、千问、腾讯、讯飞、听悟）作为同矩阵的横评对象，与被测系统同口径。

---

## 2. 数据集总盘点：白嫖 vs 自建

图例：✅ 直接白嫖 / ⚠️ 部分匹配需改造 / ❌ 公开集空白需自建

| 场景 | 子能力 | 公开集白嫖 | 必须自建 |
|---|---|---|---|
| ASR | 默认提示词·通用普通话 | ✅ SpeechIO | — |
| ASR | 默认提示词·粤语 | ✅ WSYue-ASR-eval | — |
| ASR | 默认提示词·会议多说话人 | ✅ AISHELL-4 | — |
| ASR | 默认提示词·方言 | ⚠️ KeSpeech（按需）| — |
| ASR | **自有提示词/热词增益·KRR** | ⚠️ 协议借 SlideSpeech；数据借 AISHELL-NER | ✅ **扩充自建集（带 keyword）** |
| ASR | **领域识别（法/医/金/政）** | ❌（仅医疗可挖 MultiMed-ST）| ✅ **每域 5–10h 带术语标注** |
| 翻译 | 文本 zh↔en | ✅ FLORES+ / WMT22-23 | — |
| 翻译 | 语音 zh→en | ✅ CoVoST2 | — |
| 翻译 | 同传（中英）| ✅ BSTC | — |
| 翻译 | **业务专名/领域术语** | ⚠️ HardMTBench / WMT-Term | ⚠️ 业务术语小集（可选）|
| 总结 | 会议总结 | ✅ VCSUM | — |
| 总结 | 对话/客服总结 | ✅ CSDS | — |
| 总结 | 新闻/长短文 | ✅ CNewSum / LCSTS / CLTS+ | — |
| 总结 | **结构化纪要（议题/决策/待办）** | ❌ | ✅ **基于 VCSUM 重标或自建** |

---

## 3. 分场景数据清单

### 3.1 ASR 识别

**a. 默认提示词（通用识别）— 公开集充足**

| 数据集 | 规模 | 场景/语种 | License | 获取 |
|---|---|---|---|---|
| SpeechIO test sets (ZH00000~26) | 65.5h / 43,178 条，难度分级 | 新闻/访谈/解说/直播/播客 普通话 | 开源 | github.com/SpeechColab/Leaderboard |
| WSYue-ASR-eval | builder 可见 Short 7,060 + Long 370 | 粤语，含中英混读、背景音 | **CC BY-NC 4.0（非商用）** | github.com/ASLP-lab/WenetSpeech-Yue |
| AISHELL-4 | 会议 ~120h，多说话人 | 真实会议 | Apache-2.0 | openslr.org/111 |
| KeSpeech（按需）| 1542h，8 方言 | 方言鲁棒性 | 学术开放 | openreview KeSpeech |
| Common Voice zh（对照）| 众包验证集 | 口音多样跨域 | CC0 | HF mozilla-foundation/common_voice |

**b. 自有提示词 / 热词增益 — 需改造 + 自建**

- 评测协议白嫖 SlideSpeech：U-WER / B-WER（有偏/无偏 WER）/ Recall / KER。
- 中文数据：⚠️ AISHELL-NER（实体当热词，test 3393 实体），作为 KRR 的底座。
- 测法：同批音频跑两遍（默认 prompt vs 注入热词/领域 prompt），报 **ΔCER / ΔKRR / Δ术语命中**。

### 3.2 翻译

**文本翻译 — 公开集充足**

| 数据集 | 规模 | 方向/领域 | License | 用途 |
|---|---|---|---|---|
| FLORES-200 / FLORES+ | devtest ~1012 句 ×（zh↔en）| Wikipedia 综合，含简繁 | CC-BY-SA 4.0 | 精度基线（口径干净）|
| WMT22/23 General zh-en | 各 ~1,875+ 段 | 新闻/对话/电商/社媒 | 学术开放 | 多域贴业务 |
| HardMTBench（2026）| 10k 源句/20k 条目 | zh↔en，12 难域含金融/法律/医疗，逐条带术语+难度标注 | ⚠️ repo 无 LICENSE 文件 | ✅ 已下 `translation/hardmtbench/`（github.com/jasonNLP/HardMTBench 单文件 JSONL）|
| WMT Terminology | zh→en('23)/金融繁中('25)| 带术语词典 | 开放 | 术语命中率（注意简繁）|

**语音翻译 — 基本够用**

| 数据集 | 规模 | 方向 | License | 备注 |
|---|---|---|---|---|
| CoVoST 2 | fixie-ai 镜像，test 每语向 4,898 条 | zh→en、en→zh | CC0 | 音频内嵌、免登录的语音翻译标准集 |
| BSTC（百度同传）| ~70h / ~40k 句 | zh→en，含真人同传译文 | 学术非商用 | 唯一开放中英**同传**集 |
| GigaST test（按需）| 人工校验 | en→zh | CC-BY-NC（非商用）| 注意非商用 |

业务专名/客户专属术语翻译：公开集补不全，⚠️ 可选自建小集（对应 `simult_interpreting` 的 hotwords）。

### 3.3 总结

| 数据集 | 规模 | 类型 | 参考 | License | 获取 |
|---|---|---|---|---|---|
| VCSUM | 239 场真实会议 / >230h | **中文会议总结** | 人工，多形态 | 学术 | github.com/hahahawu/VCSum |
| CSDS | 客服对话 | 中文对话/客服总结 | 人工，角色导向 | 学术 | github.com/xiaolinAndy/CSDS |
| CNewSum | 30.4 万篇 | 新闻长文摘要 | 人工高抽象 | 学术开放 | dqwang122.github.io/projects/CNewSum |
| LCSTS / CLTS+ | LCSTS 240万 / CLTS+ 18万 | 短文 / 长文 | 人工 | 学术 | arXiv 1506.05865 / github lxj5957/CLTS-Dataset |

结构化纪要（议题/决策/待办）：❌ 公开集（含 VCSUM）均无此字段，需基于 VCSUM 重标或自建。

---

## 4. 自建数据清单与优先级（深挖后大幅缩水）

> 二轮穷尽检索后，原"必须自建"的医疗 ASR、中文热词 benchmark、待办抽取、术语翻译都找到了公开集可白嫖。
> **真正只能自建的，收敛到中文金融/法律/政务三类领域语音**——其余都能白嫖或仅需小幅补标。

### 4.1 先用公开集替代（原以为要自建，实际能白嫖）

| 原自建项 | 公开集替代 | 获取 | 省掉多少 |
|---|---|---|---|
| 中文热词增益 benchmark | **SeACo 开放热词测试集**（基于 AISHELL-1，自带 B-WER 评测）+ **AISHELL-NER** | github.com/R1ckShi/SeACo-Paraformer · github.com/Alibaba-NLP/AISHELL-NER | 中文热词评测基本免自建 |
| 医疗领域 ASR | **MED-IT**（英文真实问诊+医学术语热词） | 当前本地 MED-IT | 英文医疗测评免自建；不能据此宣称中文医疗已补齐 |
| 会议待办(action item) | **AMC-A**（中文 AliMeeting action item 标注）+ **AMI**（英文 decision/action 四段标注，标注标杆）| arXiv 2303.16763（需确认下载）· groups.inf.ed.ac.uk/ami | 待办抽取评测基本免自建 |
| 业务术语翻译 | **HardMTBench**（中英+金融/法律/医疗+术语标注）+ WMT Terminology | arXiv 2605.28315 · statmt.org/wmt25/terminology | 术语翻译评测免自建 |

### 4.2 真正必须自建（公开集确认空白）

| # | 自建内容 | 为何补不了 | 建议规模 | 备注 |
|---|---|---|---|---|
| **1** | 中文**金融**语音 ASR（客服/电话会议）| 中文金融语音全空白，只有英文(SPGISpeech/Earnings) | 5–10h，带术语标注 | 英文集可做方法/对照 |
| **2** | 中文**法律**庭审语音 ASR | 确认零开放集（iFlytek 庭审数据不公开）| 5–10h | 最难获取，优先级可后置 |
| **3** | 中文**政务**热线(12345)语音 ASR | 确认空白 | 5–10h | 声学风格可借 MagicData-RAMC 近似 |
| **4** | 业务专属热词场景（机器人讲解/儿童陪伴）| 业务特色，公开集无 | 每场景 +300–500 条带 keyword | 扩充已有自建集，成本最低 |
| (5) | 中文会议 **decision** 字段补标 | 中文 action 有(AMC-A)但 decision 弱 | 借 AMI 方案在 AMC-A/VCSUM 上补标 | 少量标注，非从零 |

**自建顺序**：#4（扩已有集，最快）→ #1 金融（可获取性最高）→ #3 政务 → #5 补标 → #2 法律（最难，最后）。
医疗不在自建清单——直接用 MED-IT。

### 自建集标注 Schema（金融/法律/政务/热词扩充 共用）

每条样本一行 JSONL：

```jsonc
{
  "id": "sb2_robot_000123",
  "audio_path": "audio/robot/000123.wav",   // WAV/16k/单声道, VAD切片 3-15s, 边界保护
  "ref_text": "标注文本（原始，未规一化）",
  "lang": "zh",                              // zh | yue | zh-en
  "scene": "robot_guide",                    // 业务场景标签
  "domain": "general",                       // general | legal | medical | finance | gov
  "dims": ["噪音干扰", "固定表达"],          // 维度标签
  "keywords": [                              // 热词/固定表达，支持别名
    {"surface": "兴业银锡", "aliases": ["兴业银锡"]},
    {"surface": "GPT 5", "aliases": ["GPT 5", "GPT five"]}
  ],
  "split": "lite|full"                       // 分层抽样标记，Lite 是 Full 子集
}
```

标注要点：说话重叠只标主说话人；keyword 覆盖行业术语/专名/新词/熟语；别名匹配命中任一即算 KRR 命中。

---

## 5. 两档规模设计（核心：快速验证 vs 全量）

### 5.1 设计原则

- **Lite = Full 的分层抽样子集 + 关闭重裁判**，不是另一套数据。靠 `split` 字段和 `--tier` 开关。
- 精度可离线批量并发跑；**流式性能必须单条串行采集**（并发污染延迟），所以 Lite 的流式样本要刻意小。
- 重裁判（COMET / G-Eval / Embedding）慢且贵，Lite 默认关闭或只抽很小一撮。

### 5.2 快速验证（Lite）— 目标 10–15 分钟跑完

用途：改 prompt / 换模型 / CI 回归门禁，看趋势而非出排名。

| 场景 | 数据来源 | Lite 条数 | 跑哪些指标 |
|---|---|---|---|
| ASR 通用 | SpeechIO 分层抽样（27 集各 ~30）| ~800 | CER |
| ASR 粤语 | WSYue 抽样 | ~300 | CER |
| ASR 热词增益 | 自建集（覆盖每维度/语种）| ~400 | CER + KRR + ΔPrompt |
| ASR 流式性能 | 上述抽样中再抽 | ~50–100（串行）| TTFT + 定稿延迟 |
| 翻译 文本 | FLORES+ 抽样 | ~200 | chrF + BLEU（关 COMET）|
| 翻译 语音 | CoVoST2 抽样 | ~200 | chrF + BLEU |
| 总结 | VCSUM 抽样 | ~10–20 场 | ROUGE（关 G-Eval，或仅抽 5 场跑 judge）|
| **合计** | | **~1,800 条 + 10–20 场会议** | |

> 经验：~300–500 条/场景足够稳定地"看趋势、抓回归"，但**不足以出可发布排名**——那是 Full 的事。

### 5.3 全量评测（Full）— 用于对外横评报告

用途：对外报告、版本验收。走开源集全量 + 自建全量，开全部指标。

| 场景 | 数据集 | Full 规模 |
|---|---|---|
| ASR 普通话 | SpeechIO 全量 | 43,178 条 / 65.5h |
| ASR 粤语 | WSYue Short + Long | 7,430 条 |
| ASR 会议（按需）| AISHELL-4 test | 全量 |
| ASR 热词/领域 | 自建集全量（扩充后 + 领域集）| ~数千–1万条 |
| 翻译 文本 | FLORES+ devtest + WMT22/23 | ~2,000 + ~1,875 |
| 翻译 语音 | CoVoST2 zh→en/en→zh test + FLEURS/ACL6060 | CoVoST2 每语向 4,898 + 其他公开集 |
| 总结 | VCSUM test + CNewSum 抽样 | VCSUM 全 test + CNewSum 抽样 |
| **量级** | | **约 6–7 万条 + 数百场会议** |

指标全开：CER/WER/Emb/KRR/ΔPrompt + COMET/chrF/BLEU/judge + G-Eval/ROUGE/BERTScore/Faithfulness + 全套性能。

### 5.4 统计有效性备注

- CER 横评要稳定，单个"型号 × 维度"格子建议 ≥ 200 条；维度细分多时 Full 才够铺满，Lite 只保证每格 ≥ 30 抓趋势。
- 流式延迟方差大，Full 建议每型号 ≥ 300 条串行采样取 P50/P95。
- 说话重叠、极少数维度样本量小，结论仅作趋势参考。

---

## 6. 指标 → 数据集 → 库 映射（白嫖速查）

| 指标 | 库（pip 白嫖）| License | 适用数据集 |
|---|---|---|---|
| CER / WER | jiwer | Apache | 全 ASR 集 |
| 文本规一化 | whisper_normalizer + speechio/chinese_text_normalization + OpenCC | MIT | 全 ASR（自研 ~50 行串起来）|
| Embedding 相似度 | Qwen3-Embedding-4B | 开源 | 全 ASR |
| KRR / B-WER | 自研脚本（借 SlideSpeech 协议）| — | 自建集 / AISHELL-NER |
| COMET | unbabel-comet | Apache | FLORES/WMT/CoVoST2 |
| BLEU / chrF | sacrebleu | Apache | 翻译全 |
| 翻译 judge | deepeval (GEval) | Apache | 翻译全 |
| ROUGE / BERTScore | rouge-score / bert-score | Apache/MIT | 总结全 |
| 总结 G-Eval | deepeval (GEval) | Apache | VCSUM/CSDS |
| 流式延迟 TTFT/定稿/RTFx | **自研压测脚本（~150–200 行）**| — | 所有流式 |

**只有两块必须自研**：① 中文/粤语统一规一化函数 ② 流式延迟压测脚本。其余全部 pip 白嫖。

---

## 7. License 合规速查

| 数据集 | License | 商用/对外报告可用性 |
|---|---|---|
| SpeechIO / AISHELL / Common Voice / FLORES+ | 开源 / Apache / CC0 / CC-BY-SA | ✅ 可用（CC-BY-SA 注意署名）|
| WSYue | CC BY-NC 4.0 | ⚠️ 非商用；仅限研究用途，商业用途需单独审查 |
| CoVoST2 | CC0 | ✅ |
| BSTC / VCSUM / CSDS / CNewSum | 学术 | ⚠️ 对外报告仅引用指标，注明来源；非商用条款需法务确认 |
| GigaST / KeSpeech 部分 | CC-BY-NC / 学术 | ⚠️ 非商用，仅限研究评测 |
| HardMTBench | 待确认 | ⚠️ 用前确认仓库与协议 |
| 自建集 | 自有 | ✅ 完全可控 |

> 对外横评报告优先用许可宽松的集（SpeechIO/CoVoST2/FLORES+）作主结论。
> WSYue、WMT25-Term、KeSpeech 等非商用或学术许可数据只作研究评测，再分发前需单独审查。

---

## 8. 落地顺序（Roadmap）

1. **复现 ASR 横评**：规一化包 + ASR runner + 公开集，验证 CER/KRR 口径。
2. **接公开集 + 流式压测**：SpeechIO / WSYue 全量 + 流式延迟脚本。
3. **扩充自建集 #1**（机器人讲解/儿童陪伴 + keyword）。
4. **翻译场景**：FLORES+ / CoVoST2 + COMET/chrF。
5. **总结场景**：VCSUM + G-Eval/ROUGE。
6. **领域 ASR #2 + 结构化纪要 #3**（自建，周期较长，并行启动采集/标注）。
7. **炫酷看板**：results JSON → 双轴散点 + 维度热力图 + 失败案例抽屉。

---

## 附：关键链接

- SpeechIO Leaderboard: https://github.com/SpeechColab/Leaderboard
- WenetSpeech-Yue / WSYue: https://github.com/ASLP-lab/WenetSpeech-Yue
- AISHELL-4: https://www.openslr.org/111/ · AISHELL-NER: https://arxiv.org/abs/2202.08533
- FLORES+: https://huggingface.co/datasets/openlanguagedata/flores_plus
- WMT23: https://www2.statmt.org/wmt23/translation-task.html
- CoVoST2: https://arxiv.org/abs/2007.10310 · BSTC: https://ai.baidu.com/broad/introduction?dataset=bstc
- VCSUM: https://github.com/hahahawu/VCSum · CSDS: https://github.com/xiaolinAndy/CSDS
- CNewSum: https://dqwang122.github.io/projects/CNewSum/
- SlideSpeech: https://slidespeech.github.io/
- jiwer / whisper_normalizer / speechio TN / sacrebleu / unbabel-comet / deepeval / rouge-score / bert-score
