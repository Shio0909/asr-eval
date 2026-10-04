# AGENTS.md — asr-eval

给自动化编码代理和新贡献者的速查。架构与用法见 `README.md`，这里只记约定和陷阱。

## 约定

- **两阶段分离**：`infer`（调模型存 hyp）与 `score`（算指标）互相独立，模型只调用一次。
- **task 路由**：`asr` / `translate` / `summarize` / `diarization`。manifest 每行的 `task` 决定走向。
- **默认 `--workers 1` 串行**：并发会污染延迟指标。只看精度时才加 `--workers N`。
- **失败样本**：失败或空 hyp 不计入精度；成功率低于 95%（`score.py` 的 `COVERAGE_MIN`）标 `low_coverage`。
- **规一化**：ref 和 hyp 必须同口径。中文走 `normalize.py` 的 TextNorm，英文走 `normalize_en`（Whisper 口径）。
  修改英文边界规则时，同步更新 metric signature 和数字/时间/缩写黄金用例。
- **溯源**：infer 写 `__meta__` 行（含 manifest 指纹）。指纹与 score 时不一致会标 `manifest_sha_mismatch`。

## 密钥与端点

- 仓库不内置任何服务地址或密钥。端点默认为空，全部通过环境变量提供（见 `.env.example`）。
- 密钥只从环境变量或被忽略的 `eval/secrets_local.py` 读取。不得写进代码、测试、文档或日志。
- 测试里使用 `*.test` 占位域名。`tests/conftest.py` 在导入前设置占位端点。

## 接口接入验证

新增接口、修改 adapter 或补充接口参数时，先写一个独立的最小客户端做实连验证，再改 adapter 和能力契约：

- 最小客户端不得复用待验证 adapter 的组包和解析逻辑。
- 用一条短样本覆盖鉴权、参数位置（表单 / JSON / WS 首包）、默认值与枚举、事件顺序、成功返回字段、错误包装。
- 区分三层事实：服务端源码或 OpenAPI 声明、线上实测、评测 adapter 的行为。线上与源码不一致时，以实测保证兼容，并在测试中覆盖两种结构。
- 一次性客户端放 `/tmp`。实测通过后再登记 `request_schema` 和 UI 参数，并补 adapter 单测。

## 测试

```bash
uv run --with pytest pytest tests/ -q
```

改 `metrics.py` 或 `normalize.py` 必须跑全量。

## 扩展点

- 加数据集：`build_manifest.py` 加 builder，`dashboard/server.py` 的 `DATASETS` 注册。
- 加模型：`adapters.py` 加 adapter，注册到 `ADAPTERS`。
