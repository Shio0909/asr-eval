# 离线最小示例

不需要任何模型服务或音频：用预置的识别结果（`infer_demo.jsonl`）演示 score 阶段，
看到 CER/WER、KRR 与置信区间的输出格式。

```bash
uv run python eval/score.py \
  --manifest examples/mini/manifest.jsonl \
  --infer examples/mini/infer_demo.jsonl \
  --out /tmp/asr-eval-mini.json
```

接入真实模型后，用 `eval/runner.py` 一条命令完成 infer + score，见根目录 `README.md`。
