#!/bin/sh
# 把 ROOT(=/app)下写死的数据/产物路径软链到挂载的数据卷，实现"镜像装代码、卷装数据"。
# 卷挂在 $DATA_VOLUME（默认 /data）：
#   $DATA_VOLUME/asr-eval/datasets  ← 数据集（download.sh 已落这里）
#   $DATA_VOLUME/state/{results,infer,jobs.json,interfaces.json} ← 持久运行态
set -e
DATA="${DATA_VOLUME:-/data}"
FORMULA_SEED="/app/seed-data/golden/tts_formula.jsonl"

# 数据根：代码永远找 /app/datasets（eval/config.py 的 ROOT/datasets，写死无 env 旋钮）
if [ -d "$DATA/asr-eval/datasets" ]; then
    # 8 条公式 golden 很小且属于代码版本的一部分；卷上缺失时由镜像幂等补齐。
    mkdir -p "$DATA/asr-eval/datasets/golden"
    if [ ! -f "$DATA/asr-eval/datasets/golden/tts_formula.jsonl" ] && [ -f "$FORMULA_SEED" ]; then
        cp "$FORMULA_SEED" "$DATA/asr-eval/datasets/golden/tts_formula.jsonl"
    fi
    ln -sfn "$DATA/asr-eval/datasets" /app/datasets
else
    echo "[entrypoint] 警告：$DATA/asr-eval/datasets 不存在——卷没挂或数据没下，评测会找不到数据集" >&2
    mkdir -p /app/datasets/golden
    if [ -f "$FORMULA_SEED" ]; then
        cp "$FORMULA_SEED" /app/datasets/golden/tts_formula.jsonl
    fi
fi

# 持久运行态：看板历史(results/infer/manifests)、译后音频(audio_out)、任务登记(jobs.json)、自定义接口(interfaces.json)
mkdir -p "$DATA/state/results" "$DATA/state/infer" "$DATA/state/manifests" "$DATA/state/audio_out"
ln -sfn "$DATA/state/results" /app/results
ln -sfn "$DATA/state/infer"   /app/infer
ln -sfn "$DATA/state/manifests" /app/manifests
ln -sfn "$DATA/state/audio_out" /app/audio_out
[ -f "$DATA/state/jobs.json" ] || echo '{}' > "$DATA/state/jobs.json"
ln -sfn "$DATA/state/jobs.json" /app/jobs.json
# 无条件创建+软链（与 jobs.json 同款）：首启卷上没有该文件时，接口配置曾写进容器层、重启即丢
[ -f "$DATA/state/interfaces.json" ] || echo '[]' > "$DATA/state/interfaces.json"
ln -sfn "$DATA/state/interfaces.json" /app/dashboard/interfaces.json

# HF 缓存也落卷（级联/embedding 等若需）
export HF_HOME="${HF_HOME:-$DATA/hfcache}"

exec "$@"
