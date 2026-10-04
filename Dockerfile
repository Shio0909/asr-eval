# asr-eval 评测看板镜像（代码+依赖，数据走挂载卷，不入镜像）
# 构建：docker build -t asr-eval:latest .   运行：见 README
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
# 需要镜像加速时用 --build-arg PIP_INDEX_URL=... 覆盖
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org

# 系统依赖：ffmpeg 解码同传/压缩音频；libspeex1 解码讯飞 TTS；soundfile 需 libsndfile1。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates ffmpeg libsndfile1 libspeex1 tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 依赖层（仅 requirements.txt 变才重装，命中缓存）。requirements.txt 由 uv export 从 uv.lock 导出。
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
        --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
    && pip install --no-cache-dir -r requirements.txt \
        --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"

# 应用代码（datasets/results/infer 等运行态不进镜像，见 .dockerignore + entrypoint 软链到卷）
COPY eval/ ./eval/
COPY dashboard/ ./dashboard/
COPY datasets/golden/tts_formula.jsonl /app/seed-data/golden/tts_formula.jsonl
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 十二要素：上云只改这些 env（缺省=空（需自行配置端点），见 eval/config.py / .env.example）
#   - server.py 末尾 from config import → 需 eval/ 在 PYTHONPATH
#   - 密钥(DASHSCOPE_API_KEY 等)、DASH_PASS 走部署平台的环境变量/密钥注入，镜像不含任何密钥
ENV PYTHONPATH=/app/eval \
    DASH_HOST=0.0.0.0 \
    DASH_PORT=8088 \
    DATA_VOLUME=/data
EXPOSE 8088

# 健康检查:TCP 端口探活(看板默认开 HTTP Basic 鉴权,HTTP 探活会被 401 干扰;
# 端口在监听即代表 uvicorn 活着)。
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(5); sys.exit(s.connect_ex(('127.0.0.1',8088)))" || exit 1

# entrypoint 把 ROOT 下数据/产物软链到挂载卷(/data)，再起看板
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "dashboard/server.py"]
