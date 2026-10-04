"""端点 / 路径 / 服务配置 — 环境变量覆盖，缺省为空（需自行配置）。

上云只改环境变量，不动代码（十二要素）。本地零感知。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(key, default):
    return os.environ.get(key, default)


# 端点默认全部为空：公开版不内置任何服务地址，全部通过环境变量提供。
# 空串表示未配置，对应 adapter 会在首次调用时给出明确错误。
ENDPOINTS = {
    "platform": _env("ASR_PLATFORM_URL", ""),        # 自托管 ASR 平台（HTTP，Bearer 鉴权）
    "openai_audio": _env("ASR_OPENAI_AUDIO_URL", ""),  # OpenAI 兼容 audio 接口
    "ext_adv": _env("EXT_ADV_URL", ""),
    "light": _env("LIGHT_URL", ""),
    "gemma": _env("GEMMA_URL", ""),
    "sensevoice": _env("SENSEVOICE_URL", ""),
}
# WS 流式/同传端点；不填则复用 platform。
ENDPOINTS["platform_ws"] = _env("ASR_PLATFORM_WS_URL", ENDPOINTS["platform"])

# 看板服务
DASH_HOST = _env("DASH_HOST", "0.0.0.0")
DASH_PORT = int(_env("DASH_PORT", "8088"))

# 可选增强：embedding 语义相似度端点（OpenAI 兼容 /v1/embeddings）。
# 默认 siliconflow 免费 bge-m3。纯增强——端点挂/限流/无 key 时该指标降级为 null，不影响主指标。
EMBEDDING_URL = _env("EMB_URL", "https://api.siliconflow.cn/v1/embeddings")
EMBEDDING_MODEL = _env("EMB_MODEL", "BAAI/bge-m3")


def rel_path(p):
    """绝对路径 → 相对 ROOT（manifest 序列化用，跨机器可移植）。已是相对则原样返回。"""
    if not p or not os.path.isabs(p):
        return p
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def abs_path(p):
    """相对 ROOT → 绝对（读取时还原）。已是绝对则原样返回。"""
    if not p or os.path.isabs(p):
        return p
    return os.path.join(ROOT, p)
