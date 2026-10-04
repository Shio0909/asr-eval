"""模型 adapter：把各家 ASR 接口封成统一的 transcribe() 调用。

统一返回 TranscribeResult，内含 hyp 文本 + 性能字段（耗时），
性能与精度在 runner 里一起聚合。新增竞品时在这里加一个 adapter 类即可。
"""

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass

import requests

from config import ENDPOINTS, abs_path
from logconf import get_logger, log_request, log_request_preview, log_response_preview, redact
from simult_trace import SimultTraceRecorder

log = get_logger("adapters")

LONGFORM_PROGRESS_MARKER = "@@LONGFORM_PROGRESS "


def emit_longform_progress(item, *, phase="streaming", audio_s=0.0,
                           audio_total_s=0.0, segments=0, latest_text=""):
    """向 dashboard 子进程管道发送单行结构化长音频进度；普通运行不带元数据时静默。"""
    meta = item.get("_stream_progress") if isinstance(item, dict) else None
    if not isinstance(meta, dict):
        return
    payload = {
        "phase": phase,
        "sample_id": item.get("id"),
        "sample_index": meta.get("sample_index"),
        "sample_total": meta.get("sample_total"),
        "audio_s": round(float(audio_s or 0.0), 1),
        "audio_total_s": round(float(audio_total_s or 0.0), 1),
        "segments": int(segments or 0),
        "latest_text": str(latest_text or "").strip()[:240],
    }
    print(LONGFORM_PROGRESS_MARKER + json.dumps(payload, ensure_ascii=False), flush=True)


_ASR_TEXT_MARKER = "<asr_text>"
# 独立成行的对话模板角色名 / 语言声明 —— 只在确认吐过协议标记时才剔除（见下）
_ASR_SCAFFOLD_LINE = re.compile(
    r"^\s*(?:system|user|assistant|language(?:\s+[A-Za-z-]+)*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_asr_scaffolding(text: str) -> str:
    """剥掉模型偶发吐出的对话模板/协议脚手架。

    实测(ascend_00243, 4765 分之 1)：模型把整段 chat 模板吐了出来 ——
        '啊，走过的 professor �system\\n\\nuser\\n\\nassistant\\nlanguage Chinese<asr_text>啊，走过的 professor �大概有一百多位。'
    `<asr_text>` 之前是模板结构/上下文，之后才是 ASR 正文。

    ⚠️ 仅在**见到该协议标记**时才动手，避免误伤正常转写（正常语音里本来就可能出现
    "system" 等词，如 ascend_01088 "i mean for android system" 是真实内容）。
    处理口径照服务端参考客户端实现。
    """
    if not text or _ASR_TEXT_MARKER not in text:
        return text
    frames = [f.strip() for f in text.split(_ASR_TEXT_MARKER)]
    body = next((f for f in reversed(frames) if f), "")
    body = _ASR_SCAFFOLD_LINE.sub("", body).strip()
    return body or text


def _timed_text_segments(events):
    """把 adapter 内部的 (源音频已发送秒数, 定稿文本) 变成可持久化时间轴。"""
    return [{"end_s": round(float(end_s or 0.0), 3), "text": str(text).strip()}
            for end_s, text in events if str(text or "").strip()]


_WS_SECRET_KEYS = {
    "authorization", "api_key", "apikey", "secret", "token", "password",
    "cookie", "credential", "private_key", "signature", "access_token",
    "auth_token", "bearer_token", "id_token", "refresh_token",
}
_WS_AUDIO_KEYS = {"audio", "audio_data", "audio_bytes", "pcm", "base64"}
_WS_VOLATILE_KEYS = {
    "session_id", "meeting_id", "request_id", "trace_id", "task_id",
    "message_id", "connection_id", "run_id", "timestamp", "ts",
    "created_at", "started_at",
}


def is_sensitive_request_param(name):
    low = str(name or "").lower().replace("-", "_")
    return low in _WS_SECRET_KEYS or low.endswith((
        "_api_key", "_secret", "_password", "_credential",
        "_private_key", "_signature", "_token",
    ))


def _incremental_preview(data, prefix):
    """Rebuild one server-declared LocalAgreement preview snapshot."""
    confirmed = str(data.get(f"{prefix}_confirmed") or "").strip()
    draft = str(data.get(f"{prefix}_draft") or "").strip()
    if data.get(f"{prefix}_draft_replaces_confirmed") and draft:
        return draft
    if not confirmed or not draft:
        return confirmed or draft
    cjk_boundary = any("\u3400" <= char <= "\u9fff" for char in (confirmed[-1], draft[0]))
    separator = "" if cjk_boundary or confirmed[-1].isspace() or draft[0].isspace() else " "
    return f"{confirmed}{separator}{draft}"


def _sanitize_ws_control(value, depth=0):
    """Keep reproducibility fields from WS control frames without secrets/audio blobs."""
    if depth > 8:
        return "<max-depth>"
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = str(key)
            low = name.lower().replace("-", "_")
            if is_sensitive_request_param(low):
                out[name] = "<redacted>"
            elif low in _WS_AUDIO_KEYS:
                out[name] = "<audio omitted>"
            else:
                out[name] = _sanitize_ws_control(item, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_ws_control(item, depth + 1) for item in value[:128]]
    if isinstance(value, bytes):
        return f"<bytes omitted: {len(value)}>"
    if isinstance(value, str):
        return redact(value)[:2048]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))[:2048]


def _ws_control_record(endpoint, connected, task_start, task_started):
    """Persist the actual sent start frame and received handshake/ack for audit."""
    safe_connected = _sanitize_ws_control(connected)
    safe_start = _sanitize_ws_control(task_start)
    safe_started = _sanitize_ws_control(task_started)

    def digest(value):
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def without_volatile_ids(value):
        if isinstance(value, dict):
            return {
                key: without_volatile_ids(item)
                for key, item in value.items()
                if str(key).lower().replace("-", "_") not in _WS_VOLATILE_KEYS
            }
        if isinstance(value, list):
            return [without_volatile_ids(item) for item in value]
        return value

    return {
        "endpoint": str(endpoint or "").split("?", 1)[0],
        "connected_success": safe_connected,
        "task_start": safe_start,
        "task_started": safe_started,
        "task_start_sha256": digest(safe_start),
        "task_started_sha256": digest(safe_started),
        # Full ack hash is exact-run provenance; stable hash excludes per-call IDs so
        # score aggregation can reveal backend/model/route changes instead of sessions.
        "task_started_stable_sha256": digest(without_volatile_ids(safe_started)),
    }


def _load_dotenv(path=None):
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()


def _plat_token(api_key=None):
    """None=内置平台服务从密钥环境变量取；空串=显式无鉴权（自定义接口）。"""
    return os.environ.get("ASR_PLATFORM_TOKEN", "") if api_key is None else api_key


def _bearer_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def http_post(url, *, timeout=None, **kw):
    """统一 HTTP POST 切面(L2 可观测)：记端点/状态码/字节/耗时到 requests.log。
    成功失败都记一行；流式(stream=True)不读 body(免消费流) → resp_bytes 留空。
    透明返回 requests.Response / 透传异常，不改变各 adapter 原有 try/except 行为。"""
    host = url.split("?", 1)[0]   # 去掉 query(xf 把 api_key 放 query) — 另外 formatter 也会脱敏兜底
    streaming = bool(kw.get("stream"))
    t0 = time.perf_counter()
    try:
        log_request_preview("POST", url, kw, timeout)
        r = requests.post(url, timeout=timeout, **kw)
        ms = round((time.perf_counter() - t0) * 1000)
        log_request(layer="http", endpoint=host, status=r.status_code,
                    resp_bytes=(None if streaming else len(r.content)), ms=ms, ok=r.ok)
        log_response_preview(r, streaming=streaming)
        return r
    except Exception as e:
        ms = round((time.perf_counter() - t0) * 1000)
        log_request(layer="http", endpoint=host, ms=ms, ok=False, err=redact(str(e))[:200])
        raise


def http_request(method, url, *, timeout=None, **kw):
    """统一 HTTP 请求切面(L2 可观测)：method 可自定义，供高级自定义接口复用。"""
    host = url.split("?", 1)[0]
    streaming = bool(kw.get("stream"))
    t0 = time.perf_counter()
    try:
        log_request_preview(method, url, kw, timeout)
        r = requests.request(method, url, timeout=timeout, **kw)
        ms = round((time.perf_counter() - t0) * 1000)
        log_request(layer="http", endpoint=host, status=r.status_code,
                    resp_bytes=(None if streaming else len(r.content)), ms=ms, ok=r.ok)
        log_response_preview(r, streaming=streaming)
        return r
    except Exception as e:
        ms = round((time.perf_counter() - t0) * 1000)
        log_request(layer="http", endpoint=host, ms=ms, ok=False, err=redact(str(e))[:200])
        raise


def ws_connect(url, **kw):
    """统一 WebSocket 连接切面(L3 可观测)：记握手主机/耗时/成败到 requests.log。
    透明返回连接对象 / 透传异常，不改各 simult adapter 原收发逻辑(首包 ttfb·AL·LAAL 仍由 L1 记)。"""
    from websocket import create_connection
    host = url.split("?", 1)[0]   # xf 把鉴权放 query → 去掉；formatter 也兜底脱敏
    t0 = time.perf_counter()
    try:
        ws = create_connection(url, **kw)
        log_request(layer="ws_connect", endpoint=host, ms=round((time.perf_counter() - t0) * 1000), ok=True)
        return ws
    except Exception as e:
        log_request(layer="ws_connect", endpoint=host, ms=round((time.perf_counter() - t0) * 1000),
                    ok=False, err=redact(str(e))[:200])
        raise

# 平台系语言名映射(PlatformASRAdapter/PlatformSSEAdapter 共用，原先两份重复)
_LANG_PLATFORM = {"英文": "English", "中文": "Simplified Chinese", "简体中文": "Simplified Chinese",
              "西班牙文": "Spanish", "日文": "Japanese", "en": "English", "zh": "Simplified Chinese"}


try:  # 密钥不入仓：本地从 secrets_local.py 读，容器/CI 走环境变量
    from secrets_local import GEMMA_KEY
except ImportError:
    GEMMA_KEY = os.environ.get("GEMMA_KEY", "")


# soundfile(libsndfile) 啃不动或易出错的封装 → 直接走 ffmpeg（mp3 坏头/m4a/AAC/视频抽音轨）
_FFMPEG_EXT = (".mp3", ".m4a", ".mp4", ".aac", ".mov", ".avi", ".mkv",
               ".webm", ".flv", ".wma", ".amr", ".3gp", ".ts")


def _ffmpeg_mono16k(audio_path: str):
    """ffmpeg 解码任意音/视频 → 16k 单声道 float32。真实用户文件(杂格式/坏头/视频)兜底。"""
    import subprocess

    import numpy as np
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", audio_path,
                        "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                       capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"ffmpeg 解码失败({audio_path}): {p.stderr.decode('utf-8', 'ignore')[:200]}")
    return np.frombuffer(p.stdout, dtype="<f4").copy()


def _is_platform_endpoint(endpoint: str) -> bool:
    """endpoint 是否指向 ASR_PLATFORM_URL / ASR_PLATFORM_WS_URL 配置的内置平台。"""
    def _host(u):
        return re.sub(r"^\w+://", "", (u or "").strip()).rstrip("/")
    ep = _host(endpoint)
    return bool(ep) and any(
        ep.startswith(_host(u)) for u in (ENDPOINTS.get("platform"), ENDPOINTS.get("platform_ws")) if _host(u)
    )


def _decode_audio_bytes(raw: bytes, sr: int = 24000) -> bytes:
    """ffmpeg 解码任意音频字节(mp3/wav/...) → mono s16le PCM。多段 TTS MP3 各自解码后拼 PCM 用
    （直接字节拼接多个独立 MP3 会在 ID3/帧边界产生噪音）。"""
    import subprocess
    if not raw:
        return b""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le",
                        "-ac", "1", "-ar", str(sr), "pipe:1"], input=raw, capture_output=True)
    return p.stdout if (p.returncode == 0 and p.stdout) else b""


def _load_mono16k(audio_path: str):
    """把任意音/视频读成 16k 单声道 float32 波形（load_wav/pcm 共用核心）。

    wav/flac/ogg 走 soundfile；mp3/m4a/视频等走 ffmpeg；soundfile 失败也回退 ffmpeg。
    对齐报告数据形式(16k/mono)；重采样用 polyphase 抗混叠（最近邻会引入混叠失真）。
    """
    import os as _os

    import numpy as np
    if _os.path.splitext(audio_path)[1].lower() in _FFMPEG_EXT:
        return _ffmpeg_mono16k(audio_path)
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32")
    except Exception:
        return _ffmpeg_mono16k(audio_path)   # 坏头/不支持的封装 → 兜底
    if data.ndim > 1:           # 多声道→单声道
        data = data.mean(axis=1)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), 16000)
        data = resample_poly(data, 16000 // g, int(sr) // g).astype(np.float32)
    return data


def load_wav_bytes(audio_path: str) -> bytes:
    """16k/mono/16bit WAV 字节。各端点(只吃 wav)和 gemma(format=wav)统一用这个。"""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, _load_mono16k(audio_path), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load_pcm_bytes(audio_path: str) -> bytes:
    """16k/mono/s16le 裸 PCM 字节。WebSocket 流式同传(Gummy/讯飞)按帧发裸 PCM 用。"""
    import numpy as np

    pcm = np.clip(_load_mono16k(audio_path), -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def _json_path_get(data, path: str, default=None):
    """取 JSON 路径，支持 a.b[0].c / a.0.c。失败返回 default。"""
    if not path:
        return default
    cur = data
    for part in str(path).strip().split("."):
        if cur is None:
            return default
        token = part.strip()
        if not token:
            continue
        while token:
            if "[" in token:
                head, rest = token.split("[", 1)
                if head:
                    if isinstance(cur, dict):
                        cur = cur.get(head)
                    else:
                        return default
                idx, tail = rest.split("]", 1)
                try:
                    cur = cur[int(idx)]
                except Exception:
                    return default
                token = tail.lstrip(".")
                continue
            if token.isdigit():
                try:
                    cur = cur[int(token)]
                except Exception:
                    return default
            else:
                if isinstance(cur, dict):
                    cur = cur.get(token)
                else:
                    return default
            token = ""
    return default if cur is None else cur


def _extract_text_from_response(r, *, response_type="json", text_path="", fallback_paths=None):
    """从 HTTP 响应抽取文本，供 custom-http-asr 复用。"""
    fallback_paths = fallback_paths or []
    if response_type == "text":
        return (r.text or "").strip()
    if response_type == "json":
        data = r.json()
        paths = [text_path, *(fallback_paths or [])]
        for p in paths:
            val = _json_path_get(data, p)
            if val is not None and str(val).strip():
                return str(val).strip()
        if isinstance(data, dict):
            for k in ("text", "result", "transcript", "transcription", "data"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if isinstance(v, dict):
                    for kk in ("text", "transcript", "transcription", "result"):
                        vv = v.get(kk)
                        if isinstance(vv, str) and vv.strip():
                            return vv.strip()
        return ""
    return (r.text or "").strip()


def save_pcm_wav(pcm: bytes, path: str, sr: int = 16000) -> str:
    """裸 s16le PCM → wav 落盘（同传译后语音存档）。返回写入路径；空 PCM 不写、返回 ""。"""
    import wave
    if not pcm:
        return ""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return path


def save_tts_audio(raw: bytes, path_noext: str, sr: int = 16000) -> str:
    """译后语音落盘，按字节头自动识别格式：MP3/WAV 原样存，裸 PCM 包成 wav。返回实际路径(带扩展名)。

    各家 TTS 编码不一(平台=MP3、讯飞/qwen 多为裸 PCM)，统一用这个免得当 PCM 错包。
    """
    if not raw:
        return ""
    os.makedirs(os.path.dirname(path_noext) or ".", exist_ok=True)
    if raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa", b"\xff\xf2"):
        p = path_noext + ".mp3"
    elif raw[:4] == b"RIFF":
        p = path_noext + ".wav"
    else:                                  # 裸 s16le PCM → 包 wav
        return save_pcm_wav(raw, path_noext + ".wav", sr)
    with open(p, "wb") as f:
        f.write(raw)
    return p


@dataclass
class TranscribeResult:
    text: str
    elapsed_s: float          # 端到端耗时（请求→完整结果）
    extra: dict               # 模型附带信息（情感等）
    ok: bool = True
    error: str = ""


def _sanitize_nan(obj):
    """递归把 NaN/Inf 浮点换成 None。job summary 会被 json.dump 进 jobs.json 并由
    FastAPI 严格 JSON 编码返回——NaN/Inf 会让 /api/jobs、/api/overview 直接 500。"""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _runtime_request_parts(adapter) -> dict[str, dict]:
    """Split validated per-run values by transport location.

    The dashboard allowlists these values against the persisted OpenAPI/manual
    schema. This second boundary still refuses credential-like fields so a
    hand-edited interfaces.json cannot turn task parameters into secret
    injection.
    """
    parts = {"form": {}, "query": {}, "json": {}, "header": {}}
    values = getattr(adapter, "request_params", None) or {}
    schema = {
        str(field.get("name")): field
        for field in (getattr(adapter, "request_schema", None) or [])
        if isinstance(field, dict) and field.get("name")
    }
    for name, value in values.items():
        field = schema.get(str(name))
        if not field:
            continue
        if is_sensitive_request_param(name):
            continue
        location = str(field.get("in") or "form").lower()
        if location in parts:
            parts[location][str(name)] = value
    return parts


def _task_request_parts(adapter, *, language="", target_lang="", hotwords="") -> dict[str, dict]:
    """Route editable task-owned fields back to their registered wire names."""
    parts = {"form": {}, "query": {}, "json": {}, "header": {}}
    values = {"language": language, "target_lang": target_lang, "hotwords": hotwords}
    for field in getattr(adapter, "request_schema", None) or []:
        if not isinstance(field, dict):
            continue
        managed = field.get("managed_by")
        value = values.get(managed)
        if managed not in values or value in (None, ""):
            continue
        location = str(field.get("in") or "form").lower()
        if location in parts:
            parts[location][str(field.get("name"))] = value
    return parts


def _form_params(values: dict) -> dict:
    out = {}
    for key, value in values.items():
        if isinstance(value, bool):
            out[key] = str(value).lower()
        elif isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = "" if value is None else str(value)
    return out


# 运行页参数契约。任务级字段仍由 runner 负责正确路由，但会在「接口 × 数据集」
# 面板中开放覆盖；端点字段则通过 request_params 透传。平台条目来自该服务
# 在线 openapi.json，其余条目来自本文件实际 adapter 请求结构。
_PLAT_TARGET_LANGS = [
    "None", "Arabic", "Azerbaijani", "Bosnian", "Bulgarian", "Cantonese",
    "Catalan", "Chinese", "Croatian", "Czech", "Dutch", "English", "Estonian",
    "French", "Galician", "German", "Indonesian", "Italian", "Japanese", "Korean",
    "Latvian", "Malay", "Persian", "Portuguese", "Romanian", "Russian", "Slovak",
    "Spanish", "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese",
]
_PLAT_WS_TARGET_LANGS = [
    "Arabic", "Cantonese", "Chinese", "Dutch", "English", "French", "German",
    "Indonesian", "Italian", "Japanese", "Korean", "Malay", "Portuguese",
    "Russian", "Spanish", "Thai", "Turkish", "Urdu", "Vietnamese",
]


def _task_field(name, *, location="form", default=None, enum=None, managed_by=None,
                description="", required=False, wire_name="", api_default=None,
                eval_default=None):
    out = {"name": name, "in": location, "type": "boolean" if isinstance(default, bool) else "string",
           "managed_by": managed_by or name}
    if default is not None:
        out["default"] = default
    if enum:
        out["enum"] = enum
    if description:
        out["description"] = description
    if required:
        out["required"] = True
    if wire_name:
        out["wire_name"] = wire_name
    if api_default is not None:
        out["api_default"] = api_default
    if eval_default is not None:
        out["eval_default"] = eval_default
    return out


_LANGUAGE_FORM = _task_field(
    "language", default="auto", managed_by="language",
    description="输入语音语言；auto 自动识别，多个候选用英文逗号分隔。",
)
_TARGET_FORM = _task_field(
    "target_lang", enum=_PLAT_TARGET_LANGS, managed_by="target_lang",
    description="目标翻译语言；不指定或选择 None 表示不翻译。",
)
_HOTWORDS_FORM = _task_field(
    "hotwords", default="", managed_by="hotwords",
    description="英文逗号分隔；最多 64 项，单项最多 64 字符，规范化后总长度最多 2048 字符。",
)
_AUDIO_FILE_FORM = {
    "name": "file", "in": "form", "type": "string", "format": "binary",
    "required": True, "managed_by": "audio", "editable": False,
    "description": "音频由数据集清单提供；运行时自动转成接口要求的上传文件。",
}
_LITE_AUDIO_FORM = {
    **_AUDIO_FILE_FORM, "name": "audio", "wire_name": "audio",
}
_TEXT_INPUT_JSON = {
    "name": "text", "in": "json", "type": "string", "required": True,
    "managed_by": "source_text", "editable": False,
    "description": "待处理文本由数据集样本的 source_text 提供。",
}

BUILTIN_REQUEST_CONTRACTS = {
    "light": {"source": "light-openapi", "status": "verified", "kind": "endpoint",
             "endpoint": "/asr_lite", "protocol": "http-multipart",
             "verified_version": "0.2.0", "verified_at": "2026-08-03", "request_schema": [
        _LITE_AUDIO_FORM,
        _task_field("language", default="auto", managed_by="language",
                    enum=["auto", "中文", "英文", "日文"],
                    description="语言：auto 自动检测，或指定中文、英文、日文。"),
        {"name": "itn", "in": "form", "type": "boolean", "default": False,
         "api_default": True, "eval_default": False,
         "description": "接口默认开启 ITN；标准 CER/WER 评测默认关闭。"},
        _task_field("hotwords", default="", managed_by="hotwords",
                    description="热词列表，逗号分隔。"),
    ]},
    "light-mlt": {"source": "light-openapi", "status": "verified", "kind": "endpoint",
                  "endpoint": "/asr_mlt_nano", "protocol": "http-multipart",
                  "verified_version": "0.2.0", "verified_at": "2026-08-07", "request_schema": [
        _LITE_AUDIO_FORM,
        _task_field("language", default="auto", managed_by="language",
                    enum=["auto", "中文", "英文", "粤语", "日文", "韩文", "越南语", "印尼语",
                          "泰语", "马来语", "菲律宾语", "阿拉伯语", "印地语", "保加利亚语",
                          "克罗地亚语", "捷克语", "丹麦语", "荷兰语", "爱沙尼亚语", "芬兰语",
                          "希腊语", "匈牙利语", "爱尔兰语", "拉脱维亚语", "立陶宛语", "马耳他语",
                          "波兰语", "葡萄牙语", "罗马尼亚语", "斯洛伐克语", "斯洛文尼亚语", "瑞典语"],
                    description="语言：auto 自动检测，或使用服务 OpenAPI 公布的语言枚举。"),
        {"name": "itn", "in": "form", "type": "boolean", "default": False,
         "api_default": True, "eval_default": False,
         "description": "接口默认开启 ITN；标准 CER/WER 评测默认关闭。"},
        _task_field("hotwords", default="", managed_by="hotwords",
                    description="热词列表，逗号分隔。"),
    ]},
    "std": {"source": "platform-openapi", "status": "verified", "kind": "endpoint",
               "endpoint": "/api/asr/std", "protocol": "http-multipart",
               "verified_version": "1.70.6", "verified_at": "2026-08-03", "request_schema": [
        _AUDIO_FILE_FORM,
        {"name": "enable_word_timestamps", "in": "form", "type": "boolean", "default": False,
         "api_default": False, "eval_default": False,
         "description": "启用字级时间戳，会额外调用 高级档强制对齐。"},
        _LANGUAGE_FORM,
    ]},
    "adv": {"source": "platform-openapi", "status": "verified", "kind": "endpoint",
            "endpoint": "/api/asr/adv", "protocol": "http-multipart",
            "verified_version": "1.70.6", "verified_at": "2026-08-03", "request_schema": [
        _AUDIO_FILE_FORM,
        {"name": "enable_word_timestamps", "in": "form", "type": "boolean", "default": False,
         "api_default": False, "eval_default": False,
         "description": "启用字级时间戳。"},
        _TARGET_FORM,
        {"name": "enable_speaker_diarization", "in": "form", "type": "boolean", "default": False,
         "api_default": False, "eval_default": False,
         "description": "启用说话人识别；会调用专用说话人分离服务。"},
        {"name": "enable_text_edit", "in": "form", "type": "boolean", "default": False,
         "api_default": True, "eval_default": False,
         "description": "接口默认开启；标准 CER/WER 评测默认关闭，可在实验中打开。"},
        {"name": "speaker_merge_gap_ms", "in": "form", "type": "integer", "default": 2000,
         "api_default": 2000, "eval_default": 2000, "minimum": 0,
         "depends_on": "enable_speaker_diarization=true",
         "description": "仅在说话人识别开启时生效；0 表示禁用相邻同说话人合并。"},
        _LANGUAGE_FORM,
    ]},
    "adv-domain": {"source": "platform-openapi", "status": "verified", "kind": "endpoint",
                   "endpoint": "/api/asr/adv-domain", "protocol": "http-multipart",
                   "verified_version": "1.70.6", "verified_at": "2026-08-03", "request_schema": [
        _AUDIO_FILE_FORM,
        {"name": "domain", "in": "form", "type": "string", "default": "medical",
         "enum": ["legal", "medical", "finance", "government_emergency"],
         "required": True, "api_default": None, "eval_default": "medical",
         "description": "领域专业识别端点的必填行业领域。"},
        {"name": "enable_word_timestamps", "in": "form", "type": "boolean", "default": False,
         "description": "启用字级时间戳。"},
        _TARGET_FORM,
        {**_HOTWORDS_FORM, "wire_name": "hot_words",
         "description": "额外热词 JSON 数组；adapter 会把评测热词转换为该字段。"},
        _LANGUAGE_FORM,
    ]},
    "sse": {"source": "platform-openapi", "status": "verified", "kind": "endpoint",
                  "endpoint": "/api/asr/sse", "protocol": "http-sse",
                  "verified_version": "1.70.6", "verified_at": "2026-08-03", "request_schema": [
        _AUDIO_FILE_FORM,
        _task_field("target_lang", enum=_PLAT_TARGET_LANGS, managed_by="target_lang"),
        {"name": "abbreviations", "in": "form", "type": "string", "default": "",
         "description": "缩写词自动替换，填写 JSON 对象，例如 {\"简称\":\"完整名称\"}。"},
        {"name": "enable_sop", "in": "form", "type": "boolean", "default": False,
         "description": "启用 SOP 模式，允许返回创建笔记等操作指令。"},
        {"name": "record_only", "in": "form", "type": "boolean", "default": False,
         "description": "仅录音模式；跳过文本模型后处理，直接返回 ASR 原始结果。"},
        _LANGUAGE_FORM, _HOTWORDS_FORM,
    ]},
    "plat-minutes": {"source": "platform-openapi", "status": "verified", "kind": "endpoint",
                   "endpoint": "/api/asr/meeting-minutes", "protocol": "http-multipart",
                   "verified_version": "1.70.6", "verified_at": "2026-08-03", "request_schema": [
        _AUDIO_FILE_FORM,
        _task_field("target_lang", enum=_PLAT_TARGET_LANGS, managed_by="target_lang"),
        {"name": "enable_diarize", "in": "form", "type": "boolean", "default": True,
         "description": "启用会议说话人识别。"},
        _LANGUAGE_FORM,
    ]},
    "plat-diar": {"source": "platform-openapi+adapter", "status": "verified",
                "kind": "evaluation_profile", "profile_of": "adv",
                "endpoint": "/api/asr/adv", "protocol": "http-multipart",
                "fixed_request": {"enable_speaker_diarization": True},
                "verified_version": "1.70.6", "verified_at": "2026-08-03", "request_schema": [
        _AUDIO_FILE_FORM,
        {"name": "enable_speaker_diarization", "wire_name": "enable_speaker_diarization",
         "in": "form", "type": "boolean", "default": True, "api_default": False,
         "eval_default": True, "managed_by": "profile", "editable": False,
         "description": "该评测 profile 固定开启说话人识别。"},
        {"name": "speaker_merge_gap_ms", "in": "form", "type": "integer", "default": 2000,
         "minimum": 0, "description": "相邻同说话人分段的合并阈值（毫秒）；0 表示不合并。"},
        _LANGUAGE_FORM,
    ]},
    "plat-formula": {"source": "platform-openapi", "status": "no_editable_params",
                   "kind": "endpoint", "endpoint": "/api/text/tts_formula_helper",
                   "protocol": "http-sse", "verified_version": "1.70.6",
                   "verified_at": "2026-08-03", "request_schema": [_TEXT_INPUT_JSON]},
    "sensevoice": {"source": "adapter", "status": "verified", "request_schema": [
        _task_field("language", location="query", default="zh", managed_by="language"),
    ]},
    "gemma-text": {"source": "adapter", "status": "verified", "request_schema": [
        _task_field("target_lang", location="json", managed_by="target_lang"),
        {"name": "temperature", "in": "json", "type": "number", "default": 0,
         "minimum": 0, "maximum": 2},
        {"name": "max_tokens", "in": "json", "type": "integer", "default": 512,
         "minimum": 1},
    ]},
    "cascade-gemma": {"source": "adapter", "status": "verified", "request_schema": [
        _task_field("target_lang", location="json", managed_by="target_lang"),
    ]},
    "plat-simult": {"source": "platform-schema+adapter", "status": "verified",
                   "kind": "endpoint", "endpoint": "/ws/audio/simult-interpreting",
                   "protocol": "websocket-binary-pcm",
                   "fixed_request": {
                       "audio_setting": {"sample_rate": 16000, "format": "pcm", "channel": 1},
                       "vad_setting": {"threshold": 0.5, "silence_duration": 700,
                                       "min_speech_duration": 300, "soft_max_duration": 15000,
                                       "hard_max_duration": 30000, "soft_silence_duration": 300},
                       "tts_setting.enable": "由是否保存译后音频管理",
                   },
                   "request_schema": [
        _task_field("target_lang", managed_by="target_lang", required=True,
                    description="目标翻译语言，接口接受任意非空语言名称。"),
        _LANGUAGE_FORM,
        {**_HOTWORDS_FORM, "wire_name": "voice_input_setting.hot_words"},
        {"name": "abbreviations", "in": "json", "type": "object", "default": {},
         "wire_name": "voice_input_setting.abbreviations",
         "description": "缩写词到完整名称的映射。"},
        {"name": "pipeline_mode", "in": "json", "type": "string", "default": "single_stage",
         "wire_name": "voice_input_setting.pipeline_mode",
         "enum": ["single_stage", "multi_stage"]},
        {"name": "return_asr_text", "in": "json", "type": "boolean", "default": False,
         "wire_name": "voice_input_setting.return_asr_text",
         "depends_on": "pipeline_mode=multi_stage",
         "description": "旧部署兼容字段；当前服务始终在 segment done 返回 original_text。"},
        {"name": "incremental_enabled", "in": "json", "type": "boolean", "default": False,
         "wire_name": "voice_input_setting.incremental_enabled",
         "description": "开启同段增量预览；多阶段复用 Qwen3-ASR 流式会话并输出 asr/translated 两阶段更新。"},
        {"name": "incremental_interval_ms", "in": "json", "type": "integer", "default": 1000,
         "wire_name": "voice_input_setting.incremental_interval_ms",
         "minimum": 500, "maximum": 5000, "depends_on": "incremental_enabled=true",
         "description": "同段增量识别的基础调度间隔（毫秒）。"},
        {"name": "incremental_holdback_tokens", "in": "json", "type": "integer", "default": 1,
         "wire_name": "voice_input_setting.incremental_holdback_tokens",
         "minimum": 0, "maximum": 16, "depends_on": "incremental_enabled=true",
         "description": "LocalAgreement 暂缓确认的尾部 token 数；越大通常越稳但确认更慢。"},
        {"name": "vad_threshold", "in": "json", "type": "number", "default": 0.5,
         "wire_name": "vad_setting.threshold", "minimum": 0, "maximum": 1,
         "description": "VAD 语音概率阈值；会影响切段、质量和延迟。"},
        {"name": "vad_silence_duration_ms", "in": "json", "type": "integer", "default": 700,
         "wire_name": "vad_setting.silence_duration", "minimum": 0,
         "description": "普通模式下触发切段的静音时长（毫秒）。"},
        {"name": "vad_min_speech_duration_ms", "in": "json", "type": "integer", "default": 300,
         "wire_name": "vad_setting.min_speech_duration", "minimum": 0,
         "description": "最短有效语音时长（毫秒）。"},
        {"name": "vad_soft_max_duration_ms", "in": "json", "type": "integer", "default": 15000,
         "wire_name": "vad_setting.soft_max_duration", "minimum": 0,
         "description": "超过该段长后进入敏感切段模式（毫秒）。"},
        {"name": "vad_hard_max_duration_ms", "in": "json", "type": "integer", "default": 30000,
         "wire_name": "vad_setting.hard_max_duration", "minimum": 0,
         "description": "强制切段的最大段长（毫秒），不能小于 soft max。"},
        {"name": "vad_soft_silence_duration_ms", "in": "json", "type": "integer", "default": 300,
         "wire_name": "vad_setting.soft_silence_duration", "minimum": 0,
         "description": "敏感模式下触发切段的静音时长（毫秒）。"},
        {"name": "tts_mode", "in": "json", "type": "string", "default": "preset",
         "wire_name": "tts_setting.mode", "enum": ["preset", "clone"],
         "description": "译后语音模式；只有启用保存译后音频时才会执行 TTS。"},
        {"name": "tts_output_sample_rate", "in": "json", "type": "integer", "default": 16000,
         "wire_name": "tts_setting.output_sample_rate", "minimum": 8000, "maximum": 48000,
         "description": "译后 PCM 输出采样率。"},
        {"name": "tts_voice_id", "in": "json", "type": "string", "default": "",
         "wire_name": "tts_setting.voice_id",
         "depends_on": "tts_mode=preset", "description": "预置 TTS 音色 ID；留空由服务端按目标语言选择。"},
    ]},
    "plat-simult-ws": {"source": "platform-schema+adapter", "status": "verified",
                     "kind": "evaluation_profile", "profile_of": "voice-input",
                     "endpoint": "/ws/audio/voice-input", "protocol": "websocket-binary-pcm",
                     "fixed_request": {
                         "audio_setting": {"sample_rate": 16000, "format": "pcm", "channel": 1},
                         "vad_setting": {"threshold": 0.5, "silence_duration": 600,
                                         "min_speech_duration": 300, "soft_max_duration": 15000,
                                         "hard_max_duration": 30000, "soft_silence_duration": 300},
                         "voice_input_setting.hot_words": [],
                         "voice_input_setting.abbreviations": {},
                         "voice_input_setting.summary_interval_s": 0,
                     },
                     "request_schema": [
        _task_field("target_lang", managed_by="target_lang", enum=_PLAT_WS_TARGET_LANGS),
        _LANGUAGE_FORM,
        {"name": "vad_threshold", "in": "json", "type": "number", "default": 0.5,
         "wire_name": "vad_setting.threshold", "minimum": 0, "maximum": 1,
         "description": "VAD 语音概率阈值；会影响切段、质量和延迟。"},
        {"name": "vad_silence_duration_ms", "in": "json", "type": "integer", "default": 600,
         "wire_name": "vad_setting.silence_duration", "minimum": 0,
         "description": "触发普通切段的静音时长（毫秒）。"},
        {"name": "vad_min_speech_duration_ms", "in": "json", "type": "integer", "default": 300,
         "wire_name": "vad_setting.min_speech_duration", "minimum": 0,
         "description": "最短有效语音时长（毫秒）。"},
        {"name": "vad_soft_max_duration_ms", "in": "json", "type": "integer", "default": 15000,
         "wire_name": "vad_setting.soft_max_duration", "minimum": 0,
         "description": "超过该段长后进入敏感切段模式（毫秒）。"},
        {"name": "vad_hard_max_duration_ms", "in": "json", "type": "integer", "default": 30000,
         "wire_name": "vad_setting.hard_max_duration", "minimum": 0,
         "description": "强制切段的最大段长（毫秒），不能小于 soft max。"},
        {"name": "vad_soft_silence_duration_ms", "in": "json", "type": "integer", "default": 300,
         "wire_name": "vad_setting.soft_silence_duration", "minimum": 0,
         "description": "敏感模式下触发切段的静音时长（毫秒）。"},
        {"name": "summary_interval_s", "in": "json", "type": "integer", "default": 0,
         "wire_name": "voice_input_setting.summary_interval_s", "minimum": 0,
         "description": "服务端阶段性总结间隔；同传质量/延迟横评默认关闭。"},
    ]},
    "plat-simult-post": {"source": "platform-openapi", "status": "verified",
                       "kind": "hidden_legacy_endpoint",
                       "endpoint": "/api/asr/simult-interpreting-sse",
                       "protocol": "http-sse", "verified_version": "1.70.6",
                       "verified_at": "2026-08-03", "request_schema": [
        _AUDIO_FILE_FORM,
        _task_field("target_lang", managed_by="target_lang", required=True,
                    description="目标翻译语言（必填）。"),
        _HOTWORDS_FORM,
        {"name": "enable_tts", "in": "form", "type": "boolean", "default": False},
        {"name": "tts_mode", "in": "form", "type": "string", "default": "preset",
         "enum": ["preset", "clone"]},
        {"name": "tts_gender", "in": "form", "type": "string", "default": "",
         "enum": ["", "f", "m"]},
        {"name": "tts_voice_id", "in": "form", "type": "string", "default": ""},
        _LANGUAGE_FORM,
    ]},
    "plat-realtime": {"source": "platform-openapi+adapter", "status": "verified",
                    "kind": "endpoint", "endpoint": "/v1/realtime",
                    # 版本实测自线上 /health，与源码声明一致。
                    "protocol": "websocket", "verified_version": "1.70.33",
                    "verified_at": "2026-09-14", "request_schema": [
        _LANGUAGE_FORM,
    ]},
    "qwen3-asr-ws": {"source": "qwen3-asr-openapi+adapter", "status": "verified",
                     "kind": "endpoint", "endpoint": "/ws", "protocol": "websocket",
                     "verified_version": "qwen-asr==0.0.6", "verified_at": "2026-09-14",
                     "request_schema": [_LANGUAGE_FORM]},
    "qwen-simult": {"source": "adapter", "status": "verified", "request_schema": [
        _task_field("target_lang", managed_by="target_lang",
                    enum=["English", "Chinese", "Japanese", "Korean", "Spanish",
                          "Cantonese", "French", "German", "Russian"]),
    ]},
    "qwen-simult-offline": {"source": "adapter", "status": "verified", "request_schema": [
        _task_field("target_lang", managed_by="target_lang",
                    enum=["English", "Chinese", "Japanese", "Korean", "Spanish",
                          "Cantonese", "French", "German", "Russian"]),
    ]},
    "doubao-simult": {"source": "adapter", "status": "verified", "request_schema": [
        _task_field("target_lang", managed_by="target_lang", enum=["English", "Chinese"]),
    ]},
    "xf-simult": {"source": "adapter", "status": "verified", "request_schema": [
        _task_field("target_lang", managed_by="target_lang", enum=["English"],
                    description="当前已核实服务仅支持中文源语音翻译为英文。"),
    ]},
    "xf-spark-slm-iat": {"source": "adapter", "status": "no_editable_params", "request_schema": []},
}


# 能力契约描述“接口真实能做什么”；与 request_schema（用户能改什么）分离。
# 未核实的限制保留 unknown，而不是根据 adapter 能绕到的其他端点推导能力。
BUILTIN_CAPABILITY_CONTRACTS = {
    "light": {"status": "verified", "source": "light-openapi", "kind": "endpoint",
             "endpoint": "/asr_lite", "verified_version": "0.2.0",
             "tasks": ["asr"], "input_modalities": ["audio_file"],
             "output_modalities": ["final_text"],
             "features": {"hotwords": True, "domain": False, "itn": True,
                          "timestamps": False}},
    "light-mlt": {"status": "verified", "source": "light-openapi", "kind": "endpoint",
                  "endpoint": "/asr_mlt_nano", "verified_version": "0.2.0",
                  "tasks": ["asr"], "input_modalities": ["audio_file"],
                  "output_modalities": ["final_text"],
                  "features": {"hotwords": True, "domain": False, "itn": True,
                               "timestamps": False}},
    "std": {"status": "verified", "source": "platform-openapi", "kind": "endpoint",
               "endpoint": "/api/asr/std", "verified_version": "1.70.6",
               "tasks": ["asr"], "input_modalities": ["audio_file"],
               "output_modalities": ["final_text", "utterances", "audio_info", "word_timestamps"],
               "output_conditions": {"word_timestamps": "enable_word_timestamps=true"},
               "features": {"hotwords": False, "domain": False, "diarization": False,
                            "speech_translation": False}},
    "adv": {"status": "verified", "source": "platform-openapi", "kind": "endpoint",
            "endpoint": "/api/asr/adv", "verified_version": "1.70.6",
            "tasks": ["asr", "speech_translation"], "input_modalities": ["audio_file"],
            "output_modalities": ["result", "raw_result", "alt_result", "translation",
                                  "word_timestamps", "speaker_segments", "speaker_overlap",
                                  "pipeline_status", "pipeline_warnings", "audio_info"],
            "output_conditions": {"translation": "target_lang is set",
                                  "word_timestamps": "enable_word_timestamps=true",
                                  "speaker_segments": "enable_speaker_diarization=true"},
            "features": {"hotwords": False, "domain": False, "diarization": True,
                         "overlap_detection": True, "word_timestamps": True,
                         "degraded_status": True}},
    "adv-domain": {"status": "verified", "source": "platform-openapi", "kind": "endpoint",
                   "endpoint": "/api/asr/adv-domain", "verified_version": "1.70.6",
                   "tasks": ["asr", "speech_translation"], "input_modalities": ["audio_file"],
                   "output_modalities": ["edited_text", "utterances", "translation",
                                         "word_timestamps", "audio_info"],
                   "output_conditions": {"translation": "target_lang is set",
                                         "word_timestamps": "enable_word_timestamps=true"},
                   "features": {"hotwords": True, "domain": True, "diarization": False},
                   "domains": ["legal", "medical", "finance", "government_emergency"]},
    "qwen3-asr-ws": {"status": "verified", "source": "qwen3-asr-openapi+adapter", "kind": "endpoint",
                     "endpoint": "/ws", "verified_version": "qwen-asr==0.0.6",
                     "tasks": ["asr"], "input_modalities": ["audio_stream"],
                     "output_modalities": ["final_text", "incremental_text"],
                     "request_events": ["start", "finish"],
                     "response_events": ["started", "result", "error"],
                     "streaming": {"transport": "websocket", "input": True, "output": True,
                                   "granularity": "segment", "realtime": True, "paced_1x": False},
                     "output_conditions": {"incremental_text": "每 chunk 回一版累积文本",
                                           "final_text": "finish 后 final=true 的终稿"},
                     "features": {"hotwords": False, "diarization": False, "timestamps": False,
                                  "explicit_done_event": True, "binary_float32_frames": True,
                                  "concurrency_cap": 2,
                                  "concurrency_note": "线上同传产品共用此端点；评测并发必须 ≤2"}},
    "plat-realtime": {"status": "verified", "source": "platform-openapi+adapter", "kind": "endpoint",
                    "endpoint": "/v1/realtime", "verified_version": "1.70.6",
                    "tasks": ["asr"], "input_modalities": ["audio_stream"],
                    "output_modalities": ["final_text", "segment_text"],
                    "request_events": ["session.update", "input_audio_buffer.append",
                                       "input_audio_buffer.commit"],
                    "response_events": ["session.created", "session.updated",
                                        "input_audio_buffer.speech_started",
                                        "input_audio_buffer.speech_stopped",
                                        "input_audio_buffer.commited",
                                        "conversation.item.created",
                                        "conversation.item.input_audio_transcription.delta",
                                        "conversation.item.input_audio_transcription.completed",
                                        "error"],
                    "streaming": {"transport": "websocket", "input": True, "output": True,
                                  "granularity": "segment", "realtime": True, "paced_1x": True},
                    "output_conditions": {"segment_text": "以 .completed 逐段回显，无终止事件"},
                    "features": {"hotwords": False, "diarization": False, "timestamps": False,
                                 "base64_json_audio_frames": True, "binary_frames": False,
                                 "explicit_done_event": False}},
    "sse": {"status": "verified", "source": "platform-openapi+adapter", "kind": "endpoint",
                  "endpoint": "/api/asr/sse", "verified_version": "1.70.6",
                  "tasks": ["asr", "speech_translation"], "input_modalities": ["audio_file"],
                  "output_modalities": ["raw_text", "edited_text", "translation", "sop_actions",
                                        "formatting_metadata", "processing_stats"],
                  "response_events": ["start", "asr_result", "sop", "done", "error"],
                  "streaming": {"transport": "sse", "input": False, "output": True,
                                "granularity": "stage_event", "token_delta": False},
                  "features": {"hotwords": True, "dual_result": True, "sop": True,
                               "record_only": True, "abbreviations": True}},
    "plat-simult": {"status": "verified", "source": "platform-schema+adapter", "kind": "endpoint",
                  "endpoint": "/ws/audio/simult-interpreting",
                  "tasks": ["simultaneous_translation"], "input_modalities": ["audio_stream"],
                  "output_modalities": ["incremental_translation", "segment_final_translation",
                                        "translated_audio", "source_audio", "segment_timings",
                                        "intermediate_asr_text"],
                  "output_conditions": {"translated_audio": "tts_setting.enable=true",
                                        "intermediate_asr_text": "pipeline_mode=multi_stage and return_asr_text=true"},
                  "language_pairs": [{"source": ["*"], "target": ["*"]}],
                  "language_pair_semantics": "accepted_by_protocol; quality and TTS voice coverage are separate",
                  "request_events": ["task_start", "task_pause", "task_resume", "task_finish"],
                  "response_events": ["connected_success", "task_started", "si_segment_start",
                                      "si_incremental_update", "si_incremental_end",
                                      "si_token", "si_segment_done", "si_segment_error",
                                      "si_tts_start", "si_tts_file", "si_tts_end",
                                      "si_source_audio", "task_paused", "task_resumed",
                                      "task_finished", "task_failed"],
                  "streaming": {"transport": "websocket", "input": True, "output": True,
                                "granularity": "token+segment", "realtime": True, "paced_1x": True},
                  "context": {"mode": "rolling_previous_asr_tail", "scope": "session",
                              "condition": "pipeline_mode=multi_stage"},
                  "features": {"hotwords": True, "abbreviations": True,
                               "incremental_same_segment": True, "tts": True,
                               "tts_modes": ["preset", "clone"],
                               "partial_on_disconnect": True}},
    "plat-simult-ws": {"status": "verified", "source": "platform-schema+adapter",
                     "kind": "evaluation_profile", "profile_of": "voice-input",
                     "endpoint": "/ws/audio/voice-input",
                     "tasks": ["simultaneous_translation"], "input_modalities": ["audio_stream"],
                     "output_modalities": ["source_segment_text", "segment_final_translation",
                                           "segment_timings", "final_summary"],
                     "language_pairs": [{"source": ["*"], "target": _PLAT_WS_TARGET_LANGS}],
                     "request_events": ["task_start", "task_pause", "task_resume", "task_finish"],
                     "response_events": ["connected_success", "task_started", "result_final",
                                         "result_summary", "task_paused", "task_resumed",
                                         "task_finished", "task_failed"],
                     "streaming": {"transport": "websocket", "input": True, "output": True,
                                   "granularity": "segment", "realtime": True, "paced_1x": True},
                     "context": {"mode": "server_session", "scope": "session"},
                     "features": {"hotwords": False, "abbreviations": False,
                                  "tts": False, "partial_on_disconnect": True}},
    "plat-minutes": {"status": "verified", "source": "platform-openapi", "kind": "endpoint",
                   "endpoint": "/api/asr/meeting-minutes", "verified_version": "1.70.6",
                   "tasks": ["meeting_summary"], "input_modalities": ["audio_file"],
                   "output_modalities": ["markdown_summary", "transcript", "speaker_segments",
                                         "translation", "topics", "decisions", "participants",
                                         "commitments"],
                   "processing": {"mode": "synchronous", "long_timeout_recommended": True},
                   "features": {"diarization": True, "long_audio": True,
                                "structured_analysis": True}},
    "plat-formula": {"status": "verified", "source": "platform-openapi", "kind": "endpoint",
                   "endpoint": "/api/text/tts_formula_helper", "verified_version": "1.70.6",
                   "tasks": ["formula_normalization"], "input_modalities": ["text"],
                   "output_modalities": ["incremental_spoken_text", "spoken_text"],
                   "response_events": ["start", "token", "done", "error"],
                   "streaming": {"transport": "sse", "input": False, "output": True,
                                 "granularity": "token"}},
    "plat-diar": {"status": "verified", "source": "platform-openapi+adapter",
                "kind": "evaluation_profile", "profile_of": "adv",
                "endpoint": "/api/asr/adv", "tasks": ["diarization"],
                "input_modalities": ["audio_file"],
                "output_modalities": ["speaker_segments", "speaker_overlap", "diarization_backend",
                                      "pipeline_status"],
                "evaluation_projection": ["speaker_segments"],
                "features": {"speaker_merge": True, "overlap_detection": True, "long_audio": True}},
    "plat-simult-post": {"status": "verified", "source": "platform-openapi",
                       "kind": "hidden_legacy_endpoint",
                       "endpoint": "/api/asr/simult-interpreting-sse",
                       "verified_version": "1.70.6", "tasks": ["speech_translation"],
                       "input_modalities": ["audio_file"],
                       "output_modalities": ["incremental_translation", "final_translation",
                                             "translated_audio"],
                       "response_events": ["start", "token", "done", "tts_audio", "error"],
                       "streaming": {"transport": "sse", "input": False, "output": True,
                                     "granularity": "token"},
                       "features": {"hotwords": True, "tts": True,
                                    "tts_modes": ["preset", "clone"]}},
    "sensevoice": {"status": "verified", "source": "adapter", "tasks": ["asr"],
                   "input_modalities": ["audio_file"], "output_modalities": ["final_text", "emotion"]},
    "xf-spark-slm-iat": {"status": "verified", "source": "adapter", "tasks": ["asr"],
                         "input_modalities": ["audio_stream"], "output_modalities": ["incremental_text", "final_text"],
                         "features": {"dialect_auto_detection": True}},
    "gemma-text": {"status": "verified", "source": "adapter", "tasks": ["text_translation", "summarization"],
                   "input_modalities": ["text"], "output_modalities": ["final_text"]},
    "cascade-gemma": {"status": "verified", "source": "adapter", "tasks": ["speech_translation"],
                    "input_modalities": ["audio_file"], "output_modalities": ["translation", "intermediate_asr_text"],
                    "pipeline": ["ext-adv", "gemma-text"]},
    "qwen-simult": {"status": "verified", "source": "adapter", "tasks": ["simultaneous_translation"],
                    "input_modalities": ["audio_stream"], "output_modalities": ["incremental_text", "final_text"],
                    "language_pairs": [{"source": ["en", "zh", "ja", "ko", "es", "yue", "fr", "de", "ru"],
                                        "target": ["en", "zh", "ja", "ko", "es", "yue", "fr", "de", "ru"]}],
                    "streaming": {"transport": "websocket", "input": True, "output": True,
                                  "granularity": "cumulative_text", "revision": True,
                                  "realtime": True, "paced_1x": True}},
    "qwen-simult-offline": {"status": "verified", "source": "adapter", "tasks": ["offline_speech_translation"],
                            "input_modalities": ["audio_file"], "output_modalities": ["final_text"],
                            "language_pairs": [{"source": ["en", "zh", "ja", "ko", "es", "yue", "fr", "de", "ru"],
                                                "target": ["en", "zh", "ja", "ko", "es", "yue", "fr", "de", "ru"]}],
                            "streaming": {"transport": "websocket", "input": True, "output": True,
                                          "granularity": "cumulative_text", "revision": True,
                                          "realtime": False, "paced_1x": False}},
    "doubao-simult": {"status": "unavailable", "source": "adapter+live-smoke", "tasks": ["simultaneous_translation"],
                      "input_modalities": ["audio_stream"], "output_modalities": ["incremental_text"],
                      "language_pairs": [{"source": ["zh"], "target": ["en"]},
                                         {"source": ["en"], "target": ["zh"]}],
                      "streaming": {"transport": "websocket", "input": True, "output": True,
                                    "granularity": "incremental", "realtime": True, "paced_1x": True}},
    "xf-simult": {"status": "verified", "source": "adapter+live-smoke", "tasks": ["simultaneous_translation"],
                  "input_modalities": ["audio_stream"], "output_modalities": ["incremental_text", "final_text"],
                  "language_pairs": [{"source": ["zh"], "target": ["en"]}],
                  "streaming": {"transport": "websocket", "input": True, "output": True,
                                "granularity": "segment", "realtime": True, "paced_1x": True}},
}

TEMPLATE_REQUEST_CONTRACTS = {
    "asr_lite": {"source": "template", "status": "template_default", "request_schema": [
        _LANGUAGE_FORM,
        {"name": "itn", "in": "form", "type": "boolean", "default": False},
        {"name": "return_timestamps", "in": "form", "type": "boolean", "default": False},
        _HOTWORDS_FORM,
    ]},
    "sensevoice": BUILTIN_REQUEST_CONTRACTS["sensevoice"],
    "plat-sse": BUILTIN_REQUEST_CONTRACTS["sse"],
    "plat-sse-dual": BUILTIN_REQUEST_CONTRACTS["sse"],
    "openai-audio": {"source": "template", "status": "template_default", "request_schema": [
        _task_field("language", default="auto", managed_by="language"),
        _task_field("target_language", managed_by="target_lang"),
        {"name": "prompt", "in": "form", "type": "string"},
        {"name": "temperature", "in": "form", "type": "number", "default": 0,
         "minimum": 0, "maximum": 1},
    ]},
    "openai-chat": {"source": "template", "status": "template_default", "request_schema": [
        _task_field("target_lang", location="json", managed_by="target_lang"),
        {"name": "temperature", "in": "json", "type": "number", "default": 0,
         "minimum": 0, "maximum": 2},
        {"name": "top_p", "in": "json", "type": "number", "default": 1,
         "minimum": 0, "maximum": 1},
        {"name": "max_tokens", "in": "json", "type": "integer", "minimum": 1},
    ]},
    "openai-chat-audio": {"source": "template", "status": "template_default", "request_schema": [
        _task_field("language", location="json", default="auto", managed_by="language"),
        {"name": "temperature", "in": "json", "type": "number", "default": 0,
         "minimum": 0, "maximum": 2},
        {"name": "max_tokens", "in": "json", "type": "integer", "minimum": 1},
    ]},
    "custom-http-asr": {"source": "template", "status": "template_default", "request_schema": [
        _task_field("language", location="query", default="auto", managed_by="language"),
        _task_field("hotwords", location="header", default="", managed_by="hotwords"),
    ]},
}


def request_contract_for(model_id="", cfg=None):
    """Return the best available contract without mutating persisted interface config."""
    cfg = cfg or {}
    explicit = cfg.get("request_schema") or []
    if explicit:
        return {"source": cfg.get("request_schema_source") or "registered",
                "status": "verified", "request_schema": explicit}
    if model_id in BUILTIN_REQUEST_CONTRACTS:
        return BUILTIN_REQUEST_CONTRACTS[model_id]
    template = cfg.get("template") or cfg.get("tmpl") or ""
    if template == "plat-multipart":
        path = str(cfg.get("path") or "adv").strip("/").rsplit("/", 1)[-1]
        contract = BUILTIN_REQUEST_CONTRACTS.get(path, BUILTIN_REQUEST_CONTRACTS["adv"])
        endpoint = str(cfg.get("base_url") or cfg.get("url") or cfg.get("endpoint") or "")
        return contract if _is_platform_endpoint(endpoint) else {
            **contract, "source": "plat-template", "status": "template_default",
        }
    if template in TEMPLATE_REQUEST_CONTRACTS:
        contract = TEMPLATE_REQUEST_CONTRACTS[template]
        if template in ("plat-sse", "plat-sse-dual"):
            endpoint = str(cfg.get("base_url") or cfg.get("url") or cfg.get("endpoint") or "")
            if not _is_platform_endpoint(endpoint):
                return {**contract, "source": "plat-template", "status": "template_default"}
        return contract
    return {"source": "unverified", "status": "unverified", "request_schema": []}


def capability_contract_for(model_id="", cfg=None):
    """Return a factual capability declaration, never inferred from route fallbacks."""
    cfg = cfg or {}
    if isinstance(cfg.get("capability_contract"), dict):
        return cfg["capability_contract"]
    if model_id in BUILTIN_CAPABILITY_CONTRACTS:
        return BUILTIN_CAPABILITY_CONTRACTS[model_id]
    caps = list(cfg.get("caps") or [])
    scenarios = list(cfg.get("scenarios") or [])
    task_map = {
        "asr": "asr", "translate_audio": "speech_translation",
        "translate_text": "text_translation", "summarize": "summarization",
        "simult": "simultaneous_translation", "diar": "diarization",
    }
    tasks = list(dict.fromkeys(task_map[c] for c in caps if c in task_map))
    return {
        "status": "declared" if caps else "template_default",
        "source": "interfaces.json" if caps else "template",
        "tasks": tasks or scenarios,
        "input_modalities": ["text"] if cfg.get("template") == "openai-chat" else ["audio_file"],
        "output_modalities": ["final_text"],
        "features": {"hotwords": "hotwords" in caps},
    }


class SenseVoiceAdapter:
    """SenseVoice STT — POST /api/asr_transcribe (multipart file + ?language)。

    入参仅 file + language(zh/en/yue/ja/ko/nospeech)，无热词 → 只测 ASR 场景 a(默认提示词)。
    返回 {"text","emotion"}。
    """

    name = "sensevoice"

    def __init__(self, base_url: str = ENDPOINTS["sensevoice"], timeout: float = 60.0):
        self.url = base_url.rstrip("/") + "/api/asr_transcribe"
        self.timeout = timeout

    def transcribe(self, audio_path: str, language: str = "zh") -> TranscribeResult:
        try:
            wav = load_wav_bytes(audio_path)
            files = {"file": ("audio.wav", wav, "audio/wav")}
            runtime = _runtime_request_parts(self)
            query = {"language": language, **runtime["query"]}
            data = _form_params(runtime["form"])
            t0 = time.perf_counter()
            r = http_post(
                self.url, params=query, data=data, headers=runtime["header"],
                files=files, timeout=self.timeout
            )
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            d = r.json()
            return TranscribeResult(
                text=d.get("text", ""),
                elapsed_s=elapsed,
                extra={"emotion": d.get("emotion")},
            )
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class LightASRAdapter:
    """轻量 ASR 服务 — POST /asr_lite 或 /asr_mlt_nano。

    支持 hotwords → 可测 ASR 场景 b(自有提示词)。不同 base_url 可对应不同后端。
    评测口径对齐旧报告: itn=false(关闭逆文本规范化)。
    """

    supports_hotwords = True   # infer --hotwords 时把 manifest keywords 传进来(空格分隔)

    def __init__(self, base_url: str, name: str = "asr-lite", timeout: float = 60.0,
                 default_itn: bool = False,
                 return_timestamps: bool | None = None,
                 path: str = "/asr_lite"):
        self.url = base_url.rstrip("/") + "/" + path.strip("/")
        self.name = name
        self.timeout = timeout
        self.default_itn = default_itn
        self.return_timestamps = return_timestamps

    def transcribe(self, audio_path: str, language: str = "auto",
                   hotwords: str = "", itn: bool | None = None) -> "TranscribeResult":
        try:
            wav = load_wav_bytes(audio_path)
            files = {"audio": ("audio.wav", wav)}
            runtime = _runtime_request_parts(self)
            use_itn = self.default_itn if itn is None else itn
            data = {"language": language, "itn": str(use_itn).lower()}
            if self.return_timestamps is not None:
                data["return_timestamps"] = str(self.return_timestamps).lower()
            if hotwords:
                data["hotwords"] = hotwords
            data.update(_form_params(runtime["form"]))
            t0 = time.perf_counter()
            r = http_post(self.url, data=data, files=files, params=runtime["query"],
                          headers=runtime["header"], timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            d = r.json()
            return TranscribeResult(text=d.get("text", ""), elapsed_s=elapsed,
                                    extra={"language": d.get("language"),
                                           "timestamps": d.get("timestamps")})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class CustomHTTPASRAdapter:
    """高级自定义 HTTP ASR：支持自由 URL / method / headers / body / 返回字段抽取。

    适合大多数 HTTP 类接口；不覆盖 WebSocket/HMAC 分帧协议。"""

    def __init__(self, cfg, timeout: float = 60.0):
        self.cfg = cfg or {}
        self.name = self.cfg.get("id", "custom-http-asr")
        self.url = (self.cfg.get("base_url") or "").rstrip()
        self.method = (self.cfg.get("method") or "POST").upper()
        self.timeout = timeout
        self.headers = self._parse_json(self.cfg.get("headers_json") or self.cfg.get("headers") or {})
        self.body_json = self._parse_json(self.cfg.get("body_json") or self.cfg.get("body") or {})
        self.body_type = self.cfg.get("body_type") or "multipart"
        self.audio_field = self.cfg.get("audio_field") or "file"
        self.audio_filename = self.cfg.get("audio_filename") or "audio.wav"
        self.audio_mime = self.cfg.get("audio_mime") or "audio/wav"
        self.response_type = self.cfg.get("response_type") or "json"
        self.text_path = self.cfg.get("text_path") or ""
        self.fallback_paths = self._parse_list(self.cfg.get("fallback_paths") or [])
        self.error_path = self.cfg.get("error_path") or ""
        self.query_json = self._parse_json(self.cfg.get("query_json") or {})
        # Optional cleanup for models that decorate the transcript (e.g.
        # the model's inline "Speaker 0:" labels). Applied before scoring so
        # the decoration does not count as recognition errors.
        self.strip_regex = self.cfg.get("strip_regex") or ""

    def _parse_json(self, raw):
        if isinstance(raw, (dict, list)):
            return raw
        if not raw:
            return {}
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return {}

    def _parse_list(self, raw):
        if isinstance(raw, list):
            return [x for x in raw if str(x).strip()]
        if isinstance(raw, str):
            return [x.strip() for x in raw.splitlines() if x.strip()]
        return []

    def transcribe(self, audio_path: str, language: str = "auto", hotwords: str = "",
                   target_lang: str = "") -> TranscribeResult:
        if not audio_path:
            return TranscribeResult("", 0.0, {}, ok=False, error="缺 audio_path")
        if not self.url:
            return TranscribeResult("", 0.0, {}, ok=False, error="缺 base_url")
        try:
            wav = load_wav_bytes(audio_path)
            runtime = _runtime_request_parts(self)
            task_runtime = _task_request_parts(
                self, language=language, target_lang=target_lang, hotwords=hotwords,
            )
            for location in runtime:
                runtime[location].update(task_runtime[location])
            headers = dict(self.headers)
            headers.update({k: str(v) for k, v in runtime["header"].items()})
            has_hotwords_field = any(
                f.get("managed_by") == "hotwords"
                for f in (getattr(self, "request_schema", None) or []) if isinstance(f, dict)
            )
            if hotwords and not has_hotwords_field and "hotwords" not in self.body_json:
                headers.setdefault("X-Hotwords", hotwords)
            query = dict(self.query_json)
            query.update(runtime["query"])
            has_language_field = any(
                f.get("managed_by") == "language"
                for f in (getattr(self, "request_schema", None) or []) if isinstance(f, dict)
            )
            if (not has_language_field and language and language != "auto"
                    and "language" not in self.body_json and "language" not in query):
                query.setdefault("language", language)
            t0 = time.perf_counter()
            if self.body_type == "multipart":
                files = {self.audio_field: (self.audio_filename, wav, self.audio_mime)}
                data = _form_params({
                    **{k: v for k, v in self.body_json.items() if k != self.audio_field},
                    **runtime["form"],
                })
                r = http_request(self.method, self.url, headers=headers, params=query, data=data, files=files,
                                 timeout=self.timeout)
            elif self.body_type == "raw_binary":
                headers.setdefault("Content-Type", self.audio_mime)
                r = http_request(self.method, self.url, headers=headers, params=query, data=wav, timeout=self.timeout)
            else:  # json_base64
                body = dict(self.body_json)
                body.update(runtime["json"])
                body[self.audio_field] = base64.b64encode(wav).decode()
                body.setdefault("audio_format", "wav")
                body.setdefault("sample_rate", 16000)
                r = http_request(self.method, self.url, headers=headers, params=query, json=body, timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            txt = _extract_text_from_response(r, response_type=self.response_type,
                                              text_path=self.text_path, fallback_paths=self.fallback_paths)
            if txt and self.strip_regex:
                try:
                    txt = re.sub(self.strip_regex, " ", txt)
                    txt = re.sub(r"\s+", " ", txt).strip()
                except re.error:
                    pass
            extra = {"response_type": self.response_type, "status_code": r.status_code}
            if not txt and self.error_path:
                try:
                    extra["error_detail"] = str(_json_path_get(r.json(), self.error_path, "") or "")
                except Exception:
                    extra["error_detail"] = (r.text or "")[:200]
            return TranscribeResult(text=txt, elapsed_s=elapsed, extra=extra, ok=bool(txt.strip()),
                                    error="返回空文本" if not txt.strip() else "")
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


# 端点预设
def ext_adv(**kw):
    return LightASRAdapter(ENDPOINTS["ext_adv"], name="asr-adv", **kw)


def light_fun512(**kw):
    return LightASRAdapter(ENDPOINTS["light"], name="asr-lite", **kw)


def light_mlt_nano(**kw):
    return LightASRAdapter(
        ENDPOINTS["light"], name="asr-mlt-nano", path="/asr_mlt_nano", **kw,
    )


class GemmaAudioAdapter:
    """gemma-4-12B-it (vLLM, OpenAI 兼容) — input_audio base64。

    LLM 式 ASR：音频 + 提示词 → 文本。天然支持"自有提示词"(场景 b)。
    音频 ≤30s。注意 LLM 听不清会"编"通顺错文，需结合 CER 谨慎看。
    """
    name = "gemma-audio"
    per_item_lang = True  # 提示词按语言切换 → infer 按 manifest 行内 lang 传，而非全局 --language

    def __init__(self, base_url=ENDPOINTS["gemma"],
                 key=GEMMA_KEY,
                 timeout=90.0):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.key = key
        self.timeout = timeout

    def transcribe(self, audio_path: str, language: str = "zh",
                   prompt: str = "") -> "TranscribeResult":
        import base64
        if not prompt:
            prompt = ("Transcribe this English audio. Output only the transcription."
                      if language == "en"
                      else "请逐字转写这段音频，只输出转写文本，不要标点。")
        try:
            b64 = base64.b64encode(load_wav_bytes(audio_path)).decode()
            body = {"model": "gemma-4-12B-it", "temperature": 0, "max_tokens": 300,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}
            t0 = time.perf_counter()
            r = http_post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                              json=body, timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            return TranscribeResult(text=txt, elapsed_s=elapsed, extra={})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class QwenASRAdapter:
    """阿里 DashScope Qwen3-ASR-Flash（OpenAI 兼容 /chat/completions，input_audio data-URI）。

    ⚠️ Flash 是 Qwen3-ASR 家族的独立商用 API 版（技术报告：Qwen3-ASR-Flash-1208 serves as
    an API），分数高于开源 Qwen3-ASR-1.7B/0.6B，阿里不公开其参数量——勿当"1.7B 小模型"对标。
    音频 <10MB、≤5min。语言经 asr_options.language 传（auto 则不带，让其自检）。
    口径对齐：enable_itn=false（与平台一致）。返回在 choices[0].message.content。
    """
    name = "qwen-asr"
    per_item_lang = True  # 按行内 lang 传 asr_options.language 提升准确率

    def __init__(self, base_url="https://dashscope.aliyuncs.com/compatible-mode",
                 model="qwen3-asr-flash", api_key="", timeout=90.0):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.timeout = timeout

    def transcribe(self, audio_path: str, language: str = "auto",
                   prompt: str = "") -> "TranscribeResult":
        import base64
        if not self.key:
            return TranscribeResult("", 0.0, {}, ok=False, error="缺 DASHSCOPE_API_KEY")
        try:
            b64 = base64.b64encode(load_wav_bytes(audio_path)).decode()
            data_uri = f"data:audio/wav;base64,{b64}"
            asr_opts = {"enable_itn": False}
            if language in ("zh", "en", "ja", "ko", "yue"):
                asr_opts["language"] = language
            body = {"model": self.model, "stream": False, "asr_options": asr_opts,
                    "messages": [{"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {"data": data_uri}}]}]}
            t0 = time.perf_counter()
            r = http_post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                          json=body, timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            txt = r.json()["choices"][0]["message"]["content"]
            return TranscribeResult(text=txt, elapsed_s=elapsed, extra={})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class OpenAIChatAudioAdapter:
    """OpenAI Chat 形态的音频转写：/chat/completions + input_audio data-URI。

    与 /audio/transcriptions multipart 不是同一协议。用于 MiMo-V2.5-ASR 等
    复用 OpenAI Chat schema、但把音频放在 messages content 中的服务。
    """
    per_item_lang = True

    def __init__(self, base_url, model, api_key="", prefix="/v1", timeout=120.0):
        self.url = base_url.rstrip("/") + prefix.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = api_key
        self.timeout = timeout

    def transcribe(self, audio_path: str, language: str = "auto",
                   hotwords: str = "") -> "TranscribeResult":
        try:
            runtime = _runtime_request_parts(self)
            b64 = base64.b64encode(load_wav_bytes(audio_path)).decode()
            data_uri = f"data:audio/wav;base64,{b64}"
            lang = language if language in ("auto", "zh", "en") else "auto"
            body = {
                "model": self.model,
                "stream": False,
                "asr_options": {"language": lang},
                "messages": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": data_uri}},
                ]}],
            }
            body.update(runtime["json"])
            headers = {"Authorization": f"Bearer {self.key}",
                       **{k: str(v) for k, v in runtime["header"].items()}}
            t0 = time.perf_counter()
            r = http_post(self.url, headers=headers, params=runtime["query"],
                          json=body, timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            txt = r.json()["choices"][0]["message"]["content"]
            return TranscribeResult(text=txt, elapsed_s=elapsed, extra={})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class GemmaTextAdapter:
    """gemma-4-12B-it 文本任务：翻译 / 总结（文字→文字）。

    用于翻译场景(原文+提示词→目标语)和总结场景(文字+提示词→总结)。
    gemma 文本能力强(实测翻译质量好)。统一 generate(item) 接口。
    """
    name = "gemma-text"

    def __init__(self, base_url=ENDPOINTS["gemma"],
                 key=GEMMA_KEY,
                 timeout=120.0):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.key = key
        self.timeout = timeout

    def _prompt(self, item):
        task = item.get("task")
        if task == "translate":
            return f"把下面的文本翻译成{item.get('target_lang', '英文')}，只输出译文，不要解释：\n{item['source_text']}"
        if task == "summarize":
            return f"为下面的内容写一段简洁准确的中文摘要，只输出摘要本身：\n{item['source_text']}"
        return item.get("source_text", "")

    def generate(self, item) -> "TranscribeResult":
        try:
            runtime = _runtime_request_parts(self)
            body = {"model": "gemma-4-12B-it", "temperature": 0, "max_tokens": 512,
                    "messages": [{"role": "user", "content": self._prompt(item)}]}
            body.update(runtime["json"])
            headers = {"Authorization": f"Bearer {self.key}",
                       **{k: str(v) for k, v in runtime["header"].items()}}
            t0 = time.perf_counter()
            r = http_post(self.url, headers=headers, params=runtime["query"],
                          json=body, timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            return TranscribeResult(text=r.json()["choices"][0]["message"]["content"].strip(),
                                    elapsed_s=elapsed, extra={})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class CascadeLightGemmaAdapter:
    """级联语音翻译：ext-adv 转写 → gemma 文本翻译。

    用于带 audio_path 的 translate 清单(如 FLEURS)。和 gemma-text(直接拿
    source_text 转写文本去翻)对比，差值就是 ASR 误差对翻译的损耗。
    elapsed = 两段之和；extra.asr_text 留中间转写便于归因。
    """
    name = "cascade-gemma"

    def __init__(self, timeout: float = 120.0):
        self.asr = ext_adv(timeout=timeout)
        self.mt = GemmaTextAdapter(timeout=timeout)
        self.url = f"{self.asr.url} → {self.mt.url}"

    def generate(self, item) -> "TranscribeResult":
        a = self.asr.transcribe(item["audio_path"], language="auto")
        if not a.ok:
            return a
        m = self.mt.generate({"task": "translate", "source_text": a.text,
                              "target_lang": item.get("target_lang", "英文")})
        if not m.ok:
            return m
        return TranscribeResult(text=m.text, elapsed_s=a.elapsed_s + m.elapsed_s,
                                extra={"asr_text": a.text})


class PlatformASRAdapter:
    """自托管 ASR 平台（HTTP multipart）— 被测主体。

    一个服务的多个独立 multipart 同步端点；adapter 实例绑定单一端点，不跨身份换路由：
      - ASR    : POST /api/asr/{adv|std|compat}  adv 带 enable_text_edit=false(对齐口径)
      - 翻译   : POST /api/asr/{adv|compat}  target_lang=X，逐句 utterances[].translation 拼接
      - 热词   : 仅独立 PlatformAdvDomainAdapter → POST /api/asr/adv-domain
    base_url 参数化 → 同协议换台机器只改 url；endpoint 参数化 → 一个 adapter 测三个接口。
    """
    name = "adv"
    supports_hotwords = False
    _LANG = _LANG_PLATFORM  # 中文名→平台英文语言名(模块级常量)

    def __init__(self, base_url=ENDPOINTS["platform"], timeout=90.0,
                 domain=None, endpoint="adv", api_key=None):
        self.base = base_url.rstrip("/")
        if "/api/asr/" in self.base:
            self.base = self.base.split("/api/asr/", 1)[0]
        self.endpoint = endpoint                          # adv | std | compat
        self.api_key = _plat_token(api_key)
        if domain and endpoint != "adv-domain":
            raise ValueError(f"{endpoint} 不支持 domain；请显式使用 adv-domain 接口")
        self.domain = domain or "medical"
        self.url = f"{self.base}/api/asr/{endpoint}"    # __meta__ 与真实请求端点一致
        if endpoint != "adv":
            self.name = endpoint  # std | compat(旧版)
        self.timeout = timeout

    def _post(self, path, audio_path, fields):
        wav = load_wav_bytes(audio_path)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        headers = _bearer_headers(self.api_key)
        runtime = _runtime_request_parts(self)
        headers.update({k: str(v) for k, v in runtime["header"].items()})
        fields = {**fields, **_form_params(runtime["form"])}
        t0 = time.perf_counter()
        r = http_post(self.base + path, files=files, data=fields, headers=headers,
                      params=runtime["query"],
                      timeout=self.timeout)
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        return r.json(), elapsed

    def transcribe(self, audio_path, language="auto", hotwords="", target_lang="") -> "TranscribeResult":
        try:
            tgt = self._LANG.get(target_lang, target_lang) if target_lang else ""
            if self.endpoint == "adv-domain":
                hw = [w for w in hotwords.replace(",", " ").split() if w]
                fields = {"domain": self.domain, "language": language or "auto",
                          "hot_words": __import__("json").dumps(hw, ensure_ascii=False)}
                if tgt:
                    fields["target_lang"] = tgt
                d, el = self._post("/api/asr/adv-domain",
                                   audio_path, fields)
            else:
                if hotwords:
                    raise ValueError(f"{self.endpoint} 不支持 hotwords；请显式使用 adv-domain 接口")
                if tgt and self.endpoint == "std":
                    raise ValueError("std 不支持语音翻译；请使用 adv 或 adv-domain 接口")
                fields = {"enable_text_edit": "false", "language": language or "auto"}
                if tgt:
                    fields["target_lang"] = tgt
                d, el = self._post(f"/api/asr/{self.endpoint}", audio_path, fields)
            result = d.get("result", {})
            translated = "".join(u.get("translation", "") for u in result.get("utterances", [])).strip()
            return TranscribeResult(text=translated if tgt and translated else result.get("text", ""),
                                    elapsed_s=el, extra={"asr_text": result.get("text", "")} if tgt else {})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))

    def generate(self, item) -> "TranscribeResult":
        """音频→译文（FLEURS 等带 audio_path 的 translate 清单）。"""
        if not item.get("audio_path"):
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                    error="adv 翻译需要 audio_path（语音翻译），该清单无音频")
        try:
            tgt = self._LANG.get(item.get("target_lang", "英文"), item.get("target_lang", "English"))
            if self.endpoint not in ("adv", "compat", "adv-domain"):
                return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                        error=f"{self.endpoint} 不支持语音翻译")
            fields = {"target_lang": tgt}
            if item.get("request_language") and item["request_language"] != "auto":
                fields["language"] = item["request_language"]
            if self.endpoint == "adv-domain":
                fields["domain"] = self.domain
                fields["hot_words"] = json.dumps(
                    [w for w in str(item.get("hotwords") or "").replace(",", " ").split() if w],
                    ensure_ascii=False,
                )
            d, el = self._post(f"/api/asr/{self.endpoint}", item["audio_path"], fields)
            utts = d.get("result", {}).get("utterances", [])
            text = "".join(u.get("translation", "") for u in utts).strip() \
                or d.get("result", {}).get("text", "")
            return TranscribeResult(text=text, elapsed_s=el,
                                    extra={"asr_text": d.get("result", {}).get("text", "")})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class PlatformRealtimeAdapter:
    """平台流式 ASR WebSocket：`/v1/realtime?model=realtime-transcribe`。

    与本包其他 WS 同传端点不同，此端点**只收 Base64 JSON 音频帧**——二进制帧直接报
    `unsupported_audio_frame`；故发送走 `input_audio_buffer.append`(audio=b64) 而非 send_binary()。
    且**没有 task_finished 类终止事件**：逐段 `.completed` 后流即静默，收尾靠「文本落地 + 连续静默」判据。

    协议(实连核对，2026-09-14)：
      session.created → session.update → session.updated(回显式生效配置)
      → input_audio_buffer.append ×N(每帧 ≤100ms/3200B) → input_audio_buffer.commit
      → speech_started/speech_stopped/committed → conversation.item…delta → …completed

    精度口径：取 `.completed` 的 `transcript`(终稿)。elapsed_s 记到**最后一字落地**而非会话关闭——
    客户端静默等待不算进延迟，否则吞吐口径被自己的收尾策略污染。
    """
    name = "plat-realtime"
    supports_hotwords = False   # transcription.keywords 字段存在，但尚未验证真改变识别结果 → 不声明

    # 100ms @ 16k = 1600 samples = 3200 bytes；服务端拒收 <10ms 帧，残块须补齐
    _FRAME_BYTES = 3200
    _MIN_FRAME_BYTES = 320   # 10ms @ 16k s16le
    _RECV_TIMEOUT = 0.5      # 短超时 = 静默判据的分辨率；拉长会把延迟虚高
    # 实测：全速喂完音频后，VAD 切出的各段是服务端连着吐回来的，段间只隔 0.11~0.13s
    # （3 条多段样本实测）。1.5s 是该间隔的 ~11 倍余量，足以保段又不空等（原值 6.0s 是 46 倍）。
    # ⚠️ 该端点无终止事件，只能靠静默判收尾；若某条真被提前收尾，extra.fanel_tail_s 会显示
    #    刚好卡在阈值附近且 n_seg 偏少 —— 复核时优先看这个。
    # 最后一道兜底：最后一字落地后连续静默这么久 → 收尾（只计 commit 之后）。
    # ⚠️ 别设太小：pace=True 时服务端有积压，末段可能几秒后才回，过早收工会丢段
    #    （实测 1x 节奏下 3 次里 2 次只剩前半段）。主判据是 session.finished，
    #    这条只在它一直不来时才生效，故可以放大而不影响正常速度（正常 0.4s 就收到 finished）。
    _IDLE_SILENCE_S = 15.0
    # 无文本兜底：极短/无语音样本服务端可能一个文本都不回（如 ascend_00774 0.31s "this"）。
    # 此时 last_text_at 恒为 None、静默判据永不触发 → 会一路耗到 60s 硬上限 ×3 次重试 = 180s/条。
    # 实测这类样本正是「掉速到 1~3 条/分」的成因。服务约 20x 实时，5s 足够等到首段文本。
    _NO_TEXT_TIMEOUT_S = 5.0
    _FINISH_GRACE_S = 0.5   # 配平达成后再等 authority 的 session.finished 这么久

    def __init__(self, base_url=None, timeout=60.0, pace=False, api_key=None,
                 trail_silence_s=1.0, context_enabled=False, context_max_items=8,
                 context_max_chars=2000):
        base_url = base_url or ENDPOINTS.get("platform_ws")
        if not base_url:
            raise RuntimeError("plat-realtime 未配置端点：设 ASR_PLATFORM_WS_URL 环境变量或看板填 WS 地址")
        self.ws_url = (base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
                       + "/v1/realtime?model=realtime-transcribe")
        # 跨 item 上下文（x_platform.context）：A/B 用，默认关保持与历史跑分同口径。
        # CLI 也可用 --request-params-json '{"context_enabled": true}' 覆盖。
        self.context_enabled = bool(context_enabled)
        self.context_max_items = int(context_max_items)
        self.context_max_chars = int(context_max_chars)
        self.timeout = timeout
        # pace 默认关：按 1x 实时喂音频只服务同传场景，纯精度评测会把自己的喂音频时间
        # 算进 elapsd_s 而虚增延迟（实测同 6 条：pace=开 4.86s / 关 0.52s，CER 一字不差）。
        # 要测真同传节奏就显式 pace=True。
        self.pace = pace
        # 尾部补静音：本端点的 server_vad 需要 silence_duration_ms(默认700ms) 的**尾部静音**
        # 才判定「说完了」并提交该段。真实麦克风会持续送静音，而离线评测「发完文件即断」——
        # 短音频/句尾无停顿的音频因此被 VAD 丢弃或截尾（实测：0.17s 样本 item_count=0 无任何结果；
        # 补 2s 静音后正常出字）。补一段静音让离线评测贴近真实使用。
        self.trail_silence_s = float(trail_silence_s or 0.0)
        token = _plat_token(api_key)
        self.ws_headers = [f"Authorization: Bearer {token}"] if token else []

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _context_config(self):
        """跨 item 上下文配置：构造参数与 request_params 任一开启即生效，默认关。

        返回 None 表示不下发 context 字段，保持与未开启时完全一致的 session.update。
        """
        params = getattr(self, "request_params", None) or {}
        enabled = self._as_bool(params.get("context_enabled", self.context_enabled))
        if not enabled:
            return None
        try:
            max_items = int(params.get("context_max_items", self.context_max_items))
            max_chars = int(params.get("context_max_chars", self.context_max_chars))
        except (TypeError, ValueError):
            max_items, max_chars = self.context_max_items, self.context_max_chars
        return {"enabled": True, "max_items": max_items, "max_chars": max_chars}

    def _session_update(self, language, context=None):
        x_platform = {
            "sentence_timestamps": {"enabled": False},
            "word_timestamps": {"enabled": False},
            "speaker_diarization": {"enabled": False, "preserve_overlap": False},
            "translation": {"enabled": False, "target_language": None},
            "smart_edit": {"enabled": False, "scope": "item"},
        }
        if context is not None:
            x_platform["context"] = context
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"model": "realtime-transcribe", "prompt": "",
                                      "keywords": [], "languages": [language or "auto"]},
                    "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                       "prefix_padding_ms": 300, "silence_duration_ms": 700}}},
                "x_platform": x_platform,
            },
        }

    def transcribe(self, audio_path: str, language: str = "auto", prompt: str = "",
                   context=None) -> "TranscribeResult":
        import base64
        t0 = time.perf_counter()
        context = self._context_config() if context is None else context
        try:
            pcm = load_pcm_bytes(audio_path)
            src_dur = len(pcm) / 32000.0
            # 尾部补静音（见 __init__ 说明）：让离线评测贴近真实麦克风的持续供流
            if self.trail_silence_s > 0:
                pcm = pcm + b"\x00" * int(32000 * self.trail_silence_s)
            ws = ws_connect(self.ws_url, timeout=self._RECV_TIMEOUT, header=self.ws_headers)
            try:
                connected = json.loads(ws.recv())          # session.created
                self._throw_if_error(connected)
                update = self._session_update(language, context)
                ws.send(json.dumps(update, ensure_ascii=False))
                updated = json.loads(ws.recv())            # session.updated(生效配置回显)
                self._throw_if_error(updated)
                ws_control = _ws_control_record(self.ws_url, connected, update, updated)

                base = time.perf_counter()
                for k, i in enumerate(range(0, len(pcm), self._FRAME_BYTES)):
                    chunk = pcm[i:i + self._FRAME_BYTES]
                    if len(chunk) < self._MIN_FRAME_BYTES:   # 残块补齐(服务端拒收 <10ms 帧)
                        chunk = chunk + b"\x00" * (self._MIN_FRAME_BYTES - len(chunk))
                    ws.send(json.dumps({"type": "input_audio_buffer.append",
                                        "audio": base64.b64encode(chunk).decode("ascii")},
                                       ensure_ascii=False))
                    if self.pace:
                        dt = base + (k + 1) * 0.1 - time.perf_counter()
                        if dt > 0:
                            time.sleep(dt)
                # ⚠️ 本端点开着 server_vad（见 _session_update 的 turn_detection）：服务端已按静音
                # 自动分段并提交，**不能再发 input_audio_buffer.commit** —— 缓冲已空，会回
                # input_audio_buffer_commit_empty 报错并触发无谓重试（实测踩到）。平台文档
                # 平台示例客户端明确：server_vad 下由服务端分段，
                # 结束会话只发 platform.session.finish。
                # 该 finish 是本服务扩展事件：服务端排空缓冲/ASR/增强后回 platform.session.finished，
                # 这是唯一的「服务端说它做完了」信号，收到即可收工，不必靠静默猜。
                # 契约见 docs/websocket-realtime-transcriptions.md:435(请求)/:730(响应)。
                # 不发 client_seq：该扩展只在需要 durable 重连时才要，发了反而刷屏 platform.client_ack。
                ws.send(json.dumps({"type": "platform.session.finish"}))
                # 收尾判据的基线 = 喂完音频并发出 session.finish 的时刻（此后才可能在等结果）
                commit_at = time.perf_counter() - t0
                ferd_s = commit_at                       # 喂音频耗时（客户端节奏，单独报，不计入服务延迟）

                text, n_seg, last_text_at, first_text_at = "", 0, None, None
                # 确定性收尾：服务端每切一段发一条 input_audio_buffer.committed，每个 committed 的
                # item 必对应一条 ...transcription.completed（含空文本的 committed 也会回 completed）
                # —— 实测 3 类样本(多段/单段/无语音)均严格配平。故 completed >= committed 即可收工，
                # 不必猜静默。这是本端点无终止事件时最接近"服务端说完了"的信号。
                committed_n = completed_n = 0
                finished_ev = None
                pair_at = None
                # ⚠️ 服务端推 completed 的顺序**不确定**（同一段音频两次跑可能正序/反序），
                # 按到达顺序拼会把文本打乱（实测 aishell 出现「后半句跑到前面」，
                # 如 ref『其市管…政策也均进行调整』→ hyp『也均进行调整。其市管…政策』）。
                # speech_started 带权威时间轴 audio_start_ms，故按它排序拼段。
                seg_starts = {}     # item_id → audio_start_ms
                seg_parts = []      # [(start_ms|null, 到达序, 文本)]
                while True:
                    try:
                        raw = ws.recv()
                    except Exception:
                        raw = None   # 静默：超时即视为一次空拉
                    now = time.perf_counter() - t0
                    if raw:
                        if not str(raw).startswith("{"):
                            continue
                        ev = json.loads(raw)
                        t = ev.get("type") or ""
                        if t in ("error", "task_failed"):
                            return TranscribeResult("", now,
                                                    {"src_dur_s": round(src_dur, 2), "ws_control": ws_control},
                                                    ok=False, error=str(ev.get("error"))[:200])
                        if t == "platform.session.finished":
                            finished_ev = ev        # 服务端终态：buffer/ASR/增强已排空
                        elif t == "input_audio_buffer.speech_started":
                            iid = ev.get("item_id")
                            if iid is not None:
                                seg_starts[iid] = ev.get("audio_start_ms")
                        elif t == "input_audio_buffer.committed":
                            committed_n += 1
                        elif t == "conversation.item.input_audio_transcription.completed":
                            completed_n += 1        # 空文本也算已回（配平用）
                            seg = (ev.get("transcript") or "").strip()
                            if seg:
                                seg_parts.append((seg_starts.get(ev.get("item_id")),
                                                  len(seg_parts), seg))
                                n_seg += 1
                                last_text_at = now
                                if first_text_at is None:
                                    first_text_at = now
                    # 收尾判据：commit 已发（音频喂完）且最后一字落地后连续静默达阈；不比静默次数，
                    # 免把「等下一段」的长音频误收尾。commit 前不判（音频还在喂，无文本属正常）。
                    # 主判据：服务端已回 session.finished（buffer/ASR/增强全部排空）→ 立刻收
                    if finished_ev is not None:
                        break
                    # 次判据（兼容未实现该扩展的部署）：committed/completed 配平即认为处理完。
                    # ⚠️ 必须 now >= commit_at —— pace=True 时服务端会在喂音频途中就提交前几段，
                    #    那时也会配平，若不守卫会提前收工、丢掉后面所有内容。
                    if now >= commit_at and committed_n > 0 and completed_n >= committed_n:
                        if pair_at is None:
                            pair_at = now       # 配平已达成；再宽限一小会儿等权威的 finished
                        elif now - pair_at >= self._FINISH_GRACE_S:
                            break
                    else:
                        pair_at = None
                    # 兜底：万一某条 committed 迟迟等不到 completed，退回静默判据
                    if last_text_at is not None and now - max(last_text_at, commit_at) >= self._IDLE_SILENCE_S:
                        break
                    # 一条文本都没等到 → 别耗到硬上限（该类样本会连累整批吞吐）
                    if last_text_at is None and now - commit_at >= max(self._NO_TEXT_TIMEOUT_S, src_dur):
                        break
                    if now > max(self.timeout, src_dur * 4 + 20):
                        break   # 硬上限兜底(防坏会话永不收尾)
                # 按 speech_started 的时间轴拼段（缺失时间的段按到达序兜底排在其后）
                if seg_parts:
                    ordered = sorted(
                        seg_parts,
                        key=lambda p: (p[0] if p[0] is not None else float("inf"), p[1]),
                    )
                    text = "".join(p[2] for p in ordered)
                text = _clean_asr_scaffolding(text)   # 剥偶发模板脚手架(见该函数说明)
                return TranscribeResult(
                    text=text,
                    # 延迟记到最后一字落地(而非收尾后)：客户端静默等待是我们的收尾策略，不算服务延迟
                    elapsed_s=last_text_at if text else 0.0,
                    extra={"src_dur_s": round(src_dur, 2), "n_seg": n_seg,
                           "server_finished": finished_ev is not None,
                           "item_count": (finished_ev or {}).get("item_count"),
                           "server_status": (finished_ev or {}).get("status"),
                           "usage_seconds": (finished_ev or {}).get("usage_seconds"),
                           "degraded_stages": (finished_ev or {}).get("degraded_stages"),
                           "recv_timeout_s": self._RECV_TIMEOUT,
                           "ferd_s": round(ferd_s, 2),
                           "ttfb_s": round(first_text_at, 3) if first_text_at is not None else None,
                           "fanel_tail_s": round(max(0.0, (time.perf_counter() - t0) - (last_text_at or 0.0)), 2),
                           "ws_control": ws_control},
                    ok=bool(text), error="" if text else "无转写文本",
                )
            finally:
                ws.close()
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))

    @staticmethod
    def _throw_if_error(ev):
        """握手期事件里的错误包装 → 抛 RuntimeError（与 _ws_control_record 的容错口径一致）。"""
        if not isinstance(ev, dict):
            return
        t = ev.get("type") or ev.get("event")
        if t in ("error", "task_failed"):
            raise RuntimeError(str(ev.get("error"))[:200])


_QWEN_WS_LANG = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ja": "Japanese",
                 "ko": "Korean", "ar": "Arabic", "de": "German", "fr": "French",
                 "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
                 "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "tr": "Turkish",
                 "ms": "Malay", "nl": "Dutch"}


class Qwen3ASRWSAdapter:
    """Qwen3-ASR 真流式 WebSocket：`ws://<host>/ws`。

    ⚠️ 与同机的 HTTP 兜底路（`/api/start|chunk|finish`，页面 demo 走的那条）是两套协议：
    真 WS 用长连接不碰多 replica 的会话粘滞问题，且逐 chunk **增量**吐字、`finish` 有显式终态。

    协议（照服务端参考客户端，已实连核对）：
      connect → {"type":"start", language?} → {"type":"started", session_id, language}
      → 裸二进制 float32le chunk(默认 1000ms) → {"type":"result", text} 逐块增量
      → {"type":"finish"} → {"type":"result", final:true, text} 终稿

    无鉴权。精度取终态文本；增量 emissions 供首字/TTFB/AL 族延迟口径用。
    ⚠️ 该端点线上同传产品共用 → 评测并发必须 ≤2（见 class 属性 _MAX_WORKERS）。
    """
    name = "qwen3-asr-ws"
    supports_hotwords = False
    _MAX_WORKERS = 2       # 线上同传共用此端点；infer 侧并发上限（防打满产品）
    _CHUNK_MS = 1000       # 与参考客户端 QWEN3_ASR_CHUNK_MS 一致

    def __init__(self, base_url=None, timeout=60.0, pace=False, api_key=None,
                 trail_silence_s=1.0):
        base_url = base_url or os.environ.get("QWEN3_ASR_WS_URL", "")
        self.ws_url = base_url.rstrip("/")
        if not self.ws_url.endswith("/ws"):     # 允许只给 host:port，自动补路径
            self.ws_url = self.ws_url + "/ws"
        self.timeout = timeout
        self.pace = pace
        # 尾部补静音：本端点无 VAD、不依赖静音收段（补不补都行）。默认与 /v1/realtime 取同一
        # 值，使两条接口**喂完全相同的音频**，对照不受输入差异干扰。实测补 1s 对 10 条样本
        # (含 4 条难样本)的 CER 与输出**一字不差**，故无副作用。
        self.trail_silence_s = float(trail_silence_s or 0.0)
        self.ws_headers = []

    def transcribe(self, audio_path: str, language: str = "auto", prompt: str = "") -> "TranscribeResult":
        t0 = time.perf_counter()
        ws = None
        finised = False
        try:
            pcm = load_pcm_bytes(audio_path)
            src_dur = len(pcm) / 32000.0
            if self.trail_silence_s > 0:
                pcm = pcm + b"\x00" * int(32000 * self.trail_silence_s)
            # s16le → float32le（该端点吃裸 float32，非 Base64 JSON）
            import numpy as np
            f32 = (np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0).astype("<f4")
            ws = ws_connect(self.ws_url, timeout=self.timeout)
            # 服务端在收到 start 前不推任何东西（参考客户端同样先发后收），故此处不预读

            start = {"type": "start"}
            if language and language != "auto":
                start["language"] = _QWEN_WS_LANG.get(language, language)
            ws.send(json.dumps(start, ensure_ascii=False))
            started = json.loads(ws.recv())
            if started.get("type") == "error":
                raise RuntimeError(str(started.get("error"))[:200])

            n_samples = int(16000 * self._CHUNK_MS / 1000)
            base = time.perf_counter()
            emissions, ttfb, text = [], None, ""
            last_seg_at = None
            for k, i in enumerate(range(0, len(f32), n_samples)):
                seg = f32[i:i + n_samples]
                if len(seg) == 0:
                    continue
                ws.send_binary(seg.tobytes())
                ev = json.loads(ws.recv())
                if ev.get("type") == "error":
                    return TranscribeResult(text, time.perf_counter() - t0,
                                            {"src_dur_s": round(src_dur, 2)},
                                            ok=False, error=str(ev.get("error"))[:200])
                seg_text = (ev.get("text") or "").strip()
                if seg_text:
                    if ttfb is None:
                        ttfb = time.perf_counter() - t0
                    text = seg_text
                    last_seg_at = time.perf_counter() - t0
                    # emission 的时钟 = 已消耗源秒数（与本包同传 AL/LAAL 口径一致）
                    emissions.append((round(min(src_dur, (i + n_samples) / 16000.0), 3), seg_text))
                if self.pace:
                    dt = base + (k + 1) * (self._CHUNK_MS / 1000.0) - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)

            ws.send(json.dumps({"type": "finish"}))
            finised = True
            fin = json.loads(ws.recv())
            if fin.get("type") == "error":
                raise RuntimeError(str(fin.get("error"))[:200])
            final_text = (fin.get("text") or "").strip()
            if final_text:
                text = final_text
            # 与平台那条同口径：elapsd_s 记到**最后一字落地**，不含收尾/喂音频的客户端时间；
            # 这样两条的「服务延迟」才可直接比。
            el = last_seg_at if last_seg_at is not None else (time.perf_counter() - t0)
            text = _clean_asr_scaffolding(text)   # 剥偶发模板脚手架(见该函数说明)
            return TranscribeResult(
                text=text, elapsed_s=el,
                extra={"src_dur_s": round(src_dur, 2), "ttfb_s": round(ttfb, 3) if ttfb else None,
                       "n_seg": len(emissions), "final_event": bool(fin.get("final")),
                       "translation_segments": _timed_text_segments(emissions),
                       "fanel_tail_s": round(max(0.0, (time.perf_counter() - t0) - el), 2),
                       "rtf": round(el / src_dur, 3) if src_dur else None},
                ok=bool(text), error="" if text else "无转写文本",
            )
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=time.perf_counter() - t0, extra={},
                                    ok=False, error=str(e))
        finally:
            if ws is not None:
                # 失败早退（错误事件/超时）时也必须告诉服务端收尾，否则会话会赖在端上
                # ——该端点 cap 小、不超时回收，漏一个就少一个名额（曾把端点占满）。
                if not finised:
                    try:
                        ws.send(json.dumps({"type": "abort"}))
                    except Exception:
                        pass
                try:
                    ws.close()
                except Exception:
                    pass


class PlatformAdvDomainAdapter(PlatformASRAdapter):
    """Dedicated identity for /adv-domain; never masquerades as Standard or Advanced."""
    name = "adv-domain"
    supports_hotwords = True

    def __init__(self, base_url=ENDPOINTS["platform"], timeout=90.0, domain=None, api_key=None):
        super().__init__(base_url=base_url, timeout=timeout, domain=domain,
                         endpoint="adv-domain", api_key=api_key)


class PlatformSSEAdapter:
    """平台的 SSE 流式接口（POST 上传 → text/event-stream 逐事件返回）：
      - mode="sse"    : /api/asr/sse  流式识别
      - mode="simult" : /api/asr/simult-interpreting-sse  同声传译(target_lang 必填)
    精度口径：识别用 asr_result 事件的原始文本(等价 enable_text_edit=false)；
    流式专属指标：extra.ttfb_s = 发出请求 → 首个含文字事件（score 聚合成 ttfb_p50_s）。
    """
    name = "sse"
    supports_hotwords = True   # 接口原生 hotwords 逗号分隔参数
    _LANG = _LANG_PLATFORM

    def __init__(self, base_url=ENDPOINTS["platform"], timeout=180.0, mode="sse",
                 audio_out_dir=None, api_key=None):
        self.base = base_url.rstrip("/")
        self.mode = mode
        path = ("/api/asr/sse" if mode == "sse"
                else "/api/asr/simult-interpreting-sse")
        self.url = self.base + path
        if mode != "sse":
            self.name = "plat-simult"
        self.timeout = timeout
        self.audio_out_dir = audio_out_dir   # 非空 → enable_tts 收 tts_audio 存 wav(16k PCM)
        self.headers = _bearer_headers(_plat_token(api_key))

    def _stream(self, audio_path, fields, until_audio=False):
        """收完整个 event-stream → (events, 首包延迟, 总耗时)。
        until_audio=True 时不在 done 停（tts_audio 在 done 之后才来），等到 tts_audio/error。"""
        wav = load_wav_bytes(audio_path)
        runtime = _runtime_request_parts(self)
        fields = {**fields, **_form_params(runtime["form"])}
        headers = {**self.headers, **{k: str(v) for k, v in runtime["header"].items()}}
        t0 = time.perf_counter()
        r = http_post(self.url, files={"file": ("audio.wav", wav, "audio/wav")},
                      data=fields, headers=headers, params=runtime["query"],
                      stream=True, timeout=self.timeout)
        r.raise_for_status()
        events, ttfb, cur = [], None, None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                cur = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                except Exception:
                    data = {}
                events.append((cur, data))
                if ttfb is None and (data.get("text") or data.get("content")):
                    ttfb = time.perf_counter() - t0
                # 取音频时读到流结束(server 收尾自闭，tts 可能分多块)；不取音频则 done/tts_audio 即停
                if cur == "error" or (not until_audio and cur in ("done", "tts_audio")):
                    break
        return events, ttfb, time.perf_counter() - t0

    def transcribe(self, audio_path, language="auto", hotwords="", target_lang="") -> "TranscribeResult":
        try:
            fields = {}
            if hotwords:
                fields["hotwords"] = hotwords.replace(" ", ",")
            # language != auto → 注入 form 字段（后端 42ee000 起支持，作为 LLM recognition hint）
            if language and language != "auto":
                fields["language"] = language
            if target_lang:
                fields["target_lang"] = self._LANG.get(target_lang, target_lang)
            events, ttfb, el = self._stream(audio_path, fields)
            err = next((d for e, d in events if e == "error"), None)
            if err:
                return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                        error=str(err)[:300])
            done = next((d for e, d in events if e == "done"), {})
            # 原始识别口径：asr_result 事件；done.text 是 LLM 修饰稿，仅留作参考
            asr = next((d.get("text", "") for e, d in events if e == "asr_result"), "")
            text = (done.get("text", "") if target_lang else "") or asr or done.get("asr_text", "") or done.get("text", "")
            return TranscribeResult(text=text, elapsed_s=el,
                                    extra={"ttfb_s": round(ttfb, 3) if ttfb else None,
                                           "edited_text": done.get("text", "")})  # 完整保留：score --use-edited 拿它打分，截断会虚高 CER
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))

    def generate(self, item) -> "TranscribeResult":
        """同传：音频 → 目标语译文（token 增量拼接，done 为准）。"""
        if not item.get("audio_path"):
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                    error="同传需要 audio_path，该清单无音频")
        try:
            tgt = self._LANG.get(item.get("target_lang", "英文"), item.get("target_lang", "English"))
            fields = {"target_lang": tgt}
            if item.get("hotwords"):
                fields["hotwords"] = str(item["hotwords"]).replace(" ", ",")
            if item.get("request_language") and item["request_language"] != "auto":
                fields["language"] = item["request_language"]
            if self.audio_out_dir:
                fields["enable_tts"] = "true"   # 开 TTS → done 后多一个 tts_audio 事件
            events, ttfb, el = self._stream(item["audio_path"], fields,
                                            until_audio=bool(self.audio_out_dir))
            err = next((d for e, d in events if e == "error"), None)
            if err:
                return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                        error=str(err)[:300])
            done = next((d for e, d in events if e == "done"), {})
            text = (done.get("translation") or done.get("text")
                    or "".join(d.get("content", "") for e, d in events if e == "token")).strip()
            extra = {"ttfb_s": round(ttfb, 3) if ttfb else None,
                     "asr_text": done.get("asr_text", "")[:300]}
            if self.audio_out_dir:   # 译后语音落盘(拼接所有 tts_audio 块，16k PCM)
                pcm = bytearray()
                for e, d in events:
                    if e == "tts_audio" and (d.get("audio") or d.get("tts_audio")):
                        try:
                            pcm += base64.b64decode(d.get("audio") or d.get("tts_audio"))
                        except Exception:
                            pass
                if pcm:
                    extra["tts_audio"] = save_tts_audio(bytes(pcm),
                        os.path.join(self.audio_out_dir, f"{self.name}__{item.get('id', 'x')}"), sr=16000)
            return TranscribeResult(text=text, elapsed_s=el, extra=extra)
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class PlatformMinutesAdapter:
    """会议纪要(离线) — POST /api/asr/meeting-minutes，长音频 → Markdown 纪要。

    响应取 render.final_summary.markdown。单场会议 15-30 分钟音频，
    服务端三段式处理（ASR→说话人→纪要生成），耗时数分钟级 → 超时给足。
    评测口径：ROUGE 对 4MUG 人工关键句锚点（见 build_minutes），G-Eval 二期。
    """
    name = "plat-minutes"

    def __init__(self, base_url=ENDPOINTS["platform"], timeout=1200.0, api_key=None):
        self.base = base_url.rstrip("/")
        self.url = self.base + "/api/asr/meeting-minutes"
        self.timeout = timeout
        self.headers = _bearer_headers(_plat_token(api_key))

    def generate(self, item) -> "TranscribeResult":
        if not item.get("audio_path"):
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                    error="minutes 需要 audio_path")
        try:
            wav = load_wav_bytes(item["audio_path"])  # 8ch 远场 → mono 16k
            runtime = _runtime_request_parts(self)
            data = {"enable_diarize": "true", **_form_params(runtime["form"])}
            request_language = str(item.get("request_language") or "auto")
            data["language"] = request_language
            target_lang = item.get("target_lang")
            if target_lang and str(target_lang).lower() != "none":
                data["target_lang"] = _LANG_PLATFORM.get(target_lang, target_lang)
            headers = {**self.headers, **{k: str(v) for k, v in runtime["header"].items()}}
            t0 = time.perf_counter()
            r = http_post(self.url, files={"file": ("audio.wav", wav, "audio/wav")},
                          data=data, headers=headers, params=runtime["query"],
                          timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            render = (r.json() or {}).get("render") or {}
            md = ((render.get("final_summary") or {}).get("markdown") or "").strip()
            if not md:
                return TranscribeResult(text="", elapsed_s=elapsed, extra={}, ok=False,
                                        error="render.final_summary.markdown 为空")
            return TranscribeResult(text=md, elapsed_s=elapsed,
                                    extra={"n_transcript": len(render.get("live_transcript") or [])})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class PlatformDiarAdapter:
    """说话人分离 — POST /api/asr/adv + enable_speaker_diarization,返回带 speaker_id 的分段。
    分段(秒)编码进 text(JSON),score 用 pyannote DER 对 TextGrid 参考算分。长音频,超时给足。"""
    name = "plat-diar"

    def __init__(self, base_url=ENDPOINTS["platform"], timeout=1200.0, api_key=None):
        self.base = base_url.rstrip("/")
        self.url = self.base + "/api/asr/adv"
        self.timeout = timeout
        self.headers = _bearer_headers(_plat_token(api_key))

    def transcribe(self, audio_path, language="auto", hotwords="") -> "TranscribeResult":
        try:
            wav = load_wav_bytes(audio_path)
            runtime = _runtime_request_parts(self)
            data = {"enable_speaker_diarization": "true", "language": language or "auto",
                    **_form_params(runtime["form"])}
            headers = {**self.headers, **{k: str(v) for k, v in runtime["header"].items()}}
            t0 = time.perf_counter()
            r = http_post(self.url, files={"file": ("audio.wav", wav, "audio/wav")},
                          data=data, headers=headers, params=runtime["query"],
                          timeout=self.timeout)
            el = time.perf_counter() - t0
            r.raise_for_status()
            utts = (r.json().get("result") or {}).get("utterances") or []
            segs = [[u["start_time"] / 1000.0, u["end_time"] / 1000.0, str(u.get("speaker_id", "?"))]
                    for u in utts if u.get("start_time") is not None]
            if not segs:
                return TranscribeResult(text="", elapsed_s=el, extra={}, ok=False, error="无分段返回")
            return TranscribeResult(text=json.dumps(segs), elapsed_s=el,
                                    extra={"n_spk_hyp": len(set(s[2] for s in segs))})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class PlatformFormulaAdapter:
    """TTS 公式转写 — POST /api/text/tts_formula_helper (JSON {text}) → token SSE。

    文本→口语化朗读文本。打分口径：字级 CER 对 golden 理想转译（诊断集 8 条）。
    """
    name = "plat-formula"

    def __init__(self, base_url=ENDPOINTS["platform"], timeout=180.0, api_key=None):
        self.base = base_url.rstrip("/")
        self.url = self.base + "/api/text/tts_formula_helper"
        self.timeout = timeout
        self.headers = _bearer_headers(_plat_token(api_key))

    def generate(self, item) -> "TranscribeResult":
        try:
            t0 = time.perf_counter()
            r = http_post(self.url, json={"text": item.get("source_text", "")},
                          headers=self.headers, stream=True, timeout=self.timeout)
            r.raise_for_status()
            toks, done, ttfb, cur = [], {}, None, None
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("event:"):
                    cur = line[6:].strip()
                elif line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                    except Exception:
                        data = {}
                    if cur == "token" and data.get("content"):
                        toks.append(data["content"])
                        if ttfb is None:
                            ttfb = time.perf_counter() - t0
                    elif cur in ("done", "error"):
                        done = data
                        break
            el = time.perf_counter() - t0
            text = (done.get("text") or "".join(toks)).strip()
            if not text:
                return TranscribeResult(text="", elapsed_s=el, extra={}, ok=False,
                                        error=str(done)[:200] or "空输出")
            return TranscribeResult(text=text, elapsed_s=el,
                                    extra={"ttfb_s": round(ttfb, 3) if ttfb else None})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class OpenAIAudioAdapter:
    """OpenAI audio 兼容适配器；内置 v3 端点已下架，当前只供自定义接口/厂商预设复用。

    base_url 必须由 interfaces.json、厂商预设或 ASR_V3_URL 显式提供；
    不默认连接任何地址，需显式配置 ASR_OPENAI_AUDIO_URL。
    """
    supports_hotwords = False  # generic OpenAI-audio 不收非标准 hot_words；需显式开启
    _TGT = {"英文": "en", "中文": "zh", "简体中文": "zh", "日文": "ja", "韩文": "ko",
            "西班牙文": "es", "en": "en", "zh": "zh"}

    def __init__(self, base_url=ENDPOINTS["openai_audio"], model="asr-lite",
                 api_key="sk-eval", timeout=120.0, stream=False, prefix="/api/v3",
                 supports_hotwords=False):
        self.base = base_url.rstrip("/")
        self.prefix = prefix.rstrip("/")          # OpenAI 标准 /v1；竞品按嗅探结果
        self.model = model
        self.stream = stream   # stream=true → SSE(transcript.text.delta/done)，lite 会 400
        self.name = ("legacy-stream" if stream
                     else f"legacy-{model.replace('asr-lite', 'asr').replace('asr-', '')}")
        self.url = f"{self.base}{self.prefix}/audio/transcriptions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self.supports_hotwords = supports_hotwords

    def _err(self, r):
        try:
            return str(r.json().get("error", {}).get("message", r.text))[:300]
        except Exception:
            return r.text[:300]

    def _post(self, path, audio_path, fields):
        wav = load_wav_bytes(audio_path)
        runtime = _runtime_request_parts(self)
        headers = {**self.headers, **{k: str(v) for k, v in runtime["header"].items()}}
        fields = {**fields, **_form_params(runtime["form"])}
        t0 = time.perf_counter()
        r = http_post(self.base + path, headers=headers, params=runtime["query"],
                          files={"file": ("audio.wav", wav, "audio/wav")},
                          data=fields, timeout=self.timeout)
        elapsed = time.perf_counter() - t0
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code}: {self._err(r)}")
        return r.json(), elapsed

    def transcribe(self, audio_path, language="auto", hotwords="", target_lang="") -> "TranscribeResult":
        try:
            if target_lang:
                tgt = self._TGT.get(target_lang, target_lang)
                d, el = self._post(f"{self.prefix}/audio/translations", audio_path,
                                   {"model": self.model, "target_language": tgt})
                return TranscribeResult(text=d.get("text", ""), elapsed_s=el,
                                        extra={"translated": True})
            fields = {"model": self.model, "response_format": "json"}
            if language and language != "auto":
                fields["language"] = language
            if hotwords and self.supports_hotwords:
                fields["hot_words"] = hotwords.replace(" ", ",")
            if self.stream:
                return self._transcribe_stream(audio_path, fields)
            d, el = self._post(f"{self.prefix}/audio/transcriptions", audio_path, fields)
            return TranscribeResult(text=d.get("text", ""), elapsed_s=el, extra={})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))

    def _transcribe_stream(self, audio_path, fields):
        """stream=true：OpenAI 风格 SSE（transcript.text.delta/.done），记首包延迟。"""
        fields["stream"] = "true"
        wav = load_wav_bytes(audio_path)
        runtime = _runtime_request_parts(self)
        headers = {**self.headers, **{k: str(v) for k, v in runtime["header"].items()}}
        fields.update(_form_params(runtime["form"]))
        t0 = time.perf_counter()
        r = http_post(self.url, headers=headers, params=runtime["query"],
                          files={"file": ("audio.wav", wav, "audio/wav")},
                          data=fields, stream=True, timeout=self.timeout)
        if not r.ok:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                    error=f"HTTP {r.status_code}: {self._err(r)}")
        deltas, done, ttfb, cur = [], {}, None, None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                cur = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                except Exception:
                    data = {}
                if cur == "transcript.text.delta" and data.get("delta"):
                    deltas.append(data["delta"])
                    if ttfb is None:
                        ttfb = time.perf_counter() - t0
                elif cur == "transcript.text.done":
                    done = data
                    break
        el = time.perf_counter() - t0
        text = (done.get("text") or "".join(deltas)).strip()
        return TranscribeResult(text=text, elapsed_s=el,
                                extra={"ttfb_s": round(ttfb, 3) if ttfb else None})

    def generate(self, item) -> "TranscribeResult":
        """语音翻译：/audio/translations，target_language 为 ISO code(默认 en)。"""
        if not item.get("audio_path"):
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False,
                                    error="v3 translations 需要 audio_path")
        try:
            tgt = self._TGT.get(item.get("target_lang", "英文"), "en")
            d, el = self._post(f"{self.prefix}/audio/translations", item["audio_path"],
                               {"model": self.model, "target_language": tgt})
            return TranscribeResult(text=d.get("text", ""), elapsed_s=el, extra={})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


class QwenLiveTranslateAdapter:
    """阿里通义 Qwen3.5-LiveTranslate 视听同传 — OpenAI Realtime 协议（WebSocket）。

    wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-livetranslate-flash-realtime，
    Header Bearer 鉴权（复用 DASHSCOPE_API_KEY，与厂商预设 qwen3-asr 同一把 key）。
    流程：session.update(设目标语+纯文本) → input_audio_buffer.append 按 1x 实时节奏推裸 PCM(base64)
         → response.text.delta 收译文增量 → session.finish → session.finished。
    采 AL/LAAL（emission 相对已消耗源音频秒数）+ ttfb，与 plat-simult/xf 同口径对打。
    ⚠️ 事件/字段依百炼 Realtime 文档实现，需活体冒烟校验 delta/finished 事件名。
    """
    name = "qwen-simult"
    _TGT = {"英文": "en", "English": "en", "中文": "zh", "简体中文": "zh", "Chinese": "zh",
            "日文": "ja", "Japanese": "ja", "韩文": "ko", "Korean": "ko",
            "西班牙文": "es", "Spanish": "es", "粤语": "yue", "Cantonese": "yue",
            "法文": "fr", "French": "fr", "德文": "de", "German": "de",
            "俄文": "ru", "Russian": "ru",
            "en": "en", "zh": "zh", "ja": "ja", "ko": "ko"}

    def __init__(self, base_url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
                 api_key=None, model="qwen3.5-livetranslate-flash-realtime", timeout=120.0,
                 pace=True, audio_out_dir=None):
        self.url = f"{base_url}?model={model}"
        self.key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.timeout = timeout
        # pace=True：1x 实时节奏喂音频 → 真同传延迟(AL/LAAL 有意义)；
        # pace=False：不限速灌满 → 近离线质量上限，做"非实时质量保留率"的分母(offline 变体)。
        self.pace = pace
        # audio_out_dir 非空 → 开 audio 模态收译后语音存 wav(同传栏目用)。Qwen 输出 24k PCM。
        self.audio_out_dir = audio_out_dir
        self.name = "qwen-simult" if pace else "qwen-simult-offline"

    def generate(self, item) -> "TranscribeResult":
        if not item.get("audio_path"):
            return TranscribeResult("", 0.0, {}, ok=False, error="Qwen 同传需要 audio_path")
        if not self.key:
            return TranscribeResult("", 0.0, {}, ok=False, error="缺 DASHSCOPE_API_KEY")
        try:
            import base64
            import threading

            from websocket import create_connection  # pip/uv: websocket-client

            from metrics import average_lagging
            tgt = self._TGT.get(item.get("target_lang", "英文"), "en")
            pcm = load_pcm_bytes(item["audio_path"])
            src_dur = len(pcm) / 32000.0          # 16k*2B/sample → 源音频秒数
            ws = ws_connect(self.url, timeout=self.timeout,
                                   header=[f"Authorization: Bearer {self.key}"])
            try:
                t0 = time.perf_counter()
                sess = {"modalities": ["text", "audio"] if self.audio_out_dir else ["text"],
                        "input_audio_format": "pcm", "sample_rate": 16000,
                        "translation": {"language": tgt}}
                if self.audio_out_dir:
                    sess["output_audio_format"] = "pcm"   # 收译后语音(24k PCM)
                ws.send(json.dumps({"type": "session.update", "session": sess}))
                while True:  # 等 session 就绪再推音频
                    ty = json.loads(ws.recv()).get("type", "")
                    if ty in ("session.created", "session.updated"):
                        break
                    if ty == "error":
                        raise RuntimeError("session.update 失败")
                # 发送线程：100ms 帧按 1x 实时节奏 append，st['sent_s'] 记已消耗源音频秒数（AL 时间轴）
                st = {"sent_s": 0.0}

                def _send():
                    base = time.perf_counter()
                    for k, i in enumerate(range(0, len(pcm), 3200)):  # 3200B=100ms
                        ws.send(json.dumps({"type": "input_audio_buffer.append",
                                            "audio": base64.b64encode(pcm[i:i + 3200]).decode()}))
                        st["sent_s"] = min(src_dur, (i + 3200) / 32000.0)
                        if self.pace:  # 实时节奏；offline 变体不限速灌满
                            dt = base + (k + 1) * 0.1 - time.perf_counter()
                            if dt > 0:
                                time.sleep(dt)
                    ws.send(json.dumps({"type": "session.finish"}))

                sender = threading.Thread(target=_send, daemon=True)
                sender.start()
                # 接收线程(主)：边发边收。response.text.text 是**累积**全文(逐渐变长)，
                # 取增量 = 当前减上次前缀，打 (已消耗源秒数, 增量) 供 AL；done 为该段权威全文。
                emissions, ttfb = [], None
                done_texts, done_events, cur = [], [], ""
                trace = SimultTraceRecorder(target_lang=tgt, granularity="token")
                n_upd = n_rev = 0          # 改写率：累积更新次数 / 其中"已吐文本被回改"次数
                pcm_out = bytearray()      # 译后语音(audio 模态)
                while True:
                    raw = ws.recv()
                    if isinstance(raw, (bytes, bytearray)):
                        pcm_out += raw      # 个别实现走二进制音频帧
                        continue
                    ev = json.loads(raw)
                    ty = ev.get("type", "")
                    # 译文文本：纯文本模态走 response.text.text；audio 模态走 response.audio_transcript.text
                    if ty in ("response.text.text", "response.audio_transcript.text"):
                        new = ev.get("text", "") or ""
                        if new != cur:
                            trace.replace_partial(new, st["sent_s"], time.perf_counter() - t0)
                            n_upd += 1
                            if not new.startswith(cur):   # 前缀变了 = 已显示的译文被改写(闪烁)
                                n_rev += 1
                        delta = new[len(cur):] if new.startswith(cur) else new
                        cur = new
                        if delta.strip():
                            if ttfb is None:
                                ttfb = time.perf_counter() - t0
                            emissions.append((st["sent_s"], delta))
                    elif ty in ("response.text.done", "response.audio_transcript.done"):
                        if ev.get("text"):
                            done_texts.append(ev["text"].strip())
                            done_events.append((st["sent_s"], ev["text"].strip()))
                            trace.commit(ev["text"], st["sent_s"], time.perf_counter() - t0)
                        cur = ""                       # 下一段 response 重新累积
                    elif ty == "response.audio.delta":
                        try:
                            pcm_out += base64.b64decode(ev.get("delta", "") or "")
                        except Exception:
                            pass
                    elif ty == "session.finished":
                        break
                    elif ty == "error":
                        raise RuntimeError(str(ev.get("error") or ev)[:300])
                el = time.perf_counter() - t0
                sender.join(timeout=1.0)
            finally:
                ws.close()
            text = " ".join(done_texts).strip() or "".join(t for _, t in emissions).strip()
            saved_trace = trace.finish(text, src_dur, el)
            al, laal = average_lagging(emissions, src_dur, item.get("ref_text", ""), tgt)
            extra = {"ttfb_s": round(ttfb, 3) if ttfb else None,
                     "al_s": al, "laal_s": laal,
                     "revision_rate": round(n_rev / n_upd, 3) if n_upd else None,
                     "src_dur_s": round(src_dur, 2), "n_seg": len(emissions),
                     "translation_segments": _timed_text_segments(done_events),
                     "simult_trace": saved_trace,
                     "stable_ttfb_s": saved_trace["summary"]["ttfb_stable_s"],
                     "churn_rate": saved_trace["summary"]["churn_rate"]}
            if self.audio_out_dir and pcm_out:   # 译后语音落盘(Qwen 输出 24k PCM)
                extra["tts_audio"] = save_tts_audio(bytes(pcm_out),
                    os.path.join(self.audio_out_dir, f"{self.name}__{item.get('id', 'x')}"), sr=24000)
            return TranscribeResult(text, el, _sanitize_nan(extra))
        except Exception as e:
            _L = locals()   # 长音频整段跑：断连时保留已收译文 + 已喂音频秒数(断点位置)
            _parts = _L.get("done_texts") or _L.get("finals") or _L.get("texts") or []
            _em = _L.get("emissions") or []
            _txt = " ".join(_parts).strip() or "".join(t for _, t in _em).strip()
            _fed = (_L.get("st") or {}).get("sent_s", 0.0)
            return TranscribeResult(_txt, _L.get("el", 0.0) or 0.0,
                                    {"audio_fed_s": round(_fed, 1), "n_seg": len(_parts), "partial": bool(_txt)},
                                    ok=False, error=str(e))


class XfSimultAdapter:
    """科大讯飞同声传译 — WebSocket，hmac-sha256 签名 URL 鉴权。

    凭据 XF_APPID/XF_API_KEY/XF_API_SECRET（讯飞开放平台开通"同声传译"能力后获取，见 .env.example）。
    逐帧发裸 PCM(status 0/1/2 标首/中/尾) → payload.streamtrans_results.text(base64→json 取 dst)。
    接口强带 TTS 译文音频(tts_results)，评测只取文本、忽略音频。
    ⚠️ domain=ist_ed_open(教育域)；输出语种对有限(中英日韩等)。字段依官方文档实现，待活体冒烟。
    """
    name = "xf-simult"
    _FROM = {"中文": "cn", "简体中文": "cn", "英文": "en", "zh": "cn", "en": "en"}
    _TO = {"英文": "en", "English": "en", "中文": "cn", "Chinese": "cn",
           "日文": "ja", "Japanese": "ja", "韩文": "ko", "Korean": "ko",
           "en": "en", "zh": "cn"}
    _LANG = {"中文": "zh_cn", "简体中文": "zh_cn", "英文": "en_us", "zh": "zh_cn", "en": "en_us"}
    # 同传 API 强制要 TTS 发音人(vcn 必填)，按目标语挑个常驻音色；译文音频我们忽略，只取文本
    _VCN = {"en": "x2_john", "cn": "x2_xiaozhong", "ja": "x2_haoyu", "ko": "x2_haoyu"}

    def __init__(self, appid=None, api_key=None, api_secret=None,
                 host="ws-api.xf-yun.com", path="/v1/private/simult_interpretation",
                 timeout=120.0, src_lang="中文", audio_out_dir=None):
        self.appid = appid or os.environ.get("XF_APPID", "")
        self.api_key = api_key or os.environ.get("XF_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("XF_API_SECRET", "")
        self.host, self.path, self.timeout = host, path, timeout
        self.src_lang = src_lang   # 源语种固定项(讯飞同传需显式 from)；跨语向建模待二期
        self.audio_out_dir = audio_out_dir   # 非空 → 收 tts_results.audio 存 wav(讯飞 16k PCM)

    def _signed_url(self) -> str:
        """RFC1123 date + hmac-sha256(host/date/request-line) → base64 authorization 拼进 query。"""
        import base64
        import hashlib
        import hmac
        from email.utils import formatdate
        from urllib.parse import urlencode
        date = formatdate(usegmt=True)
        origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        sig = base64.b64encode(hmac.new(self.api_secret.encode(), origin.encode(),
                                        hashlib.sha256).digest()).decode()
        auth = (f'api_key="{self.api_key}", algorithm="hmac-sha256", '
                f'headers="host date request-line", signature="{sig}"')
        qs = urlencode({"authorization": base64.b64encode(auth.encode()).decode(),
                        "date": date, "host": self.host})
        return f"wss://{self.host}{self.path}?{qs}"

    def generate(self, item) -> "TranscribeResult":
        if not item.get("audio_path"):
            return TranscribeResult("", 0.0, {}, ok=False, error="讯飞同传需要 audio_path")
        if not (self.appid and self.api_key and self.api_secret):
            return TranscribeResult("", 0.0, {}, ok=False,
                                    error="缺 XF_APPID/XF_API_KEY/XF_API_SECRET")
        try:
            import base64
            import threading

            from websocket import create_connection  # pip/uv: websocket-client

            from metrics import average_lagging
            # ist.language 永远 zh_cn(ASR 引擎档位=中英混合模式，非源语种！改成 en_us 会 convertParam err)；
            # 翻译方向只由 streamtrans.from/to 定。源语种从 item 语向取。
            src_short = item.get("lang", "zh-en").split("-")[0]
            frm = {"zh": "cn", "en": "en", "yue": "yue"}.get(src_short, "cn")
            lang = "zh_cn"
            to = self._TO.get(item.get("target_lang", "英文"), "en")
            pcm = load_pcm_bytes(item["audio_path"])
            src_dur = len(pcm) / 32000.0
            frames = [pcm[i:i + 1280] for i in range(0, len(pcm), 1280)] or [b""]  # 40ms@16k=1280
            ws = ws_connect(self._signed_url(), timeout=self.timeout)
            try:
                t0 = time.perf_counter()
                st = {"sent_s": 0.0, "n_seg": 0, "latest_text": ""}
                emit_longform_progress(item, phase="started", audio_total_s=src_dur)

                def _send():  # 1x 实时节奏推帧，status 0/1/2 标首/中/尾
                    base = time.perf_counter()
                    for idx, chunk in enumerate(frames):
                        status = 0 if idx == 0 else (2 if idx == len(frames) - 1 else 1)
                        frame = {"header": {"app_id": self.appid, "status": status},
                                 "payload": {"data": {"audio": base64.b64encode(chunk).decode(),
                                                      "encoding": "raw", "sample_rate": 16000,
                                                      "status": status}}}
                        if idx == 0:  # 首帧带业务参数(tts.vcn 必填，按目标语取音色)
                            # ⚠️ 讯飞 simult 源识别只支持中文(accent=mandarin 必填且 en_us 会 convertParam err)
                            # → 实测只能 zh→en；en→zh 不支持，会失败(讯飞侧限制)
                            frame["parameter"] = {
                                "ist": {"language": lang, "domain": "ist_ed_open", "accent": "mandarin"},
                                "streamtrans": {"from": frm, "to": to},
                                "tts": {"vcn": self._VCN.get(to, "x2_john"),
                                        "tts_results": {"encoding": "speex-wb", "sample_rate": 16000}}}
                        ws.send(json.dumps(frame))
                        st["sent_s"] = min(src_dur, (idx + 1) * 1280 / 32000.0)
                        if idx == 0 or (idx + 1) % 125 == 0 or idx == len(frames) - 1:
                            emit_longform_progress(
                                item, audio_s=st["sent_s"], audio_total_s=src_dur,
                                segments=st["n_seg"], latest_text=st["latest_text"],
                            )
                        dt = base + (idx + 1) * (1280 / 32000.0) - time.perf_counter()
                        if dt > 0:
                            time.sleep(dt)

                sender = threading.Thread(target=_send, daemon=True)
                sender.start()
                # 讯飞 streamtrans_results：dst 段内累积(is_final=0 中间修订/1 定稿)。
                # 文本只取 is_final 定稿段(去重)；AL 用段内累积前缀的增量打时间戳；改写率同 qwen。
                emissions, ttfb = [], None
                finals, final_events, cur = [], [], ""
                trace = SimultTraceRecorder(target_lang=to, granularity="token")
                n_upd = n_rev = 0
                tts_speex, tts_done = bytearray(), False   # speex-wb 帧, 收齐(tts_results.status==2)后解码
                while True:
                    msg = ws.recv()
                    if not msg:
                        break
                    resp = json.loads(msg)
                    h = resp.get("header", {})
                    if h.get("code", 0) != 0:
                        raise RuntimeError(f"{h.get('code')}: {h.get('message')}")
                    if self.audio_out_dir:   # 收 speex-wb 帧(逐帧[1字节长度][帧]，收齐后 speex 解码)
                        tr = (resp.get("payload") or {}).get("tts_results") or {}
                        if tr.get("audio"):
                            try:
                                tts_speex += base64.b64decode(tr["audio"])
                            except Exception:
                                pass
                        if tr.get("status") == 2:   # 该会话 TTS 收齐
                            tts_done = True
                    b64 = (resp.get("payload") or {}).get("streamtrans_results", {}).get("text")
                    if b64:
                        seg = json.loads(base64.b64decode(b64).decode("utf-8"))
                        dst = seg.get("dst", "") or ""
                        if dst and dst != cur:
                            trace.replace_partial(dst, st["sent_s"], time.perf_counter() - t0)
                            n_upd += 1
                            revised = bool(cur and not dst.startswith(cur))
                            if revised:
                                n_rev += 1
                            delta = dst[len(cur):] if dst.startswith(cur) else dst
                            cur = dst
                            st["latest_text"] = dst.strip()
                            emit_longform_progress(
                                item, phase="revision" if revised else "partial",
                                audio_s=st["sent_s"], audio_total_s=src_dur,
                                segments=st["n_seg"], latest_text=st["latest_text"],
                            )
                            if delta.strip():
                                if ttfb is None:
                                    ttfb = time.perf_counter() - t0
                                emissions.append((st["sent_s"], delta))
                        if dst and seg.get("is_final"):
                            finals.append(dst.strip())
                            final_events.append((st["sent_s"], dst.strip()))
                            trace.commit(dst, st["sent_s"], time.perf_counter() - t0)
                            st["n_seg"] = len(finals)
                            st["latest_text"] = dst.strip()
                            emit_longform_progress(
                                item, phase="segment", audio_s=st["sent_s"],
                                audio_total_s=src_dur, segments=st["n_seg"],
                                latest_text=st["latest_text"],
                            )
                            cur = ""               # 下一段重新累积
                    if h.get("status") == 2:
                        break    # 翻译终态即完成；TTS 是可选产物，缺少独立终态不能阻塞主评测
                el = time.perf_counter() - t0
                sender.join(timeout=1.0)
                emit_longform_progress(
                    item, phase="completed", audio_s=src_dur, audio_total_s=src_dur,
                    segments=len(finals), latest_text=finals[-1] if finals else "",
                )
            finally:
                ws.close()
            text = " ".join(finals).strip() or "".join(t for _, t in emissions).strip()
            saved_trace = trace.finish(text, src_dur, el)
            al, laal = average_lagging(emissions, src_dur, item.get("ref_text", ""), to)
            extra = {"ttfb_s": round(ttfb, 3) if ttfb else None,
                     "al_s": al, "laal_s": laal,
                     "revision_rate": round(n_rev / n_upd, 3) if n_upd else None,
                     "src_dur_s": round(src_dur, 2), "n_seg": len(finals),
                     "translation_segments": _timed_text_segments(final_events),
                     "simult_trace": saved_trace,
                     "stable_ttfb_s": saved_trace["summary"]["ttfb_stable_s"],
                     "churn_rate": saved_trace["summary"]["churn_rate"]}
            if self.audio_out_dir and tts_done and tts_speex:   # 只解码完整的 speex-wb 音频
                try:
                    import speex_decode
                    pcm = speex_decode.decode(bytes(tts_speex))
                    if pcm:
                        extra["tts_audio"] = save_pcm_wav(pcm,
                            os.path.join(self.audio_out_dir, f"{self.name}__{item.get('id', 'x')}.wav"), sr=16000)
                except Exception:
                    pass
            return TranscribeResult(text, el, _sanitize_nan(extra))
        except Exception as e:
            _L = locals()   # 长音频整段跑：断连时保留已收译文 + 已喂音频秒数(断点位置)
            _parts = _L.get("done_texts") or _L.get("finals") or _L.get("texts") or []
            _em = _L.get("emissions") or []
            _txt = " ".join(_parts).strip() or "".join(t for _, t in _em).strip()
            _fed = (_L.get("st") or {}).get("sent_s", 0.0)
            emit_longform_progress(
                item, phase="failed", audio_s=_fed, audio_total_s=_L.get("src_dur", 0.0),
                segments=len(_parts), latest_text=_parts[-1] if _parts else "",
            )
            return TranscribeResult(_txt, _L.get("el", 0.0) or 0.0,
                                    {"audio_fed_s": round(_fed, 1), "n_seg": len(_parts), "partial": bool(_txt)},
                                    ok=False, error=str(e))


class XfSparkSlmIatAdapter:
    """讯飞方言识别大模型 — WebSocket，hmac-sha256 签名 URL 鉴权。

    凭据 XF_APPID/XF_API_KEY/XF_API_SECRET；官方协议为 16k/8k mono s16le PCM，最长 60s。
    方言大模型文档说明普通话、简单英语和 202 种方言免切换，因此不传具体方言参数。"""
    name = "xf-spark-slm-iat"

    def __init__(self, appid=None, api_key=None, api_secret=None,
                 host="iat.cn-huabei-1.xf-yun.com", path="/v1", timeout=90.0):
        self.appid = appid or os.environ.get("XF_APPID", "")
        self.api_key = api_key or os.environ.get("XF_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("XF_API_SECRET", "")
        self.host = os.environ.get("XF_IAT_HOST", host)
        self.path = os.environ.get("XF_IAT_PATH", path)
        self.timeout = float(os.environ.get("XF_IAT_TIMEOUT", timeout))

    def _signed_url(self) -> str:
        import hashlib
        import hmac
        from email.utils import formatdate
        from urllib.parse import urlencode
        date = formatdate(usegmt=True)
        origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        sig = base64.b64encode(hmac.new(self.api_secret.encode(), origin.encode(),
                                        hashlib.sha256).digest()).decode()
        auth = (f'api_key="{self.api_key}", algorithm="hmac-sha256", '
                f'headers="host date request-line", signature="{sig}"')
        qs = urlencode({"authorization": base64.b64encode(auth.encode()).decode(),
                        "date": date, "host": self.host})
        return f"wss://{self.host}{self.path}?{qs}"

    @staticmethod
    def _decode_result_text(b64_text: str) -> str:
        if not b64_text:
            return ""
        raw = base64.b64decode(b64_text).decode("utf-8", "ignore")
        try:
            data = json.loads(raw)
        except Exception:
            return raw.strip()
        if isinstance(data, dict):
            words = []
            for seg in data.get("ws") or []:
                cws = seg.get("cw") or []
                if cws:
                    words.append(str(cws[0].get("w", "")))
            if words:
                return "".join(words).strip()
        return raw.strip()

    def transcribe(self, audio_path: str, language: str = "auto", hotwords: str = "") -> TranscribeResult:
        if not audio_path:
            return TranscribeResult("", 0.0, {}, ok=False, error="讯飞方言大模型需要 audio_path")
        if not (self.appid and self.api_key and self.api_secret):
            return TranscribeResult("", 0.0, {}, ok=False,
                                    error="缺 XF_APPID/XF_API_KEY/XF_API_SECRET")
        try:
            pcm = load_pcm_bytes(audio_path)
            frames = [pcm[i:i + 1280] for i in range(0, len(pcm), 1280)] or [b""]
            ws = ws_connect(self._signed_url(), timeout=self.timeout)
            final = {}
            parts = []
            sid = ""
            try:
                t0 = time.perf_counter()
                for idx, chunk in enumerate(frames):
                    status = 0 if idx == 0 else (2 if idx == len(frames) - 1 else 1)
                    frame = {
                        "header": {"app_id": self.appid, "status": status},
                        "payload": {"audio": {"encoding": "raw", "sample_rate": 16000,
                                                "channels": 1, "bit_depth": 16,
                                                "status": status,
                                                "audio": base64.b64encode(chunk).decode()}},
                    }
                    if idx == 0:
                        frame["parameter"] = {
                            "iat": {
                                "language": "zh_cn",
                                "accent": "mulacc",
                                "domain": "slm",
                                "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
                            }
                        }
                    ws.send(json.dumps(frame, ensure_ascii=False))
                    if status != 2:
                        time.sleep(0.04)
                while True:
                    resp = json.loads(ws.recv())
                    h = resp.get("header") or {}
                    sid = h.get("sid") or sid
                    if h.get("code", 0) != 0:
                        raise RuntimeError(f"{h.get('code')}: {h.get('message')}")
                    text = self._decode_result_text(((resp.get("payload") or {}).get("result") or {}).get("text", ""))
                    if text:
                        # rpl 结果会替换历史片段；用 sn 粗粒度保留最后版本。
                        try:
                            raw = json.loads(base64.b64decode(((resp.get("payload") or {}).get("result") or {}).get("text", "")).decode("utf-8", "ignore"))
                            sn = int(raw.get("sn", len(final) + 1))
                            final[sn] = text
                        except Exception:
                            parts.append(text)
                    if h.get("status") == 2:
                        break
                el = time.perf_counter() - t0
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
            out = "".join(final[k] for k in sorted(final)) if final else "".join(parts)
            return TranscribeResult(out.strip(), el, {"sid": sid})
        except Exception as e:
            return TranscribeResult("", 0.0, {}, ok=False, error=str(e))


class PlatformWSVoiceInputAdapter:
    """平台「语音输入」WS 流式（/ws/audio/voice-input）→ **真流式同传**。

    专门的 simult-interpreting 只有 POST 单发(whole-file，算不了 AL)；此 WS 端点
    voice_input_setting.target_language 开翻译，逐段 result_final.data.translation 增量出译文，
    1x 喂可算真 AL/LAAL，对齐 qwen/xf 口径。被测主体平台的真同传版(此前 WSAdapter 方案搁置，现启用)。
    流程：connected_success → task_start → 二进制 PCM → result_final → task_finish。
    """
    name = "plat-simult-ws"
    _TGT = {
        "英文": "en", "English": "en", "中文": "zh", "简体中文": "zh", "Chinese": "zh",
        "粤语": "yue", "Cantonese": "yue", "日文": "ja", "Japanese": "ja",
        "韩文": "ko", "Korean": "ko", "Arabic": "ar", "Dutch": "nl", "French": "fr",
        "German": "de", "Indonesian": "id", "Italian": "it", "Malay": "ms",
        "Portuguese": "pt", "Russian": "ru", "Spanish": "es", "Thai": "th",
        "Turkish": "tr", "Urdu": "ur", "Vietnamese": "vi",
        "en": "en", "zh": "zh", "yue": "yue", "ja": "ja", "ko": "ko",
    }

    def __init__(self, base_url=ENDPOINTS["platform"], timeout=180.0, pace=True, api_key=None):
        self.ws_url = (base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
                       + "/ws/audio/voice-input")
        self.timeout = timeout
        self.pace = pace
        token = _plat_token(api_key)
        self.ws_headers = [f"Authorization: Bearer {token}"] if token else []

    def _vad_setting(self):
        runtime_json = _runtime_request_parts(self)["json"]
        return {
            "silence_duration": runtime_json.get("vad_silence_duration_ms", 600),
            "min_speech_duration": runtime_json.get("vad_min_speech_duration_ms", 300),
            "soft_max_duration": runtime_json.get("vad_soft_max_duration_ms", 15000),
            "hard_max_duration": runtime_json.get("vad_hard_max_duration_ms", 30000),
            "soft_silence_duration": runtime_json.get("vad_soft_silence_duration_ms", 300),
            "threshold": runtime_json.get("vad_threshold", 0.5),
        }

    def _voice_input_setting(self, item, target_language):
        runtime_json = _runtime_request_parts(self)["json"]
        return {
            "language": item.get("request_language") or "auto",
            "target_language": target_language,
            "hot_words": [],
            "abbreviations": {},
            "summary_interval_s": runtime_json.get("summary_interval_s", 0),
        }

    def generate(self, item) -> "TranscribeResult":
        if not item.get("audio_path"):
            return TranscribeResult("", 0.0, {}, ok=False, error="平台 WS 同传需要 audio_path")
        try:
            import threading

            from websocket import create_connection

            from metrics import average_lagging
            raw_target = item.get("target_lang", "英文")
            tgt = self._TGT.get(raw_target, raw_target)
            pcm = load_pcm_bytes(item["audio_path"])
            src_dur = len(pcm) / 32000.0
            ws = ws_connect(self.ws_url, timeout=self.timeout, header=self.ws_headers)
            try:
                t0 = time.perf_counter()
                while True:  # 等 connected_success
                    ev = json.loads(ws.recv())
                    if ev.get("event") == "connected_success":
                        connected_event = ev
                        break
                    if ev.get("event") in ("error", "task_failed"):
                        raise RuntimeError(str(ev)[:200])
                start_frame = {
                    "event": "task_start", "model": "voice-input",
                    "audio_setting": {"sample_rate": 16000, "format": "pcm", "channel": 1},
                    "vad_setting": self._vad_setting(),
                    "voice_input_setting": self._voice_input_setting(item, tgt),
                }
                ws.send(json.dumps(start_frame))
                while True:  # 等 task_started 再发音频
                    ev = json.loads(ws.recv())
                    if ev.get("event") == "task_started":
                        task_started_event = ev
                        break
                    if ev.get("event") in ("error", "task_failed"):
                        raise RuntimeError(str(ev)[:200])
                ws_control = _ws_control_record(
                    self.ws_url, connected_event, start_frame, task_started_event
                )
                st = {"sent_s": 0.0, "n_seg": 0, "latest_text": ""}
                emit_longform_progress(item, phase="started", audio_total_s=src_dur)

                def _send():
                    base = time.perf_counter()
                    for k, i in enumerate(range(0, len(pcm), 3200)):
                        ws.send_binary(pcm[i:i + 3200])
                        st["sent_s"] = min(src_dur, (i + 3200) / 32000.0)
                        if k == 0 or (k + 1) % 50 == 0 or st["sent_s"] >= src_dur:
                            emit_longform_progress(
                                item, audio_s=st["sent_s"], audio_total_s=src_dur,
                                segments=st["n_seg"], latest_text=st["latest_text"],
                            )
                        if self.pace:
                            dt = base + (k + 1) * 0.1 - time.perf_counter()
                            if dt > 0:
                                time.sleep(dt)
                    st["audio_done_elapsed_s"] = time.perf_counter() - t0
                    ws.send(json.dumps({"event": "task_finish"}))
                    st["task_finish_sent_elapsed_s"] = time.perf_counter() - t0

                sender = threading.Thread(target=_send, daemon=True)
                sender.start()
                emissions, ttfb, finals = [], None, []   # result_final 逐段译文(段内已定稿)
                trace = SimultTraceRecorder(target_lang=tgt, granularity="segment-final")
                while True:
                    ev = json.loads(ws.recv())
                    e = ev.get("event")
                    if e == "result_final":
                        tr = (ev.get("data") or {}).get("translation", "") or ""
                        if tr:
                            if ttfb is None:
                                ttfb = time.perf_counter() - t0
                            emissions.append((st["sent_s"], tr))
                            finals.append(tr.strip())
                            trace.commit(tr, st["sent_s"], time.perf_counter() - t0)
                            st["n_seg"] = len(finals)
                            st["latest_text"] = tr.strip()
                            emit_longform_progress(
                                item, phase="segment", audio_s=st["sent_s"],
                                audio_total_s=src_dur, segments=st["n_seg"],
                                latest_text=st["latest_text"],
                            )
                    elif e in ("task_finished", "task_stopped", "task_done"):
                        break
                    elif e in ("error", "task_failed"):
                        raise RuntimeError(str(ev)[:200])
                el = time.perf_counter() - t0
                sender.join(timeout=1.0)
                emit_longform_progress(
                    item, phase="completed", audio_s=src_dur, audio_total_s=src_dur,
                    segments=len(finals), latest_text=finals[-1] if finals else "",
                )
            finally:
                ws.close()
            al, laal = average_lagging(emissions, src_dur, item.get("ref_text", ""), tgt)
            text = " ".join(finals).strip()
            saved_trace = trace.finish(text, src_dur, el)
            finish_tail_s = max(0.0, el - st.get("audio_done_elapsed_s", el))
            return TranscribeResult(text, el,
                                    {"ttfb_s": round(ttfb, 3) if ttfb else None,
                                     "al_s": al, "laal_s": laal,
                                     "src_dur_s": round(src_dur, 2), "n_seg": len(finals),
                                     "finish_tail_s": round(finish_tail_s, 3),
                                     "translation_segments": _timed_text_segments(emissions),
                                     "simult_trace": saved_trace,
                                     "ws_control": ws_control,
                                     "stable_ttfb_s": saved_trace["summary"]["ttfb_stable_s"],
                                     "churn_rate": saved_trace["summary"]["churn_rate"]})
        except Exception as e:
            _L = locals()   # 长音频整段跑：断连时保留已收译文 + 已喂音频秒数(断点位置)
            _parts = _L.get("done_texts") or _L.get("finals") or _L.get("texts") or []
            _em = _L.get("emissions") or []
            _txt = " ".join(_parts).strip() or "".join(t for _, t in _em).strip()
            _fed = (_L.get("st") or {}).get("sent_s", 0.0)
            emit_longform_progress(
                item, phase="failed", audio_s=_fed, audio_total_s=_L.get("src_dur", 0.0),
                segments=len(_parts), latest_text=_parts[-1] if _parts else "",
            )
            _extra = {"audio_fed_s": round(_fed, 1), "n_seg": len(_parts), "partial": bool(_txt)}
            if _L.get("ws_control"):
                _extra["ws_control"] = _L["ws_control"]
            return TranscribeResult(_txt, _L.get("el", 0.0) or 0.0,
                                    _extra,
                                    ok=False, error=str(e))


class PlatformWSSimultAdapter:
    """平台同声传译 WS 流式（/ws/audio/simult-interpreting）— 真流式同传 + TTS。

    一个端点全给：si_token 增量译文(算 AL) + si_segment_done(终稿 + 服务端 timings) +
    si_tts_file(译后语音 mp3) + si_source_audio(分段源音频)。task_start/task_finish 协议。
    1x 实时节奏喂裸 PCM；emission 相对已消耗源秒数算 AL，与 qwen/xf 同口径。
    """
    name = "plat-simult"
    supports_hotwords = True
    _TGT = {"英文": "en", "中文": "zh", "简体中文": "zh", "日文": "ja", "韩文": "ko",
            "en": "en", "zh": "zh"}

    def __init__(self, base_url=None, timeout=180.0, pace=True,
                 audio_out_dir=None, tts_mode="preset", stream_log=None, api_key=None):
        base_url = base_url or ENDPOINTS.get("platform_ws")
        if not base_url:
            raise RuntimeError("plat-simult 未配置端点：看板「接口管理」编辑 plat-simult 填 WS 地址即可；"
                               "CLI/容器跑则设 ASR_PLATFORM_WS_URL 环境变量（私有隧道地址不入仓）")
        self.ws_url = (base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
                       + "/ws/audio/simult-interpreting")
        self.timeout = timeout
        self.pace = pace
        self.audio_out_dir = audio_out_dir
        self.tts_mode = tts_mode   # 合并后新增：preset=服务端预置音色(默认,横评用,可比) / clone=克隆输入说话人
        self.stream_log = stream_log   # 长音频整段跑：每段定稿即时追加落盘(进程被杀也不丢)
        self._slog_lk = threading.Lock()   # stream_log 跨线程(--workers>1)写互斥,防交错出半行 JSON
        token = _plat_token(api_key)
        self.ws_headers = [f"Authorization: Bearer {token}"] if token else []

    def _voice_input_setting(self, item):
        raw_target = item.get("target_lang", "英文")
        runtime_json = _runtime_request_parts(self)["json"]
        return {
            "language": item.get("request_language") or "auto",
            "target_language": self._TGT.get(raw_target, raw_target),
            "hot_words": [w for w in str(item.get("hotwords") or "").replace(",", " ").split() if w],
            "abbreviations": runtime_json.get("abbreviations") or {},
            "pipeline_mode": runtime_json.get("pipeline_mode", "single_stage"),
            "return_asr_text": bool(runtime_json.get("return_asr_text", False)),
            "incremental_enabled": bool(runtime_json.get("incremental_enabled", False)),
            "incremental_interval_ms": runtime_json.get("incremental_interval_ms", 1000),
            "incremental_holdback_tokens": runtime_json.get("incremental_holdback_tokens", 1),
        }

    def _tts_setting(self):
        runtime_json = _runtime_request_parts(self)["json"]
        setting = {
            "enable": bool(self.audio_out_dir),
            "mode": runtime_json.get("tts_mode", self.tts_mode),
            "output_sample_rate": runtime_json.get("tts_output_sample_rate", 16000),
        }
        if runtime_json.get("tts_voice_id"):
            setting["voice_id"] = runtime_json["tts_voice_id"]
        return setting

    def _vad_setting(self):
        runtime_json = _runtime_request_parts(self)["json"]
        return {
            "threshold": runtime_json.get("vad_threshold", 0.5),
            "silence_duration": runtime_json.get("vad_silence_duration_ms", 700),
            "min_speech_duration": runtime_json.get("vad_min_speech_duration_ms", 300),
            "soft_max_duration": runtime_json.get("vad_soft_max_duration_ms", 15000),
            "hard_max_duration": runtime_json.get("vad_hard_max_duration_ms", 30000),
            "soft_silence_duration": runtime_json.get("vad_soft_silence_duration_ms", 300),
        }

    def generate(self, item) -> "TranscribeResult":
        if not item.get("audio_path"):
            return TranscribeResult("", 0.0, {}, ok=False, error="平台 WS 同传需要 audio_path")
        try:
            import threading

            from websocket import create_connection

            from metrics import average_lagging
            voice_setting = self._voice_input_setting(item)
            tgt = voice_setting["target_language"]
            pcm = load_pcm_bytes(item["audio_path"])
            src_dur = len(pcm) / 32000.0
            ws = ws_connect(self.ws_url, timeout=self.timeout, header=self.ws_headers)
            try:
                t0 = time.perf_counter()
                while True:  # 等 connected_success
                    ev = json.loads(ws.recv())
                    if ev.get("event") == "connected_success":
                        connected_event = ev
                        break
                    if ev.get("event") in ("error", "task_failed"):
                        raise RuntimeError(str(ev)[:200])
                start_frame = {
                    "event": "task_start", "model": "simult-interpreting",
                    "audio_setting": {"sample_rate": 16000, "format": "pcm", "channel": 1},
                    "vad_setting": self._vad_setting(),
                    "voice_input_setting": voice_setting,
                    "tts_setting": self._tts_setting(),
                }
                ws.send(json.dumps(start_frame))
                while True:  # 等 task_started
                    ev = json.loads(ws.recv())
                    if ev.get("event") == "task_started":
                        task_started_event = ev
                        break
                    if ev.get("event") in ("error", "task_failed"):
                        raise RuntimeError(str(ev)[:200])
                ws_control = _ws_control_record(
                    self.ws_url, connected_event, start_frame, task_started_event
                )
                st = {"sent_s": 0.0, "n_seg": 0, "latest_text": ""}
                emit_longform_progress(item, phase="started", audio_total_s=src_dur)

                def _send():
                    base = time.perf_counter()
                    for k, i in enumerate(range(0, len(pcm), 3200)):
                        ws.send_binary(pcm[i:i + 3200])
                        st["sent_s"] = min(src_dur, (i + 3200) / 32000.0)
                        if k == 0 or (k + 1) % 50 == 0 or st["sent_s"] >= src_dur:
                            emit_longform_progress(
                                item, audio_s=st["sent_s"], audio_total_s=src_dur,
                                segments=st["n_seg"], latest_text=st["latest_text"],
                            )
                        if self.pace:
                            dt = base + (k + 1) * 0.1 - time.perf_counter()
                            if dt > 0:
                                time.sleep(dt)
                    for _ in range(10):   # 补尾 ~1s 静音 → 触发 VAD silence 终结末段(否则末段不出 done)
                        ws.send_binary(b"\x00" * 3200)
                        if self.pace:
                            time.sleep(0.1)
                    # 末段竞态修复：服务端 si_segment_done 排在 inline TTS await 之后(中文音色 ~0.7s+)，
                    # 音频一送完就 task_finish 会赶在末段 TTS 完成前拆会话 → 末段 done/TTS 双丢。
                    # 启用 TTS 时，等末段定稿(audio_done 之后才到的 si_segment_done)或 8s 上限再收尾。
                    st["audio_done_t"] = time.perf_counter()
                    st["audio_done_elapsed_s"] = st["audio_done_t"] - t0
                    if self.audio_out_dir:
                        # 末段可能被 VAD 切成多段：等到"距最近一次 si_segment_done 已静默 ≥1.2s"再收尾，
                        # 而非见首个 done 即停(否则后续段 done/TTS 仍会被 task_finish 拆掉)。12s 硬上限兜底。
                        while time.perf_counter() - st["audio_done_t"] < 12.0:
                            sd = st.get("seg_done_t", 0.0)
                            if sd > st["audio_done_t"] and time.perf_counter() - sd > 1.2:
                                break
                            time.sleep(0.2)
                    ws.send(json.dumps({"event": "task_finish"}))
                    st["task_finish_sent_elapsed_s"] = time.perf_counter() - t0

                sender = threading.Thread(target=_send, daemon=True)
                sender.start()
                emissions, ttfb, finals, segment_records = [], None, [], []
                segment_done_count = empty_segment_count = segment_error_count = 0
                trace = SimultTraceRecorder(target_lang=tgt, granularity="token")
                incremental_updates = []
                incremental_phase_counts = {"asr": 0, "translated": 0}
                incremental_asr_ttfb = incremental_translation_ttfb = None
                incremental_asr_trace = SimultTraceRecorder(
                    target_lang=item.get("request_language") or "auto",
                    granularity="incremental-snapshot",
                )
                incremental_translation_trace = SimultTraceRecorder(
                    target_lang=tgt, granularity="incremental-snapshot",
                )
                tts_chunks, e2e = [], []   # si_token 增量(非累积)；TTS 兼容新 PCM 与旧编码音频事件
                while True:
                    ev = json.loads(ws.recv())
                    e = ev.get("event")
                    d = ev.get("data") or {}
                    if e == "heartbeat":
                        continue
                    if e == "si_incremental_update":
                        wall_s = time.perf_counter() - t0
                        phase = str(d.get("phase") or "")
                        if phase in incremental_phase_counts:
                            incremental_phase_counts[phase] += 1
                        if len(incremental_updates) < 4096:
                            incremental_updates.append({
                                "segment_epoch": d.get("segment_epoch"),
                                "segment_key": d.get("segment_key"),
                                "update_index": d.get("update_index"),
                                "audio_ms": d.get("audio_ms"),
                                "received_at_s": round(wall_s, 3),
                                "phase": phase,
                                "original_confirmed": d.get("original_confirmed") or "",
                                "original_draft": d.get("original_draft") or "",
                                "original_draft_replaces_confirmed": bool(
                                    d.get("original_draft_replaces_confirmed")
                                ),
                                "translation_confirmed": d.get("translation_confirmed") or "",
                                "translation_draft": d.get("translation_draft") or "",
                                "translation_draft_replaces_confirmed": bool(
                                    d.get("translation_draft_replaces_confirmed")
                                ),
                                "timing": d.get("timing") or {},
                            })
                        if phase == "asr":
                            preview = _incremental_preview(d, "original")
                            if preview:
                                if incremental_asr_ttfb is None:
                                    incremental_asr_ttfb = wall_s
                                incremental_asr_trace.replace_partial(
                                    preview, st["sent_s"], wall_s
                                )
                        elif phase == "translated":
                            preview = _incremental_preview(d, "translation")
                            if preview:
                                if incremental_translation_ttfb is None:
                                    incremental_translation_ttfb = wall_s
                                incremental_translation_trace.replace_partial(
                                    preview, st["sent_s"], wall_s
                                )
                    elif e == "si_token":
                        c = d.get("content", "") or ""
                        if c:
                            if ttfb is None:
                                ttfb = time.perf_counter() - t0
                            emissions.append((st["sent_s"], c))
                            trace.append_partial(c, st["sent_s"], time.perf_counter() - t0)
                    elif e == "si_segment_done":
                        segment_done_count += 1
                        st["seg_done_t"] = time.perf_counter()   # 末段竞态修复用：标记定稿到达时刻
                        if d.get("text"):
                            final_text = d["text"].strip()
                            finals.append(final_text)
                            trace.commit(final_text, st["sent_s"], time.perf_counter() - t0)
                            timing = d.get("timing") or d.get("timings") or {}
                            try:
                                start_s = float(d.get("timestamp_start")) / 1000
                            except (TypeError, ValueError):
                                start_s = None
                            try:
                                end_s = float(d.get("timestamp_end")) / 1000
                            except (TypeError, ValueError):
                                end_s = st["sent_s"]
                            source_text = str(d.get("asr_text") or d.get("original_text") or "").strip()
                            if incremental_updates:
                                incremental_translation_trace.commit(
                                    final_text, st["sent_s"], time.perf_counter() - t0
                                )
                                if source_text:
                                    incremental_asr_trace.commit(
                                        source_text, st["sent_s"], time.perf_counter() - t0
                                    )
                            segment_record = {
                                "segment_id": d.get("segment_id"),
                                "start_s": round(start_s, 3) if start_s is not None else None,
                                "end_s": round(end_s, 3),
                                "emitted_at_s": round(st["sent_s"], 3),
                                "text": final_text,
                                "asr_text": source_text,
                                "pipeline_mode": d.get("pipeline_mode") or voice_setting["pipeline_mode"],
                                "timing": timing,
                            }
                            segment_reason = (d.get("segment_reason") or d.get("cut_reason")
                                              or d.get("vad_reason") or d.get("reason"))
                            if segment_reason:
                                segment_record["segment_reason"] = str(segment_reason)
                            segment_records.append(segment_record)
                            st["n_seg"] = len(finals)
                            st["latest_text"] = final_text
                            emit_longform_progress(
                                item, phase="segment", audio_s=st["sent_s"],
                                audio_total_s=src_dur, segments=st["n_seg"],
                                latest_text=st["latest_text"],
                            )
                            if self.stream_log:   # 整段跑：即时落盘(已喂音频秒, 译文段),加锁防并发交错
                                try:
                                    with self._slog_lk, open(self.stream_log, "a", encoding="utf-8") as _f:
                                        _f.write(json.dumps({"audio_t": round(st["sent_s"], 1),
                                                             "text": d["text"].strip()}, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass
                        timing = d.get("timing") or d.get("timings") or {}
                        if timing.get("e2e_ms") is not None:
                            e2e.append(timing["e2e_ms"])
                        if not d.get("text"):
                            empty_segment_count += 1
                    elif e == "si_segment_error":
                        segment_error_count += 1
                    elif e == "si_tts_file" and d.get("audio"):
                        try:
                            tts_chunks.append({
                                "audio": base64.b64decode(d["audio"]),
                                "format": str(d.get("format") or ""),
                                "sample_rate": int(d.get("sample_rate") or (
                                    16000 if d.get("format") == "pcm_s16le_mono" else 48000
                                )),
                            })
                        except Exception:
                            pass
                    elif e in ("task_finished", "task_stopped", "task_done"):
                        break
                    elif e in ("error", "task_failed"):
                        raise RuntimeError(str(ev)[:200])
                el = time.perf_counter() - t0
                sender.join(timeout=1.0)
                emit_longform_progress(
                    item, phase="completed", audio_s=src_dur, audio_total_s=src_dur,
                    segments=len(finals), latest_text=finals[-1] if finals else "",
                )
            finally:
                ws.close()
            # 正常取 si_segment_done 的终稿；若服务端定稿卡住（如中文 TTS 回归：
            # en→zh + tts.enable=True 时 token 照来但 si_segment_done 不出、TTS 也不出），
            # 退而用 si_token 增量拼回译文——译文/AL 不丢，只是没有终稿级标点修订与 TTS 音频。
            text = " ".join(finals).strip()
            seg_done = bool(finals)
            if not text and emissions:
                text = "".join(c for _, c in emissions).strip()   # 中文目标无需空格；英文目标走 finals 不触此路
            if not text:   # 连 token 都没有 → 真空返回，判失败让 infer 退避重试
                return TranscribeResult("", el, {}, ok=False, error="WS 同传无译文输出(token/定稿均空)")
            saved_trace = trace.finish(text, src_dur, el)
            al, laal = average_lagging(emissions, src_dur, item.get("ref_text", ""), tgt)
            finish_tail_s = max(0.0, el - st.get("audio_done_elapsed_s", el))
            extra = {"ttfb_s": round(ttfb, 3) if ttfb else None, "al_s": al, "laal_s": laal,
                     "src_dur_s": round(src_dur, 2), "n_seg": len(finals),
                     "finish_tail_s": round(finish_tail_s, 3),
                     "pipeline_mode": voice_setting["pipeline_mode"],
                     "translation_segments": segment_records,
                     "segment_done_count": segment_done_count,
                     "empty_segment_count": empty_segment_count,
                     "segment_error_count": segment_error_count,
                     "simult_trace": saved_trace,
                     "ws_control": ws_control,
                     "stable_ttfb_s": saved_trace["summary"]["ttfb_stable_s"],
                     "churn_rate": saved_trace["summary"]["churn_rate"]}
            source_text = " ".join(s["asr_text"] for s in segment_records if s.get("asr_text")).strip()
            if source_text:
                extra["source_text"] = source_text
                if voice_setting["pipeline_mode"] == "multi_stage":
                    extra["asr_text"] = source_text
            if incremental_updates:
                inc_translation = incremental_translation_trace.finish(text, src_dur, el)
                inc_asr = incremental_asr_trace.finish(source_text, src_dur, el)
                extra.update({
                    "incremental_enabled": voice_setting["incremental_enabled"],
                    "incremental_updates": incremental_updates,
                    "incremental_update_count": len(incremental_updates),
                    "incremental_phase_counts": incremental_phase_counts,
                    "incremental_asr_ttfb_s": (
                        round(incremental_asr_ttfb, 3)
                        if incremental_asr_ttfb is not None else None
                    ),
                    "incremental_translation_ttfb_s": (
                        round(incremental_translation_ttfb, 3)
                        if incremental_translation_ttfb is not None else None
                    ),
                    "incremental_asr_trace": inc_asr,
                    "incremental_translation_trace": inc_translation,
                    "incremental_asr_churn_rate": inc_asr["summary"]["churn_rate"],
                    "incremental_translation_churn_rate": (
                        inc_translation["summary"]["churn_rate"]
                    ),
                })
            if not seg_done:   # token 兜底：服务端未给终稿（中文 TTS 回归连坐），译文为 token 重建
                extra["text_from"] = "si_token"
            if e2e:
                extra["e2e_ms_mean"] = round(sum(e2e) / len(e2e))   # 服务端报的端到端延迟
            for key in ("queue_ms", "asr_ms", "text_translate_ms", "text_ttft_ms",
                        "text_decode_ms", "tts_queue_ms", "tts_ms", "total_ms"):
                values = [s["timing"].get(key) for s in segment_records
                          if isinstance(s.get("timing"), dict)
                          and isinstance(s["timing"].get(key), (int, float))]
                if values:
                    extra[f"{key}_mean"] = round(sum(values) / len(values))
            if self.audio_out_dir and tts_chunks:
                tts_sr = tts_chunks[0]["sample_rate"]
                parts = []
                for chunk in tts_chunks:
                    if chunk["format"] == "pcm_s16le_mono" and chunk["sample_rate"] == tts_sr:
                        parts.append(chunk["audio"])
                    else:  # 兼容旧服务返回的每段 MP3/WAV
                        decoded = _decode_audio_bytes(chunk["audio"], tts_sr)
                        if decoded:
                            parts.append(decoded)
                gap = b"\x00" * (tts_sr * 2 * 30 // 1000)   # 30ms 静音消段边界 click
                pcm = gap.join(parts)
                if pcm:
                    extra["tts_audio"] = save_pcm_wav(pcm,
                        os.path.join(self.audio_out_dir, f"{self.name}__{item.get('id', 'x')}.wav"), sr=tts_sr)
            return TranscribeResult(text, el, _sanitize_nan(extra))
        except Exception as e:
            _L = locals()   # 长音频整段跑：断连时保留已收译文 + 已喂音频秒数(断点位置)
            _parts = _L.get("done_texts") or _L.get("finals") or _L.get("texts") or []
            _em = _L.get("emissions") or []
            _txt = " ".join(_parts).strip() or "".join(t for _, t in _em).strip()
            _fed = (_L.get("st") or {}).get("sent_s", 0.0)
            emit_longform_progress(
                item, phase="failed", audio_s=_fed, audio_total_s=_L.get("src_dur", 0.0),
                segments=len(_parts), latest_text=_parts[-1] if _parts else "",
            )
            _extra = {"audio_fed_s": round(_fed, 1), "n_seg": len(_parts), "partial": bool(_txt)}
            if _L.get("ws_control"):
                _extra["ws_control"] = _L["ws_control"]
            return TranscribeResult(_txt, _L.get("el", 0.0) or 0.0,
                                    _extra,
                                    ok=False, error=str(e))


class DoubaoSimultAdapter:
    """豆包同声传译 2.0 (Seed LiveInterpret 2.0, 火山引擎 AST) — protobuf over WebSocket。

    wss://openspeech.bytedance.com/api/v4/ast/v2/translate，头 X-Api-App-Key/Access-Key/Resource-Id。
    与其它三家(JSON)不同，请求/响应是 protobuf(TranslateRequest/Response，proto 类在 eval/_doubao_pb)。
    流程：StartSession(配置 s2s) → TaskRequest 发 wav 分片(1x) → FinishSession →
         收 TranslateResponse(text=译文增量, data=ogg_opus 译后语音), 到 SessionFinished。
    凭据 DOUBAO_APPID/ACCESS_TOKEN。译后语音 ogg_opus → 解码存 wav。
    """
    name = "doubao-simult"
    _TGT = {"英文": "en", "English": "en", "中文": "zh", "简体中文": "zh",
            "Chinese": "zh", "en": "en", "zh": "zh"}

    def __init__(self, base_url="wss://openspeech.bytedance.com/api/v4/ast/v2/translate",
                 appid=None, token=None, resource_id="volc.service_type.10053",
                 timeout=120.0, pace=True, audio_out_dir=None):
        self.url = base_url
        self.appid = appid or os.environ.get("DOUBAO_APPID", "")
        self.token = token or os.environ.get("DOUBAO_ACCESS_TOKEN", "")
        self.resource_id = resource_id
        self.timeout = timeout
        self.pace = pace
        self.audio_out_dir = audio_out_dir

    def _pb(self):
        import sys as _sys
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "_doubao_pb")
        for p in (base, os.path.join(base, "python_protogen")):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        from common.events_pb2 import Type
        from products.understanding.ast.ast_service_pb2 import TranslateRequest, TranslateResponse
        return TranslateRequest, TranslateResponse, Type

    def generate(self, item) -> "TranscribeResult":
        if not item.get("audio_path"):
            return TranscribeResult("", 0.0, {}, ok=False, error="豆包同传需要 audio_path")
        if not (self.appid and self.token):
            return TranscribeResult("", 0.0, {}, ok=False, error="缺 DOUBAO_APPID/ACCESS_TOKEN")
        try:
            import threading
            import uuid

            from websocket import create_connection

            from metrics import average_lagging
            TranslateRequest, TranslateResponse, Type = self._pb()
            tgt = self._TGT.get(item.get("target_lang", "英文"), "en")
            src = (item.get("lang", "zh-en").split("-")[0]) or ("zh" if tgt == "en" else "en")
            pcm = load_pcm_bytes(item["audio_path"])
            src_dur = len(pcm) / 32000.0           # 16k/mono/16bit 裸 PCM → 秒
            sid = uuid.uuid4().hex

            def build(event, chunk=b""):
                r = TranslateRequest()
                r.request_meta.SessionID = sid
                r.event = event
                r.user.uid = "ast_eval"
                r.user.did = "ast_eval"
                r.source_audio.format = "pcm"
                r.source_audio.rate = 16000
                r.source_audio.bits = 16
                r.source_audio.channel = 1
                if chunk:
                    r.source_audio.binary_data = chunk
                r.target_audio.format = "ogg_opus"
                r.target_audio.rate = 24000
                r.request.mode = "s2s"
                r.request.source_language = src
                r.request.target_language = tgt
                return r.SerializeToString()

            ws = ws_connect(self.url, timeout=self.timeout,
                                   header=[f"X-Api-App-Key: {self.appid}",
                                           f"X-Api-Access-Key: {self.token}",
                                           f"X-Api-Resource-Id: {self.resource_id}",
                                           f"X-Api-Connect-Id: {uuid.uuid4().hex}"])
            try:
                t0 = time.perf_counter()
                ws.send_binary(build(Type.StartSession))
                resp = TranslateResponse()
                resp.ParseFromString(ws.recv())
                if resp.event != Type.SessionStarted:
                    raise RuntimeError(f"StartSession 失败: {resp.response_meta.Message}")
                st = {"sent_s": 0.0}

                def _send():
                    base = time.perf_counter()
                    for k, i in enumerate(range(0, len(pcm), 3200)):
                        ws.send_binary(build(Type.TaskRequest, pcm[i:i + 3200]))
                        st["sent_s"] = min(src_dur, (i + 3200) / 32000.0)
                        if self.pace:
                            dt = base + (k + 1) * 0.1 - time.perf_counter()
                            if dt > 0:
                                time.sleep(dt)
                    ws.send_binary(build(Type.FinishSession))

                sender = threading.Thread(target=_send, daemon=True)
                sender.start()
                emissions, ttfb, texts, text_events, opus = [], None, [], [], bytearray()
                trace = SimultTraceRecorder(target_lang=tgt, granularity="token")
                while True:
                    r = TranslateResponse()
                    r.ParseFromString(ws.recv())
                    e = r.event
                    if e in (Type.SessionFailed, Type.SessionCanceled):
                        raise RuntimeError(f"会话失败: {r.response_meta.Message}")
                    if e == Type.SessionFinished:
                        break
                    # SourceSubtitle*(源文ASR/中文) 忽略；只取译文
                    if e == Type.TranslationSubtitleResponse and r.text:   # 译文逐词增量 → AL
                        if ttfb is None:
                            ttfb = time.perf_counter() - t0
                        emissions.append((st["sent_s"], r.text))
                        trace.append_partial(r.text, st["sent_s"], time.perf_counter() - t0)
                    elif e == Type.TranslationSubtitleEnd and r.text:       # 译文定稿
                        texts.append(r.text.strip())
                        text_events.append((st["sent_s"], r.text.strip()))
                        trace.commit(r.text, st["sent_s"], time.perf_counter() - t0)
                    elif e == Type.TTSResponse and r.data:                  # 译后语音 ogg_opus
                        opus += r.data
                el = time.perf_counter() - t0
                sender.join(timeout=1.0)
            finally:
                ws.close()
            text = "".join(texts).strip()
            saved_trace = trace.finish(text, src_dur, el)
            al, laal = average_lagging(emissions, src_dur, item.get("ref_text", ""), tgt)
            extra = {"ttfb_s": round(ttfb, 3) if ttfb else None, "al_s": al, "laal_s": laal,
                     "src_dur_s": round(src_dur, 2), "n_seg": len(emissions),
                     "translation_segments": _timed_text_segments(text_events),
                     "simult_trace": saved_trace,
                     "stable_ttfb_s": saved_trace["summary"]["ttfb_stable_s"],
                     "churn_rate": saved_trace["summary"]["churn_rate"]}
            if self.audio_out_dir and opus:   # ogg_opus → 解码存 wav
                pcm = _decode_audio_bytes(bytes(opus), 24000)
                if pcm:
                    extra["tts_audio"] = save_pcm_wav(pcm,
                        os.path.join(self.audio_out_dir, f"{self.name}__{item.get('id', 'x')}.wav"), sr=24000)
            return TranscribeResult(text, el, _sanitize_nan(extra))
        except Exception as e:
            _L = locals()   # 长音频整段跑：断连时保留已收译文 + 已喂音频秒数(断点位置)
            _parts = _L.get("done_texts") or _L.get("finals") or _L.get("texts") or []
            _em = _L.get("emissions") or []
            _txt = " ".join(_parts).strip() or "".join(t for _, t in _em).strip()
            _fed = (_L.get("st") or {}).get("sent_s", 0.0)
            return TranscribeResult(_txt, _L.get("el", 0.0) or 0.0,
                                    {"audio_fed_s": round(_fed, 1), "n_seg": len(_parts), "partial": bool(_txt)},
                                    ok=False, error=str(e))


class WhisperLocalAdapter:
    """本地 faster-whisper 纯 ASR —— 译后语音保真度(tts_err)回转专用度量工具。
    为什么用它:部分多语模型有翻译能力,会把英文 TTS"翻"成中文 → tts_err 失真。
    Whisper 纯转写、无翻译倾向、language 硬约束、可复现、不依赖抖动内网。
    模型档位 env WHISPER_MODEL(默认 small;可 medium/large-v3),CPU int8;进程内缓存模型避免每条重载。"""
    name = "whisper-local"
    _MODELS = {}   # size -> WhisperModel
    _LANGS = {"en", "zh", "ja", "ko", "de", "fr", "es", "ru"}

    def __init__(self, model_size=None, device="cpu", compute_type="int8"):
        self.size = model_size or os.environ.get("WHISPER_MODEL", "small")
        self.device = device
        self.compute_type = compute_type

    def _model(self):
        if self.size not in self._MODELS:
            from faster_whisper import WhisperModel
            self._MODELS[self.size] = WhisperModel(self.size, device=self.device,
                                                   compute_type=self.compute_type)
        return self._MODELS[self.size]

    def transcribe(self, audio_path, language="auto", hotwords="", **kw) -> "TranscribeResult":
        try:
            lang = "zh" if language == "yue" else (language if language in self._LANGS else None)
            segs, _ = self._model().transcribe(audio_path, language=lang, beam_size=1)
            return TranscribeResult("".join(s.text for s in segs).strip(), 0.0, {})
        except Exception as e:
            return TranscribeResult("", 0.0, {}, ok=False, error=str(e))


ADAPTERS = {
    "whisper-local": WhisperLocalAdapter,   # 本地纯 ASR(tts_err 回转用,替代会翻译的多语模型)
    "sensevoice": SenseVoiceAdapter,   # 阿里 SenseVoice 基线
    "ext-adv": ext_adv,            # asr_lite 协议的第二个端点
    "gemma-audio": GemmaAudioAdapter,  # gemma LLM 式 ASR
    "gemma-text": GemmaTextAdapter,    # gemma 文本翻译/总结
    "cascade-gemma": CascadeLightGemmaAdapter,  # 级联语音翻译 ASR→MT
    "doubao-simult": DoubaoSimultAdapter,      # 竞品同传：豆包 Seed LiveInterpret 2.0(protobuf WS)
    "qwen-simult": QwenLiveTranslateAdapter,  # 竞品同传：Qwen3.5-LiveTranslate(Realtime 协议, Bearer)
    "qwen-simult-offline": lambda **kw: QwenLiveTranslateAdapter(pace=False, **kw),  # 不限速版=质量保留率分母
    "xf-simult": XfSimultAdapter,             # 竞品同传：科大讯飞同声传译(WS, hmac 签名)
    "xf-spark-slm-iat": XfSparkSlmIatAdapter,  # 竞品 ASR：讯飞方言识别大模型(WS, hmac 签名)
    # 竞品 ASR：阿里 DashScope Qwen3-ASR-Flash（OpenAI 兼容层，Bearer=DASHSCOPE_API_KEY）。
    # ⚠️ Flash 是 Qwen3-ASR 家族的独立 API 商用版(技术报告：Qwen3-ASR-Flash-1208 serves as an API)，
    #    分数高于开源 Qwen3-ASR-1.7B/0.6B，非同一模型；阿里不公开其参数量。音频 <10MB、≤5min。
    "qwen-asr": QwenASRAdapter,
    # ★ 被测主体 = 四档模型
    "light": light_fun512,             # asr_lite 协议
    "light-mlt": light_mlt_nano,       # asr_mlt_nano 协议
    "std": lambda **kw: PlatformASRAdapter(endpoint="std", **kw),  
    "adv": PlatformASRAdapter,                                           
    "adv-domain": PlatformAdvDomainAdapter,                              # 独立领域/热词端点，不借用 Standard/Advanced 身份
    "sse": PlatformSSEAdapter,                                     # SSE 接口
    "sse-local": lambda **kw: PlatformSSEAdapter(
        base_url="http://127.0.0.1:8000", api_key=kw.pop("api_key", ""), **kw
    ),  # 本地8000热词版本（显式不携带平台 Token）
    "compat": lambda **kw: PlatformASRAdapter(endpoint="compat", **kw),    # 旧版(保留)
    # ★ 挂在档位上的功能/场景接口（非独立档位）
    "plat-simult": PlatformWSSimultAdapter,                                   # 同声传译(WS 真流式+TTS，端点走 ASR_PLATFORM_WS_URL)
    "plat-simult-post": lambda **kw: PlatformSSEAdapter(mode="simult", **kw), # 旧 POST 单发版(无 AL,留作对照)
    "plat-simult-ws": PlatformWSVoiceInputAdapter,                            # 平台真流式同传(voice-input WS,有 AL)
    "plat-realtime": PlatformRealtimeAdapter,                                 # 流式 ASR(WS /v1/realtime,Base64 JSON 帧)
    "qwen3-asr-ws": Qwen3ASRWSAdapter,                                  # 竞品真流式 ASR(WS /ws,增量;共用服务→并发≤2)
    "plat-minutes": PlatformMinutesAdapter,                                   # 会议纪要(离线,长音频)
    "plat-formula": PlatformFormulaAdapter,                                   # TTS 公式转写(文本)
    "plat-diar": PlatformDiarAdapter,                                        # 说话人分离(DER)
        # OpenAIAudioAdapter 类保留：它是 openai-audio 自定义接口/厂商预设模板的实现，勿删。
}


def model_supports_hotwords(model_id: str) -> bool:
    """看板起跑前探测：不支持热词的模型带热词提交应显式拒绝，而不是静默降级产出假 __hw 结果。"""
    try:
        a = ADAPTERS[model_id]()
        return bool(getattr(a, "supports_hotwords", False))
    except Exception:
        return False


class GenericChatAdapter:
    """通用文本 LLM（OpenAI /chat/completions）：文本翻译 / 总结。竞品文本模型用。"""
    def __init__(self, base_url, model="gpt-4o-mini", api_key="sk-eval", prefix="/v1", timeout=120.0,
                 prompt_tpl=""):
        self.url = base_url.rstrip("/") + prefix.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        # 翻译提示词模板({src}/{tgt} 占位)：MT 专用模型按官方模板喂，偏离训练格式会低估成绩(如 Hy-MT2)
        self.prompt_tpl = prompt_tpl

    # 中文语言名 → 英文名：官方模板多为纯英文句式，插入中文语言名(如"英文")属于偏离训练格式——
    # 实测 Hy-MT2 7B 在 wmt25term_2022_0 上因此确定性早停(7 token EOS)，换 English 即痊愈
    _TGT_EN = {"英文": "English", "中文": "Chinese", "简体中文": "Simplified Chinese", "繁体中文": "Traditional Chinese",
               "日文": "Japanese", "韩文": "Korean", "德文": "German", "法文": "French",
               "西班牙文": "Spanish", "俄文": "Russian"}

    def _prompt(self, item):
        task = item.get("task")
        if task == "translate":
            tgt = item.get("target_lang", "英文")
            if self.prompt_tpl:
                return self.prompt_tpl.replace("{tgt}", self._TGT_EN.get(tgt, tgt)).replace("{src}", item["source_text"])
            return f"把下面的文本翻译成{tgt}，只输出译文，不要解释：\n{item['source_text']}"
        if task == "summarize":
            return f"为下面的内容写一段简洁准确的中文摘要，只输出摘要本身：\n{item['source_text']}"
        return item.get("source_text", "")

    def generate(self, item) -> "TranscribeResult":
        try:
            runtime = _runtime_request_parts(self)
            # max_tokens 按源文长度给预算：定值 2048 会把书章级文档译文拦腰砍断(wmt25 实测覆盖率 35%)。
            # 超过服务 max-model-len 时会 400 → 退回 2048 重试一次。
            src = item.get("source_text", "") or ""
            mt = min(8192, max(2048, 2 * len(src)))  # 中→英字符膨胀 ~3.5×，1×源长的 token 预算不够(30B 实测撞 length)
            body = {"model": self.model, "temperature": 0, "max_tokens": mt,
                    "messages": [{"role": "user", "content": self._prompt(item)}]}
            body.update(runtime["json"])
            headers = {**self.headers, **{k: str(v) for k, v in runtime["header"].items()}}
            t0 = time.perf_counter()
            r = http_post(self.url, headers=headers, params=runtime["query"],
                          json=body, timeout=self.timeout)
            if r.status_code == 400 and mt > 2048:
                body["max_tokens"] = 2048
                r = http_post(self.url, headers=headers, params=runtime["query"],
                              json=body, timeout=self.timeout)
            el = time.perf_counter() - t0
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            return TranscribeResult(text=txt, elapsed_s=el, extra={})
        except Exception as e:
            return TranscribeResult(text="", elapsed_s=0.0, extra={}, ok=False, error=str(e))


# ── 自定义接口（看板「＋添加接口」注册，配置存 dashboard/interfaces.json，不入仓）──

# 厂商预设：看板「厂商」下拉选一项 → 回填 base_url/template/path/model；密钥默认走 key_env
# 指向的环境变量（容器化友好，不落进 interfaces.json）。只收公网真·ASR 厂商，已实测协议归属。
#   ⚠️ MiniMax 无 ASR 接口（/audio/transcriptions 404，只有 TTS/声音克隆），故不在此列。
PROVIDERS = {
    "openai": {  # OpenAI 原生：/v1/audio/transcriptions，单 key
        "label": "OpenAI", "template": "openai-audio",
        "base_url": "https://api.openai.com", "path": "/v1",
        "models": ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
        "key_env": "OPENAI_API_KEY", "scenarios": ["ASR"],
        "language_spec": {
            "supported": True, "values": [],
            "note": "auto 表示不发送 language；其他值按 OpenAI Audio API 原样发送，建议使用 ISO-639-1 代码。",
        },
    },
    "dashscope": {  # 阿里 DashScope OpenAI 兼容层：qwen3-asr-flash，音频 <10MB（评测样本均短，够用）
        "label": "阿里 DashScope · Qwen3-ASR", "template": "openai-audio",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode", "path": "/v1",
        "models": ["qwen3-asr-flash"],
        "key_env": "DASHSCOPE_API_KEY", "scenarios": ["ASR"],
        "language_spec": {
            "supported": True, "values": [],
            "note": "auto 表示不发送 language；其他值由兼容接口原样接收。Qwen3-ASR-Flash 支持多语种，但该接口未返回固定枚举。",
        },
    },
    "qwen3-asr-1.7b": {
        "label": "远程自建 · Qwen3-ASR-1.7B", "template": "asr_lite",
        "base_url": os.environ.get("QWEN3_ASR_URL", ""), "path": "",
        "models": [], "scenarios": ["ASR"],
        "itn": False, "return_timestamps": False,
        "language_spec": {
            "supported": True,
            "values": [
                "auto", "zh", "en", "yue", "ja", "ko", "ar", "de", "fr", "es", "pt",
                "id", "it", "ru", "th", "vi", "tr", "hi", "ms", "nl", "sv", "da", "fi",
                "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro", "anhui", "dongbei",
                "fujian", "gansu", "guizhou", "hebei", "henan", "hubei", "hunan",
                "jiangxi", "ningxia", "shandong", "shaanxi", "shanxi", "sichuan",
                "tianjin", "yunnan", "zhejiang", "yue_hk", "yue_gd", "wu", "minnan",
            ],
            "note": "/asr_lite：使用代码值；包含 30 种语言与 22 种方言。",
        },
    },
    "mimo": {
        "label": "小米 MiMo · MiMo-V2.5-ASR", "template": "openai-chat-audio",
        "base_url": "https://api.xiaomimimo.com", "path": "/v1",
        "models": ["mimo-v2.5-asr"],
        "key_env": "MIMO_API_KEY", "scenarios": ["ASR"],
        "language_spec": {
            "supported": True, "values": ["auto", "zh", "en"],
            "note": "作为 asr_options.language 发送；当前连接器开放 auto / zh / en。",
        },
    },
}


def _resolve_key(c):
    """密钥解析：显式 api_key 优先（本地手测/一次性），否则 key_env 指向的环境变量（容器化部署）。
    两者皆空 → 空串（本地 keyless 端点用）。这样 interfaces.json 可只存 key_env、不落明文密钥。"""
    if c.get("api_key"):
        return c["api_key"]
    env = c.get("key_env")
    return os.environ.get(env, "") if env else ""


_TEMPLATES = {
    # template → 按配置实例化（path 字段含义随模板而异，见各行注释）
    "openai-audio": lambda c: OpenAIAudioAdapter(base_url=c["base_url"], model=c.get("model") or "whisper-1",
                                              api_key=_resolve_key(c) or "sk-eval",
                                              prefix=c.get("path") or "/v1"),          # path=前缀
    "plat-multipart": lambda c: PlatformASRAdapter(base_url=c["base_url"],
                                             endpoint=c.get("path") or "adv",
                                             api_key=_resolve_key(c)),                 # path=adv/std/compat
    "asr_lite": lambda c: LightASRAdapter(
        c["base_url"], name=c["id"], default_itn=c.get("itn", False),
        return_timestamps=c.get("return_timestamps", False),
        path=c.get("path") or "/asr_lite"),
    "sensevoice": lambda c: SenseVoiceAdapter(c["base_url"]),
    "plat-sse": lambda c: PlatformSSEAdapter(base_url=c["base_url"], api_key=_resolve_key(c)),
    "plat-sse-dual": lambda c: PlatformSSEAdapter(base_url=c["base_url"], api_key=_resolve_key(c)),
    "openai-chat": lambda c: GenericChatAdapter(c["base_url"], c.get("model") or "gpt-4o-mini",
                                                _resolve_key(c) or "sk-eval", c.get("path") or "/v1",
                                                prompt_tpl=c.get("prompt_tpl") or ""),
    "openai-chat-audio": lambda c: OpenAIChatAudioAdapter(
        c["base_url"], c.get("model") or "mimo-v2.5-asr", _resolve_key(c) or "sk-eval",
        c.get("path") or "/v1"),
    "custom-http-asr": lambda c: CustomHTTPASRAdapter(c),
}

# 每个模板能跑的场景（前端按此约束多选；翻译=语音翻译/文本翻译看 adapter，总结=文本）
TEMPLATE_SCENARIOS = {
    "openai-audio": ["ASR", "翻译"], "plat-multipart": ["ASR", "翻译"], "plat-sse": ["ASR", "翻译"],
    "plat-sse-dual": ["ASR", "翻译"],
    "asr_lite": ["ASR"], "sensevoice": ["ASR"], "openai-chat": ["翻译", "总结"],
    "openai-chat-audio": ["ASR"],
    "custom-http-asr": ["ASR", "翻译"],
}


def make_custom_adapter(cfg, request_params=None):
    ad = _TEMPLATES[cfg["template"]](cfg)
    ad.name = cfg["id"]
    ad.request_schema = request_contract_for(cfg.get("id", ""), cfg)["request_schema"]
    ad.request_params = dict(request_params or {})
    if ("hotwords" in (cfg.get("caps") or [])
            or any(field.get("managed_by") == "hotwords"
                   for field in ad.request_schema if isinstance(field, dict))):
        ad.supports_hotwords = True   # 能力声明开热词：如自托管 openai-audio 端点实际收 hot_words
    return ad


def _load_custom_interfaces():
    import os as _os
    p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "dashboard", "interfaces.json")
    if not _os.path.exists(p):
        return
    try:
        cfgs = json.load(open(p, encoding="utf-8"))
    except Exception:
        return
    for c in cfgs:
        if c.get("id") and c.get("template") in _TEMPLATES and c["id"] not in ADAPTERS:
            ADAPTERS[c["id"]] = (
                lambda cfg: (lambda **kw: make_custom_adapter(cfg, **kw))
            )(c)  # noqa: E731


_load_custom_interfaces()
