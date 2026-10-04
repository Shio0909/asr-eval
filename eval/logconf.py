"""中央日志配置 — 学 lm-eval(环境变量控级别) + OpenCompass(文件落盘/每任务日志)。

设计要点：
- 控制台 handler 走 **stderr**：不污染 infer 打到 stdout、被 dashboard 正则解析的进度行
  (`n/N…` / `待跑 N`)。dashboard 跑子进程时 stderr 已并入 stdout 管道，故日志仍被捕获。
- 文件 handler 走 `logs/<file>`，RotatingFileHandler 轮转(5MB×3)防无限膨胀。
- 级别由环境变量 `ASR_LOG_LEVEL` 控(默认 INFO)；`ASR_LOG_FILE=0` 关文件只留控制台
  (dashboard 给子进程设 0：其输出已被每任务日志捕获，避免多进程争抢同一文件轮转)。
- **密钥防泄漏**：所有经本配置的日志统一过 redact()，抹掉 Bearer/api_key/sk- 串，
  杜绝端点鉴权信息进日志文件或前端。
"""
import base64
import json
import logging
import os
import re
import shlex
import threading
from urllib.parse import urlencode, urlsplit, urlunsplit
from logging.handlers import RotatingFileHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")

_LEVEL = getattr(logging, os.environ.get("ASR_LOG_LEVEL", "INFO").upper(), logging.INFO)

# ── 密钥脱敏（三遍扫，顺序重要）──
# ① Bearer/Basic <token>：先扫，否则下面 keyword 遍会把 6 字符的 "Bearer" 当 secret 吃掉、漏真 token。
_BEARER_RE = re.compile(r"(?i)((?:bearer|basic)\s+)([A-Za-z0-9+/._\-]{4,}={0,2})")
# ② key: value 形式。分隔符 [\s:=\"']+ 多字符，兼容 header(`: `)/kwarg(`=`)/JSON(`": "`)；
#    不加前导 \b——环境变量名常下划线连写(XF_API_SECRET)，\b 在 `_` 处不成立会漏；
#    词中误匹配由「后随必须有分隔符」挡住(如 "tokens" 后无分隔符不算)。
_SECRET_RE = re.compile(
    r"(?i)((?:x-)?api[_-]?key|api[_-]?secret|access[_-]?token|authorization|token|secret|"
    r"password|passwd|pwd|app[_-]?id|app[_-]?key)([\s:=\"']+)([A-Za-z0-9+/._\-]{6,})")
# ③ URL 内嵌凭据 scheme://user:pass@host → 抹 pass；④ sk- 前缀长串兜底。
_URLCRED_RE = re.compile(r"(://[^/@\s:]+:)([^@\s/]+)(@)")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9]{12,}")


def redact(text):
    """把可能的密钥替换成 ***，防止进日志/前端。容错：任意输入都转字符串处理。"""
    if not text:
        return text
    s = _BEARER_RE.sub(lambda m: m.group(1) + "***", str(text))
    s = _SECRET_RE.sub(lambda m: m.group(1) + m.group(2) + "***", s)
    s = _URLCRED_RE.sub(lambda m: m.group(1) + "***" + m.group(3), s)
    return _SK_RE.sub("sk-***", s)


class _RedactFormatter(logging.Formatter):
    """在 Formatter 层脱敏：一处覆盖 message + args + exc_text(traceback) + stack_info。
    比 Filter 只动 message 更彻底——traceback 里 adapter 的 Authorization 头也会被抹。"""
    def format(self, record):
        return redact(super().format(record))


_FMT = _RedactFormatter("%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
_FH_CACHE = {}          # file -> 单例 RotatingFileHandler（多 logger 共享，避免轮转竞争）
_CONFIGURED = set()
_JOB_LOG_FILE = None


def _file_handler(file):
    if file in _FH_CACHE:
        return _FH_CACHE[file]
    path = os.path.join(LOG_DIR, file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = RotatingFileHandler(path,
                             maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(_FMT)
    _FH_CACHE[file] = fh
    return fh


def get_logger(name, file="eval.log"):
    """取一个配置好的 logger：stderr 控制台 + (可选)轮转文件。重复调用幂等。"""
    log = logging.getLogger(name)
    if name in _CONFIGURED:
        return log
    log.setLevel(_LEVEL)
    log.propagate = False   # 不冒泡到 root，避免重复打印
    sh = logging.StreamHandler()   # 默认 stderr
    sh.setFormatter(_FMT)
    log.addHandler(sh)
    if os.environ.get("ASR_LOG_FILE", "1") != "0":
        try:
            log.addHandler(_file_handler(file))
            if _JOB_LOG_FILE:
                log.addHandler(_file_handler(_JOB_LOG_FILE))
        except Exception:
            pass   # 文件不可写(只读盘/权限)时静默降级为只控制台
    _CONFIGURED.add(name)
    return log


def attach_job_log(job_id):
    """Mirror all CLI runner logs into logs/jobs/<job_id>.log."""
    global _JOB_LOG_FILE
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(job_id or "")):
        raise ValueError("bad job id")
    _JOB_LOG_FILE = f"jobs/{job_id}.log"
    handler = _file_handler(_JOB_LOG_FILE)
    for name in _CONFIGURED:
        logger = logging.getLogger(name)
        if handler not in logger.handlers:
            logger.addHandler(handler)
    return _JOB_LOG_FILE


# ────────────────────────────────────────────────────────────────────────
# 请求级可观测（L1 调用边界 / L2 HTTP / L3 WS 共用）
#   - 专用 sink logs/requests.log，key=value 单行文本(可 grep/awk)。
#   - 线程局部上下文 set_ctx() 把 job/model/sample id 注入同线程内后续每条请求日志，
#     让 work() 起的 HTTP/WS 调用都自动带上关联键(infer 用线程池，故用 threading.local)。
# ────────────────────────────────────────────────────────────────────────
_CTX = threading.local()
_REQ_LOG = None


def _req_logger():
    global _REQ_LOG
    if _REQ_LOG is None:
        _REQ_LOG = get_logger("req", file="requests.log")
    return _REQ_LOG


def set_ctx(**fields):
    """设当前线程的请求上下文(job/model/id)；None 值剔除。每条样本开头调，结尾 clear_ctx()。"""
    _CTX.data = {k: v for k, v in fields.items() if v is not None}


def clear_ctx():
    _CTX.data = {}


def _fmt_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s or "0"
    s = str(v)
    if s == "" or any(c in s for c in ' \t"='):
        return '"' + s.replace('"', "'") + '"'
    return s


def kv(**fields):
    """key=value 单行；None 值跳过，含空格/引号的值加引号。"""
    return " ".join(f"{k}={_fmt_val(v)}" for k, v in fields.items() if v is not None)


def log_request(**fields):
    """记一条请求级事件到 requests.log。ok=False → WARNING，否则 INFO。
    自动并入线程上下文(job/model/id)；输出经 _RedactFormatter 统一脱敏。"""
    merged = {**_ctx(), **fields}
    msg = kv(**merged)
    lg = _req_logger()
    (lg.info if fields.get("ok", True) else lg.warning)("%s", msg)


def _ctx():
    return getattr(_CTX, "data", {})


_REQUEST_PREVIEW_LOCK = threading.Lock()
_REQUEST_PREVIEWED_JOBS = set()
REQUEST_PREVIEW_MARKER = "REQUEST_PREVIEW_B64="
_RESPONSE_PREVIEW_LOCK = threading.Lock()
_RESPONSE_PREVIEWED_JOBS = set()
RESPONSE_PREVIEW_MARKER = "RESPONSE_PREVIEW_B64="
_RESPONSE_PREVIEW_BYTES = 32 * 1024


def _safe_request_value(value, key=""):
    low = str(key).lower()
    norm = re.sub(r"[^a-z0-9]", "", low)
    secret_key = (
        norm == "authorization"
        or norm.endswith(("apikey", "apisecret", "accesstoken", "password", "passwd",
                          "appid", "appkey", "accesskey", "signature"))
        or (norm.endswith("token") and norm not in ("maxtoken", "maxtokens"))
    )
    if secret_key:
        return "<Bearer token hidden>" if norm == "authorization" else "***"
    if low in ("file", "audio", "audio_file", "input_audio", "binary_data"):
        return "<audio omitted>"
    if isinstance(value, dict):
        return {str(k): _safe_request_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_request_value(v, key) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary omitted>"
    text = str(value)
    if text.startswith("data:audio/") or len(text) > 500:
        return "<audio or large payload omitted>"
    return redact(text)


def _safe_url(url, params=None):
    clean = redact(url)
    if not params:
        return clean
    parsed = urlsplit(clean)
    query = parsed.query
    encoded = urlencode(_safe_request_value(params), doseq=True)
    query = "&".join(x for x in (query, encoded) if x)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))


def build_request_preview(method, url, kwargs=None, timeout=None):
    """Build a representative, copyable curl without secrets or audio paths/content."""
    kwargs = kwargs or {}
    headers = _safe_request_value(kwargs.get("headers") or {})
    params = _safe_request_value(kwargs.get("params") or {})
    form = _safe_request_value(kwargs.get("data") or {}) if isinstance(kwargs.get("data"), dict) else {}
    json_body = _safe_request_value(kwargs.get("json")) if kwargs.get("json") is not None else None
    files = {}
    for field, spec in (kwargs.get("files") or {}).items():
        mime = ""
        if isinstance(spec, tuple) and len(spec) >= 3:
            mime = str(spec[2] or "")
        files[str(field)] = {"value": "<audio omitted>", "content_type": mime or None}

    safe_url = _safe_url(url, params)
    lines = ["curl --fail-with-body --show-error", f"--request {str(method).upper()}",
             shlex.quote(safe_url)]
    for key, value in headers.items():
        lines.append("--header " + shlex.quote(f"{key}: {value}"))
    for key, value in form.items():
        lines.append("--form " + shlex.quote(f"{key}={value}"))
    for key, meta in files.items():
        suffix = f";type={meta['content_type']}" if meta.get("content_type") else ""
        lines.append("--form " + shlex.quote(f"{key}=@<audio-file-hidden>{suffix}"))
    if json_body is not None:
        if not any(str(k).lower() == "content-type" for k in headers):
            lines.append("--header 'Content-Type: application/json'")
        lines.append("--data " + shlex.quote(json.dumps(json_body, ensure_ascii=False)))
    elif kwargs.get("data") is not None and not isinstance(kwargs.get("data"), dict):
        lines.append("--data '<binary-body-hidden>'")
    curl = " \\\n  ".join(lines)
    return {
        "method": str(method).upper(),
        "url": safe_url,
        "headers": headers,
        "form": form,
        "files": files,
        "json": json_body,
        "timeout_s": timeout,
        "curl": curl,
        "audio_hidden": True,
    }


def log_request_preview(method, url, kwargs=None, timeout=None):
    """Log exactly one sanitized representative request for each dashboard job."""
    job = os.environ.get("DASH_JOB") or _ctx().get("job")
    if not job:
        return
    with _REQUEST_PREVIEW_LOCK:
        if job in _REQUEST_PREVIEWED_JOBS:
            return
        _REQUEST_PREVIEWED_JOBS.add(job)
    preview = build_request_preview(method, url, kwargs, timeout)
    preview["context"] = {
        k: v for k, v in _ctx().items() if k in ("job", "model", "id")
    }
    payload = base64.urlsafe_b64encode(
        json.dumps(preview, ensure_ascii=False, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    _req_logger().info(
        "%s%s\n代表请求（仅记录本任务第一条调用；鉴权和音频已隐藏）:\n%s",
        REQUEST_PREVIEW_MARKER, payload, preview["curl"],
    )


def _safe_response_value(value, key="", depth=0):
    """Bound and redact a response tree before it is written to a job log."""
    low = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if low == "authorization" or low.endswith((
        "apikey", "apisecret", "accesstoken", "password", "passwd", "appid",
        "appkey", "accesskey", "signature", "secret", "cookie", "credential",
        "privatekey",
    )) or (low.endswith("token") and low not in ("maxtoken", "maxtokens")):
        return "***"
    if depth >= 6:
        return "<nested content omitted>"
    if isinstance(value, dict):
        items = list(value.items())
        out = {str(k): _safe_response_value(v, str(k), depth + 1) for k, v in items[:50]}
        if len(items) > 50:
            out["<omitted>"] = f"{len(items) - 50} more fields"
        return out
    if isinstance(value, (list, tuple)):
        out = [_safe_response_value(v, key, depth + 1) for v in value[:40]]
        if len(value) > 40:
            out.append(f"<{len(value) - 40} more items omitted>")
        return out
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary omitted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = redact(str(value))
    compact = re.sub(r"\s+", "", text)
    binary_key = any(x in low for x in ("audio", "pcm", "waveform", "binary", "base64"))
    looks_b64 = (len(compact) > 512 and len(compact) % 4 == 0
                 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", compact) is not None)
    if text.startswith("data:audio/") or binary_key and len(text) > 128 or looks_b64:
        return f"<audio/base64 omitted: {len(text)} chars>"
    if len(text) > 4000:
        return text[:4000] + f"… <{len(text) - 4000} chars truncated>"
    return text


def sanitize_response_value(value):
    """Public wrapper used by dashboard's normalized-result fallback."""
    return _safe_response_value(value)


def build_response_preview(response, streaming=False):
    """Capture one bounded response example without retaining audio/Base64 payloads."""
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "")
    status = int(getattr(response, "status_code", 0) or 0)
    preview = {
        "source": "http",
        "status": status,
        "reason": redact(str(getattr(response, "reason", "") or "")),
        "content_type": content_type,
        "response_bytes": None,
        "body_type": "stream" if streaming else "text",
        "body": "<streaming response not captured>" if streaming else None,
        "truncated": False,
    }
    for header in ("x-request-id", "x-trace-id", "request-id"):
        if headers.get(header):
            preview[header.replace("-", "_")] = redact(str(headers.get(header)))[:200]
    if streaming:
        return preview
    raw = bytes(getattr(response, "content", b"") or b"")
    preview["response_bytes"] = len(raw)
    if content_type.lower().startswith("audio/") or "application/octet-stream" in content_type.lower():
        preview.update(body_type="binary", body=f"<binary response omitted: {len(raw)} bytes>")
        return preview
    stripped = raw.lstrip()
    wants_json = "json" in content_type.lower() or stripped.startswith((b"{", b"["))
    if wants_json and len(raw) <= 8 * 1024 * 1024:
        try:
            preview["body_type"] = "json"
            preview["body"] = _safe_response_value(json.loads(raw.decode("utf-8", "replace")))
        except (ValueError, TypeError):
            preview["body"] = redact(raw[:_RESPONSE_PREVIEW_BYTES].decode("utf-8", "replace"))
    elif len(raw) > 8 * 1024 * 1024:
        preview.update(body_type="omitted", body=f"<response too large to preview: {len(raw)} bytes>",
                       truncated=True)
    else:
        text = redact(raw[:_RESPONSE_PREVIEW_BYTES].decode("utf-8", "replace"))
        if len(raw) > _RESPONSE_PREVIEW_BYTES:
            text += f"\n… <{len(raw) - _RESPONSE_PREVIEW_BYTES} bytes truncated>"
            preview["truncated"] = True
        preview["body"] = text
    rendered = json.dumps(preview.get("body"), ensure_ascii=False, separators=(",", ":"))
    if len(rendered.encode()) > _RESPONSE_PREVIEW_BYTES:
        preview["body"] = rendered[:_RESPONSE_PREVIEW_BYTES] + "… <preview truncated>"
        preview["body_type"] += "-text"
        preview["truncated"] = True
    return preview


def log_response_preview(response, streaming=False):
    """Log exactly one bounded response example for each dashboard job."""
    job = os.environ.get("DASH_JOB") or _ctx().get("job")
    if not job:
        return
    with _RESPONSE_PREVIEW_LOCK:
        if job in _RESPONSE_PREVIEWED_JOBS:
            return
        _RESPONSE_PREVIEWED_JOBS.add(job)
    preview = build_response_preview(response, streaming=streaming)
    preview["context"] = {k: v for k, v in _ctx().items() if k in ("job", "model", "id")}
    payload = base64.urlsafe_b64encode(
        json.dumps(preview, ensure_ascii=False, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    _req_logger().info(
        "%s%s response status=%s type=%s bytes=%s truncated=%s",
        RESPONSE_PREVIEW_MARKER, payload, preview.get("status"), preview.get("body_type"),
        preview.get("response_bytes"), preview.get("truncated"),
    )
