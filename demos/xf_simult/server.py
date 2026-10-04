"""Local microphone demo for iFLYTEK simultaneous interpretation (zh -> en)."""

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
from contextlib import suppress
from email.utils import formatdate
from pathlib import Path
from urllib.parse import urlencode, urlparse

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from websocket import WebSocketTimeoutException, create_connection


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
XF_HOST = "ws-api.xf-yun.com"
XF_PATH = "/v1/private/simult_interpretation"
FRAME_BYTES = 1280  # 40 ms of 16 kHz, mono, signed 16-bit PCM


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load the three server-side credentials without adding a dotenv dependency."""
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except FileNotFoundError:
        pass


load_dotenv()


def credentials() -> tuple[str, str, str]:
    return (
        os.environ.get("XF_APPID", ""),
        os.environ.get("XF_API_KEY", ""),
        os.environ.get("XF_API_SECRET", ""),
    )


def signed_url(api_key: str, api_secret: str, date: str | None = None) -> str:
    """Build the documented RFC1123 + HMAC-SHA256 WebSocket URL."""
    date = date or formatdate(usegmt=True)
    origin = f"host: {XF_HOST}\ndate: {date}\nGET {XF_PATH} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(api_secret.encode(), origin.encode(), hashlib.sha256).digest()
    ).decode()
    authorization = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    query = urlencode({
        "authorization": base64.b64encode(authorization.encode()).decode(),
        "date": date,
        "host": XF_HOST,
        "serviceId": "simult_interpretation",
    })
    return f"wss://{XF_HOST}{XF_PATH}?{query}"


def audio_frame(appid: str, audio: bytes, seq: int, status: int) -> dict:
    """Create one documented audio request frame; business params live on frame 0."""
    frame = {
        "header": {"app_id": appid, "status": status},
        "payload": {"data": {
            "audio": base64.b64encode(audio).decode(),
            "encoding": "raw",
            "sample_rate": 16000,
            "seq": seq,
            "status": status,
        }},
    }
    if status == 0:
        frame["parameter"] = {
            "ist": {
                "accent": "mandarin",
                "domain": "ist_ed_open",
                "language": "zh_cn",
                "vto": 15000,
                "eos": 150000,
            },
            "streamtrans": {"from": "cn", "to": "en"},
            "tts": {
                "vcn": "x2_john",
                "tts_results": {
                    "encoding": "raw",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "frame_size": 0,
                },
            },
        }
    return frame


def response_events(raw_message: str) -> list[dict]:
    """Reduce an iFLYTEK response to the only browser events this demo needs."""
    response = json.loads(raw_message)
    header = response.get("header") or {}
    code = int(header.get("code") or 0)
    if code:
        return [{
            "type": "error",
            "code": code,
            "message": str(header.get("message") or "讯飞服务返回错误"),
        }]

    events = []
    payload = response.get("payload") or {}
    result = payload.get("streamtrans_results") or {}
    if result.get("text"):
        decoded = json.loads(base64.b64decode(result["text"]).decode("utf-8"))
        events.append({
            "type": "translation",
            "source": str(decoded.get("src") or "").strip(),
            "translation": str(decoded.get("dst") or "").strip(),
            "is_final": bool(decoded.get("is_final")),
            "begin_ms": decoded.get("wb"),
            "end_ms": decoded.get("we"),
        })
    tts = payload.get("tts_results") or {}
    if tts.get("audio") and tts.get("encoding") == "raw":
        events.append({
            "type": "tts_audio",
            "audio": base64.b64decode(tts["audio"]),
        })
    if header.get("status") == 2:
        events.append({"type": "done"})
    return events


app = FastAPI(title="讯飞同声传译 Demo", docs_url=None, redoc_url=None)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "configured": all(credentials())})


def _local_origin(origin: str | None) -> bool:
    if not origin:
        return True
    return (urlparse(origin).hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}


async def _send_remote(remote, frame: dict) -> None:
    payload = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
    await asyncio.to_thread(remote.send, payload)


async def _relay_responses(remote, browser: WebSocket, done: asyncio.Event) -> None:
    try:
        while not done.is_set():
            try:
                raw_message = await asyncio.to_thread(remote.recv)
            except WebSocketTimeoutException:
                continue
            if not raw_message:
                break
            for event in response_events(raw_message):
                if event["type"] == "tts_audio":
                    await browser.send_bytes(event["audio"])
                else:
                    await browser.send_json(event)
                if event["type"] in {"done", "error"}:
                    done.set()
                    return
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        if not done.is_set():
            with suppress(Exception):
                await browser.send_json({"type": "error", "message": "讯飞连接中断，请重试"})
    finally:
        done.set()


@app.websocket("/ws")
async def microphone_proxy(browser: WebSocket) -> None:
    if not _local_origin(browser.headers.get("origin")):
        await browser.close(code=1008)
        return
    await browser.accept()

    appid, api_key, api_secret = credentials()
    if not all((appid, api_key, api_secret)):
        await browser.send_json({
            "type": "error",
            "message": "请先在项目 .env 配置 XF_APPID、XF_API_KEY、XF_API_SECRET",
        })
        await browser.close(code=1011)
        return

    try:
        remote = await asyncio.to_thread(
            create_connection, signed_url(api_key, api_secret), timeout=15,
        )
        remote.settimeout(1.0)
    except Exception:
        await browser.send_json({"type": "error", "message": "无法连接讯飞服务，请检查网络和凭据"})
        await browser.close(code=1011)
        return

    done = asyncio.Event()
    relay = asyncio.create_task(_relay_responses(remote, browser, done))
    pending = bytearray()
    seq = 0
    finalized = False

    async def finalize() -> None:
        nonlocal seq, finalized
        if finalized or done.is_set():
            return
        finalized = True
        if seq == 0:
            await _send_remote(remote, audio_frame(appid, bytes(pending), seq, 0))
            pending.clear()
            seq += 1
        await _send_remote(remote, audio_frame(appid, bytes(pending), seq, 2))
        pending.clear()
        seq += 1

    try:
        await browser.send_json({"type": "ready"})
        while not done.is_set():
            try:
                message = await asyncio.wait_for(browser.receive(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                pending.extend(message["bytes"])
                while len(pending) >= FRAME_BYTES:
                    chunk = bytes(pending[:FRAME_BYTES])
                    del pending[:FRAME_BYTES]
                    status = 0 if seq == 0 else 1
                    await _send_remote(remote, audio_frame(appid, chunk, seq, status))
                    seq += 1
                continue
            if message.get("text"):
                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    command = {}
                if command.get("type") == "stop":
                    await finalize()
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(done.wait(), timeout=20)
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        with suppress(Exception):
            await browser.send_json({"type": "error", "message": "会话异常，请重新开始"})
    finally:
        if not finalized and not done.is_set():
            with suppress(Exception):
                await finalize()
        done.set()
        with suppress(Exception):
            await asyncio.to_thread(remote.close)
        relay.cancel()
        with suppress(asyncio.CancelledError):
            await relay


def main() -> None:
    parser = argparse.ArgumentParser(description="本地讯飞中文→英文同声传译 Demo")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
