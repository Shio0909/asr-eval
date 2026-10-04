"""轻量公共工具 — 无第三方依赖（score 纯打分路径不应拖上 soundfile/requests）。"""

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager


def manifest_sha(path: str) -> str:
    """manifest 指纹（12 位）：只对**样本内容**算，路径/键序/空白不影响。

    同指纹 = 同一份样本，结果才可比。绝对路径不入指纹 → 换机器重建 manifest
    （路径变）不再触发 mismatch（修复 v2：旧口径是整文件字节哈希，含绝对路径）。
    """
    h = hashlib.sha256()
    for line in open(path, "rb"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            h.update(line)  # 解析失败回退到原字节，保守
            continue
        # 内容字段：识别看 id+ref，文本任务再看 source+target；audio_path 故意排除
        key = "\x1f".join(str(d.get(k, "")) for k in
                          ("id", "ref_text", "source_text", "target_lang", "task"))
        h.update(key.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:12]


def _audio_identity(row: dict):
    """取稳定的音频身份；优先显式摘要/id，路径只保留可迁移的尾部。"""
    for key in ("audio_sha256", "audio_md5", "audio_id", "audio_url"):
        if row.get(key):
            return {key: row[key]}
    path = str(row.get("audio_path") or "").replace("\\", "/")
    if not path:
        return None
    if "/datasets/" in path:
        path = "datasets/" + path.split("/datasets/", 1)[1]
    elif os.path.isabs(path):
        # 临时构建目录通常不同；保留尾三段，兼顾跨机器和同名文件区分度。
        parts = [p for p in path.split("/") if p]
        path = "/".join(parts[-3:])
    return {"audio_path": path}


def manifest_sha_v2(path: str) -> str:
    """更完整的 manifest 指纹（12 位），与 v1 并列记录、暂不参与排名。

    v2 覆盖评分/路由相关字段和音频身份；JSON 键序、空白及常见机器根目录
    差异不影响结果。解析失败时仍纳入原始行，避免静默碰撞。
    """
    h = hashlib.sha256()
    fields = (
        "id", "task", "ref_text", "source_text", "target_lang", "lang",
        "keywords", "ref", "n_spk_ref", "dialect", "domain",
    )
    with open(path, "rb") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                h.update(b"raw:")
                h.update(line)
            else:
                value = {key: row.get(key) for key in fields if key in row}
                audio = _audio_identity(row)
                if audio is not None:
                    value["audio_identity"] = audio
                h.update(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))
            h.update(b"\x1e")
    return h.hexdigest()[:12]


def validate_json_file(path: str) -> None:
    with open(path, encoding="utf-8") as src:
        json.load(src)


def validate_jsonl_file(path: str) -> None:
    with open(path, encoding="utf-8") as src:
        for lineno, line in enumerate(src, 1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno} 不是有效 JSON") from exc


def ensure_unique_ids(rows, *, source="manifest") -> None:
    """Reject duplicate sample IDs before inference or scoring can double-weight them."""
    seen = set()
    duplicate_ids = set()
    duplicates = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        if sample_id in seen and sample_id not in duplicate_ids:
            duplicate_ids.add(sample_id)
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = " ..." if len(duplicates) > 5 else ""
        raise ValueError(
            f"{source} 包含 {len(duplicates)} 个重复样本 ID: {preview}{suffix}"
        )


@contextmanager
def atomic_text_writer(path: str, *, validator=None, encoding="utf-8"):
    """在目标同目录完整写临时文件，校验成功后再原子替换旧文件。"""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent)
    stream = None
    try:
        mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o644
        os.chmod(tmp, mode)
        stream = os.fdopen(fd, "w", encoding=encoding)
        fd = -1
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = None
        if validator:
            validator(tmp)
        os.replace(tmp, path)
        tmp = ""
        # 尽力把目录项也刷盘；不支持目录 fsync 的平台直接降级。
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if stream is not None:
            stream.close()
        elif fd >= 0:
            os.close(fd)
        if tmp:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass


def atomic_write_json(path: str, value, *, indent=1) -> None:
    with atomic_text_writer(path, validator=validate_json_file) as out:
        json.dump(value, out, ensure_ascii=False, indent=indent)
        out.write("\n")


def code_revision() -> str | None:
    """返回当前代码 revision；取不到时诚实返回 None，不影响评测执行。"""
    for key in ("GIT_COMMIT", "SOURCE_VERSION"):
        value = os.environ.get(key)
        if value:
            return value[:40]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=repo,
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.SubprocessError):
        return None
