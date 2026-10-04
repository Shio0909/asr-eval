"""推理阶段 — 调模型存 hyp（借鉴 UltraEval-Audio 的 infer/score 分离）。

与打分分离的价值：模型只调一次存下来，之后换指标/改口径/加 Embedding 都不必重调模型。
特性：断点续跑（增量落盘 + 跳过已完成 id）、失败重试（退避）、可选并发。

infer 文件每行: {id, hyp, elapsed_s, audio_s, ok, error, extra}
默认串行（性能延迟有效）；--workers>1 提速但延迟数据失真（score 会标注）。
"""

import argparse
import hashlib
import inspect
import json
import os
import re
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import soundfile as sf

from adapters import ADAPTERS, capability_contract_for, is_sensitive_request_param
from config import abs_path
from logconf import clear_ctx, get_logger, log_request, redact, set_ctx
from util import (atomic_text_writer, code_revision, manifest_sha, manifest_sha_v2,
                  ensure_unique_ids, validate_jsonl_file)

log = get_logger("infer")


def _pct(xs, p):
    """百分位(线性插值)；空 → None。仅用于 rollup 汇总。"""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


def audio_duration_s(path: str) -> float:
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def load_done(path: str) -> set:
    """读已有 infer 文件，返回已成功的 id 集合（失败的会重试）。"""
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("ok"):
                    done.add(d["id"])
            except Exception:
                pass
    return done


def language_tag(language: str) -> str:
    """language hint 文件名标签：短且稳定，避免传/不传同名结果互相覆盖。"""
    s = (language or "").strip()
    if not s:
        return ""
    keep = "".join(ch for ch in s if ch.isalnum())[:12]
    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:6]
    return f"__lang-{keep or 'hint'}-{digest}"


def infer_file_path(model: str, manifest: str, hotwords: bool = False, domain: str = "",
                    language: str = "", config_hash: str = "") -> str:
    base = os.path.splitext(os.path.basename(manifest))[0]
    os.makedirs("infer", exist_ok=True)
    # 场景 b(带热词)/领域提示词 单独存文件，避免与裸跑的 hyp 混在一起续跑
    cfg = re.sub(r"[^a-f0-9]", "", (config_hash or "").lower())[:16]
    return (f"infer/{model}__{base}{'__hw' if hotwords else ''}"
            f"{f'__dom-{domain}' if domain else ''}{language_tag(language)}"
            f"{f'__cfg-{cfg}' if cfg else ''}.jsonl")


def migrate_legacy_auto_infer(legacy_path: str, target_path: str, manifest: str,
                              model: str, endpoint: str, *, hotwords: bool = False,
                              domain: str = "", n_total: int = 0) -> bool:
    """Safely copy an untagged legacy auto infer into the language-tagged path.

    Old dashboard runs used the same untagged path for an empty language (which
    could resolve to a dataset default such as zh) and explicit auto.  Migration
    is therefore fail-closed: every recorded session must prove the same model,
    endpoint, manifest, and auto request parameters.
    """
    if os.path.exists(target_path) or not os.path.isfile(legacy_path):
        return False

    def meta_value(meta, key):
        value = meta.get(key)
        if value is None:
            value = (meta.get("run_spec") or {}).get(key)
        return value

    def canonical_endpoint(value):
        return str(value or "").strip().rstrip("/")

    def reject(reason):
        log.info("旧 auto infer 不迁移(%s): %s", os.path.basename(legacy_path), reason)
        return False

    current_endpoint = canonical_endpoint(endpoint)
    if not current_endpoint:
        return reject("当前端点为空，无法证明服务一致")

    metas = []
    done = set()
    try:
        with open(legacy_path, encoding="utf-8") as src:
            for lineno, line in enumerate(src, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("id"):
                    return reject(f"第 {lineno} 行缺少 id")
                if row["id"] == "__meta__":
                    metas.append(row)
                elif row.get("ok"):
                    done.add(str(row["id"]))
    except (OSError, json.JSONDecodeError) as exc:
        return reject(f"文件不可完整验证: {exc}")

    if not metas:
        return reject("缺少会话元数据")

    current_sha_v2 = manifest_sha_v2(manifest)
    current_domain = domain or ""
    for meta in metas:
        if str(meta_value(meta, "language") or "").strip().lower() != "auto":
            return reject("存在非 auto 或未声明 language 的会话")
        if meta_value(meta, "model") != model:
            return reject("模型不一致")
        if meta_value(meta, "manifest_sha_v2") != current_sha_v2:
            return reject("manifest 指纹不一致或缺失")
        if canonical_endpoint(meta_value(meta, "endpoint")) != current_endpoint:
            return reject("端点不一致或缺失")
        if bool(meta_value(meta, "hotwords")) != bool(hotwords):
            return reject("hotwords 配置不一致")
        if (meta_value(meta, "domain") or "") != current_domain:
            return reject("domain 配置不一致")

    source_meta = metas[-1]
    source_run_spec = dict(source_meta.get("run_spec") or {})
    source_run_spec["migrated_from"] = os.path.basename(legacy_path)
    migrated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    migration_meta = {
        "id": "__meta__",
        "schema_version": 2,
        "run_id": source_meta.get("run_id"),
        "run_spec": source_run_spec,
        "code_revision": source_meta.get("code_revision"),
        "metric_signature": source_meta.get("metric_signature"),
        "output_variant_used": source_meta.get("output_variant_used"),
        "model": model,
        "endpoint": endpoint,
        "language": "auto",
        "workers": source_meta.get("workers"),
        "hotwords": bool(hotwords),
        "domain": domain or None,
        "client_host": source_meta.get("client_host"),
        "note": source_meta.get("note"),
        "manifest": manifest,
        "manifest_sha": manifest_sha(manifest),
        "manifest_sha_v2": current_sha_v2,
        "started": source_meta.get("started"),
        "n_total": n_total,
        "n_todo": max(n_total - len(done), 0),
        "migration": {
            "kind": "legacy_untagged_auto",
            "source": os.path.basename(legacy_path),
            "source_runs": len(metas),
            "source_run_ids": [m.get("run_id") for m in metas if m.get("run_id")],
            "migrated_at": migrated_at,
            "code_revision": code_revision(),
        },
    }
    with atomic_text_writer(target_path, validator=validate_jsonl_file) as dst:
        ended_with_newline = True
        with open(legacy_path, encoding="utf-8") as src:
            for line in src:
                dst.write(line)
                ended_with_newline = line.endswith("\n")
        if not ended_with_newline:
            dst.write("\n")
        dst.write(json.dumps(migration_meta, ensure_ascii=False) + "\n")

    log.info("安全迁移旧 auto infer: %s → %s（复用成功样本 %d 条）",
             legacy_path, target_path, len(done))
    return True


def item_hotwords(r) -> str:
    """manifest keywords → 空格分隔热词串（FunASR 系接口惯例）。"""
    surfaces = []
    for k in (r.get("keywords") or []):
        surfaces += [k] if isinstance(k, str) else (k.get("aliases") or [k.get("surface", "")])
    return " ".join(dict.fromkeys(s for s in surfaces if s))


def _ctor_accepts(factory, name: str) -> bool:
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _aply_worker_cap(model: str, adapter, workers: int) -> int:
    """端点自有并发上限 → 压回上限。绝不静默超用把线上共用的产品打满。"""
    cap = getattr(adapter, "_MAX_WORKERS", None)
    if cap and workers > cap:
        log.warning("%s 端点并发上限 %d（线上共用）；--workers %d 已压回 %d",
                    model, cap, workers, cap)
        return cap
    return workers


def _make_adapter(model: str, kwargs: dict):
    factory = ADAPTERS[model]
    used, dropped = {}, []
    for k, v in kwargs.items():
        if _ctor_accepts(factory, k):
            used[k] = v
        else:
            dropped.append(k)
    adapter = None
    while adapter is None:
        try:
            adapter = factory(**used)
        except TypeError as e:
            # **kw 工厂(lambda/def f(**kw))会骗过签名检查，底层 __init__ 不收该参时这里才炸 → 剔除降级重试
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not (m and m.group(1) in used):
                raise
            dropped.append(m.group(1))
            used.pop(m.group(1))
    if "domain" in dropped:
        log.warning("%s 不支持 domain，--domain 被忽略", model)
    if "audio_out_dir" in dropped:
        log.warning("%s 不支持 audio_out_dir，--save-audio 被忽略", model)
    if "request_params" in dropped:
        # 内置 adapter 的构造器历史上不接 request_params；统一在实例化后挂载，
        # 请求切面按契约决定 form/query/json/header，避免为每个构造器重复加参数。
        from adapters import request_contract_for
        schema = request_contract_for(model)["request_schema"]
        if not schema:
            raise ValueError(f"{model} 未接入运行时 request_params，拒绝生成参数未生效的结果")
        adapter.request_schema = schema
        adapter.request_params = dict(kwargs.get("request_params") or {})
    return adapter, used


def _safe_request_params(values):
    return {
        str(k): v for k, v in (values or {}).items()
        if not is_sensitive_request_param(k)
    }


def run(manifest, model, language="auto", workers=1, retries=2, out=None,
        resume=True, limit=0, hotwords=False, domain="", save_audio=False, note="",
        target_lang="", hotwords_text="", request_params=None, config_hash=""):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    ensure_unique_ids(rows, source=manifest)
    for r in rows:  # manifest 存相对路径 → 用前还原绝对（跨机器可移植）
        if r.get("audio_path"):
            r["audio_path"] = abs_path(r["audio_path"])
    if limit:
        rows = rows[:limit]
    akw = {}
    if save_audio:  # 同传译后语音存档：audio_out/<model>__<manifest>/（仅 simult 系 adapter 受理）
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _cfg = re.sub(r"[^a-f0-9]", "", (config_hash or "").lower())[:16]
        _base = (f"{model}__{os.path.splitext(os.path.basename(manifest))[0]}"
                 f"{language_tag(language)}{f'__cfg-{_cfg}' if _cfg else ''}")
        akw["audio_out_dir"] = os.path.join(_root, "audio_out", _base)
    ctor_kwargs = dict(akw)
    if domain:  # 领域提示词仅由独立 adv-domain adapter 接受；其他端点会明确拒绝
        ctor_kwargs["domain"] = domain
    if request_params:
        ctor_kwargs["request_params"] = request_params
    adapter, used_kwargs = _make_adapter(model, ctor_kwargs)
    workers = _aply_worker_cap(model, adapter, workers)
    if "domain" not in used_kwargs:
        domain = ""
    hotwords_text = str(hotwords_text or "").strip()
    hotwords_enabled = bool(hotwords or hotwords_text)
    if hotwords_enabled and not getattr(adapter, "supports_hotwords", False):
        log.warning("%s 不支持 hotwords，--hotwords 被忽略", model)
        hotwords = False
        hotwords_text = ""
        hotwords_enabled = False
    endpoint = getattr(adapter, "url", "") or getattr(adapter, "ws_url", "")
    requested_out = out
    out = out or infer_file_path(model, manifest, hotwords_enabled, domain, language, config_hash)
    if (resume and requested_out is None
            and str(language or "").strip().lower() == "auto"
            and not config_hash):
        legacy = infer_file_path(model, manifest, hotwords_enabled, domain, "")
        migrate_legacy_auto_infer(
            legacy, out, manifest, model, endpoint,
            hotwords=hotwords_enabled, domain=domain, n_total=len(rows),
        )
    done = load_done(out) if resume else set()
    todo = [r for r in rows if r["id"] not in done]
    row_position = {r["id"]: i + 1 for i, r in enumerate(rows)}
    longform_progress = any(r.get("longform") for r in rows)
    simult_progress = "simultaneous_translation" in (
        capability_contract_for(model).get("tasks") or []
    )
    # 这行含「待跑 N」：dashboard 正则解析它显示进度（经 stderr 并入子进程 stdout 管道）
    log.info("%s on %s: 共 %d，已完成 %d，待跑 %d，workers=%d%s",
             model, os.path.basename(manifest), len(rows), len(done), len(todo), workers,
             "，热词开" if hotwords_enabled else "")

    cur_sha = manifest_sha(manifest)
    cur_sha_v2 = manifest_sha_v2(manifest)
    run_id = uuid.uuid4().hex
    run_spec = {
        "model": model,
        "endpoint": endpoint,
        "manifest": os.path.basename(manifest),
        "manifest_sha": cur_sha,
        "manifest_sha_v2": cur_sha_v2,
        "language": language,
        "target_lang": target_lang or None,
        "workers": workers,
        "retries": retries,
        "resume": bool(resume),
        "limit": limit or None,
        "hotwords": hotwords_enabled,
        "dataset_hotwords": bool(hotwords),
        "hotwords_text": hotwords_text or None,
        "domain": domain or None,
        "save_audio": bool(save_audio),
        "request_params": _safe_request_params(request_params),
        "config_hash": config_hash or None,
        "adapter_kwargs": {
            k: (_safe_request_params(v) if k == "request_params" else v)
            for k, v in used_kwargs.items()
        },
    }
    fout = open(out, "a", encoding="utf-8")
    if todo:  # 每个运行会话记一条 __meta__（provenance：谁、何时、何参数跑的）
        fout.write(json.dumps({
            "id": "__meta__", "schema_version": 2,
            "run_id": run_id, "run_spec": run_spec,
            "code_revision": code_revision(),
            "metric_signature": None, "output_variant_used": None,
            "model": model, "endpoint": endpoint,
            "language": language, "workers": workers, "hotwords": hotwords_enabled,
            "dataset_hotwords": bool(hotwords), "hotwords_text": hotwords_text or None,
            "target_lang": target_lang or None,
            "domain": domain or None, "client_host": socket.gethostname(),
            "request_params": _safe_request_params(request_params),
            "config_hash": config_hash or None,
            "note": note or None,
            "manifest": manifest,
            "manifest_sha": cur_sha,
            "manifest_sha_v2": cur_sha_v2,
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            "n_total": len(rows), "n_todo": len(todo)}, ensure_ascii=False) + "\n")
        fout.flush()
    job = os.environ.get("DASH_JOB", "")   # 看板子进程会设；CLI 跑则空 → 关联键留空
    jobstore_managed = os.environ.get("JOBSTORE_MANAGED") == "1"
    done_before = len(rows) - len(todo)
    if jobstore_managed:
        import jobstore
        jobstore.update(job, stage="infer+score", progress=f"{done_before}/{len(rows)}")
    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0, "n": 0, "calls": 0, "consec_fail": 0}
    lats = []   # 成功样本的端到端延迟(s)，供 rollup 算 p50/p95/p99
    # 熔断：连续 N 条(重试耗尽后)全失败≈端点挂了，中止而不是傻跑完全量出一份废结果。已跑样本保留可续跑。
    max_consec = int(os.environ.get("ASR_MAX_CONSEC_FAIL", "20"))
    fuse = threading.Event()

    def work(r):
        # L1 调用边界：设线程上下文(job/model/id)，本样本内 adapter 的 HTTP/WS 调用自动带关联键
        set_ctx(job=job or None, model=model, id=r["id"])
        try:
            is_text = r.get("task") in ("translate", "summarize", "minutes", "formula")
            task = r.get("task") or "asr"
            dur = 0.0 if (is_text and not r.get("audio_path")) else audio_duration_s(r.get("audio_path", ""))
            # per_item_lang 的 adapter(提示词分语言)用行内 lang，其余用全局 --language
            lang = (r.get("lang") if getattr(adapter, "per_item_lang", False) and r.get("lang")
                    else language)
            kw = {}
            hw_parts = []
            if hotwords:
                item_hw = item_hotwords(r)
                if item_hw:
                    hw_parts.append(item_hw)
            if hotwords_text:
                hw_parts.append(hotwords_text)
            if hw_parts:
                kw["hotwords"] = " ".join(hw_parts)
            if target_lang:
                kw["target_lang"] = target_lang
            if is_text and not hasattr(adapter, "generate"):  # 纯 ASR adapter 无 generate → 优雅失败
                log_request(layer="call", task=task, ok=False, err="adapter 不支持该任务")
                return {"id": r["id"], "hyp": "", "elapsed_s": 0.0, "audio_s": round(dur, 2),
                        "ok": False, "workers": workers, "error": f"{model} 不支持 {r.get('task')} 任务",
                        "_attempts": 0}
            last_err = ""
            attempts = 0
            for attempt in range(retries + 1):
                attempts = attempt + 1
                if is_text:
                    call_item = dict(r)
                    call_item["request_language"] = language
                    if r.get("longform") and r.get("audio_path"):
                        call_item["_stream_progress"] = {
                            "sample_index": row_position.get(r["id"]),
                            "sample_total": len(rows),
                        }
                    if target_lang:
                        call_item["target_lang"] = target_lang
                    if kw.get("hotwords"):
                        call_item["hotwords"] = kw["hotwords"]
                    res = adapter.generate(call_item)
                else:
                    res = adapter.transcribe(r["audio_path"], language=lang, **kw)
                ex = res.extra or {}
                # L1：每次尝试(含重试)一条 — 延迟/RTF/字数/同传 ttfb·AL·LAAL；ok=False 自动升 WARNING
                log_request(layer="call", endpoint=endpoint, task=task, attempt=attempts, ok=res.ok,
                            latency_ms=round(res.elapsed_s * 1000) if res.elapsed_s else 0,
                            audio_s=round(dur, 2) or None,
                            rtf=round(res.elapsed_s / dur, 3) if dur else None,
                            chars=len(res.text or "") if res.ok else None,
                            ttfb_ms=round(ex["ttfb_s"] * 1000) if ex.get("ttfb_s") is not None else None,
                            al_s=ex.get("al_s"), laal_s=ex.get("laal_s"),
                            err=None if res.ok else (res.error or "")[:200])
                if res.ok:
                    return {"id": r["id"], "hyp": res.text, "elapsed_s": round(res.elapsed_s, 3),
                            "audio_s": round(dur, 2), "ok": True, "workers": workers,
                            "extra": res.extra, "_attempts": attempts}
                last_err = res.error
                if attempt < retries:
                    # 端点抽风/退避重试留痕。日志侧不手动 redact——_RedactFormatter 输出层统一做(也覆盖 traceback)。
                    log.warning("%s id=%s 第%d/%d次失败，退避重试: %s",
                                model, r["id"], attempt + 1, retries, last_err)
                    time.sleep(0.5 * (2 ** attempt))  # 退避
            log.error("%s id=%s 重试耗尽，记为失败: %s", model, r["id"], last_err)
            # error 会落进 infer/*.jsonl → 被 score 拷进 results → /api/result 返回，
            # 不经日志层脱敏，故须在落盘前 redact(如 xf WS URL 里的 api_key)。
            return {"id": r["id"], "hyp": "", "elapsed_s": 0.0,
                    "audio_s": round(dur, 2), "ok": False, "workers": workers,
                    "error": redact(last_err), "_attempts": attempts}
        finally:
            clear_ctx()

    def emit(rec):
        with lock:
            att = rec.pop("_attempts", 1)
            # 重试次数属于稳定性口径，必须随 infer 落盘；只看进程日志无法在后续
            # score-only 或上传共享卷后还原每条样本实际调用了几次模型。
            rec["attempts"] = att
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            counts["ok"] += rec["ok"]
            counts["fail"] += (not rec["ok"])
            counts["calls"] += att
            counts["n"] += 1
            counts["consec_fail"] = 0 if rec["ok"] else counts["consec_fail"] + 1
            if max_consec and counts["consec_fail"] >= max_consec:
                fuse.set()
            if rec["ok"] and rec.get("elapsed_s"):
                lats.append(rec["elapsed_s"])
            # 同传通常串行且样本量小，逐条上报；其他长音频同样逐场上报；普通短音频每 10 条。
            if (simult_progress or longform_progress or counts["n"] % 10 == 0
                    or counts["n"] == len(todo)):
                print(f"  {counts['n']}/{len(todo)}…", flush=True)  # dashboard 靠这行解析进度
                if jobstore_managed:
                    jobstore.update(job, stage="infer+score",
                                    progress=f"{done_before + counts['n']}/{len(rows)}")

    if workers <= 1:
        for r in todo:
            emit(work(r))
            if fuse.is_set():
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(work, r) for r in todo]):
                emit(fut.result())
                if fuse.is_set():
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    fout.close()
    if fuse.is_set():
        log.error("熔断中止：连续 %d 条失败（端点疑似不可用，最后错误见上）。已跑 %d/%d 保留，修复端点后重跑即续跑。",
                  max_consec, counts["n"], len(todo))
        raise SystemExit(3)
    log.info("完成: 新跑 ok=%d fail=%d calls=%d → %s", counts["ok"], counts["fail"], counts["calls"], out)
    # rollup：一条汇总进 requests.log（成功率 + 延迟分位）
    done_n = counts["ok"] + counts["fail"]
    log_request(layer="rollup", job=job or None, model=model, manifest=os.path.basename(manifest),
                calls=counts["calls"], ok_n=counts["ok"], fail_n=counts["fail"],
                success_rate=round(counts["ok"] / done_n, 3) if done_n else None,
                p50_ms=round(_pct(lats, 50) * 1000) if lats else None,
                p95_ms=round(_pct(lats, 95) * 1000) if lats else None,
                p99_ms=round(_pct(lats, 99) * 1000) if lats else None)
    return out, (workers > 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--language", default="auto")
    ap.add_argument("--target-lang", default="",
                    help="覆盖 manifest 的 target_lang（翻译/同传实验）")
    ap.add_argument("--workers", type=int, default=1, help=">1 提速但延迟失真")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--hotwords", action="store_true",
                    help="把 manifest keywords 作为热词传给模型(场景 b)，infer 文件名带 __hw")
    ap.add_argument("--hotwords-text", default="",
                    help="每条样本都追加的自定义热词；与 --hotwords 的 manifest keywords 合并")
    ap.add_argument("--save-audio", action="store_true",
                    help="同传：存译后语音到 audio_out/（仅 simult 系 adapter）")
    ap.add_argument("--domain", default="",
                    help="领域提示词(legal/medical/finance/government_emergency)→ 强制走 adv-domain，文件名带 __dom-X")
    ap.add_argument("--request-params-json", default="{}",
                    help="已由接口 request_schema 校验的运行时请求参数 JSON")
    ap.add_argument("--config-hash", default="",
                    help="接口配置+运行参数指纹；用于隔离 infer 续跑文件")
    args = ap.parse_args()
    request_params = json.loads(args.request_params_json or "{}")
    run(args.manifest, args.model, args.language, args.workers, args.retries,
        args.out, not args.no_resume, args.limit, args.hotwords, args.domain, args.save_audio,
        target_lang=args.target_lang, hotwords_text=args.hotwords_text,
        request_params=request_params, config_hash=args.config_hash)


if __name__ == "__main__":
    main()
