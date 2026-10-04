"""任务登记(共享)——CLI(runner/longrun)起的评测也写进 jobs.json,看板 /api/jobs 合并展示。

否则前端只看得到从看板按钮起的任务(JOBS 纯内存)。文件锁(fcntl)防并发交错写。
看板自身起的子进程会设环境变量 DASH_JOB,此时 runner 不再重复登记(看板已在内存跟踪)。
"""
import fcntl
import json
import os
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_FILE = os.path.join(ROOT, "jobs.json")


def _rw(mutate):
    """加锁 读→改→写 jobs.json。"""
    try:
        f = open(JOBS_FILE, "a+", encoding="utf-8")
    except Exception:
        return
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read().strip()
        data = json.loads(raw) if raw else {}
        mutate(data)
        f.seek(0)
        f.truncate()
        f.write(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        except Exception:
            pass


def start(model, dataset, tier="cli", **extra):
    """登记一个 CLI 运行中任务,返回 jid;看板已跟踪(DASH_JOB)则跳过返回 None。"""
    if os.environ.get("DASH_JOB"):
        return None
    queued_jid = os.environ.get("BATCH_JOB")
    if queued_jid:
        def _claim(d):
            if queued_jid in d:
                d[queued_jid].update(
                    status="running", stage="CLI 运行中",
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    started_ts=time.time(), pid=os.getpid(), **extra,
                )
        _rw(_claim)
        return queued_jid
    jid = "cli_" + uuid.uuid4().hex[:6]
    rec = {"id": jid, "model": model, "dataset": dataset, "tier": tier,
           "status": "running", "stage": "CLI 运行中", "origin": "cli",
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "started_ts": time.time(), "pid": os.getpid(), **extra}
    _rw(lambda d: d.__setitem__(jid, rec))
    return jid


def enqueue(model, dataset, tier="full", **extra):
    """预登记一个持久批处理任务；runner 通过 BATCH_JOB 认领同一条记录。"""
    jid = "batch_" + uuid.uuid4().hex[:8]
    rec = {
        "id": jid, "model": model, "dataset": dataset, "tier": tier,
        "status": "running", "stage": "批量排队（等待前序批次）", "origin": "cli",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "created_ts": time.time(), "pid": os.getpid(), **extra,
    }
    _rw(lambda d: d.__setitem__(jid, rec))
    return jid


def finish(jid, status="done", **extra):
    if not jid:
        return
    def _m(d):
        if jid in d:
            stage = extra.pop("stage", "完成" if status == "done" else status)
            d[jid].update(status=status, stage=stage,
                          finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **extra)
            if d[jid].get("started_ts"):
                d[jid]["took"] = round(time.time() - d[jid]["started_ts"])
    _rw(_m)


def update(jid, **extra):
    """Update metadata for a live CLI job after its id-dependent resources exist."""
    if not jid:
        return
    def _m(d):
        if jid in d:
            d[jid].update(extra)
    _rw(_m)
