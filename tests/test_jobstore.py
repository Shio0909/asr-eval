import json


def test_enqueued_batch_job_is_claimed_without_creating_duplicate(tmp_path, monkeypatch):
    import jobstore

    jobs_file = tmp_path / "jobs.json"
    monkeypatch.setattr(jobstore, "JOBS_FILE", str(jobs_file))
    jid = jobstore.enqueue("std", "commonvoice_en", workers=8, note="展示")

    monkeypatch.setenv("BATCH_JOB", jid)
    claimed = jobstore.start("std", "commonvoice_en", workers=8, note="展示")
    data = json.loads(jobs_file.read_text(encoding="utf-8"))

    assert claimed == jid
    assert list(data) == [jid]
    assert data[jid]["stage"] == "CLI 运行中"
    assert data[jid]["started_ts"] > data[jid]["created_ts"]

    jobstore.finish(jid, "cancelled", stage="端点已下线")
    data = json.loads(jobs_file.read_text(encoding="utf-8"))
    assert data[jid]["status"] == "cancelled"
    assert data[jid]["stage"] == "端点已下线"
