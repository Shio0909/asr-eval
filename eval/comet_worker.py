"""在隔离 Python 环境中计算 COMET；stdin/stdout 只传 JSON。"""

import importlib.util
import json
import sys


def main():
    if sys.argv[1:] == ["--check"]:
        raise SystemExit(0 if importlib.util.find_spec("comet") is not None else 1)

    from comet import download_model, load_from_checkpoint

    payload = json.load(sys.stdin)
    triples = payload.get("triples") or []
    data = [{"src": s or "", "ref": r or "", "mt": h or ""} for s, r, h in triples]
    checkpoint = download_model(payload.get("model") or "Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(checkpoint)
    # unbabel-comet 2.2 + current torch rejects its default combination of
    # multiprocessing_context with num_workers=0. One worker is enough for the
    # isolated CPU scorer and avoids the DataLoader ValueError.
    out = model.predict(data, batch_size=int(payload.get("batch") or 8),
                        gpus=0, num_workers=1, progress_bar=False)
    scores = out.get("scores") if isinstance(out, dict) else getattr(out, "scores", None)
    json.dump({"scores": [round(float(x), 4) for x in scores]}, sys.stdout)


if __name__ == "__main__":
    main()
