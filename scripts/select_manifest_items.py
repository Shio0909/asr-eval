#!/usr/bin/env python3
"""Select named items from one or more JSONL manifests in a fixed order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    by_id = {}
    for path in args.inputs:
        for line in path.read_text().splitlines():
            if line:
                item = json.loads(line)
                by_id[item["id"]] = item
    missing = [item_id for item_id in args.ids if item_id not in by_id]
    if missing:
        raise SystemExit(f"missing ids: {', '.join(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(
            json.dumps(by_id[item_id], ensure_ascii=False) + "\n"
            for item_id in args.ids
        )
    )
    print(json.dumps({"out": str(args.out), "ids": args.ids}, ensure_ascii=False))


if __name__ == "__main__":
    main()
