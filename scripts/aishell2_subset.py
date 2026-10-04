#!/usr/bin/env python3
"""从 AISHELL-2 iOS 官方 82G zip 里抽样子集，只解被引用的 wav——不落全量。

背景：共享卷无法从本机直传，myairbridge 原始链接已失效，
全量 82G 上云代价大且评测本来就是抽样跑。本脚本产出几百 MB 的自包含子集目录，
好传、可复现（说话人分层抽样 + 固定种子），云上/本地共用同一份 → manifest 指纹一致。

用法（默认 400 说话人 × 5 条 = 2000 条，约 300MB）:
  python scripts/aishell2_subset.py
  python scripts/aishell2_subset.py --zip ~/Downloads/myairbridge-iOS.zip --spk 400 --per-spk 5 --seed 42

产出: datasets/asr/aishell2_ios/{trans_subset.tsv, wav/<SPK>/<utt>.wav, subset_meta.json}
之后: build_manifest.py --dataset aishell2 正常构建清单。
"""
import argparse
import io
import json
import os
import random
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.expanduser("~/Downloads/myairbridge-iOS.zip"))
    ap.add_argument("--spk", type=int, default=400, help="抽多少说话人(=解多少个 tar，主导耗时)")
    ap.add_argument("--per-spk", type=int, default=5, help="每说话人抽几条")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets/asr/aishell2_ios"))
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = set(zf.namelist())

    def _read(member):
        return zf.read(member).decode("utf-8", errors="replace")

    # trans.txt: "IC0001W0001\t文本"；wav.scp: "IC0001W0001\twav/C0001/IC0001W0001.wav"
    trans = {}
    for line in _read("iOS/data/trans.txt").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            trans[parts[0]] = parts[1]
    by_spk = {}   # spk → [(utt, tar 内相对路径)]
    for line in _read("iOS/data/wav.scp").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or parts[0] not in trans:
            continue
        utt, rel = parts
        spk = rel.split("/")[1] if rel.count("/") >= 2 else ""   # wav/C0001/xx.wav → C0001
        if spk and f"iOS/data/wav/{spk}.tar.gz" in names:
            by_spk.setdefault(spk, []).append((utt, rel))

    rng = random.Random(args.seed)
    spks = rng.sample(sorted(by_spk), min(args.spk, len(by_spk)))
    plan = {s: rng.sample(sorted(by_spk[s]), min(args.per_spk, len(by_spk[s]))) for s in spks}
    total = sum(len(v) for v in plan.values())
    print(f"抽样计划: {len(plan)} 说话人 × ≤{args.per_spk} 条 = {total} 条 (seed={args.seed})")

    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    rows, done_n = [], 0
    for si, (spk, items) in enumerate(sorted(plan.items()), 1):
        want = {os.path.basename(rel): utt for utt, rel in items}   # tar 内成员按 basename 匹配(容忍前缀差异)
        spk_dir = os.path.join(args.out, "wav", spk)
        os.makedirs(spk_dir, exist_ok=True)
        if not all(os.path.exists(os.path.join(spk_dir, b)) for b in want):   # 幂等续跑：齐了就跳过
            with zf.open(f"iOS/data/wav/{spk}.tar.gz") as f, \
                    tarfile.open(fileobj=io.BufferedReader(f), mode="r|gz") as tf:
                got = 0
                for m in tf:
                    b = os.path.basename(m.name)
                    if b in want and m.isfile():
                        with open(os.path.join(spk_dir, b), "wb") as w:
                            w.write(tf.extractfile(m).read())
                        got += 1
                        if got == len(want):
                            break
        ok_rows = [(utt, spk, b) for b, utt in want.items() if os.path.exists(os.path.join(spk_dir, b))]
        rows += ok_rows
        done_n += len(ok_rows)
        if si % 20 == 0 or si == len(plan):
            print(f"  {si}/{len(plan)} 说话人，累计 {done_n} 条…", flush=True)

    with open(os.path.join(args.out, "trans_subset.tsv"), "w", encoding="utf-8") as w:
        for utt, spk, b in sorted(rows):
            w.write(f"{utt}\t{spk}\t{trans[utt]}\n")
    json.dump({"source_zip": os.path.basename(args.zip), "zip_bytes": os.path.getsize(args.zip),
               "spk": len(plan), "per_spk": args.per_spk, "seed": args.seed, "n": len(rows),
               "sampling": "speaker-stratified"},
              open(os.path.join(args.out, "subset_meta.json"), "w"), ensure_ascii=False, indent=1)
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(args.out, "wav")) for f in fs)
    print(f"完成: {len(rows)} 条 → {args.out} (wav 共 {sz/1e6:.0f} MB)。上卷传这个目录即可。")


if __name__ == "__main__":
    main()
