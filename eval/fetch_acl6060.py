"""一次性物化 ACL 60/60（HF ymoslem/acl-6060）→ 本地 wav + tsv，供 build_acl6060 离线读取。

ACL 60/60：ACL 2022 英文演讲音频，专业翻译成 10 语（含中文）。这里只取 en→zh
（英文音频 + text_zh 中文翻译参考），用于 en→zh 语音翻译/同传评测（IWSLT 官方评测集）。

落盘布局（datasets/ 不入仓）：
  datasets/translation/acl6060/
    audio/<split>/<index>.wav     源英文音频(原始 wav 字节，infer 时 load_wav_bytes 再转 16k)
    acl6060.tsv                   split \t index \t text_en \t text_zh
    .done                         幂等标记

跑法（需 datasets 库，dashboard 运行环境不带，故单独一次性跑）：
  uv run --with datasets --with soundfile python eval/fetch_acl6060.py
"""
import os

from config import ROOT

OUT = os.path.join(ROOT, "datasets/translation/acl6060")


def main():
    from datasets import Audio, load_dataset

    done = os.path.join(OUT, ".done")
    if os.path.exists(done):
        print(f"已物化，跳过（删 {done} 可重来）")
        return
    ds = load_dataset("ymoslem/acl-6060").cast_column("audio", Audio(decode=False))
    rows = []
    for split in ("dev", "eval"):
        adir = os.path.join(OUT, "audio", split)
        os.makedirs(adir, exist_ok=True)
        for ex in ds[split]:
            idx = str(ex["index"])
            with open(os.path.join(adir, f"{idx}.wav"), "wb") as f:
                f.write(ex["audio"]["bytes"])
            en = (ex.get("text_en") or "").replace("\t", " ").replace("\n", " ").strip()
            zh = (ex.get("text_zh") or "").replace("\t", " ").replace("\n", " ").strip()
            rows.append((split, idx, en, zh))
    with open(os.path.join(OUT, "acl6060.tsv"), "w", encoding="utf-8") as f:
        for split, idx, en, zh in rows:
            f.write(f"{split}\t{idx}\t{en}\t{zh}\n")
    open(done, "w").write("ok")
    print(f"物化完成：{len(rows)} 条 → {OUT}")


if __name__ == "__main__":
    main()
