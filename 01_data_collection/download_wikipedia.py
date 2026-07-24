#!/usr/bin/env python3
"""Download a small slice of Wikipedia articles into raw/ as .jsonl shards."""

from datasets import load_dataset
from pathlib import Path
import json

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SHARD_SIZE = 10_000
MAX_ARTICLES = 50_000

print("Loading wikimedia/wikipedia (20231101.en) ...")
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

shard_idx = 0
article_count = 0
buffer = []

for example in ds:
    article_count += 1
    buffer.append({"text": example["text"], "source": f"wiki/{example['id']}"})

    if len(buffer) >= SHARD_SIZE or article_count >= MAX_ARTICLES:
        shard_path = RAW_DIR / f"wiki_shard_{shard_idx:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            for doc in buffer:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"  wrote {len(buffer):>6} articles -> {shard_path.name}")
        buffer = []
        shard_idx += 1

    if article_count >= MAX_ARTICLES:
        break

if buffer:
    shard_path = RAW_DIR / f"wiki_shard_{shard_idx:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as f:
        for doc in buffer:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  wrote {len(buffer):>6} articles -> {shard_path.name}")

print(f"\nDone. {article_count} articles saved to {RAW_DIR}/")