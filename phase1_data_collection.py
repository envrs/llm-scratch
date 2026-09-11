#!/usr/bin/env python3
"""
phase1_data_collection.py

Implements Phase 1 (Data Collection) of the "Build an LLM From Scratch" pipeline:
    - Ingest raw documents (.txt, .jsonl, .csv)
    - Exact deduplication (hash-based)
    - Near-duplicate deduplication (MinHash, optional - falls back gracefully)
    - Quality filtering (length, repetition, symbol ratio heuristics)
    - Language filtering (optional langdetect, falls back to ASCII heuristic)
    - Basic toxic-content filtering (keyword-list based, swap in a real
      classifier / moderation API for production use)
    - PII redaction (emails, phone numbers, SSNs, credit-card-like numbers)
    - Train / validation / test split
    - Dataset statistics report

Usage:
    python phase1_data_collection.py \\
        --input-dir ./01_data_collection/raw \\
        --output-dir ./01_data_collection/processed \\
        --val-frac 0.02 --test-frac 0.02

Input formats supported in --input-dir (scanned recursively):
    *.txt      -> one document per file
    *.jsonl    -> one JSON object per line, with a "text" field (configurable)
    *.csv      -> one row per document, with a text column (configurable)

Output:
    <output-dir>/train.jsonl
    <output-dir>/val.jsonl
    <output-dir>/test.jsonl
    <output-dir>/stats.json
    <output-dir>/rejected_sample.jsonl   (sample of filtered-out docs, for auditing)

This is a reference implementation meant to be adapted: swap in real toxicity /
PII models, distributed dedup (e.g. Spark + MinHashLSH), and cloud storage as
your corpus scales past what fits on one machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# --------------------------------------------------------------------------
# Optional dependencies - the pipeline degrades gracefully if these are
# missing, rather than hard-failing.
# --------------------------------------------------------------------------

try:
    from langdetect import detect as _langdetect_detect
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    HAVE_LANGDETECT = True
except ImportError:
    HAVE_LANGDETECT = False

try:
    from datasketch import MinHash, MinHashLSH

    HAVE_DATASKETCH = True
except ImportError:
    HAVE_DATASKETCH = False


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    text_field: str = "text"           # field/column name for jsonl / csv input
    min_chars: int = 200               # reject documents shorter than this
    max_chars: int = 1_000_000         # reject documents longer than this
    max_symbol_ratio: float = 0.3      # reject if too much non-alnum content
    max_repeated_line_ratio: float = 0.3   # reject boilerplate/spam-like repetition
    target_languages: tuple = ("en",)  # only kept if langdetect available
    near_dup_threshold: float = 0.85   # MinHash Jaccard similarity threshold
    val_frac: float = 0.02
    test_frac: float = 0.02
    seed: int = 42
    rejected_sample_size: int = 200


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def iter_raw_documents(cfg: PipelineConfig) -> Iterator[dict]:
    """Yield {"text": ..., "source": ...} dicts from all supported files under input_dir."""
    input_dir = cfg.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for path in sorted(input_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".txt":
                yield from _read_txt(path)
            elif suffix == ".jsonl":
                yield from _read_jsonl(path, cfg.text_field)
            elif suffix == ".csv":
                yield from _read_csv(path, cfg.text_field)
        except Exception as e:
            print(f"  [warn] failed to read {path}: {e}", file=sys.stderr)


def _read_txt(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield {"text": text, "source": str(path)}


def _read_jsonl(path: Path, text_field: str) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field, "")
            if text:
                yield {"text": text, "source": f"{path}:{i}"}


def _read_csv(path: Path, text_field: str) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row.get(text_field, "")
            if text:
                yield {"text": text, "source": f"{path}:{i}"}


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------

def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def dedup_exact(docs: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    kept = []
    removed = 0
    for doc in docs:
        h = exact_hash(doc["text"])
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        kept.append(doc)
    return kept, removed


def _shingles(text: str, k: int = 5) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def dedup_near(docs: list[dict], threshold: float) -> tuple[list[dict], int]:
    """MinHash-LSH near-duplicate removal. No-op (with a warning) if datasketch is unavailable."""
    if not HAVE_DATASKETCH:
        print("  [info] datasketch not installed - skipping near-duplicate dedup "
              "(pip install datasketch to enable).")
        return docs, 0

    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    kept = []
    removed = 0
    for i, doc in enumerate(docs):
        mh = MinHash(num_perm=64)
        for shingle in _shingles(normalize_for_hash(doc["text"])):
            mh.update(shingle.encode("utf-8"))
        key = f"doc-{i}"
        if lsh.query(mh):
            removed += 1
            continue
        lsh.insert(key, mh)
        kept.append(doc)
    return kept, removed


# --------------------------------------------------------------------------
# Quality filtering
# --------------------------------------------------------------------------

_SYMBOL_RE = re.compile(r"[^a-zA-Z0-9\s]")


def quality_reason(doc: dict, cfg: PipelineConfig) -> str | None:
    """Return a rejection reason string, or None if the document passes."""
    text = doc["text"]
    n = len(text)

    if n < cfg.min_chars:
        return "too_short"
    if n > cfg.max_chars:
        return "too_long"

    symbol_ratio = len(_SYMBOL_RE.findall(text)) / max(n, 1)
    if symbol_ratio > cfg.max_symbol_ratio:
        return "high_symbol_ratio"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        line_counts = Counter(lines)
        most_common_count = line_counts.most_common(1)[0][1]
        if most_common_count / len(lines) > cfg.max_repeated_line_ratio and len(lines) > 5:
            return "repetitive_lines"

    return None


# --------------------------------------------------------------------------
# Language filtering
# --------------------------------------------------------------------------

_ASCII_RE = re.compile(r"[\x00-\x7F]")


def detect_language(text: str) -> str:
    if HAVE_LANGDETECT:
        try:
            return _langdetect_detect(text[:2000])
        except Exception:
            return "unknown"
    # Fallback heuristic: mostly-ASCII text is treated as English.
    ascii_ratio = len(_ASCII_RE.findall(text)) / max(len(text), 1)
    return "en" if ascii_ratio > 0.9 else "unknown"


# --------------------------------------------------------------------------
# Toxic content filtering (basic keyword heuristic)
#
# NOTE: this is a placeholder. For production use, swap in a real
# classifier or moderation API - keyword lists are easy to evade and prone
# to false positives.
# --------------------------------------------------------------------------

_TOXIC_KEYWORDS = {
    # Deliberately left generic/minimal - replace with a proper moderation
    # model or vetted lexicon appropriate for your data and use case.
}


def is_flagged_toxic(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in _TOXIC_KEYWORDS)


# --------------------------------------------------------------------------
# PII redaction
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CC_RE = re.compile(r"(?<!\d)(?:\d{4}[-\s]?){3}\d{4}(?!\d)")


def redact_pii(text: str) -> tuple[str, dict]:
    counts = {"email": 0, "phone": 0, "ssn": 0, "credit_card": 0}

    def _sub(pattern, tag, s):
        def repl(m):
            counts[tag] += 1
            return f"[REDACTED_{tag.upper()}]"
        return pattern.sub(repl, s)

    text = _sub(_EMAIL_RE, "email", text)
    text = _sub(_SSN_RE, "ssn", text)
    text = _sub(_CC_RE, "credit_card", text)
    text = _sub(_PHONE_RE, "phone", text)
    return text, counts


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def split_dataset(docs: list[dict], val_frac: float, test_frac: float, seed: int):
    if not (0 <= val_frac <= 1 and 0 <= test_frac <= 1 and val_frac + test_frac <= 1):
        raise ValueError("val_frac and test_frac must be in [0, 1] and sum to at most 1")
    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return train, val, test


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)

    print(f"[1/7] Reading raw documents from {cfg.input_dir} ...")
    docs = list(iter_raw_documents(cfg))
    print(f"      loaded {len(docs)} documents")
    if not docs:
        print("No documents found - check --input-dir and file extensions (.txt/.jsonl/.csv).")
        return

    rejected_sample: list[dict] = []

    def maybe_sample_reject(doc: dict, reason: str):
        if len(rejected_sample) < cfg.rejected_sample_size:
            rejected_sample.append({"reason": reason, "source": doc.get("source"),
                                     "preview": redact_pii(doc["text"])[0][:200]})

    print("[2/7] Exact deduplication ...")
    docs, n_exact_dup = dedup_exact(docs)
    print(f"      removed {n_exact_dup} exact duplicates, {len(docs)} remain")

    print("[3/7] Near-duplicate deduplication ...")
    docs, n_near_dup = dedup_near(docs, cfg.near_dup_threshold)
    print(f"      removed {n_near_dup} near-duplicates, {len(docs)} remain")

    print("[4/7] Quality filtering ...")
    kept = []
    reason_counts: Counter = Counter()
    for doc in docs:
        reason = quality_reason(doc, cfg)
        if reason:
            reason_counts[reason] += 1
            maybe_sample_reject(doc, reason)
            continue
        kept.append(doc)
    docs = kept
    print(f"      removed {sum(reason_counts.values())} docs "
          f"({dict(reason_counts)}), {len(docs)} remain")

    print("[5/7] Language filtering ...")
    kept = []
    lang_removed = 0
    for doc in docs:
        lang = detect_language(doc["text"])
        if lang in cfg.target_languages:
            kept.append(doc)
        else:
            lang_removed += 1
            maybe_sample_reject(doc, f"language_{lang}")
    docs = kept
    print(f"      removed {lang_removed} non-target-language docs, {len(docs)} remain")

    print("[6/7] Toxic-content filtering + PII redaction ...")
    kept = []
    toxic_removed = 0
    pii_totals: Counter = Counter()
    for doc in docs:
        if is_flagged_toxic(doc["text"]):
            toxic_removed += 1
            maybe_sample_reject(doc, "toxic_keyword_match")
            continue
        redacted_text, counts = redact_pii(doc["text"])
        for k, v in counts.items():
            pii_totals[k] += v
        doc["text"] = redacted_text
        kept.append(doc)
    docs = kept
    print(f"      removed {toxic_removed} flagged docs; "
          f"redacted PII counts: {dict(pii_totals)}")

    print("[7/7] Splitting and writing output ...")
    train, val, test = split_dataset(docs, cfg.val_frac, cfg.test_frac, cfg.seed)

    _write_jsonl(cfg.output_dir / "train.jsonl", train)
    _write_jsonl(cfg.output_dir / "val.jsonl", val)
    _write_jsonl(cfg.output_dir / "test.jsonl", test)
    _write_jsonl(cfg.output_dir / "rejected_sample.jsonl", rejected_sample)

    stats = {
        "raw_documents": len(docs) + n_exact_dup + n_near_dup + sum(reason_counts.values())
                          + lang_removed + toxic_removed,
        "exact_duplicates_removed": n_exact_dup,
        "near_duplicates_removed": n_near_dup,
        "quality_filtered": dict(reason_counts),
        "language_filtered": lang_removed,
        "toxic_filtered": toxic_removed,
        "pii_redactions": dict(pii_totals),
        "final_document_count": len(docs),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "approx_total_chars": sum(len(d["text"]) for d in docs),
        "approx_total_words": sum(len(d["text"].split()) for d in docs),
        "used_langdetect": HAVE_LANGDETECT,
        "used_datasketch_near_dedup": HAVE_DATASKETCH,
    }
    (cfg.output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print("\nDone.")
    print(json.dumps(stats, indent=2))


def _write_jsonl(path: Path, docs: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data Collection pipeline.")
    parser.add_argument("--input-dir", required=True, type=Path,
                         help="Directory containing raw .txt/.jsonl/.csv files (scanned recursively).")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Directory to write train/val/test jsonl + stats.")
    parser.add_argument("--text-field", default="text",
                         help="Field/column name holding document text in jsonl/csv input.")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--test-frac", type=float, default=0.02)
    parser.add_argument("--languages", nargs="+", default=["en"],
                         help="Target language codes to keep (requires langdetect for real detection).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_field=args.text_field,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        target_languages=tuple(args.languages),
        seed=args.seed,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()