"""
build_training_format.py
------------------------
Converts the preprocessed Parquet splits (train / val / test) into
training-ready JSONL files where every row has:

    {
        "review_id":  str,          # UUID from the processed split
        "text":       str,          # the cleaned review sentence
        "input_text": str,          # "Review: <text>" — model input string
Converts the preprocessed Parquet splits (train / val / test) into three
training-ready JSONL files:

1. master.jsonl  — one record per sentence with ALL aspect/sentiment pairs
   {
     "review_id": "...",
     "text": "...",
     "aspects": [{"category": "battery", "sentiment": "negative"}, ...]
   }

2. aspect_category.jsonl  — multi-label category classification
   {
     "review_id": "...",
     "text": "...",
     "labels": ["battery", "customer_support"]
   }

3. aspect_sentiment.jsonl  — per-aspect sentiment classification
   {
     "review_id": "...",
     "aspect": "battery",
     "text": "...",
     "label": "negative"
   }

All three files are written for each split (train / val / test) inside:
    data/training_format/
        ├── train/
        │   ├── master.jsonl
        │   ├── aspect_category.jsonl
        │   └── aspect_sentiment.jsonl
        ├── val/  ...
        ├── test/ ...
        └── label_vocab.json

Usage
-----
    python scripts/build_training_format.py          # all 3 splits
    python scripts/build_training_format.py --split train
    python scripts/build_training_format.py --out-dir data/training_format
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical label vocabulary — sorted alphabetically, N/A excluded.
# ---------------------------------------------------------------------------
LABEL_VOCAB: list[str] = sorted([
    "aesthetics",
    "compatibility",
    "cost",
    "effectiveness",
    "efficiency",
    "enjoyability",
    "general",
    "learnability",
    "reliability",
    "safety",
    "security",
    "usability",
])

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
INVALID_VALS = {"N/A", "NA", "", None}
LABEL_TO_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_VOCAB)}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _parse_aspects(raw_metadata: dict) -> list[dict]:
    """Extract valid (category, sentiment) pairs from raw_metadata."""
    annots = raw_metadata.get("aware_annotations", [])
    seen: set[tuple] = set()
    aspects: list[dict] = []
    for a in annots:
        if not isinstance(a, dict):
            continue
        cat = a.get("aspect_category")
        pol = a.get("polarity")
        if cat in INVALID_VALS or cat not in LABEL_TO_IDX:
            continue
        if pol in INVALID_VALS:
            pol = None          # keep category even when sentiment is missing
        key = (cat, pol)
        if key not in seen:
            seen.add(key)
            aspects.append({"category": cat, "sentiment": pol})
    return aspects


def convert_split(df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Returns three parallel lists of records:
        master_records, category_records, sentiment_records
    """
    master: list[dict] = []
    category: list[dict] = []
    sentiment: list[dict] = []

    for _, row in df.iterrows():
        text = str(row["review_text"]).strip()
        rid  = str(row["id"])
        if not text:
            continue

        aspects = _parse_aspects(row["raw_metadata"])

        # ── 1. Master record ──────────────────────────────────────────────
        master.append({
            "review_id": rid,
            "text":      text,
            "aspects":   aspects,
        })

        # ── 2. Aspect-category record (multi-label) ───────────────────────
        labels = sorted({a["category"] for a in aspects})
        category.append({
            "review_id": rid,
            "text":      text,
            "labels":    labels,
        })

        # ── 3. Aspect-sentiment records (one per aspect) ──────────────────
        for asp in aspects:
            pol = asp["sentiment"]
            if pol is None:         # skip aspects with unknown sentiment
                continue
            sentiment.append({
                "review_id": rid,
                "aspect":    asp["category"],
                "text":      text,
                "label":     pol,
            })

    return master, category, sentiment


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    → {path}  ({len(records):,} records)")


def print_stats(split: str, master: list, category: list, sentiment: list) -> None:
    print(f"\n  [{split}] master={len(master):,}  "
          f"category={len(category):,}  "
          f"sentiment={len(sentiment):,}")

    cat_counts = Counter(lbl for r in category for lbl in r["labels"])
    print("  Category distribution:")
    for lbl in LABEL_VOCAB:
        print(f"    {lbl:<20s} {cat_counts.get(lbl, 0):>5,}")

    sent_counts = Counter(r["label"] for r in sentiment)
    print("  Sentiment distribution:", dict(sent_counts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build master / aspect-category / aspect-sentiment JSONL datasets."
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="all")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir",  default="data/training_format")
    args = parser.parse_args()

    splits   = ["train", "val", "test"] if args.split == "all" else [args.split]
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    print(f"Label vocab ({len(LABEL_VOCAB)} categories): {LABEL_VOCAB}\n")

    for split in splits:
        parquet_path = data_dir / f"{split}.parquet"
        if not parquet_path.exists():
            print(f"[WARN] {parquet_path} not found — skipping.", file=sys.stderr)
            continue

        df = pd.read_parquet(parquet_path)
        master, category_recs, sentiment_recs = convert_split(df)

        print_stats(split, master, category_recs, sentiment_recs)

        split_dir = out_dir / split
        write_jsonl(master,         split_dir / "master.jsonl")
        write_jsonl(category_recs,  split_dir / "aspect_category.jsonl")
        write_jsonl(sentiment_recs, split_dir / "aspect_sentiment.jsonl")

    # Write the shared label vocab
    vocab_path = out_dir / "label_vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w") as f:
        json.dump({"label_vocab": LABEL_VOCAB, "label_to_idx": LABEL_TO_IDX}, f, indent=2)
    print(f"\nLabel vocab → {vocab_path}")


if __name__ == "__main__":
    main()
