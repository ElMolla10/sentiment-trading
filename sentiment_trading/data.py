"""
FinancialPhraseBank loader with stratified 70/15/15 splits (paper Section III-B).
Produces bronze (raw) → gold (tokenized) medallion tiers.
"""
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .config import (
    BRONZE_DIR, GOLD_DIR, DATASET_NAME, DATASET_CONFIG,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, MAX_SEQ_LEN,
)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
ORDINAL = {"negative": -1, "neutral": 0, "positive": 1}


def _load_raw() -> list[dict]:
    """Download FinancialPhraseBank and persist to bronze tier."""
    cache = BRONZE_DIR / "financial_phrasebank.json"
    if cache.exists():
        return json.loads(cache.read_text())

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)
    # sentences_50agree has a single "train" split — we re-split ourselves
    split = ds["train"]
    rows = [{"sentence": row["sentence"], "label": row["label"]} for row in split]
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def get_splits(force: bool = False) -> tuple[list, list, list]:
    """Return (train, val, test) as lists of {"sentence", "label"} dicts."""
    cache = BRONZE_DIR / "splits.pkl"
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            return pickle.load(f)

    rows = _load_raw()
    labels = [r["label"] for r in rows]

    # Stratified split: first carve out test, then split remainder into train/val
    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_idx, rest_idx = train_test_split(
        range(len(rows)), test_size=val_test_ratio, stratify=labels, random_state=RANDOM_SEED
    )
    rest_labels = [labels[i] for i in rest_idx]
    relative_test = TEST_RATIO / val_test_ratio
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=relative_test, stratify=rest_labels, random_state=RANDOM_SEED
    )

    train = [rows[i] for i in train_idx]
    val = [rows[i] for i in val_idx]
    test = [rows[i] for i in test_idx]

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump((train, val, test), f)
    return train, val, test


class SentimentDataset:
    """Tokenized dataset for a single split."""

    def __init__(self, rows: list[dict], tokenizer_name: str, max_len: int = MAX_SEQ_LEN):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        sentences = [r["sentence"] for r in rows]
        enc = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        # token_type_ids only if model uses them
        self.token_type_ids = enc.get("token_type_ids")
        self.labels = [r["label"] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item
