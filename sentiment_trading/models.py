"""
Transformer sentiment model wrappers + Loughran-McDonald lexicon fallback.

Each wrapper normalises the model's native label ordering to the canonical:
  0 = negative, 1 = neutral, 2 = positive
and exposes:
  predict_class(sentences)  -> list[int]   (argmax class index)
  predict_score(sentences)  -> list[float] (continuous score in [-1, +1])
"""
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_SEQ_LEN, LATENCY_BUDGET_SECONDS

# Canonical label order used throughout this system
CANONICAL = {"negative": 0, "neutral": 1, "positive": 2}
ORDINAL = np.array([-1.0, 0.0, 1.0])  # indexed by canonical class


class BaseSentimentModel(ABC):
    name: str

    @abstractmethod
    def predict_class(self, sentences: list[str]) -> list[int]:
        """Return canonical class index (0=neg, 1=neu, 2=pos) per sentence."""

    @abstractmethod
    def predict_score(self, sentences: list[str]) -> list[float]:
        """Return continuous score in [-1, +1] per sentence."""

    def latency_ok(self, sentences: list[str], budget: float = LATENCY_BUDGET_SECONDS) -> bool:
        probe = sentences[:2] if sentences else ["test"]
        t0 = time.perf_counter()
        self.predict_class(probe)
        return (time.perf_counter() - t0) < budget


class TransformerSentimentModel(BaseSentimentModel):
    """Wraps any HuggingFace sequence-classification model."""

    def __init__(self, model_key: str, hf_name: str, device: Optional[str] = None):
        self.name = model_key
        self.hf_name = hf_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_name)
        self.model.to(self.device)
        self.model.eval()

        # Build a remapping from model's native label index -> canonical index
        id2label = self.model.config.id2label  # {0: "positive", 1: "negative", ...}
        self._remap = {}
        for native_idx, label_str in id2label.items():
            label_str = label_str.lower().strip()
            # Handle variations: "pos"/"neg"/"neu" prefixes
            if label_str.startswith("pos"):
                canonical_idx = CANONICAL["positive"]
            elif label_str.startswith("neg"):
                canonical_idx = CANONICAL["negative"]
            else:
                canonical_idx = CANONICAL["neutral"]
            self._remap[int(native_idx)] = canonical_idx

        # Inverse: canonical_idx -> native_idx (for training targets)
        self._inv_remap = {v: k for k, v in self._remap.items()}

    def _encode(self, sentences: list[str]):
        enc = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def _logits(self, sentences: list[str]) -> torch.Tensor:
        enc = self._encode(sentences)
        out = self.model(**enc)
        return out.logits  # (B, num_labels)

    @torch.no_grad()
    def _canonical_probs(self, sentences: list[str]) -> np.ndarray:
        """Return (B, 3) probability matrix in canonical label order."""
        logits = self._logits(sentences)
        native_probs = F.softmax(logits, dim=-1).cpu().numpy()  # (B, num_native)
        B = native_probs.shape[0]
        canonical_probs = np.zeros((B, 3), dtype=np.float32)
        for native_idx, can_idx in self._remap.items():
            if native_idx < native_probs.shape[1]:
                canonical_probs[:, can_idx] += native_probs[:, native_idx]
        return canonical_probs

    def predict_class(self, sentences: list[str]) -> list[int]:
        probs = self._canonical_probs(sentences)
        return probs.argmax(axis=1).tolist()

    def predict_score(self, sentences: list[str]) -> list[float]:
        """Weighted sum of ordinal values over class probabilities."""
        probs = self._canonical_probs(sentences)
        scores = probs @ ORDINAL  # (B,)
        return scores.tolist()

    def native_label_for(self, canonical_idx: int) -> int:
        """Used by trainer to map canonical training targets to native head indices."""
        return self._inv_remap.get(canonical_idx, canonical_idx)


class LexiconFallbackModel(BaseSentimentModel):
    """Loughran-McDonald lexicon fallback (paper Section III-C, III-H)."""

    name = "lm-lexicon"

    def __init__(self):
        self._pos: set[str] = set()
        self._neg: set[str] = set()
        self._loaded = False
        self._try_load()

    def _try_load(self):
        try:
            import pysentiment2 as ps2
            lm = ps2.LM()
            self._lm = lm
            self._loaded = True
            self._use_ps2 = True
        except Exception:
            self._use_ps2 = False
            self._loaded = True  # fall back to minimal hardcoded lists
            self._pos = {
                "gain", "gains", "profit", "profits", "positive", "growth",
                "increase", "increases", "strong", "improved", "improve",
                "record", "exceed", "exceeded", "rise", "rose", "beat",
            }
            self._neg = {
                "loss", "losses", "decline", "declines", "negative", "weak",
                "decrease", "decreases", "fall", "fell", "miss", "missed",
                "below", "concern", "risk", "risks", "delay", "delays",
            }

    def _score_sentence(self, sentence: str) -> float:
        tokens = re.findall(r"[a-z]+", sentence.lower())
        if not tokens:
            return 0.0
        if self._use_ps2:
            result = self._lm.get_score(tokens)
            polarity = result.get("Polarity", 0.0)
            return float(max(-1.0, min(1.0, polarity)))
        pos = sum(1 for t in tokens if t in self._pos)
        neg = sum(1 for t in tokens if t in self._neg)
        total = pos + neg
        if total == 0:
            return 0.0
        return float((pos - neg) / total)

    def predict_score(self, sentences: list[str]) -> list[float]:
        return [self._score_sentence(s) for s in sentences]

    def predict_class(self, sentences: list[str]) -> list[int]:
        classes = []
        for s in self.predict_score(sentences):
            if s > 0.1:
                classes.append(CANONICAL["positive"])
            elif s < -0.1:
                classes.append(CANONICAL["negative"])
            else:
                classes.append(CANONICAL["neutral"])
        return classes
