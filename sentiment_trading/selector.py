"""
Automatic deployment-time model selection (paper Section III-D).

Benchmarks every candidate on the held-out test split, selects by:
  1. highest macro-F1  (primary)
  2. lowest MAE        (tie-break)
  3. lowest latency    (second tie-break)
"""
import json
import time
from pathlib import Path
from typing import Optional

from .config import CANDIDATE_MODELS, CHECKPOINTS_DIR, RESULTS_DIR
from .data import get_splits
from .evaluator import compute_metrics
from .models import TransformerSentimentModel, LexiconFallbackModel, BaseSentimentModel


class ModelSelector:
    def __init__(self, candidates: Optional[dict[str, str]] = None):
        self._candidates = candidates or CANDIDATE_MODELS
        self._results: dict[str, dict] = {}
        self._selected: Optional[BaseSentimentModel] = None
        self._fallback = LexiconFallbackModel()

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(self, force: bool = False) -> dict[str, dict]:
        """
        Evaluate all candidates on the test split.
        Loads fine-tuned checkpoints when available, otherwise zero-shot.
        """
        results_file = RESULTS_DIR / "benchmark.json"
        if results_file.exists() and not force:
            self._results = json.loads(results_file.read_text())
            return self._results

        _, _, test_rows = get_splits()
        sentences = [r["sentence"] for r in test_rows]
        y_true = [r["label"] for r in test_rows]

        self._results = {}
        for key, hf_name in self._candidates.items():
            print(f"\nBenchmarking {key}...")
            ckpt_dir = CHECKPOINTS_DIR / key
            load_path = str(ckpt_dir) if (ckpt_dir / "config.json").exists() else hf_name

            try:
                model = TransformerSentimentModel(key, load_path)
                t0 = time.perf_counter()
                y_pred = model.predict_class(sentences)
                latency = (time.perf_counter() - t0) / len(sentences)
                metrics = compute_metrics(y_true, y_pred)
                metrics["latency_per_sample_s"] = latency
            except Exception as e:
                print(f"  ERROR loading {key}: {e}")
                metrics = {"accuracy": 0.0, "f1": 0.0, "mae": 999.0, "latency_per_sample_s": 999.0}

            self._results[key] = metrics
            print(
                f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
                f"mae={metrics['mae']:.4f}"
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps(self._results, indent=2))
        return self._results

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self) -> BaseSentimentModel:
        """Return the best model loaded and ready for inference."""
        if not self._results:
            self.benchmark()

        def sort_key(item):
            m = item[1]
            return (-m.get("f1", 0.0), m.get("mae", 999.0), m.get("latency_per_sample_s", 999.0))

        ranked = sorted(self._results.items(), key=sort_key)
        best_key = ranked[0][0]
        best_hf = self._candidates[best_key]

        print(f"\nSelected model: {best_key} (F1={self._results[best_key]['f1']:.4f})")

        ckpt_dir = CHECKPOINTS_DIR / best_key
        load_path = str(ckpt_dir) if (ckpt_dir / "config.json").exists() else best_hf
        self._selected = TransformerSentimentModel(best_key, load_path)
        return self._selected

    @property
    def selected_model(self) -> Optional[BaseSentimentModel]:
        return self._selected

    @property
    def fallback(self) -> LexiconFallbackModel:
        return self._fallback

    def print_table(self):
        """Print results table matching Table I of the paper."""
        if not self._results:
            print("No results yet. Run benchmark() first.")
            return
        print(f"\n{'Model':<30} {'Accuracy':>10} {'F1':>10} {'MAE':>8}")
        print("-" * 62)
        for key, m in self._results.items():
            marker = " *" if key == getattr(self._selected, "name", None) else ""
            print(
                f"{key:<30} {m['accuracy']:>9.4f}  {m['f1']:>9.4f}  {m['mae']:>7.4f}{marker}"
            )
