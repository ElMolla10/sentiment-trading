"""
End-to-end TradingPipeline (paper Section III-A).

Layer 1: data ingestion (price + news)
Layer 2: sentiment prediction with auto-selected model + lexicon fallback
Layer 3: signal generation (multiplicative modulation)
Layer 4: execution-ready signal output
"""
import time
from dataclasses import dataclass
from typing import Optional

from .config import TEMPORAL_DECAY_TAU_MINUTES, LATENCY_BUDGET_SECONDS
from .models import BaseSentimentModel, LexiconFallbackModel
from .modulator import modulate_with_decay
from .selector import ModelSelector


@dataclass
class TradingSignal:
    raw_price_signal: float       # base p before modulation
    sentiment_score: float        # s after decay
    final_signal: float           # modulated output
    model_used: str               # which model produced s
    dt_minutes: float             # news age


class TradingPipeline:
    """
    Orchestrates model selection → sentiment scoring → multiplicative modulation.

    Usage:
        pipeline = TradingPipeline()
        pipeline.setup()                 # benchmarks & selects best model
        signal = pipeline.generate(
            base_signal=-0.7,
            news_text="Apple reports record quarterly earnings",
            dt_minutes=15.0,
        )
    """

    def __init__(self, tau: float = TEMPORAL_DECAY_TAU_MINUTES):
        self.tau = tau
        self._selector = ModelSelector()
        self._primary: Optional[BaseSentimentModel] = None
        self._fallback: LexiconFallbackModel = LexiconFallbackModel()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, force_benchmark: bool = False) -> dict:
        """Benchmark all candidates and select the best model."""
        results = self._selector.benchmark(force=force_benchmark)
        self._primary = self._selector.select()
        self._selector.print_table()
        return results

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _score_sentiment(self, text: str) -> tuple[float, str]:
        """Returns (score, model_name) using primary → fallback degradation."""
        if self._primary is None:
            score = self._fallback.predict_score([text])[0]
            return score, self._fallback.name

        # Check latency budget; fall back if primary is too slow
        try:
            t0 = time.perf_counter()
            score = self._primary.predict_score([text])[0]
            latency = time.perf_counter() - t0
            if latency > LATENCY_BUDGET_SECONDS:
                raise TimeoutError(f"Primary model exceeded {LATENCY_BUDGET_SECONDS}s budget")
            return score, self._primary.name
        except Exception:
            score = self._fallback.predict_score([text])[0]
            return score, self._fallback.name

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate(
        self,
        base_signal: float,
        news_text: Optional[str],
        dt_minutes: float = 0.0,
    ) -> TradingSignal:
        """
        Produce a modulated trading signal.

        base_signal : price-based prediction; sign = direction (neg = short).
        news_text   : raw news headline/sentence; None = no news available.
        dt_minutes  : minutes elapsed since news publication.
        """
        if news_text is None:
            return TradingSignal(
                raw_price_signal=base_signal,
                sentiment_score=0.0,
                final_signal=base_signal,
                model_used="none",
                dt_minutes=dt_minutes,
            )

        sentiment_score, model_name = self._score_sentiment(news_text)
        final = modulate_with_decay(
            p=base_signal,
            s=sentiment_score,
            dt_minutes=dt_minutes,
            tau=self.tau,
            s_available=True,
        )

        return TradingSignal(
            raw_price_signal=base_signal,
            sentiment_score=sentiment_score,
            final_signal=final,
            model_used=model_name,
            dt_minutes=dt_minutes,
        )

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        base_signals: list[float],
        news_texts: list[Optional[str]],
        dt_minutes_list: Optional[list[float]] = None,
    ) -> list[TradingSignal]:
        dt_list = dt_minutes_list or [0.0] * len(base_signals)
        return [
            self.generate(p, text, dt)
            for p, text, dt in zip(base_signals, news_texts, dt_list)
        ]
