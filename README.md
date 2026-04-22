# Sentiment-Modulated Algorithmic Trading

> **Automatic model selection · Multiplicative signal modulation · Sign-aware short handling**

A production-grade Python system that integrates financial NLP sentiment into algorithmic trading signals. The architecture addresses three gaps that persist across the prior literature: models are never selected automatically at deployment time, sentiment is rarely applied as a continuous multiplicative modulator, and its sign is almost never inverted for short positions.

---

## Key Results

Results on the held-out FinancialPhraseBank test split (3-class: negative / neutral / positive):

| Model | Accuracy | F1 | MAE |
|---|---|---|---|
| **DeBERTa-finance** ✓ | **94.2%** | **0.9420** | **0.060** |
| FinBERT | 89.2% | 0.8932 | 0.114 |
| pmatorras/BERT-finance | 12.5% | 0.028 | 1.157 |

The 81.7 percentage-point gap between best and worst candidates (and the fact that `pmatorras` sits *below* the 33.3 % random baseline) is the core motivation for automatic benchmarking — without it, a pipeline may be routing live capital through an adversarially bad sentiment source.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 – Data Ingestion                                       │
│  Price feed  +  News feed  →  Bronze / Silver / Gold tiers      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Layer 2 – Sentiment Prediction (auto-selected model)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │DeBERTa-finance│ │   FinBERT    │  │ pmatorras/BERT-finance│  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └─────────────────┴──────────────────────┘              │
│                    ModelSelector (F1 → MAE → latency)           │
│                    + Loughran-McDonald lexicon fallback          │
└────────────────────────┬────────────────────────────────────────┘
                         │  s ∈ [-1, +1]
┌────────────────────────▼────────────────────────────────────────┐
│  Layer 3 – Signal Generation                                    │
│                                                                 │
│  Temporal decay:  s′ = s · exp(−dt / τ)   (τ = 90 min)         │
│                                                                 │
│  Multiplicative modulation:                                     │
│    sign(p) == sign(s) → final = p · (1 + |s′|)  [amplify]      │
│    sign(p) != sign(s) → final = p · (1 − |s′|)  [attenuate]    │
│    s == 0             → final = p               [pass-through]  │
│                                                                 │
│  Short-side auto-correction (no special case needed):           │
│    p = −0.7, s = +0.8  →  final = −0.7 × 0.2 = −0.14          │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Layer 4 – Execution                                            │
│  TradingSignal { raw, sentiment, final, model, dt }             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1 — Install dependencies

```bash
git clone https://github.com/ElMolla10/sentiment-trading.git
cd sentiment-trading
pip install -r requirements.txt
```

> Requires Python 3.11+ and PyTorch 2.1+. A CUDA GPU is recommended for training but not required.

### 2 — Run the paper's worked example

```bash
python main.py demo
```

```
=== Paper Worked Example (Section III-F) ===
  p = -0.7 (short position)
  s = +0.8 (strongly positive news)
  sign(p)=-, sign(s)=+  → OPPOSED
  final = -0.7 * (1 - |0.8|) = -0.7 * 0.2 = -0.1400
  Short conviction reduced from |-0.7| to |-0.14|
```

### 3 — Fine-tune all candidate models

```bash
python main.py train
```

Downloads FinancialPhraseBank, performs stratified 70/15/15 split, fine-tunes DeBERTa-finance, FinBERT, and pmatorras/BERT-finance with AdamW (lr=2e-5, warmup=10 %, batch=32, early stopping patience=1), and saves checkpoints to `checkpoints/`.

### 4 — Reproduce Table I (automatic benchmark)

```bash
python main.py benchmark
```

Evaluates every candidate on the held-out test split, prints accuracy / F1 / MAE, selects the best model, and highlights the best–worst gap.

### 5 — Interactive modulated prediction

```bash
python main.py predict
```

```
> -0.7 | Apple reports record quarterly earnings | 15
  [SHORT] base=-0.7000  s=+0.8200  final=-0.1034  (model=deberta-finance, dt=15min)
```

---

## Project Layout

```
sentiment-trading/
├── main.py                         # CLI: train | benchmark | predict | demo
├── requirements.txt
├── sentiment_trading/
│   ├── config.py                   # all hyperparameters and paths
│   ├── data.py                     # FinancialPhraseBank loader + medallion tiers
│   ├── models.py                   # TransformerSentimentModel + LexiconFallbackModel
│   ├── trainer.py                  # fine-tuning loop with early stopping
│   ├── evaluator.py                # accuracy, macro-F1, ordinal MAE
│   ├── selector.py                 # ModelSelector: benchmark → select best
│   ├── modulator.py                # modulate(), apply_decay(), modulate_with_decay()
│   └── pipeline.py                 # TradingPipeline (full end-to-end)
├── data/
│   ├── bronze/                     # raw FinancialPhraseBank JSON + split cache
│   ├── silver/                     # engineered features (future)
│   └── gold/                       # model-ready tokenized datasets (future)
├── checkpoints/                    # per-model fine-tuned weights
└── results/                        # benchmark.json (Table I)
```

---

## Core API

### `modulate(p, s)`

```python
from sentiment_trading import modulate

# Long position, positive news → amplified
modulate(0.6, 0.7)   # → 1.02

# Short position, positive news → attenuated (correct sign-aware behaviour)
modulate(-0.7, 0.8)  # → -0.14

# Neutral sentiment → pass-through
modulate(0.5, 0.0)   # → 0.5
```

### `TradingPipeline`

```python
from sentiment_trading import TradingPipeline

pipeline = TradingPipeline()
pipeline.setup()          # benchmarks all candidates, selects best

signal = pipeline.generate(
    base_signal=-0.7,
    news_text="Company reports record losses amid supply chain disruptions",
    dt_minutes=30,        # news is 30 minutes old
)
print(signal.final_signal)   # attenuated or amplified depending on sentiment
print(signal.model_used)     # e.g. "deberta-finance"
```

### `ModelSelector`

```python
from sentiment_trading.selector import ModelSelector

selector = ModelSelector()
results = selector.benchmark()   # runs all candidates on test split
best    = selector.select()      # loads best model into memory
selector.print_table()
```

---

## Modulation Formula

```
Given:
  p ∈ (−∞, +∞)  base price-based prediction  (sign = direction)
  s ∈ [−1, +1]  sentiment score               (sign = bullish/bearish)

if s == 0:
    final = p                       # neutral news → pass-through
elif sign(p) == sign(s):
    final = p × (1 + |s|)          # aligned → amplify
else:
    final = p × (1 − |s|)          # opposed → attenuate
```

With temporal decay (τ = 60–120 min, default 90):

```
s′ = s × exp(−dt / τ)
final = modulate(p, s′)
```

Graceful degradation: if the news pipeline is unavailable, `s` is forced to 0 and `final = p`, so the sentiment layer never injects noise when it cannot operate.

---

## Candidate Models

| Key | HuggingFace ID |
|-----|---------------|
| `deberta-finance` | `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` |
| `finbert` | `ProsusAI/finbert` |
| `pmatorras` | `pmatorras/BERT-finance` |

To add a new candidate, extend `CANDIDATE_MODELS` in `sentiment_trading/config.py` and re-run `python main.py benchmark`.

---

## Configuration

All hyperparameters live in `sentiment_trading/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `BATCH_SIZE` | `32` | Training batch size |
| `WARMUP_RATIO` | `0.10` | Linear warmup fraction |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay |
| `MAX_EPOCHS` | `4` | Maximum fine-tuning epochs |
| `EARLY_STOP_PATIENCE` | `1` | Val-F1 patience |
| `MAX_SEQ_LEN` | `128` | Tokenizer max length |
| `TEMPORAL_DECAY_TAU_MINUTES` | `90` | Sentiment decay time constant |
| `LATENCY_BUDGET_SECONDS` | `2.0` | Primary model latency cap before fallback |

---

## Dataset

[FinancialPhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (`sentences_50agree`) — 4,840 English-language financial news sentences with three-class sentiment labels (negative / neutral / positive) annotated by 16 domain experts. Class distribution: ~12 % negative, ~59 % neutral, ~28 % positive.

---

## Safety & Constraints

This system is designed for research and strategy prototyping. Before deploying in a live trading environment, the following constraints must be understood and explicitly accepted.

### What the system does well
- Automatic benchmarking eliminates the most catastrophic failure mode: a below-random sentiment source silently integrated into a live pipeline. The 81.7 pp best–worst gap makes this non-negotiable.
- Multiplicative modulation is magnitude-bounded: even a maximally confident sentiment score (`|s| = 1`) can at most double or zero out the base signal — it cannot flip its direction or produce runaway values.
- Graceful degradation guarantees `final = p` whenever sentiment is unavailable, so the trading logic degrades to its base signal rather than failing open.

### Known limitations

| # | Constraint | Risk if ignored |
|---|-----------|-----------------|
| 1 | **Linear scaling only.** The formula uses `(1 ± \|s\|)` scaling. Market reactions to news — especially around earnings, central bank announcements, or macro surprises — are empirically nonlinear. | Underweights tail events; overweights routine news. |
| 2 | **English-only sentiment.** Models are fine-tuned on English financial text. Sentiment from non-English news sources is unscored and silently treated as neutral (`s = 0`). | May systematically misprice assets whose primary news flow is non-English. |
| 3 | **Single-domain evaluation.** Benchmarks are run on FinancialPhraseBank (equity-market news sentences). Generalisation to foreign exchange, fixed income, commodities, or social-media text is unvalidated. | Reported accuracy figures do not transfer to other asset classes without re-benchmarking. |
| 4 | **Fixed decay constant τ = 90 min.** The exponential decay `exp(−dt / τ)` uses a single constant derived from published intraday studies, not estimated per-instrument. Highly liquid assets (e.g. S&P 500 futures) may decay faster; illiquid assets slower. | Mis-timed attenuation; stale sentiment treated as fresh or vice versa. |
| 5 | **Modulation strength not jointly calibrated.** The `\|s\|` weight is the sentiment model's own probability calibration. No separate modulation-strength hyperparameter is jointly estimated with the base forecaster. | Suboptimal blending ratio between price and sentiment signals. |
| 6 | **Automatic selection requires a representative validation set.** `ModelSelector` picks the best model on the held-out split. If that split is stale, class-imbalanced, or domain-shifted, the selected model may not be the best for the live distribution. | Selection is only as good as the validation data it runs on; refresh periodically. |

### Recommended safeguards before live deployment

1. **Re-benchmark on your own corpus** — do not rely solely on FinancialPhraseBank results if your strategy ingests a different news source.
2. **Cap modulation magnitude** — consider clipping `|s|` to e.g. 0.5 until live performance is validated, limiting the maximum amplification to 1.5× rather than 2×.
3. **Monitor for sentiment drift** — schedule `python main.py benchmark --force` on a regular cadence to detect silent model degradation before it affects capital.
4. **Paper-trade first** — validate the full pipeline in simulation with realistic latency and slippage before routing live orders.
