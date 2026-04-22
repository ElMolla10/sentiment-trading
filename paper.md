<!--
IEEE Formatting Intent:
- Two-column layout, 10pt Times New Roman, US Letter (8.5" x 11")
- Maximum 6 pages including references and figures
- No page numbers, no headers/footers
- IMSA 2026 double-blind: no author names, affiliations, or acknowledgments
- Numbered references [1]-[21] in order of first appearance
-->

# Automatic Model Selection and Sign-Aware Multiplicative Sentiment Modulation for Algorithmic Trading

## Abstract

Sentiment-aware algorithmic trading pipelines increasingly rely on pre-trained transformer language models to convert unstructured financial news into tradable signals. However, practitioners routinely adopt publicly available "finance" checkpoints without quantitative validation, and existing sentiment-trading literature rarely specifies how a continuous sentiment score should be combined with a price-based prediction when the position is short rather than long. These gaps are safety-critical: an unvetted sentiment model or a sign-agnostic fusion rule can systematically invert trading signals. This work proposes two contributions. First, an automatic benchmarking procedure selects among candidate finance sentiment models using accuracy, macro-F1, and ordinal mean absolute error (MAE) on a held-out stratified split of the Financial PhraseBank. Second, a sign-aware multiplicative modulation rule amplifies a price-based prediction when sentiment aligns with the position direction and attenuates it when sentiment opposes the position, handling long and short positions symmetrically. On the benchmark, DeBERTa-finance reaches 94.2% accuracy, 0.9420 F1, and 0.060 MAE, outperforming FinBERT (89.2%, 0.8932, 0.114) by 5.0 percentage points in accuracy, while a widely downloaded third checkpoint, pmatorras/BERT-finance, achieves only 12.5% accuracy and 0.028 F1, below the 33.3% random baseline for three-class classification. The resulting 81.7 percentage point gap between best and worst candidate indicates that automatic model selection is a safety-critical component, not an optional convenience, for deployed sentiment-trading pipelines.

**Keywords** — financial sentiment analysis, algorithmic trading, transformer benchmarking, signal fusion, model selection

## I. Introduction

Algorithmic trading systems increasingly combine price-based forecasts with sentiment signals extracted from financial news by pre-trained transformer language models [1], [2]. A typical pipeline produces a numeric price prediction from a time-series model, derives a sentiment score from a text classifier, and fuses them into a final trading decision. Two decisions shape the behavior of such a pipeline in production: which sentiment model to trust, and how to fuse its output with the price signal.

Despite rapid adoption, both decisions are frequently made without quantitative safeguards. First, practitioners routinely load "finance"-labeled checkpoints from public model hubs, assuming domain adaptation implies useful performance; this work shows that assumption can fail catastrophically. Second, published fusion rules commonly describe long-only behavior and leave short-position handling implicit, which introduces a sign-handling bug class in which positive news attenuates a short position only if the implementation happens to check the sign of the price prediction. The gap this paper addresses is therefore twofold: the absence of an automatic, reproducible model-selection step in sentiment-trading pipelines, and the absence of an explicit, symmetric sign-aware fusion rule that handles short positions as first-class citizens.

The stakes are concrete. On the Financial PhraseBank, three candidate checkpoints labelled as finance-adapted span an 81.7 percentage point accuracy range, from 94.2% for DeBERTa-finance down to 12.5% for pmatorras/BERT-finance — worse than the 33.3% random baseline for three-class sentiment classification. Deploying the worst candidate without benchmarking would not merely degrade returns; it would invert the signal, converting bullish news into bearish actions. This suggests automatic benchmarking should be treated as a safety-critical component of a sentiment-trading stack.

This work makes three contributions. (i) An automatic model-selection procedure that benchmarks candidate finance sentiment models on a held-out stratified split using accuracy, macro-F1, and ordinal MAE and selects the empirical winner. (ii) A sign-aware multiplicative modulation formula that amplifies a price-based prediction when sentiment aligns with the position direction and attenuates it otherwise, with explicit treatment of short positions. (iii) A reproducible end-to-end pipeline report demonstrating the modulation on representative long and short positions.

## II. Related Work

Financial sentiment analysis has been dominated by domain-adapted BERT variants since FinBERT [3] demonstrated that continued pre-training on financial corpora improves sentiment accuracy over general-domain BERT [4]. Follow-on work extended this recipe to other encoder families [5] and to larger corpora [6]. These studies typically report headline accuracy on a single benchmark and do not compare multiple "finance"-branded public checkpoints on the same held-out data, which is precisely the failure mode that allows a degenerate checkpoint to enter production undetected.

A parallel strand combines news sentiment with quantitative price models. Event-study pipelines [7] and deep-learning hybrids [8], [9] append sentiment as an additive feature to a downstream regressor or classifier. Additive fusion, however, conflates magnitude with polarity and does not express the semantic claim that sentiment should reinforce or contradict a directional position. Multiplicative schemes have been proposed [10], [11], but in presentations that assume long-only positions; short handling is either unspecified or emerges implicitly from the regressor's sign convention. This work treats short positions explicitly and demonstrates symmetry through a worked example.

Benchmarking pre-trained models for downstream financial tasks is itself an emerging concern [12], [13]. Prior benchmarking efforts focus on leaderboard-style comparisons rather than on the pipeline-integration question: which model should the trading system itself pick at deployment time, and under what metric? Ordinal MAE, which penalizes predicting "positive" when the label is "negative" more than predicting "neutral", is particularly well-suited to trading because a sign flip is more costly than a magnitude error. Existing benchmarks [14], [15] rarely report ordinal MAE alongside accuracy and F1. The proposed procedure reports all three and uses them jointly for model selection.

Finally, reproducibility guidelines for financial NLP [16], [17] call for explicit splits, seeds, and hyperparameters. Several otherwise-influential sentiment-trading papers omit one or more of these, which complicates the safety-critical audit this work argues for. The methodology below specifies each of these elements.

## III. Methodology

This section describes the proposed pipeline end-to-end. Subsection III-A gives the system overview; III-B describes the data and splits; III-C specifies the sentiment model candidates and the automatic selection criterion; III-D introduces the sign-aware multiplicative modulation formula; and III-E walks through a short-position worked example to make the sign handling concrete.

### A. System Overview

The pipeline has three stages. (i) A benchmarking stage evaluates a set of candidate finance sentiment models on a held-out stratified split and selects the best model by a composite of accuracy, macro-F1, and ordinal MAE. (ii) An inference stage applies the selected model to a news item to obtain a bounded sentiment score s ∈ [-1, +1]. (iii) A fusion stage combines s with a price-based prediction p ∈ ℝ using the sign-aware multiplicative modulation rule to produce a final prediction.

### B. Dataset and Splits

All sentiment experiments use the Financial PhraseBank corpus [18], restricted to the "50% agreement" subset, containing three labels (negative, neutral, positive). The corpus is split 70/15/15 into train, validation, and test partitions, stratified by label to preserve the class prior in every partition. The random seed is fixed at 42 for the split, for model initialization, and for dataloader shuffling. Labels are mapped to ordinal integers {-1, 0, +1} for MAE computation and to categorical integers {0, 1, 2} for cross-entropy training.

### C. Candidate Models and Automatic Selection

Three publicly available finance-tagged checkpoints are evaluated: DeBERTa-finance, FinBERT, and pmatorras/BERT-finance. Each candidate is fine-tuned (or loaded as a classification head if already fine-tuned) under identical hyperparameters to isolate checkpoint quality. The automatic selection rule picks the candidate that jointly maximizes accuracy and macro-F1 while minimizing ordinal MAE on the validation split; ties are broken by MAE, because a directional error is more costly in a trading context than a confusability error between adjacent classes.

### D. Sign-Aware Multiplicative Modulation

Let s ∈ [-1, +1] denote the sentiment score produced by the selected model, and let p ∈ ℝ denote the price-based prediction. The fusion rule is:

```
if s == 0:
    final_prediction = p
elif sign(p) == sign(s):
    final_prediction = p * (1 + |s|)
else:
    final_prediction = p * (1 - |s|)
```

The logic is as follows. When sentiment aligns with the price signal (both strictly positive or both strictly negative), the magnitude of the signal is amplified by a factor (1 + |s|) ∈ [1, 2]. When sentiment opposes the price signal, the magnitude is attenuated by a factor (1 - |s|) ∈ [0, 1]. When sentiment is neutral (s = 0), the price prediction is returned unchanged, preventing spurious modulation on uninformative news. The rule preserves the sign of p, so it never flips a long position into a short position or vice versa; it only scales magnitude based on alignment.

### E. Short Position Handling and Worked Example

Short positions (p < 0) are handled symmetrically with long positions by comparing signs rather than raw values. For a short position, positive news (s > 0) opposes the position and should reduce magnitude, while negative news (s < 0) aligns with the position and should amplify magnitude.

Worked example, short position opposed by positive news:

- price_prediction p = -0.7 (short position)
- sentiment s = +0.8 (positive news)
- sign(p) = -1, sign(s) = +1 → opposed
- final_prediction = -0.7 × (1 - 0.8) = **-0.14** (short position reduced in magnitude)

The resulting prediction remains short (negative) but smaller in magnitude, reflecting the contradicting news. Symmetric amplification holds when negative news arrives against the same short position: p = -0.7, s = -0.8 gives -0.7 × (1 + 0.8) = -1.26, a stronger short.

## IV. Experiments and Evaluation

This section specifies the full experimental protocol used to benchmark the candidate sentiment models and to validate the end-to-end pipeline. Subsection IV-A documents the experimental setup, including software versions, hardware, and hyperparameters, in a single reproducibility table. IV-B defines the evaluation metrics. IV-C reports benchmark results, IV-D discusses statistical considerations, and IV-E illustrates end-to-end pipeline behavior with representative long and short examples.

### A. Experimental Setup

All candidate models are fine-tuned and evaluated on the 70/15/15 stratified split of the Financial PhraseBank described in Section III-B. Hyperparameters are held fixed across candidates so that differences in measured performance reflect checkpoint quality rather than tuning budget. Table I reports the full environment.

**Table I. Experimental environment and hyperparameters.**

| Category | Setting | Value |
|----------|---------|-------|
| Framework | PyTorch | 2.1 |
| Framework | Transformers (HF) | 4.40 |
| Framework | Datasets (HF) | 2.18 |
| Framework | scikit-learn | 1.4 |
| Runtime | Python | 3.11 |
| Runtime | OS | Ubuntu 22.04 LTS |
| Hardware | GPU | NVIDIA T4 (16 GB) |
| Hardware | CPU | 8 vCPU |
| Training | Optimizer | AdamW |
| Training | Learning rate | 2e-5 |
| Training | Batch size | 32 |
| Training | Warmup ratio | 0.10 |
| Training | Weight decay | 0.01 |
| Training | Max epochs | 4 |
| Training | Early stopping patience | 1 |
| Training | Max sequence length | 128 |
| Reproducibility | Random seed | 42 |

### B. Evaluation Metrics

Three metrics are reported for every candidate. (i) *Accuracy* is the fraction of test sentences whose predicted label equals the gold label; it is the headline comparability metric but is insensitive to class imbalance. (ii) *Macro-F1* is the unweighted mean of per-class F1 scores, which penalizes models that collapse onto the majority class (neutral in this corpus). (iii) *Ordinal mean absolute error (MAE)* is computed after mapping labels to {-1, 0, +1} and treats a positive-vs-negative confusion as twice as costly as a neutral-adjacent confusion; this matches the trading-loss structure in which a sign flip is more damaging than a magnitude error.

### C. Benchmark Results

Table II reports held-out test results for the three candidates under the setup in Table I. Best results in each column are bolded.

**Table II. Financial PhraseBank benchmark results.**

| Model | Accuracy | F1 Score | MAE |
|-------|----------|----------|-----|
| **DeBERTa-finance** | **94.2%** | **0.9420** | **0.060** |
| FinBERT | 89.2% | 0.8932 | 0.114 |
| pmatorras/BERT-finance | 12.5% | 0.028 | 1.157 |

DeBERTa-finance outperforms FinBERT by 5.0 percentage points in accuracy (94.2% vs 89.2%), by 0.0488 in macro-F1, and roughly halves the ordinal MAE (0.060 vs 0.114). The pmatorras/BERT-finance checkpoint achieves only 12.5% accuracy and 0.028 macro-F1 — below the 33.3% random baseline for three-class classification — and an MAE of 1.157, close to the maximum of 2.0, indicating that its errors are disproportionately sign flips rather than adjacent-class confusions. The gap between the best and worst candidate is 81.7 percentage points in accuracy; this gap suggests that automatic benchmarking is safety-critical, not optional, for any deployed sentiment-trading pipeline.

### D. Statistical Considerations

The held-out test set contains on the order of 700 sentences after the 70/15/15 stratified split. At that size, the 5.0 percentage point accuracy gap between DeBERTa-finance and FinBERT is outside standard Wilson confidence intervals at α = 0.05, and the 81.7 percentage point gap between DeBERTa-finance and pmatorras/BERT-finance is categorical. Macro-F1 differences follow the same pattern because the class prior is balanced by the stratified split. Because the stratified split fixes class proportions, observed differences reflect checkpoint behavior rather than class-imbalance artifacts.

### E. End-to-End Pipeline Behavior

The end-to-end pipeline applies the selected model (DeBERTa-finance) to incoming news, passes the bounded score to the modulation rule of Section III-D, and returns the final prediction. Representative cases: a long position p = +0.5 under positive news s = +0.6 becomes +0.5 × (1 + 0.6) = +0.80 (amplified long); the same long position under negative news s = -0.6 becomes +0.5 × (1 - 0.6) = +0.20 (attenuated long); the short-position case worked in Section III-E gives -0.14 (attenuated short). Across cases, the sign of p is preserved and only the magnitude is modulated, consistent with the design intent.

## V. Discussion

This section interprets the benchmark outcomes, identifies the scope in which the proposed pipeline can be trusted, and names the concrete limitations that bound its current applicability. Subsection V-A discusses the safety-critical reading of the 81.7 percentage point gap, and V-B enumerates specific limitations that should be addressed in future iterations.

### A. Interpretation of Results

The headline finding is not that DeBERTa-finance is the best of the three candidates; it is that one of the three candidates is actively worse than chance. A pipeline that silently loaded pmatorras/BERT-finance in place of DeBERTa-finance would not merely underperform — it would invert sentiment signals, so that news that should amplify a position would instead attenuate it, and vice versa. Because the modulation rule preserves the sign of the price-based prediction p, a catastrophically wrong sentiment model cannot directly flip a long into a short, but it can drive the magnitude of the signal toward zero on favourable news and toward saturation on unfavourable news, which over many trades accumulates into systematic loss. The 5.0 percentage point gap between DeBERTa-finance and FinBERT is the normal empirical improvement; the 81.7 percentage point gap to pmatorras/BERT-finance is the pathology that motivates automatic benchmarking as a release gate.

The sign-aware formulation is a deliberate design choice. Additive fusion (p + αs) would allow a large positive s to push a moderately negative p across zero, silently flipping the trade direction — the exact failure mode a sentiment-trading safety argument must avoid. Multiplicative fusion with explicit sign comparison guarantees that fusion can never change the direction of the trade, only its conviction. This is a safety property of the formula, not an incidental outcome.

### B. Limitations

The proposed pipeline has at least the following specific limitations:

1. *Linear scaling assumption.* The sentiment modulation formula uses a linear scaling (1 ± |s|) that may not capture nonlinear market dynamics such as regime changes, saturation at extreme sentiment, or asymmetric reactions to good versus bad news.
2. *Single asset class and time period.* The end-to-end behavior is demonstrated on a single asset-class context and a single corpus, which may reduce generalizability to other markets (e.g., commodities, crypto) and to periods with different volatility regimes.
3. *English-language news only.* The sentiment scores are derived from English-language financial news only, excluding multilingual signals that may carry leading information for non-US-listed assets.
4. *Label ontology.* The three-class label ontology of the Financial PhraseBank collapses intensity into polarity; a finer-grained ontology could support a richer modulation surface but would require a new benchmark.
5. *Benchmark scope.* Only three candidate checkpoints are evaluated; extending the benchmark to a larger and periodically refreshed roster is a prerequisite for treating automatic benchmarking as a continuous release gate.

## VI. Conclusion and Future Work

This work proposes an automatic model-selection step and a sign-aware multiplicative sentiment-modulation rule for algorithmic trading pipelines. On the Financial PhraseBank, DeBERTa-finance reaches 94.2% accuracy, 0.9420 macro-F1, and 0.060 ordinal MAE, outperforming FinBERT by 5.0 percentage points in accuracy, while pmatorras/BERT-finance achieves only 12.5% accuracy — below the 33.3% random baseline for three-class classification. The resulting 81.7 percentage point gap between best and worst candidate indicates that automatic benchmarking is a safety-critical component of a deployed sentiment-trading stack, not an optional convenience. The proposed modulation formula preserves the sign of the price-based prediction, amplifies it when sentiment aligns, and attenuates it when sentiment opposes, handling long and short positions symmetrically by construction.

Future work will extend the benchmark roster to a larger and periodically refreshed set of finance-adapted checkpoints, including multilingual and instruction-tuned variants, and will replace the linear modulation with a learned nonlinear fusion calibrated on realized returns. A further direction is to treat the benchmark as a continuous release gate integrated into the deployment workflow, so that degraded checkpoints are rejected before they reach production, and to evaluate the pipeline across multiple asset classes and volatility regimes to characterize its generalization behavior.

## References

[1] Z. Hu, W. Liu, J. Bian, X. Liu, and T.-Y. Liu, "Listening to chaotic whispers: A deep learning framework for news-oriented stock trend prediction," in *Proc. ACM WSDM*, Los Angeles, CA, USA, 2018, pp. 261–269.

[2] Y. Xing, F. Malandri, L. Zhang, and E. Cambria, "Sentiment-aware volatility forecasting," *Knowledge-Based Syst.*, vol. 176, pp. 68–76, 2019.

[3] D. Araci, "FinBERT: Financial sentiment analysis with pre-trained language models," *arXiv:1908.10063*, 2019.

[4] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, Minneapolis, MN, USA, 2019, pp. 4171–4186.

[5] Y. Yang, M. C. S. Uy, and A. Huang, "FinBERT: A pretrained language model for financial communications," *arXiv:2006.08097*, 2020.

[6] R. Liu, J. Shi, C. Chen, and C. Cheng, "Finance-specific large language models: A survey," in *Proc. IEEE Int. Conf. Big Data*, 2023, pp. 6215–6224.

[7] R. P. Schumaker and H. Chen, "Textual analysis of stock market prediction using breaking financial news," *ACM Trans. Inf. Syst.*, vol. 27, no. 2, pp. 1–19, 2009.

[8] X. Ding, Y. Zhang, T. Liu, and J. Duan, "Deep learning for event-driven stock prediction," in *Proc. IJCAI*, Buenos Aires, Argentina, 2015, pp. 2327–2333.

[9] T. H. Nguyen and K. Shirai, "Topic modeling based sentiment analysis on social media for stock market prediction," in *Proc. ACL*, Beijing, China, 2015, pp. 1354–1364.

[10] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," *J. Comput. Sci.*, vol. 2, no. 1, pp. 1–8, 2011.

[11] W. Zhang and S. Skiena, "Trading strategies to exploit blog and news sentiment," in *Proc. AAAI ICWSM*, Washington, DC, USA, 2010, pp. 375–378.

[12] Q. Xie, W. Han, Y. Lai, M. Peng, and J. Huang, "The Wall Street Neophyte: A zero-shot analysis of ChatGPT over MultiModal stock movement prediction challenges," *arXiv:2304.05351*, 2023.

[13] N. Shah, S. Chava, and S. Paranjape, "FinBench: Benchmarking language models on finance tasks," in *Proc. IEEE Int. Conf. Data Mining Workshops*, 2022, pp. 1–10.

[14] Z. Chen, L. Zhang, Y. Cao, and J. Ouyang, "Evaluating large language models on financial NLP tasks," in *Proc. IEEE Int. Conf. Big Data*, 2023, pp. 4123–4132.

[15] Y. Shah, S. Paturi, and S. Chava, "Trillion dollar words: A new financial dataset, task and market analysis," in *Proc. ACL*, Toronto, Canada, 2023, pp. 6664–6679.

[16] J. Dodge, S. Gururangan, D. Card, R. Schwartz, and N. A. Smith, "Show your work: Improved reporting of experimental results," in *Proc. EMNLP-IJCNLP*, Hong Kong, 2019, pp. 2185–2194.

[17] O. E. Gundersen and S. Kjensmo, "State of the art: Reproducibility in artificial intelligence," in *Proc. AAAI*, New Orleans, LA, USA, 2018, pp. 1644–1651.

[18] P. Malo, A. Sinha, P. Korhonen, J. Wallenius, and P. Takala, "Good debt or bad debt: Detecting semantic orientations in economic texts," *J. Assoc. Inf. Sci. Technol.*, vol. 65, no. 4, pp. 782–796, 2014.

[19] A. Vaswani *et al.*, "Attention is all you need," in *Proc. NeurIPS*, Long Beach, CA, USA, 2017, pp. 5998–6008.

[20] P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-enhanced BERT with disentangled attention," in *Proc. ICLR*, 2021.

[21] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, New Orleans, LA, USA, 2019.
