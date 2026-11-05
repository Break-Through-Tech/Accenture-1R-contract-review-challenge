# TF-IDF plus cross-encoder score fusion

## Rationale

The sparse word/character model remains the primary classifier. The reranker contributes
context-sensitive scores on a high-recall candidate set, but replacing sparse scores with
reranker scores reduced AUPR on average across five seeds. This experiment blends the two
signals instead: sparse scores remain available for every chunk/category, and reranker
evidence only adjusts scores for the candidates it scored.

For sparse probability `p_s`, reranker probability `p_r`, and global weight `alpha`, the
fused logit for a candidate is

`logit(p_fused) = (1 - alpha) * logit(p_s) + alpha * logit(p_r)`.

For non-candidates, `p_fused = p_s`. This preserves sparse fallback when retrieval misses
a positive. A fixed grid `[0, .1, .2, .35, .5, .75, 1]` is evaluated on calibration
macro-AUPR, with ties favoring the smaller alpha. Per-category F1 thresholds are then fit
on the same calibration partition. The exact candidate mask is recovered from nonzero
reranker scores and checked against the saved candidate-pair count.

## Exploratory results

| Seed | Calibration-selected alpha | TF-IDF F1 | Fused F1 | Delta | TF-IDF AUPR | Fused AUPR | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.35 | 0.5331 | 0.5879 | +0.0548 | 0.5971 | 0.6538 | +0.0567 |
| 7 | 0.20 | 0.5149 | 0.5634 | +0.0484 | 0.5854 | 0.6345 | +0.0490 |
| 17 | 0.75 | 0.4969 | 0.5646 | +0.0677 | 0.5631 | 0.6191 | +0.0560 |
| 23 | 0.20 | 0.5053 | 0.5593 | +0.0540 | 0.5399 | 0.5845 | +0.0446 |
| 101 | 0.20 | 0.4725 | 0.5115 | +0.0390 | 0.5312 | 0.5850 | +0.0538 |
| Mean | | 0.5046 | 0.5573 | +0.0528 | 0.5633 | 0.6154 | +0.0520 |

On these five splits, the fusion beats the TF-IDF score on both macro metrics for all
seeds. Paired contract-bootstrap 95% intervals are positive on all five splits. About
37% to 45% of evaluation chunk/category pairs receive reranker evidence; all other scores
fall back exactly to TF-IDF. Per-category behavior remains uneven, particularly for rare
categories. See `results/tfidf_reranker_fusion/` for the alpha grid, category deltas,
per-seed results, and bootstrap output.

## Interpretation and limitations

This result is **exploratory, not confirmatory**. The five split evaluations were already
inspected in previous reranker/architecture experiments, so reuse introduces selection
and test-set peeking risk. Alpha and thresholds also share the calibration partition,
which may overfit its sparse categories. These gains do not establish a robust or
production-quality improvement. Do not promote this model based on these numbers.

A stronger follow-up needs an untouched, approved evaluation set. The preferred
implementation would fit blend/stacking parameters using training-only out-of-fold
predictions, fit final per-category thresholds on calibration only, then evaluate once on
that new holdout. Keep the current TF-IDF model as the control and retain raw sparse
scores for auditability and fallback. `data/cuad/test.json` remains unused.

## Reproduction

This analysis consumes saved predictions and does not train or access test data:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.evaluate_tfidf_reranker_fusion --bootstrap-replicates 1000
```
