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

These results are **exploratory, not confirmatory**. The five split evaluations were
already inspected in previous reranker/architecture experiments, so reuse introduces
selection bias in the final score. The follow-up below separates blend-weight selection
from threshold calibration, but it cannot undo earlier inspection of the evaluation
contracts. These gains do not establish a robust or production-quality improvement.

## Training-only out-of-fold alpha selection

A follow-up selected the global blend weight from out-of-fold predictions on only the
260 seed-42 training contracts. Five fold-specific rerankers each trained on 208 of those
contracts; the other 52 received out-of-fold predictions. The separate 66-contract
calibration partition was used for checkpoint selection in each fold and for final
per-category threshold fitting. The original 82-contract evaluation partition was not
used to choose alpha. All folds use the same predeclared alpha grid and retain TF-IDF
scores off-candidate.

| Alpha | OOF training macro-AUPR |
|---:|---:|
| 0.0 | 0.5338 |
| 0.1 | 0.5731 |
| 0.2 | **0.5843** |
| 0.35 | 0.5817 |
| 0.5 | 0.5725 |
| 0.75 | 0.5469 |
| 1.0 | 0.4704 |

With alpha fixed to 0.2 from OOF training predictions, the seed-42 evaluation comparison
was TF-IDF macro-F1 0.5331 vs. fusion 0.5752, and macro-AUPR 0.5971 vs. 0.6544. Paired
contract-bootstrap 95% intervals for the deltas were [0.0136, 0.0629] for macro-F1 and
[0.0315, 0.0795] for macro-AUPR. These intervals are conditional on the fitted models
and calibration thresholds, and the evaluation contracts were inspected in prior work.
This is a cleaner parameter-selection procedure, not a fresh independent confirmation.
Some rare categories still regress. Full fold manifests, predictions, reports, and
per-category results are under `results/reranker_oof_seed42/` and
`results/reranker_oof_fusion_seed42/`.

To reproduce the cross-fit, train, and analyze stages (GPU training required):

```bash
.venv/bin/python -m scripts.crossfit_reranker_fusion prepare
.venv/bin/python -m scripts.crossfit_reranker_fusion train
.venv/bin/python -m scripts.crossfit_reranker_fusion analyze
```

For a stronger independent confirmation, the team/advisor must approve an untouched
evaluation source and the protocol before it is opened. Keep TF-IDF as the control and
retain raw sparse scores for auditability and fallback. `data/cuad/test.json` remains
unused.

## Reproduction of the original calibration-selected experiment

This older analysis consumes saved predictions and does not train or access test data:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.evaluate_tfidf_reranker_fusion --bootstrap-replicates 1000
```
