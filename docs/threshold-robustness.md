# Threshold robustness: can better cutoffs rescue rare categories?

This follows up the error analysis (issue #18), which found that rare categories lose about 0.20
F1 between the best possible cutoff and the cutoff fit on calibration contracts. Here, four
alternatives to the current method are compared on the two leading models.

**Short answer:** there is no universal fix. For the word+char TF-IDF control, the current
per-category F1 cutoff is already the best of the five methods, and every alternative lowers
macro-F1. For the reranker, shrinking cutoffs toward a pooled value raises macro-F1 by 0.009
and beats the current method on all 5 seeds. That gain is small and comes from previously
inspected evaluation contracts, so it is a candidate for the holdout gate, not a result.

**Reproduce:** `python -m scripts.compare_thresholds`. It reads the saved scores in
`results/retrieval_reranker_seed{7,17,23,42,101}/` and retrains nothing. Outputs go to
`results/threshold_robustness/`.

## Protocol

- Every strategy is fit on **calibration contracts only** and scored on the held-out evaluation contracts of the CUAD training file. `data/cuad/test.json` was not read.
- The settings were fixed in `src/threshold_strategies.py` before any evaluation score was computed: shrinkage prior 10, rare cut-off of 10 calibration positives, recall floor 0.5. None were tuned.
- The current method reproduces the saved `thresholds.json` exactly (maximum difference 0.0 across 5 seeds and both models).
- Rare / uncommon / common buckets here use **calibration** positives (<10, 10–29, 30+), since that is what each strategy reacts to. That differs from the error analysis, which bucketed by evaluation support.
- These evaluation contracts were already used to compare models, so picking a strategy on them is a form of selection. A winner still has to pass [upgrade-validation-gate.md](upgrade-validation-gate.md).

## Strategies

| Strategy | Rule (all on calibration data) |
|---|---|
| `per_category_f1` (current) | Each category's F1-optimal cutoff; 0.5 if it has no positives |
| `global` | One cutoff for every category, maximizing pooled (micro) F1 |
| `shrunk` | `w · own + (1 − w) · pooled`, with `w = n / (n + 10)` and `n` = calibration positives |
| `rare_fallback` | Own cutoff, but the pooled one if a category has fewer than 10 positives |
| `recall_floor` | Own cutoff, lowered only if it catches under 50% of calibration positives |

## Results (mean over 5 seeds)

| Model | Strategy | Macro-F1 | Δ vs. current | Δ rare F1 | Δ common F1 | Seeds beating current | Worst seed Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | current | **0.505** | — | — | — | — | — |
| TF-IDF | recall_floor | 0.500 | −0.004 | −0.016 | −0.001 | 2 / 5 | −0.015 |
| TF-IDF | shrunk | 0.493 | −0.011 | −0.082 | +0.005 | 0 / 5 | −0.016 |
| TF-IDF | rare_fallback | 0.487 | −0.017 | −0.096 | 0.000 | 2 / 5 | −0.042 |
| TF-IDF | global | 0.478 | −0.027 | −0.096 | +0.004 | 0 / 5 | −0.064 |
| Reranker | shrunk | **0.529** | **+0.009** | **+0.042** | +0.001 | **5 / 5** | +0.002 |
| Reranker | rare_fallback | 0.525 | +0.004 | +0.024 | 0.000 | 4 / 5 | −0.002 |
| Reranker | current | 0.520 | — | — | — | — | — |
| Reranker | recall_floor | 0.518 | −0.002 | +0.005 | 0.000 | 3 / 5 | −0.022 |
| Reranker | global | 0.509 | −0.012 | +0.024 | −0.022 | 1 / 5 | −0.028 |

Macro-AUPR does not change, because thresholds do not change how chunks are ranked.

## Why the two models react differently

The pooled-cutoff strategies assume a score means the same thing in every category. That holds
for the reranker but not for TF-IDF:

- **TF-IDF** trains a separate classifier per category, so score scales differ widely. On seed 42, the median calibration score of true positives ranges from 0.02 in one category to 0.97 in another. A pooled cutoff of about 0.62 is far too high for low-scoring rare categories and wipes out their recall. Under shrinkage, Most Favored Nation falls to 3% recall and loses 0.18 F1.
- **The reranker** pushes almost every positive close to 1.0 (median positive score 0.99), so scores are comparable across categories. Pulling a noisy rare-category cutoff toward the pooled one helps: No-Solicit of Customers gains 0.12 F1, Unlimited/All-You-Can-Eat License 0.10 and Warranty Duration 0.05.

The **recall floor** helps neither model. It raises recall but adds more false positives than it
removes misses. For example, TF-IDF No-Solicit of Customers precision drops to 7%.

## What this means for the error analysis

The 0.20 "threshold gap" for rare categories was measured against a cutoff chosen on the
evaluation contracts themselves, so it is an optimistic ceiling. These results suggest that for
TF-IDF most of it is not recoverable from 4–9 calibration positives with a simple rule. The noise
is in the data, not in the method.

## Recommendation

1. **Keep the current method for TF-IDF.** None of the alternatives help it.
2. **Treat reranker + shrinkage as one candidate** for the predeclared gate, alongside the existing score-fusion candidate. The pre-registration should name it: "reranker with shrinkage prior 10, fit on calibration contracts." It must not be re-tuned on the holdout.
3. **Do not expect thresholds to close the rare-category gap.** Rare categories need more labeled examples, or should be reported as "insufficient support" with intervals in the final report.

Possible follow-ups that were **not** tried here: shrinking in logit space, or shrinking TF-IDF
cutoffs toward a per-category prior instead of a pooled one. Either must be predeclared before it
is run, so it is not chosen after seeing these results.

## Output files

| File | Contents |
|---|---|
| `per_seed.csv` | Macro-F1 overall and by calibration bucket, per model × strategy × seed |
| `summary.csv` | Means over seeds, change vs. current, seeds beating current, worst-seed change |
| `per_category.csv` | Mean F1 / precision / recall per category and strategy, with change vs. current |
| `summary.json` | The summary table and protocol notes |
