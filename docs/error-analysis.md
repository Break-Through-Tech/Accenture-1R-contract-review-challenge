# Error analysis: word + char TF-IDF control vs. retrieval reranker

This covers the October milestone item "error analysis on where the model struggles" (issue #18).
It answers four questions: which categories fail, why they fail, whether the two leading models
fail in the same places, and how far per-category numbers can be trusted.

**Reproduce:** `python -m scripts.analyze_errors`. The script reads only the saved scores and
thresholds in `results/retrieval_reranker_seed{7,17,23,42,101}/` and retrains nothing. Outputs
go to `results/error_analysis/`.

**Protocol:** Five seeded contract-level splits of the training file, each with 82 evaluation
contracts. Thresholds were fit on calibration contracts only. Every chunk-level count below is
summed or averaged over the five seeds. The retriever stage of the reranker run is the word+char
TF-IDF control, scored on the same chunks with the same thresholds. Its macro numbers match
`results/retrieval_reranker_multiseed/per_seed.csv` exactly. `data/cuad/test.json` was not read.
The seeded evaluation sets overlap, so the five seeds are not independent samples.

## Bug found and fixed first: per-category AUPR was assigned to the wrong rows

`compute_base_metrics` returns rows **sorted by support**. Four scripts then assigned
`compute_aupr(...)` positionally, but that array is in `CATEGORIES` (alphabetical) order:
`evaluate_architecture_suite.py`, `train_long_context.py`, `train_category_query.py` and
`evaluate_encoder_nested.py`. As a result, almost every per-category AUPR in those reports
belonged to another category. For example, word+char TF-IDF seed 42 listed License Grant at
AUPR 0.049 when the real value is 0.870.

- **Macro AUPR was not affected,** because a mean does not depend on row order. The README tables are still correct.
- **Fix:** `clausetriage.metrics.add_aupr_column` maps AUPR by category name. All four scripts now use it. Regression tests are in `tests/test_error_analysis.py`.
- **Saved tables repaired** from the saved `.npz` scores by `python -m scripts.repair_per_category_aupr`: `architecture_suite_seed{7,17,23,42}`, the multi-seed `per_category_metrics_by_seed.csv`, `category_query_seed42` and `long_context_transformer_seed42`. Only the `aupr` column changed. The script is idempotent.
- The other experiment scripts (reranker, graph, contrastive, set prediction, sparse long context) already mapped AUPR by name and were correct.

## Headline findings

| | Word+char TF-IDF | Reranker |
|---|---:|---:|
| Mean macro-F1 (5 seeds) | 0.505 | 0.520 |
| Mean macro-AUPR (5 seeds) | 0.563 | 0.557 |
| Chunk-level recall (all positives) | 62.7% | 64.6% |
| False positives (5 seeds) | 3,017 | 3,293 |
| Share of errors that are misses (FN) | 48% | 45% |
| False positives on chunks with **no** gold label | 39% | 34% |

### 1. Macro-F1 hides a two-tier model

| Support bucket (avg. eval positives) | Categories | TF-IDF F1 | Reranker F1 | Avg. F1 std across seeds |
|---|---:|---:|---:|---:|
| Common (30+) | 19 | 0.65 | 0.68 | ~0.05 |
| Uncommon (10–29) | 13 | 0.48 | 0.47 | ~0.11 |
| Rare (<10) | 9 | 0.24 | 0.29 | ~0.14–0.16 |

The common categories (Governing Law, Parties, Insurance, Anti-Assignment, License Grant) are
already usable. The rare categories (Affiliate License-Licensor, Unlimited/All-You-Can-Eat
License, Non-Disparagement, Third Party Beneficiary, No-Solicit of Customers, Price
Restrictions, Most Favored Nation, Source Code Escrow; No-Solicit of Employees is the
exception at 0.75) pull the macro average down. Some of them, such as Most Favored Nation, are
exactly the clauses that matter for risk scoring.

### 2. Rare categories lose F1 to threshold transfer, not just to ranking

`threshold_gap` is the best F1 any cutoff could reach on the evaluation contracts minus the F1
at the calibration-fit cutoff. It is an upper bound, not a result. For TF-IDF it averages
**0.20 on rare categories** and 0.04 on common ones. With only 4–9 calibration positives, the
F1-optimal cutoff is fit on a handful of chunks and does not transfer.

The clearest case is **Source Code Escrow**: AUPR 0.70 but F1 0.29. The scores rank well, but
the threshold is set so high that it produces ~0.2 false positives and ~4.6 misses per seed.
It also has zero evaluation positives in 2 of 5 seeds.

*Implication:* cheap, low-risk fixes are available before any new model. For example, shrink
rare-category thresholds toward a pooled cutoff, cap thresholds at a recall floor, or tune on
cross-fitted calibration folds. Any of these must be chosen on calibration data only.

### 3. Most false positives are confusions with a related clause, not boilerplate

61–66% of false positives land on chunks that carry a *different* gold label
(`results/error_analysis/false_positive_sources.csv`). The largest patterns:

| Predicted | Chunk's actual label | Model | Count | Share of that category's FPs |
|---|---|---|---:|---:|
| Affiliate License-Licensor | License Grant | Reranker | 138 | 63% |
| Exclusivity | License Grant | TF-IDF | 121 | 60% |
| Uncapped Liability | Cap On Liability | Reranker | 95 | 83% |
| Effective Date | Parties | TF-IDF | 86 | 69% |
| Affiliate License-Licensee | License Grant | Reranker | 78 | 72% |
| Termination For Convenience | Expiration Date | TF-IDF | 66 | 50% |

There are three distinct causes:
- **Sibling clauses** (license family; Cap vs. Uncapped Liability; termination vs. expiration). The models learn the topic but not the distinguishing condition: *affiliate*, *uncapped*, *for convenience*. Uncapped vs. Cap On Liability is the most important one for risk scoring, because confusing them flips the risk direction.
- **Front-matter co-location** (dates vs. Parties/Document Name). The first 512-token chunk holds all of these, and the date label is only sometimes on that chunk. Some of these "errors" are a chunk-labeling artifact rather than a model mistake, and they would mostly disappear under contract-level scoring.
- **Possible annotation gaps.** CUAD labels spans, not whole chunks, so a chunk flagged as Exclusivity inside a License Grant section may contain exclusivity language that annotators did not mark. This needs a manual read of examples before treating it as model error (see Next steps).

### 4. The two models miss different positives

Out of 7,779 evaluation positives (5 seeds), TF-IDF alone caught 796 and the reranker alone
caught 942. Recall if either model counts is **74.8%**, versus 62.7% and 64.6% individually.
The per-category pattern is strong:

- **Reranker much better:** Third Party Beneficiary (+0.28 F1), Non-Disparagement (+0.27), Source Code Escrow (+0.23), Irrevocable or Perpetual License (+0.20), Non-Compete (+0.13).
- **TF-IDF much better:** Most Favored Nation (−0.17), No-Solicit of Employees (−0.16), Warranty Duration (−0.12), Uncapped Liability (−0.11).

This explains why the existing score-fusion prototype helped. It also suggests choosing the
model per category on calibration data. Both still need the untouched-holdout gate in
[upgrade-validation-gate.md](upgrade-validation-gate.md) before any claim is made.

### 5. Per-category numbers for rare categories are not reliable at this size

For rare categories, F1 swings by 0.3–0.4 between seeds. Third Party Beneficiary ranges from
0.00 to 0.43 for TF-IDF, and Price Restrictions from 0.00 to 0.43. Any per-category comparison
on those rows needs a contract-bootstrap interval, or should be reported as "insufficient
support". The F1 spread per category is in `category_summary.csv` (`std_f1`, `min_f1`, `max_f1`).

## Next steps (proposed, for team discussion)

1. **Threshold robustness** (cheapest win): compare shrinkage and recall-floor thresholds against the current F1-optimal ones, using calibration data only.
2. **Read examples:** sample ~10 false positives each for the top four confusion pairs and label them "model error / annotation gap / co-location". This needs chunk text, so run it locally where the DistilRoBERTa tokenizer is cached. The saved `.npz` files do not store chunk IDs, so saving `contract_id`, `start` and `end` alongside scores in future runs would make this easy.
3. **Contract-level evaluation** (#24): score "does the contract contain category X". This is closer to what a reviewer needs, and it removes the front-matter co-location noise.
4. **Report rare categories with intervals,** or mark them "insufficient support" in the final report.

## Output files

| File | Contents |
|---|---|
| `category_errors_by_seed.csv` | Per model × seed × category: TP/FP/FN, P/R/F1, AUPR, threshold, oracle F1, threshold gap, FPs on unlabeled chunks, calibration support |
| `category_summary.csv` | The same averaged over seeds, plus F1 std/min/max and support bucket |
| `bucket_summary.csv` | Mean F1, AUPR and threshold gap by support bucket |
| `false_positive_sources.csv` | For each predicted category, the gold labels its false-positive chunks carried |
| `model_disagreement.csv` | Per category: positives caught by both, only TF-IDF, only the reranker, or neither, and false alarms by model |
| `summary.json` | Headline numbers quoted above |
