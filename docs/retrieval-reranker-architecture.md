# Sparse retrieval plus category-conditioned cross-encoder

## Design

A training-only word/character TF-IDF one-vs-rest model ranks chunks separately for
each contract/category. The system retains at most the top 16 chunks per pair. A
DistilRoBERTa binary cross-encoder then reads `(category name + description, chunk text)`
and reranks only those candidates. All positive training chunk/category pairs are used;
false-positive candidates from the sparse retriever supply hard negatives (up to three
per positive). Categories with no training positives are not proposed by the system.

The retrieval stage is annotation-blind at inference. Candidate recall is measured before
reranking because any missed answer is unrecoverable downstream. Candidate scores outside
the top-K are zero; thresholds are fit per category on calibration contracts only.

## Five-seed results

Five predeclared split seeds (42, 7, 17, 23, 101) use 260 training, 66 calibration, and
82 evaluation contracts each. Every split uses the same sentence-token chunker and pinned
tokenizer as its word/character control. Checkpoint selection and per-category thresholds
use calibration data only; the evaluation partition is used once for the reported metrics.

| Seed | TF-IDF macro-F1 | Reranker macro-F1 | Delta | TF-IDF macro-AUPR | Reranker macro-AUPR | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.5331 | **0.5572** | +0.0241 | **0.5971** | 0.5950 | -0.0021 |
| 7 | 0.5149 | **0.5301** | +0.0152 | **0.5854** | 0.5570 | -0.0285 |
| 17 | 0.4969 | **0.5230** | +0.0261 | 0.5631 | **0.5784** | +0.0153 |
| 23 | **0.5053** | 0.4938 | -0.0115 | **0.5399** | 0.5067 | -0.0332 |
| 101 | 0.4725 | **0.4972** | +0.0246 | 0.5312 | **0.5486** | +0.0174 |
| Mean | 0.5046 | **0.5203** | +0.0157 | **0.5633** | 0.5571 | -0.0062 |

F1 improves on four of five splits; AUPR improves on only two. Paired 1,000-replicate
bootstrap intervals resample evaluation contracts with replacement and keep calibration
thresholds fixed. Every per-seed 95% interval for both macro-F1 and macro-AUPR deltas
includes zero. These split evaluations overlap, so the five-seed means and win counts are
descriptive, not independent replications or a significance test. Macro-F1 uses a fixed
41-category macro, assigning zero to categories with no evaluation positives; macro-AUPR
averages categories with at least one positive.

The paired control uses the exact same saved retriever scores as the word/character
TF-IDF evaluation artifacts. When a category has no calibration positives, both systems
use a 0.5 threshold; otherwise thresholds are fit on the calibration partition only.
This makes the paired threshold policy explicit, including on sparse categories.

At K=16, candidate recall ranges from 0.966 to 0.979 for positive chunk/category pairs
and from 0.994 to 0.997 for positive contract/category cases. The system reranked about
40,900 to 45,000 candidate pairs per evaluation split. Category effects are heterogeneous:
mean F1 gains were large for Third Party Beneficiary (+0.277 across seeds, mean support
7.6) and Non-Disparagement (+0.266, support 6.8), while Most Favored Nation (-0.173,
support 5.4) and Warranty Duration (-0.115, support 23.2) declined. These per-category
results are exploratory and especially noisy for rare classes. Candidate scoring is much
more expensive than sparse inference; no production-latency claim is made.

**Conclusion:** the reranker has a consistent mean F1 advantage in this screening set,
but does not reliably beat the control across splits, and has lower mean AUPR. The #14
success criterion is therefore not established. Do not promote it over the TF-IDF control.

Full paired results, per-category deltas, reports, thresholds, and raw predictions are in
`results/retrieval_reranker_multiseed/` and `results/retrieval_reranker_seed{42,7,17,23,101}/`.
Reproduce the analysis (without retraining) with:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.analyze_reranker_multiseed --bootstrap-replicates 1000
```

## Reproduction and artifacts

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.train_retrieval_reranker
```

The script uses only `data/cuad/train_separate_questions.json`; it never loads
`data/cuad/test.json`. Checkpoints and the fitted retriever are written under ignored
`outputs/retrieval_reranker_seed42/`. Evaluation report, per-category metrics, calibration
thresholds, and raw calibration/evaluation predictions are in
`results/retrieval_reranker_seed42/`. Run an existing checkpoint with
`--evaluate-checkpoint outputs/retrieval_reranker_seed42/best`.
