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

## Seed-42 results

The contract-level split contains 260 training, 66 calibration, and 82 evaluation
contracts (9,025 / 1,899 / 2,196 chunks). The reranker trained for three epochs on
21,204 pairs (5,301 positives and 15,903 retrieved hard negatives); checkpoint selection
used calibration macro-AUPR.

| Model | Evaluation macro-F1 | Evaluation macro-AUPR |
|---|---:|---:|
| Word + character TF-IDF retriever | 0.533 | 0.597 |
| Retrieval + cross-encoder reranker | **0.557** | 0.595 |

Evaluation candidate recall was 0.976 over positive chunk/category pairs and 0.994 over
positive contract/category cases at K=16. Recall at K=1/4/8/16 was 0.599/0.879/0.946/0.976
for positive chunk/category pairs; contract/category recall was 0.865/0.969/0.990/0.994.
The evaluation reranked 40,918 candidate pairs.

This is a promising single-seed macro-F1 gain (+0.024), while macro-AUPR is slightly
lower (-0.002). It is not evidence of a robust improvement yet: run more predeclared
contract split seeds before promoting it. Candidate scoring is also substantially more
expensive than applying the sparse model alone. No throughput or production-latency claim
is made.

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
