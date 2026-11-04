# Supervised contrastive chunk embeddings

## Design

A pinned DistilRoBERTa chunk encoder feeds a 256-dimensional normalized projection. Each
training batch samples categories, then an anchor, a same-category positive, and a lexical
TF-IDF nearest chunk that lacks that category. The multi-label supervised-contrastive loss
treats any pair sharing an annotated category as positive; sparse similarity increases the
weight of hard negatives in the denominator. TF-IDF is fitted on training chunks only.

At inference, each category is represented by positive and negative centroids computed
from training embeddings. A chunk's score is the sigmoid of its cosine-similarity margin
to those two centroids. No calibration/evaluation examples contribute to prototypes.

## Seed-42 results

| Model | Evaluation macro-F1 | Evaluation macro-AUPR |
|---|---:|---:|
| Word + character TF-IDF | 0.533 | 0.597 |
| Contrastive encoder + prototypes | 0.365 | 0.372 |

Checkpoint selection chose epoch 1 by calibration macro-AUPR (0.345); calibration
macro-AUPR then fell to 0.343 and 0.342 in epochs 2 and 3. Prototype ranking recovered
0.950 of positive chunk/category pairs within the top 16 chunks per contract/category,
but its calibrated full classification scores were weak. This candidate does not beat the
sparse baseline in this screening run and is not a replacement candidate.

The objective and prototype temperature have not been broadly tuned, and this is one
split only. The held-out evaluation is reported once for this configuration; further
architecture choices should be selected using training/calibration data or a new split,
not by repeatedly optimizing against seed 42 evaluation.

## Reproduction and artifacts

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.train_contrastive
```

Encoder checkpoints and prototypes are stored under ignored
`outputs/contrastive_seed42/`. Reports, per-category metrics, calibration thresholds,
and raw scores are in `results/contrastive_seed42/`. Use
`--evaluate-checkpoint outputs/contrastive_seed42/best` to rerun inference without
training. Only the CUAD training file is used; `data/cuad/test.json` is never loaded.
