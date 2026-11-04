# Contract chunk graph neural network

## Design

The model encodes each chunk with the pinned DistilRoBERTa encoder, then applies two
weighted GraphSAGE layers before the multi-label chunk head. Each contract is a separate
graph. Bidirectional edges link adjacent chunks; additional edges link each chunk to its
four highest-similarity non-adjacent chunks. Lexical cosine similarities come from a
word-bigram TF-IDF vectorizer fitted on training chunks only. Labels are never used to
construct graph topology. Inference needs the same contract's ordered chunks so that
local and nonlocal messages can be passed.

This is an initial screening run, not a tuned architecture study. It used three epochs,
256 graph hidden units, two message-passing layers, and the seed-42 nested split.

## Seed-42 results

| Model | Evaluation macro-F1 | Evaluation macro-AUPR |
|---|---:|---:|
| Word + character TF-IDF | 0.533 | 0.597 |
| Weighted GraphSAGE | 0.390 | 0.426 |

Calibration macro-AUPR rose from 0.214 to 0.416 over the three epochs, so this budget
may undertrain the model. Even so, the held-out result is materially below the sparse
baseline and the previous contract-context Transformer. This graph configuration is not
a candidate for replacement. The result does not establish that graph models are
inherently unsuitable; possible follow-up work includes longer training and explicit
edge-type or section-heading features, with hyperparameters chosen without reusing the
held-out evaluation split.

The evaluation graph set contains 82 contracts and 2,196 chunks. Across all 408 split
contracts the graph builder created 103,642 directed edges (including reverse edges).
Seven one-chunk/no-neighbor contracts had no edges. The graph builder uses only train-fitted
TF-IDF vocabulary and contract text; `data/cuad/test.json` was not read.

## Reproduction and artifacts

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.train_contract_graph
```

Model/vectorizer checkpoints are under ignored `outputs/contract_graph_seed42/`. The
split hashes, training curve, graph-size summary, thresholds, per-category metrics, and
raw calibration/evaluation predictions are in `results/contract_graph_seed42/`. To score
the selected checkpoint without retraining:

```bash
.venv/bin/python -m scripts.train_contract_graph \
  --evaluate-checkpoint outputs/contract_graph_seed42/best
```
