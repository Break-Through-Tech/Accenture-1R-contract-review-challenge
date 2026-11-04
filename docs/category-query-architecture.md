# Category-query evidence architecture

This candidate gives each of CUAD's 41 clause types its own query, initialized by
encoding the category name and human-written description in the pinned DistilRoBERTa
space. Queries cross-attend to the contextualized sequence of a contract's chunk
embeddings. A query/chunk matching head predicts the chunk labels, with a local
chunk-classifier residual so the model can retain ordinary lexical signals. The same
queries produce multi-instance contract-presence scores and attention distributions
that can localize evidence chunks.

Training combines (1) chunk-level CUAD labels, (2) contract-level presence labels
formed by OR-ing chunk annotations, and (3) an attention loss that encourages positive
category queries to attend to annotated evidence chunks. Attention localization is an
evaluation signal, not a substitute for classification quality or a guarantee that a
highlight is legally sufficient.

## Reproduce

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.train_category_query \
  --config configs/category_query.yaml \
  --output outputs/category_query_seed42
```

The run uses the same contract-level 260/66/82 seed-42 train/calibration/evaluation
partition, pinned tokenizer and 512/64 chunking as the previous architectures. The
best checkpoint is selected using calibration macro-AUPR only; per-category thresholds
are fit on calibration only. The held-out evaluation scores are reported once, and
`data/cuad/test.json` is never read. The script saves report, raw scores, thresholds,
per-category metrics and a reloadable local checkpoint. Model weights remain ignored
local outputs.

## Seed-42 held-out results

| Model | Macro F1 | Macro AUPR |
|---|---:|---:|
| Word + character TF-IDF | 0.533 | 0.597 |
| Chunk encoder + inter-chunk Transformer | 0.496 | 0.507 |
| Category-query evidence model | 0.478 | 0.512 |

The category-query model does not beat the TF-IDF control and is slightly behind the
previous Transformer on macro F1. Its distinctive result is evidence localization:
on 1,031 positive contract-category instances, the top-attended chunk was annotated
positive 59.2% of the time, and mean attention mass on annotated positive chunks was
45.5%. These are held-out localization diagnostics, not additional clause F1.

The registered predictor also exposes `predict_with_evidence(texts)`, returning raw
probabilities, a `[chunks, categories]` attention matrix, and the top-attended chunk
index per category. Full metrics, calibration/evaluation scores and thresholds are in
`results/category_query_seed42/`. Do not claim overall superiority from its small AUPR
advantage over the earlier Transformer; the evaluation set is one fixed split and the
stronger word+character baseline remains the leading model.
