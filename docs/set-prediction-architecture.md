# Hungarian-matched contract-level span sets

## Design

A frozen DistilRoBERTa encoder supplies one CLS embedding per overlapped 512-token
chunk. For each contract, 8 learned slots per CUAD category cross-attend over all chunk
embeddings and predict slot presence plus start/end chunk indices. Per-category Hungarian
matching aligns slots with annotated CUAD answer spans; unmatched slots receive a no-object
loss. Chunk indices decode to end-exclusive character ranges using the original chunk
offsets. The encoder is frozen to keep full-contract representation extraction tractable.

Thresholds are distinct: category-presence thresholds are calibrated on calibration
contracts, chunk-projection thresholds on calibration chunks, and span-instance
thresholds by calibration-set IoU-0.5 F1. The slot-threshold calibration does not use the
evaluation annotations. Source spans are mapped to all overlapping chunks; no gold answer
span was uncovered in the three seed-42 partitions. Eight slots covered 98.7% of mapped
evaluation span instances (95.3% train, 97.1% calibration).

## Seed-42 results

| Model | Chunk macro-F1 | Chunk macro-AUPR |
|---|---:|---:|
| Word + character TF-IDF | 0.533 | 0.597 |
| Hungarian set predictor | 0.099 | 0.088 |

Contract-presence macro-AUPR for the set model was 0.413. At character-span IoU >= 0.5,
only 2 of 2,005 evaluation answers matched (micro F1 0.0011; 1,502 false-positive and
2,003 false-negative predicted instances). The chunk-boundary pointer representation is
far too coarse and the learned slots generalize poorly in this run. This is a negative
result, not a viable classifier or span extractor. CUAD answer snippets are not necessarily
complete legally sufficient clauses.

## Reproduction and artifacts

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.train_set_prediction
```

The frozen encoder embeddings are computed at run time; the set-decoder checkpoint is
under ignored `outputs/set_prediction_seed42/`. Reports, thresholds, per-category
metrics, split hashes, coverage statistics, and raw predictions are in
`results/set_prediction_seed42/`. Only the CUAD training file is loaded; `test.json` is
never used.
