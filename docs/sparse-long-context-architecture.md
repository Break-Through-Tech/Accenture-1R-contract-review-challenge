# Local/global sparse attention over full contract chunk sequences

## Design

This is a chunk-level Longformer-style model, not raw-token Longformer. The pinned
DistilRoBERTa encoder is frozen and produces one representation for every overlapping
512-token chunk in the full contract. Two trainable sparse-attention layers then process
the complete chunk sequence: each chunk attends to a radius-16 local window and a single
contract-global token; the global token attends to all chunks. At fixed radius the context
attention is O(NW), rather than the O(N²) chunk attention used by a dense sequence model.

Chunk scores from overlapping windows are aggregated over source-character intervals by
averaging the scores of all windows covering the interval, then taking the maximum
segment score for contract presence. Calibration and evaluation use the same contract
splits as previous experiments.

## Seed-42 results

| Metric | Word + character TF-IDF | Local/global sparse context |
|---|---:|---:|
| Chunk macro-F1 | **0.533** | 0.142 |
| Chunk macro-AUPR | **0.597** | 0.131 |
| Overlap-consensus contract macro-F1 | **0.661** | 0.432 |
| Overlap-consensus contract macro-AUPR | **0.779** | 0.443 |

The chunker covered 100% of source characters in train, calibration, and evaluation.
Evaluation contracts had at most 168 chunks. Across evaluation contracts, the model used
an estimated 58,368 sparse attention links per layer versus 125,174 dense chunk links
(46.6%). This is a real sparsity reduction, but modest at the lengths in this split; the
main asymptotic benefit appears on longer contracts. The sparse model did not approach
the TF-IDF control's classification quality in this frozen-encoder, five-epoch screening
run. No production latency claim is made.

## Reproduction and artifacts

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.train_sparse_long_context
```

The encoder is frozen and loaded from the pinned local model cache. Sparse-context
checkpoints are under ignored `outputs/sparse_long_context_seed42/`. Split hashes,
coverage/complexity summaries, calibration thresholds, per-category metrics, and raw
scores are in `results/sparse_long_context_seed42/`. The run uses only the CUAD training
file and never reads `data/cuad/test.json`.
