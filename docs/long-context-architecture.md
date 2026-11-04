# Contract-level long-context classifier

## Architecture

This is a neural hierarchical model, not a sparse-feature variant. Each shared
sentence-aware token chunk is encoded by the pretrained DistilRoBERTa backbone. The
ordered chunk embeddings for one contract are then processed by a **two-layer
inter-chunk Transformer encoder** with learned positional embeddings, and a 41-way
linear head predicts labels for each contextualized chunk. Every chunk can attend to
all other chunks in its contract, not only nearby text or its own 512-token window.
A bidirectional GRU variant is also supported for ablation (`context_type: bigru`).

Training examples are complete contract sequences with the existing per-chunk
multi-hot labels. A masked weighted BCE objective ignores padded chunk slots. The
encoder is fine-tuned jointly with the inter-chunk Transformer and classifier; gradient
checkpointing, chunk-level microbatches, and one contract per batch bound activation
memory. Inference must score all ordered chunks for one contract together; scoring
chunks independently would discard the core context mechanism.

## Reproducible protocol

The script uses `train_separate_questions.json` only and the same seed-42 nested
contract split as the word/character experiment and encoder comparison: 260 training,
66 calibration, 82 held-out evaluation contracts. It uses the pinned DistilRoBERTa
revision and shared 512/64 sentence-token chunker. The calibration partition chooses
thresholds; the disjoint evaluation partition reports per-category precision, recall,
F1, and AUPR. No test-file access occurs.

On a CUDA runtime, after installing requirements and making the tokenizer/model
available in cache, run:

```sh
MLFLOW_DISABLE_AGENT_HINT=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.train_long_context \
  --config configs/long_context.yaml --output outputs/long_context_seed42 --register
```

`--register` is optional. Without it, the checkpoint, threshold file, raw calibration
and evaluation scores, per-category metrics, and run manifest remain in the output
directory. The registered predictor API expects one contract's ordered chunk texts per
call. Do not mix chunks from different contracts in one call, because cross-chunk
attention is computed over the entire supplied sequence.

The forward pass, padding masks, loss masking, grouping, and backpropagation are covered
by unit tests. The initial CUDA run requested 20 epochs but hit a cuBLAS error starting
epoch 20; the epoch-19 best checkpoint (selected by calibration AUPR) was successfully
evaluated on the untouched evaluation contracts. The checked-in config now defaults to
19 epochs to make that completed run reproducible. Results are committed in
`results/long_context_transformer_seed42/`:

| Model (same seed-42 evaluation contracts) | Macro F1 | Macro AUPR |
|---|---:|---:|
| Word + character TF-IDF | 0.533 | 0.597 |
| Chunk encoder + inter-chunk Transformer | 0.496 | 0.507 |

The neural model is a real long-context architecture and is competitive with the
word-only baseline, but it does not beat the hybrid yet. The registered predictor
requires one contract's chunks per call. Keep `test.json` untouched.
