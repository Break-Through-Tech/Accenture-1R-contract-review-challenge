# Contract-level long-context classifier

## Architecture

This is a neural hierarchical model, not a sparse-feature variant. Each shared
sentence-aware token chunk is encoded by the pretrained DistilRoBERTa backbone. The
ordered chunk embeddings for one contract are then processed as a **bidirectional
GRU sequence**, and a 41-way linear head predicts the labels for each contextualized
chunk. Consequently, each chunk's classification can use learned signals from chunks
before and after it, across the contract—not only its own 512-token window.

Training examples are complete contract sequences with the existing per-chunk
multi-hot labels. A masked weighted BCE objective ignores padded chunk slots. The
encoder is fine-tuned jointly with the context GRU and classifier; gradient
checkpointing and one contract per batch keep activation memory manageable. Inference
must score all ordered chunks for one contract together; scoring chunks independently
would discard the core context mechanism.

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
call. Do not mix chunks from different contracts in one call, because the recurrent
state encodes within-contract context.

The forward pass, padding masks, loss masking, grouping, and backpropagation are covered
by unit tests. This environment has no CUDA device; a full real-data run across 9,025
training chunks and four epochs has **not** been performed here. CPU smoke tests cannot
establish model quality. Compare this model to the word+character model using the saved
seed-42 partitions before making any performance claim; keep `test.json` untouched.
