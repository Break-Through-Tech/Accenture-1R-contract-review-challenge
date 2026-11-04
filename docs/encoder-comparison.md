# Encoder comparison status and procedure

The unmerged `origin/init-transformer-encoder-training` branch contains the
DistilRoBERTa training, checkpoint inference, and evaluation implementation. The local
alternative-architecture branch incorporates that code, its tests, and the training
configuration. It also pins `transformers==4.57.6`, fixes `warmup_steps` to use the
supported `warmup_ratio`, and configures the encoder to use the shared sentence-aware
token chunker (512 total tokens, 64-token overlap) rather than the old fixed 800/200
character chunks.

`scripts/evaluate_encoder_nested.py` defines the aligned comparison for seed 42:

- 260 contract training set
- 66 disjoint threshold-calibration contracts
- 82 disjoint final evaluation contracts
- same `split_by_contract`, pinned tokenizer, sentence-aware chunker, label projection,
  and CUAD training source as the word-only and word+character experiments
- thresholds fitted on calibration only; final per-category P/R/F1 and AUPR reported
  on evaluation only
- `test.json` is never read

Run on a Colab T4 or another suitable CUDA runtime after installing requirements and
placing the official data at `data/cuad/train_separate_questions.json`:

```sh
python -m scripts.evaluate_encoder_nested --config configs/encoder.yaml \
  --output outputs/encoder_nested --register
```

Omit `--register` to keep the trained checkpoint and evaluation local without adding
an MLflow model version. Raw calibration/evaluation probabilities, thresholds,
per-category metrics, and split hashes are written under the output directory. Local
CPU validation was limited to an eight-chunk, one-epoch smoke fit; it demonstrates
that shared token chunking, training, checkpoint save/load, and prediction are wired
end to end, but it is not a meaningful encoder benchmark. The aligned seed-42 run contains 9,025 training, 1,899 calibration, and 2,196
evaluation chunks. The full four-epoch run has not been executed locally; use GPU
compute for the intended training run.
