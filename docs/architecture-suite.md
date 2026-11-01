# Additional architecture experiments

`src/models/naive/` now contains three alternatives to word+character TF-IDF:

1. **Neighbor-context TF-IDF** augments each center chunk with its previous and next
   chunk text while retaining center-chunk labels.
2. **Classifier chain** uses word+character TF-IDF features and conditions category
   predictions on earlier category outputs in a seeded random order. Constant labels in
   a train split are handled explicitly.
3. **Positive-prototype retrieval** ranks each category by maximum cosine similarity
   to its positive training chunks in the shared sparse TF-IDF space.

The runner includes both word-only TF-IDF and word+character TF-IDF controls and uses
seed 42's same 260/66/82 contract-level training/calibration/evaluation sets as the
prior hybrid and upcoming encoder comparison. Chunking and labels use the shared
sentence-token pipeline (512/64). Thresholds are tuned on calibration only. The
runner never reads `test.json`:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.evaluate_architecture_suite --seed 42
```

## Four-seed screening results

| Model | Mean macro F1 | Mean macro AUPR |
|---|---:|---:|
| Word-only TF-IDF | 0.441 | 0.480 |
| Word + character TF-IDF | **0.513** | **0.571** |
| Neighbor-context word + character | 0.334 | 0.354 |
| Word + character classifier chain | 0.490 | 0.533 |
| Positive-prototype retrieval | 0.166 | 0.167 |

The three proposed alternatives are implemented and evaluated across four predeclared
seeds (42, 7, 17, 23); none beat the word+character control on macro F1 or AUPR in
any of these splits. Context concatenation and positive-only retrieval are negative
results. The classifier chain is competitive but consistently behind the hybrid. Do
not describe these architectures as superior. Per-seed reports and raw scores live in
`results/architecture_suite_seed{42,7,17,23}/`; aggregate summaries and per-category
results are under `results/architecture_suite_seed42/`. These splits overlap, so the
summary is descriptive rather than inferential. The hybrid's broader five-seed results
remain under `results/word_char_tfidf_nested/`.
