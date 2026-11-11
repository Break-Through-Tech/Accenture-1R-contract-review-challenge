# Category-description cue logit boost ablation

## Question and setup

Can category-associated words or phrases add useful signal to the word/character TF-IDF
classifier? This ablation uses the existing category-description cue lexicon, matching a
cue as a whole phrase within a chunk. If any cue for a category matches, a shared additive
logit boost is applied to that chunk/category score:

`adjusted_logit = tfidf_logit + boost * cue_match`

No boost is applied to other category scores. A single boost is selected independently
for each split from `[0, 0.25, 0.5, 1, 2]` by calibration macro-AUPR; category decision
thresholds are then calibrated on that same calibration partition. The cues are derived
from category descriptions, with the repository's existing manual cues for terse
categories. They are not extracted from evaluation labels. This is a small interpretable
lexicon adjustment experiment, not evidence that a cue match establishes clause presence.

## Results

| Seed | Selected boost | Δ Macro-F1 | Δ Macro-AUPR | Evaluation cue-match fraction |
|---:|---:|---:|---:|---:|
| 42 | 1.0 | -0.0027 | +0.0098 | 37.0% |
| 7 | 0.25 | +0.0050 | +0.0035 | 38.0% |
| 17 | 0.5 | -0.0026 | +0.0045 | 36.0% |
| 23 | 0.5 | +0.0022 | +0.0053 | 38.4% |
| 101 | 0.5 | +0.0083 | +0.0072 | 36.2% |
| Mean | | **+0.0020** | **+0.0061** | |

The paired contract-bootstrap interval for macro-F1 included zero on all five seeds;
macro-AUPR intervals included zero on one seed. The direction of the F1 effect is mixed,
and average gains are modest. Per-category results and calibration-grid scores are in
`results/category_cue_boost/`. This does not establish a robust improvement and should not
replace the TF-IDF control. Cue matches affected about 42% to 45% of evaluation
chunk/category pairs, so the cue list is broad; false matches and category-specific
regressions are plausible. `data/cuad/test.json` was not accessed.

## Reproduction

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.evaluate_category_cue_boost --bootstrap-replicates 1000
```
