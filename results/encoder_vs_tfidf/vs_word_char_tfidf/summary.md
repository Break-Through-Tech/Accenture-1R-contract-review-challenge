# distilroberta vs word_char_tfidf

**Verdict: does_not_beat** (mean macro-F1 delta is not positive or the candidate wins at most half the seeds).

Seeds compared: 5 of 5; macro-F1 wins 0; bootstrap 95% lower bound above zero on 0 seeds.

| Seed | Baseline F1 | Candidate F1 | Delta F1 [95% CI] | Baseline AUPR | Candidate AUPR | Delta AUPR [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 42 | 0.533 | 0.451 | -0.082 [-0.112, -0.046] | 0.597 | 0.476 | -0.121 [-0.151, -0.085] |
| 7 | 0.515 | 0.451 | -0.064 [-0.093, -0.027] | 0.585 | 0.448 | -0.137 [-0.161, -0.095] |
| 17 | 0.494 | 0.431 | -0.063 [-0.098, -0.027] | 0.563 | 0.478 | -0.085 [-0.110, -0.047] |
| 23 | 0.505 | 0.434 | -0.072 [-0.098, -0.033] | 0.540 | 0.468 | -0.072 [-0.112, -0.034] |
| 101 | 0.473 | 0.392 | -0.081 [-0.106, -0.050] | 0.531 | 0.431 | -0.101 [-0.133, -0.070] |

| Support bucket | Categories | Baseline F1 | Candidate F1 | Mean delta F1 | Improved | Worse |
|---|---:|---:|---:|---:|---:|---:|
| common | 19 | 0.653 | 0.632 | -0.021 | 7 | 12 |
| uncommon | 13 | 0.480 | 0.355 | -0.125 | 1 | 12 |
| rare | 9 | 0.224 | 0.119 | -0.105 | 2 | 7 |

Rare categories (<10 mean evaluation positives) are listed with status "insufficient support" in per_category_summary.csv and are not ranked.
