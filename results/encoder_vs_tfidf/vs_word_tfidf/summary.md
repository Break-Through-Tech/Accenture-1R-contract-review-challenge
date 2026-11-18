# distilroberta vs word_tfidf

**Verdict: does_not_beat** (mean macro-F1 delta is not positive or the candidate wins at most half the seeds).

Seeds compared: 5 of 5; macro-F1 wins 2; bootstrap 95% lower bound above zero on 0 seeds.

| Seed | Baseline F1 | Candidate F1 | Delta F1 [95% CI] | Baseline AUPR | Candidate AUPR | Delta AUPR [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 42 | 0.476 | 0.451 | -0.025 [-0.054, +0.009] | 0.508 | 0.476 | -0.031 [-0.067, +0.005] |
| 7 | 0.455 | 0.451 | -0.004 [-0.035, +0.031] | 0.491 | 0.448 | -0.043 [-0.071, -0.005] |
| 17 | 0.409 | 0.431 | +0.022 [-0.009, +0.048] | 0.459 | 0.478 | +0.019 [-0.012, +0.055] |
| 23 | 0.423 | 0.434 | +0.010 [-0.017, +0.043] | 0.463 | 0.468 | +0.005 [-0.034, +0.045] |
| 101 | 0.396 | 0.392 | -0.004 [-0.032, +0.023] | 0.444 | 0.431 | -0.014 [-0.044, +0.014] |

| Support bucket | Categories | Baseline F1 | Candidate F1 | Mean delta F1 | Improved | Worse |
|---|---:|---:|---:|---:|---:|---:|
| common | 19 | 0.591 | 0.632 | +0.042 | 16 | 3 |
| uncommon | 13 | 0.390 | 0.355 | -0.035 | 5 | 8 |
| rare | 9 | 0.158 | 0.119 | -0.039 | 3 | 6 |

Rare categories (<10 mean evaluation positives) are listed with status "insufficient support" in per_category_summary.csv and are not ranked.
