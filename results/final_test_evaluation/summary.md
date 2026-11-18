# Final evaluation: sealed_test

Data `data/cuad/test.json` (SHA-256 `007b6a40b0c6...`), 102 contracts, 2666 chunks. Model `models/word_char_tfidf.joblib` (SHA-256 `6fd84c28600f...`). `test_data_used: true`.

## Primary: chunk-level detection, frozen bundle at its own thresholds

Intervals: contract bootstrap, BCa 95% (percentile bounds are in report.json and the CSVs).

| Metric | Result [95% BCa CI] | Development mean (range of 5 splits) | Seed 42 | Pre-registered reading |
|---|---|---|---|---|
| Macro-F1 (41 categories) | 0.539 [0.514, 0.595] | 0.504 (0.473-0.533) | 0.533 | consistent_with_development, above_every_development_split |
| Macro-AUPR (categories with positives) | 0.608 [0.567, 0.641] | 0.563 (0.531-0.597) | 0.597 | consistent_with_development, above_every_development_split |

Categories with positives: 40 of 41. Unscorable (no positives): Price Restrictions. Insufficient support (<10 positives): 5.

### By development support bucket

| Bucket | Categories | Mean F1 | Dev mean F1 | Mean AUPR | Dev mean AUPR | Under 10 positives here |
|---|---:|---:|---:|---:|---:|---:|
| common | 19 | 0.650 | 0.653 | 0.702 | 0.696 | 0 |
| uncommon | 13 | 0.500 | 0.480 | 0.559 | 0.511 | 1 |
| rare | 9 | 0.361 | 0.224 | 0.462 | 0.366 | 5 |

### Per category (sorted by support)

| Category | Support | Status | Precision | Recall | F1 [95% CI] | AUPR [95% CI] | Dev mean F1 |
|---|---:|---|---:|---:|---|---|---:|
| Parties | 137 | reported | 0.882 | 0.708 | 0.785 [0.724, 0.837] | 0.863 [0.801, 0.903] | 0.797 |
| License Grant | 107 | reported | 0.844 | 0.710 | 0.772 [0.691, 0.834] | 0.825 [0.712, 0.880] | 0.754 |
| Anti-Assignment | 106 | reported | 0.806 | 0.708 | 0.754 [0.682, 0.813] | 0.831 [0.745, 0.888] | 0.779 |
| Document Name | 106 | reported | 0.838 | 0.538 | 0.655 [0.562, 0.736] | 0.729 [0.608, 0.809] | 0.752 |
| Governing Law | 100 | reported | 0.912 | 0.830 | 0.869 [0.816, 0.913] | 0.899 [0.795, 0.949] | 0.907 |
| Agreement Date | 94 | reported | 0.753 | 0.681 | 0.715 [0.626, 0.791] | 0.754 [0.643, 0.833] | 0.721 |
| Expiration Date | 89 | reported | 0.747 | 0.697 | 0.721 [0.642, 0.791] | 0.761 [0.672, 0.833] | 0.691 |
| Cap On Liability | 86 | reported | 0.753 | 0.674 | 0.712 [0.622, 0.788] | 0.787 [0.669, 0.862] | 0.687 |
| Effective Date | 84 | reported | 0.593 | 0.381 | 0.464 [0.358, 0.566] | 0.442 [0.333, 0.542] | 0.611 |
| Minimum Commitment | 71 | reported | 0.692 | 0.380 | 0.491 [0.347, 0.632] | 0.535 [0.387, 0.678] | 0.479 |
| Audit Rights | 70 | reported | 0.684 | 0.771 | 0.725 [0.642, 0.792] | 0.802 [0.691, 0.866] | 0.752 |
| Revenue/Profit Sharing | 67 | reported | 0.586 | 0.612 | 0.599 [0.465, 0.735] | 0.571 [0.444, 0.698] | 0.561 |
| Exclusivity | 64 | reported | 0.463 | 0.578 | 0.514 [0.391, 0.633] | 0.519 [0.382, 0.655] | 0.459 |
| Post-Termination Services | 58 | reported | 0.375 | 0.517 | 0.435 [0.352, 0.522] | 0.415 [0.307, 0.515] | 0.443 |
| Insurance | 56 | reported | 0.697 | 0.821 | 0.754 [0.521, 0.875] | 0.832 [0.630, 0.930] | 0.793 |
| Ip Ownership Assignment | 49 | reported | 0.500 | 0.429 | 0.462 [0.319, 0.595] | 0.479 [0.304, 0.631] | 0.583 |
| Change Of Control | 44 | reported | 0.824 | 0.318 | 0.459 [0.286, 0.633] | 0.584 [0.436, 0.709] | 0.428 |
| Non-Compete | 43 | reported | 0.293 | 0.907 | 0.443 [0.347, 0.540] | 0.794 [0.676, 0.882] | 0.340 |
| Covenant Not To Sue | 40 | reported | 0.667 | 0.450 | 0.537 [0.381, 0.692] | 0.587 [0.453, 0.702] | 0.599 |
| Rofr/Rofo/Rofn | 39 | reported | 0.524 | 0.564 | 0.543 [0.357, 0.684] | 0.505 [0.226, 0.690] | 0.523 |
| Termination For Convenience | 38 | reported | 0.583 | 0.553 | 0.568 [0.429, 0.692] | 0.602 [0.433, 0.733] | 0.472 |
| Non-Transferable License | 33 | reported | 0.571 | 0.606 | 0.588 [0.415, 0.732] | 0.599 [0.371, 0.756] | 0.672 |
| Renewal Term | 31 | reported | 0.821 | 0.742 | 0.780 [0.634, 0.879] | 0.785 [0.547, 0.902] | 0.739 |
| Volume Restriction | 24 | reported | 0.306 | 0.458 | 0.367 [0.189, 0.564] | 0.395 [0.200, 0.602] | 0.264 |
| Competitive Restriction Exception | 22 | reported | 0.158 | 0.682 | 0.256 [0.161, 0.373] | 0.372 [0.148, 0.557] | 0.206 |
| Liquidated Damages | 21 | reported | 0.400 | 0.381 | 0.390 [0.188, 0.638] | 0.431 [0.200, 0.719] | 0.324 |
| Irrevocable Or Perpetual License | 20 | reported | 0.516 | 0.800 | 0.627 [0.400, 0.793] | 0.730 [0.395, 0.939] | 0.642 |
| Notice Period To Terminate Renewal | 20 | reported | 0.688 | 0.550 | 0.611 [0.370, 0.810] | 0.636 [0.349, 0.829] | 0.713 |
| Affiliate License-Licensee | 19 | reported | 0.480 | 0.632 | 0.545 [0.348, 0.706] | 0.565 [0.338, 0.767] | 0.326 |
| Warranty Duration | 15 | reported | 0.700 | 0.467 | 0.560 [0.250, 0.857] | 0.547 [0.245, 0.849] | 0.543 |
| Affiliate License-Licensor | 14 | reported | 0.220 | 0.786 | 0.344 [0.182, 0.489] | 0.551 [0.153, 0.842] | 0.042 |
| Third Party Beneficiary | 14 | reported | 0.263 | 0.357 | 0.303 [0.143, 0.471] | 0.243 [0.053, 0.465] | 0.180 |
| Uncapped Liability | 14 | reported | 0.571 | 0.857 | 0.686 [0.500, 0.828] | 0.781 [0.565, 0.896] | 0.633 |
| No-Solicit Of Customers | 12 | reported | 0.159 | 0.583 | 0.250 [0.100, 0.408] | 0.522 [0.170, 0.971] | 0.182 |
| No-Solicit Of Employees | 12 | reported | 0.818 | 0.750 | 0.783 [0.556, 0.951] | 0.911 [0.703, 1.000] | 0.749 |
| Joint Ip Ownership | 9 | insufficient support | 0.318 | 0.778 | 0.452 [0.211, 0.625] | 0.654 [0.182, 0.850] | 0.454 |
| Non-Disparagement | 9 | insufficient support | 0.167 | 0.222 | 0.190 [0.000, 0.476] | 0.182 [0.035, 0.548] | 0.149 |
| Unlimited/All-You-Can-Eat-License | 5 | insufficient support | 0.076 | 1.000 | 0.141 [0.027, 0.368] | 0.147 [0.020, 0.607] | 0.112 |
| Most Favored Nation | 3 | insufficient support | 0.143 | 0.667 | 0.235 [0.000, 0.632] | 0.139 [0.001, 0.500] | 0.241 |
| Source Code Escrow | 2 | insufficient support | 1.000 | 1.000 | 1.000 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.173 |
| Price Restrictions | 0 | unscorable | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a | 0.190 |

## Secondary analyses

- **S1 word-only TF-IDF reference:** macro-F1 0.480 [0.459, 0.524], macro-AUPR 0.524 [0.486, 0.559]. Word+char minus word-only (paired): macro-F1 0.058 [0.040, 0.077], macro-AUPR 0.083 [0.064, 0.104].
- **S2 contract-level presence:** macro-F1 0.678 [0.651, 0.736], macro-AUPR 0.794 [0.751, 0.823] (40 categories present in at least one contract).
- **S3 triage ranking vs perfect detection (`v2-team-proposed`):** Spearman 0.777 [0.648, 0.863], largest rank move 64, top-10 overlap 4 over 102 contracts.
- **S4 evidence:** Unknown rate 0.230 (462 of 2013 detected clauses); a grounded quote overlaps a gold answer 0.364 of the time overall and 0.583 on correctly detected chunks.

## Run record

- Commit `2ea0619b34a2b5097d92a490e0928e1e33881e51`; protocol SHA-256 `a01eebc77be5...`.
- Attempts: 1. Deviations: 0.
- Serving parity checked on 102 contracts.
