# Final evaluation: development_dry_run

Data `outputs/final_test_dry_run/development_holdout.json` (SHA-256 `acfc12054ca6...`), 82 contracts, 2196 chunks. Model `models/word_char_tfidf.joblib` (SHA-256 `6fd84c28600f...`). `test_data_used: false`.

## Primary: chunk-level detection, frozen bundle at its own thresholds

Intervals: contract bootstrap, BCa 95% (percentile bounds are in report.json and the CSVs).

| Metric | Result [95% BCa CI] | Development mean (range of 5 splits) | Seed 42 | Pre-registered reading |
|---|---|---|---|---|
| Macro-F1 (41 categories) | 0.533 [0.505, 0.569] | 0.504 (0.473-0.533) | 0.533 | consistent_with_development, within_development_range |
| Macro-AUPR (categories with positives) | 0.597 [0.546, 0.617] | 0.563 (0.531-0.597) | 0.597 | consistent_with_development, within_development_range |

Categories with positives: 41 of 41. Unscorable (no positives): none. Insufficient support (<10 positives): 9.

### By development support bucket

| Bucket | Categories | Mean F1 | Dev mean F1 | Mean AUPR | Dev mean AUPR | Under 10 positives here |
|---|---:|---:|---:|---:|---:|---:|
| common | 19 | 0.660 | 0.653 | 0.709 | 0.696 | 0 |
| uncommon | 13 | 0.488 | 0.480 | 0.532 | 0.511 | 2 |
| rare | 9 | 0.330 | 0.224 | 0.454 | 0.366 | 7 |

### Per category (sorted by support)

| Category | Support | Status | Precision | Recall | F1 [95% CI] | AUPR [95% CI] | Dev mean F1 |
|---|---:|---|---:|---:|---|---|---:|
| Parties | 102 | reported | 0.893 | 0.735 | 0.806 [0.739, 0.860] | 0.881 [0.836, 0.918] | 0.797 |
| License Grant | 89 | reported | 0.829 | 0.764 | 0.795 [0.735, 0.848] | 0.870 [0.798, 0.918] | 0.754 |
| Document Name | 83 | reported | 0.912 | 0.627 | 0.743 [0.652, 0.817] | 0.832 [0.691, 0.898] | 0.752 |
| Governing Law | 81 | reported | 0.890 | 0.901 | 0.896 [0.846, 0.935] | 0.958 [0.922, 0.979] | 0.907 |
| Anti-Assignment | 78 | reported | 0.846 | 0.705 | 0.769 [0.685, 0.836] | 0.838 [0.750, 0.897] | 0.779 |
| Expiration Date | 77 | reported | 0.806 | 0.649 | 0.719 [0.628, 0.797] | 0.769 [0.668, 0.846] | 0.691 |
| Agreement Date | 76 | reported | 0.831 | 0.711 | 0.766 [0.672, 0.836] | 0.794 [0.647, 0.880] | 0.721 |
| Effective Date | 75 | reported | 0.769 | 0.533 | 0.630 [0.512, 0.732] | 0.643 [0.505, 0.752] | 0.611 |
| Cap On Liability | 73 | reported | 0.593 | 0.658 | 0.623 [0.506, 0.699] | 0.728 [0.575, 0.810] | 0.687 |
| Minimum Commitment | 64 | reported | 0.742 | 0.359 | 0.484 [0.305, 0.637] | 0.525 [0.395, 0.665] | 0.479 |
| Exclusivity | 57 | reported | 0.378 | 0.544 | 0.446 [0.348, 0.548] | 0.459 [0.332, 0.564] | 0.459 |
| Post-Termination Services | 57 | reported | 0.453 | 0.509 | 0.479 [0.374, 0.576] | 0.496 [0.363, 0.610] | 0.443 |
| Audit Rights | 51 | reported | 0.677 | 0.824 | 0.743 [0.649, 0.816] | 0.836 [0.679, 0.918] | 0.752 |
| Termination For Convenience | 45 | reported | 0.447 | 0.378 | 0.410 [0.273, 0.549] | 0.437 [0.287, 0.553] | 0.472 |
| Non-Transferable License | 41 | reported | 0.791 | 0.829 | 0.810 [0.633, 0.922] | 0.828 [0.614, 0.937] | 0.672 |
| Warranty Duration | 39 | reported | 0.769 | 0.513 | 0.615 [0.481, 0.739] | 0.675 [0.556, 0.773] | 0.543 |
| Revenue/Profit Sharing | 37 | reported | 0.420 | 0.568 | 0.483 [0.356, 0.612] | 0.438 [0.297, 0.547] | 0.561 |
| Insurance | 32 | reported | 0.900 | 0.844 | 0.871 [0.750, 0.939] | 0.965 [0.843, 0.993] | 0.793 |
| Renewal Term | 32 | reported | 0.810 | 0.531 | 0.642 [0.473, 0.784] | 0.689 [0.521, 0.811] | 0.739 |
| Change Of Control | 31 | reported | 0.632 | 0.387 | 0.480 [0.286, 0.652] | 0.557 [0.347, 0.730] | 0.428 |
| Covenant Not To Sue | 31 | reported | 0.706 | 0.387 | 0.500 [0.286, 0.667] | 0.547 [0.330, 0.692] | 0.599 |
| Non-Compete | 30 | reported | 0.304 | 0.700 | 0.424 [0.314, 0.526] | 0.495 [0.323, 0.639] | 0.340 |
| Notice Period To Terminate Renewal | 22 | reported | 0.933 | 0.636 | 0.757 [0.552, 0.889] | 0.759 [0.534, 0.891] | 0.713 |
| Volume Restriction | 21 | reported | 0.294 | 0.238 | 0.263 [0.083, 0.465] | 0.205 [0.062, 0.361] | 0.264 |
| Ip Ownership Assignment | 18 | reported | 0.500 | 0.722 | 0.591 [0.412, 0.762] | 0.677 [0.451, 0.855] | 0.583 |
| Irrevocable Or Perpetual License | 17 | reported | 0.548 | 1.000 | 0.708 [0.467, 0.885] | 0.874 [0.658, 0.994] | 0.642 |
| Rofr/Rofo/Rofn | 16 | reported | 0.625 | 0.312 | 0.417 [0.174, 0.615] | 0.301 [0.045, 0.490] | 0.523 |
| Competitive Restriction Exception | 15 | reported | 0.170 | 0.600 | 0.265 [0.133, 0.392] | 0.124 [0.045, 0.249] | 0.206 |
| Uncapped Liability | 15 | reported | 0.500 | 0.733 | 0.595 [0.370, 0.778] | 0.676 [0.386, 0.847] | 0.633 |
| Joint Ip Ownership | 12 | reported | 0.421 | 0.667 | 0.516 [0.308, 0.765] | 0.585 [0.180, 0.785] | 0.454 |
| Unlimited/All-You-Can-Eat-License | 11 | reported | 0.147 | 0.455 | 0.222 [0.056, 0.453] | 0.286 [0.019, 0.866] | 0.112 |
| Third Party Beneficiary | 10 | reported | 0.385 | 0.500 | 0.435 [0.127, 0.769] | 0.294 [0.023, 0.614] | 0.180 |
| No-Solicit Of Employees | 9 | insufficient support | 0.875 | 0.778 | 0.824 [0.376, 1.000] | 0.859 [0.462, 1.000] | 0.749 |
| Affiliate License-Licensee | 8 | insufficient support | 0.222 | 0.500 | 0.308 [0.125, 0.526] | 0.329 [0.100, 0.615] | 0.326 |
| Source Code Escrow | 8 | insufficient support | 0.750 | 0.375 | 0.500 [0.000, 0.800] | 0.842 [0.456, 1.000] | 0.173 |
| Liquidated Damages | 6 | insufficient support | 0.222 | 0.667 | 0.333 [0.100, 0.621] | 0.603 [0.032, 1.000] | 0.324 |
| Most Favored Nation | 6 | insufficient support | 0.308 | 0.667 | 0.421 [0.100, 0.783] | 0.467 [0.027, 0.949] | 0.241 |
| No-Solicit Of Customers | 5 | insufficient support | 0.160 | 0.800 | 0.267 [0.105, 0.483] | 0.650 [0.200, 1.000] | 0.182 |
| Affiliate License-Licensor | 4 | insufficient support | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.049 [0.012, 0.196] | 0.042 |
| Price Restrictions | 4 | insufficient support | 0.188 | 0.750 | 0.300 [0.000, 0.600] | 0.605 [0.556, 1.000] | 0.190 |
| Non-Disparagement | 2 | insufficient support | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.035 [0.018, 0.074] | 0.149 |

## Secondary analyses

- **S1 word-only TF-IDF reference:** macro-F1 0.476 [0.449, 0.508], macro-AUPR 0.508 [0.464, 0.526]. Word+char minus word-only (paired): macro-F1 0.057 [0.033, 0.083], macro-AUPR 0.090 [0.067, 0.114].
- **S2 contract-level presence:** macro-F1 0.670 [0.641, 0.709], macro-AUPR 0.779 [0.726, 0.804] (41 categories present in at least one contract).
- **S3 triage ranking vs perfect detection (`v2-team-proposed`):** Spearman 0.777 [0.652, 0.862], largest rank move 50, top-10 overlap 2 over 82 contracts.
- **S4 evidence:** Unknown rate 0.192 (293 of 1529 detected clauses); a grounded quote overlaps a gold answer 0.383 of the time overall and 0.601 on correctly detected chunks.

## Run record

- Commit `e892f345c28f31113d710d7cb14389a472067c63`; protocol SHA-256 `5de80fc29a8d...`.
- Attempts: 1. Deviations: 0.
- Serving parity checked on 82 contracts.
