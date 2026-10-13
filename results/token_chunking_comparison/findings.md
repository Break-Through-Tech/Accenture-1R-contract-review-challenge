# PR #48 comparison on the saved development contracts

All five methods use the exact saved 326 development IDs, in their saved order,
from `results/token_chunking_development/report.json`. Its dataset hash, tokenizer
revision, package versions, categories, and evaluation settings were verified.
The split was not regenerated. The four existing methods' annotation and chunk-cost
results exactly match their original saved results. Runtime was remeasured for all
five methods in one process, with generation, coverage audit, label projection,
and scoring/token measurement timed separately. Downloads/loading are excluded.

Every method sees 9,175 raw and unique annotations; zero duplicates were removed.
The counting key is `(contract_id, category, start, end)`. Percentages use the unique
annotation denominator, retain different categories, and count an annotation once
per metric. Cut boundaries must be strictly inside the annotation; endpoint equality
is not a cut. These are annotated answer-span metrics, not complete-clause or
classifier-performance metrics. No validation metrics or model training were run.

| Method | Full containment | Cut without containment | Overflow >512 | Chunks | Duplicated characters | Generation / total runtime |
|---|---:|---:|---:|---:|---:|---:|
| Character 800/200 | 81.17% (7,447) | 18.83% (1,728) | 0.57% (174) | 30,637 | 33.03% | 0.04s / 5.55s |
| Character 1500/200 | 89.87% (8,246) | 10.13% (929) | 9.15% (1,302) | 14,235 | 15.16% | 0.02s / 4.10s |
| Our token windows 512/64 | 94.87% (8,704) | 5.13% (471) | 0% | 10,190 | 13.98% | 10.97s / 14.94s |
| Our sentence windows 512/64 | 98.82% (9,067) | 1.18% (108) | 0% | 10,924 | 6.29% | 34.39s / 37.93s |
| Gwen's token windows 480/120 | 97.10% (8,909) | 2.90% (266) | 0% | 12,551 | 32.64% | 3.06s / 8.25s |

Total runtime includes all four measured stages, not just boundary generation.
These are single wall-clock observations rather than repeated performance benchmarks.
Our algorithms retokenize their proposed slices during generation; Gwen's algorithm
does not, so the independent post-generation checks are particularly important.

## Provenance and budget units

Gwen's source is loaded unchanged from PR #48 commit
`94d6d3d8b332de91338625bde100259a1717728b`, path `experiments/token_chunking.py`,
SHA-256 `6be39944ce5ee89293973411e164384bae8865ae872a6347a2790c1c8f5b6379`.
Her implementation and three original tests were inspected. The adapter uses the
same, byte-identical `Chunk` implementation without merging either branch.

`window_tokens=480` counts tokens in the original document encoding **without special
tokens**. It is not equivalent to our `max_tokens=512`, which includes special tokens.
Its `overlap_tokens=120` also refers to document-token positions; each final slice and
shared substring is retokenized for reporting. Gwen's maximum final input length
was **485**, demonstrating why 480 document content tokens need not retokenize to
exactly 482 input tokens. All final lengths are measured with special tokens,
`truncation=False`, and `padding=False`. The tokenizer is
`distilbert/distilroberta-base`, revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`;
transformers 4.57.6 and tokenizers 0.22.2. All other versions are recorded in report.json.

## Verification and tradeoffs

At the requested 480/120 settings, all 326 contracts passed exact chunk/source text
equality, original offset/index/progress checks, full source coverage, and existing
any-overlap label projection. No final chunk exceeded 512 tokens. Coverage was
verified before calculating duplicated characters; no contracts or annotations
were dropped. Gwen's source was not modified to satisfy these checks.

A separate regression test using one-token Unicode windows exposes nonadvancing
outputs from her algorithm. The integration rejects those outputs rather than
repairing them. This is not a failure observed at the requested 480/120 development
configuration, and the observed zero overflow is not a universal guarantee of her
algorithm. The tokenizer's long-document warning refers to the initial whole-document
encoding, not the verified final chunk lengths.

Tests: 53 focused checks passed, followed by the full suite with **111 passing
tests** (eight new integration cases). Both runs used `PYTHONDONTWRITEBYTECODE=1`,
`python -B`, `HF_HUB_OFFLINE=1`, and pytest `-p no:cacheprovider`. The new checks
cover exact Git-blob provenance, explicit 480/120 real-tokenizer/projection behavior,
the unrepaired Unicode progress failure, saved-ID order and validation exclusion,
dataset changes, and invalid split manifests.

Gwen preserves 205 more annotations than our plain token windows, with 2,361 more
chunks and substantially more duplicated characters. Her mean actual overlap is
119.99 content tokens, versus 63.72 for our plain token windows and 27.35 for sentence
windows. Sentence windows preserve 158 more annotations than Gwen with 1,627 fewer
chunks and less duplication, but cost more generation time. Zero-overlap adjacency
is 35.96% for sentence windows and 0% for both token-window strategies. This compares
complete strategies with different budgets/overlap policies, not sentence boundaries
in isolation. The per-category table includes all 41 categories and their counts.

Representative gains/losses for Gwen relative to our sentence strategy:

- **Gain: Minimum Commitment, [1556, 2008)**, Adams Golf endorsement agreement.
  The annotation starts with the definition of "MANDATORY PRODUCTS". Our chunks
  [0, 1873) and [1873, 3776) split it with zero overlap. Gwen contains it in
  [1309, 3111), overlapping her preceding [0, 1765) chunk.
- **Loss: Renewal Term, [4146, 5868)**, Airtech franchise agreement. It starts
  "At the expiration of the term or any renewal term hereof". Our [3547, 5886)
  contains it; Gwen's [2475, 4708), [4167, 6346), and [5806, 7810) intersect it
  without any one chunk containing the full annotation.

Per-category differences also matter: Gwen's Affiliate License-Licensee containment
is 90% versus our plain token windows' 80%, across 80 annotations. Sentence windows
contain all 51 Non-Disparagement annotations versus Gwen's 90.20%. Small-category
percentages should be interpreted alongside their annotation counts. report.json
includes representative gains and losses against all four existing strategies,
with source text, contract IDs, and absolute offsets.

## Recommendation before validation

Freeze the five existing configurations without further development tuning. Treat
sentence 512/64 as the leading containment/duplication candidate, Gwen 480/120 as
the faster plain-token comparator, and our token 512/64 as a control for the sentence
strategy's budget. Retain both character baselines. Then perform one comparison on
the same saved 82 validation IDs after approval; do not choose settings by repeatedly
examining validation. Earlier exploration covered all 408 training-file contracts,
so the validation portion is not a pristine holdout.

Files: summary.csv, per_category.csv, report.json; detailed chunks.csv is generated
locally and ignored. No dependencies were installed. No production/shared chunker,
label policy, or category mapping changed. test.json was not read. Neither PR was
merged; no commits or pushes were made in this comparison step.
