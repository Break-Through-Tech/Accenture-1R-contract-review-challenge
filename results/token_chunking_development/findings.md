# Issue #12 development comparison

Evaluated 326 development contracts from the official training file, using the
existing contract split (seed 42, validation fraction 0.2). All methods saw the
same 9,175 raw/unique annotations; no duplicates were removed. Deduplication is
by contract ID, category, start, and end, preserving different categories.
Validation metrics were not calculated. Earlier exploration used all 408
training-file contracts, so the validation portion is not a pristine holdout.

| Method | Fully contained | Cut without containment | Overflow >512 | Chunks | Duplicated-character overhead |
|---|---:|---:|---:|---:|---:|
| Character 800/200 | 81.17% (7,447) | 18.83% (1,728) | 0.57% (174) | 30,637 | 33.03% |
| Character 1500/200 | 89.87% (8,246) | 10.13% (929) | 9.15% (1,302) | 14,235 | 15.16% |
| A: token 512/64 | 94.87% (8,704) | 5.13% (471) | 0% | 10,190 | 13.98% |
| B: sentence 512/64 | 98.82% (9,067) | 1.18% (108) | 0% | 10,924 | 6.29% |

All final token-method slices were retokenized with the real DistilRoBERTa fast
tokenizer, including special tokens, without truncation or padding. Full source
coverage, exact slices, ordered indices, and progress were checked before overhead.
Special-token overhead is two; the 512-token limit includes those two tokens.

| Method | Input-token median / p95 / maximum | Mean actual overlap, content tokens | Zero-overlap adjacencies | Generation time |
|---|---:|---:|---:|---:|
| Character 800/200 | 178 / 321 / 763 | 49.99 | 0% | 0.04s |
| Character 1500/200 | 330 / 587 / 1,425 | 50.07 | 0% | 0.01s |
| A: token 512/64 | 512 / 512 / 512 | 63.72 | 0% | 10.60s |
| B: sentence 512/64 | 476 / 512 / 512 | 27.35 | 35.96% | 33.74s |

Generation time includes strategy-required sentence detection and budget checks.
Character generation does not tokenize. Separate scoring/measurement times were
5.26s, 3.90s, 3.78s, and 3.46s respectively; projection times are in report.json.
These are single-process wall-clock observations, excluding downloads and loading,
not model inference benchmarks. No model was trained.

B improves containment by 8.95 percentage points over character 1500/200 and
3.96 points over A. It produces 734 more chunks than A but less duplicated text.
Its average realized overlap is much smaller. This tests complete strategies,
including packing and overlap policies, not sentence boundaries alone.

B used token-window fallback for 451 oversized detected sentences, producing 1,239
fallback chunks. All 788 fallback-internal adjacencies had nonzero overlap; all
fallback transitions had zero overlap. Regular sentence-packing adjacencies had
33.40% zero overlap. First chunks of contracts are excluded from all adjacency
denominators. Exact lengths, overlaps, and group classifications are in chunks.csv.

Standalone annotated-answer lengths (including special tokens): median 44,
p95 188, p99 350.26, maximum 1,387; 21 annotations exceeded 512. These are
diagnostics because standalone and in-context tokenization can differ. Answer
annotations are not necessarily complete legal clauses. Character containment
describes original chunks, not evidence retained by a truncating encoder dataset.
The 9.15% observed here applies specifically to development chunks at 1500/200;
it does not verify the configuration or denominator behind Gwen's original result.

Per-category changes are mixed. For example, Uncapped Liability containment rose
from 72.31% to 97.69% across 130 annotations, while Parties fell from 100% to
99.88% across 1,625 annotations. Full counts and all categories are in per_category.csv.

Examples of B versus character 1500/200 (absolute, end-exclusive offsets):

- **Improved: Rofr/Rofo/Rofn, [5041, 5485)** in the Xpresspa marketing agreement.
  The annotation begins "If Calm does not exercise its ROFR within the ROFR Period".
  Old chunks [3900, 5400) and [5200, 6700) cut it without containment; B contains it
  in [4004, 5528).
- **Worsened: Minimum Commitment, [1556, 2008)** in the Adams Golf endorsement
  agreement. The annotation begins with the definition of "MANDATORY PRODUCTS".
  The old [1300, 2800) contains it; B's [0, 1873) and [1873, 3776) split it with
  zero overlap and no full containment.

report.json includes original source text, contract IDs, annotation offsets, and
relevant chunks for both gains and losses against both baselines. It also records
the dataset hash, split IDs, package versions, detector settings, and counting policy.
Tokenizer: distilbert/distilroberta-base at
fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b; transformers 4.57.6, tokenizers 0.22.2.
Sentence detector: nupunkt 0.8.0, nonadaptive sent_spans, no layout/heading heuristics.

The observed result supports taking these fixed settings to validation after the
design/settings freeze discussion. It does not establish model-quality gains or
an optimal CUAD chunker. The shared chunker, projection, and split are unchanged;
test.json was not read. No commit or push was made.
