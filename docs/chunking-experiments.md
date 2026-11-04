# Chunking experiments

These are data-preparation experiments, not classifier evaluations. They measure CUAD answer-span containment, source coverage, token-budget compliance, duplication, and single-run timing. They do not establish that one chunker improves model quality.

## Saved development-set comparison

Five strategies were compared on the same 326 saved development contracts and 9,175 annotations. Validation metrics were not calculated; earlier exploration used all 408 training-file contracts, so the saved validation portion is not pristine.

| Strategy | Answer spans fully contained | Cut without containment | Inputs over 512 tokens | Chunks | Duplicated-character overhead |
|---|---:|---:|---:|---:|---:|
| Character 800/200 | 81.17% | 18.83% | 0.57% | 30,637 | 33.03% |
| Character 1500/200 | 89.87% | 10.13% | 9.15% | 14,235 | 15.16% |
| Token 512/64 | 94.87% | 5.13% | 0% | 10,190 | 13.98% |
| Sentence 512/64 | **98.82%** | **1.18%** | 0% | 10,924 | **6.29%** |
| PR #48 token 480/120 | 97.10% | 2.90% | 0% | 12,551 | 32.64% |

All strategies cover the source text; token counts are checked on final slices including special tokens. Sentence-aware windows gave the best containment and least text duplication in this comparison, but took longer to generate and sometimes had zero overlap between adjacent chunks. These are one-machine timings and not production benchmarks. The 480/120 and 512/64 budgets are defined differently, so they are not directly interchangeable.

The original analysis also includes per-category results, example spans, exact settings, and audit details:

- [PR #48 comparison findings](../results/token_chunking_comparison/findings.md)
- [Comparison report](../results/token_chunking_comparison/report.json)
- [Per-category results](../results/token_chunking_comparison/per_category.csv)
- [Earlier token/sentence development findings](../results/token_chunking_development/findings.md)

## Reproduction

With project dependencies installed and the pinned tokenizer cached, run from the repository root:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m scripts.evaluate_chunking
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m scripts.evaluate_token_chunking
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m scripts.compare_gwen_chunking
```

The last two commands use saved contract IDs; they do not regenerate a split. Reports are written under `results/`. The detailed chunk-level CSVs are generated locally and ignored by Git. The supplied `data/cuad/test.json` is not used.

The token methods use `distilbert/distilroberta-base` at revision
`fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`. The PR #48 adapter is evaluated from its reviewed source snapshot; see its report for provenance and limitations. Do not interpret annotated-answer containment as evidence of complete legal-clause coverage or model accuracy.
