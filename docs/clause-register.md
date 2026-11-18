# Clause register

A clause register is a table of every clause ClauseTriage found in one contract. Each row gives the clause
type, its risk rating, why it got that rating, where it is in the contract, and the exact sentence it is
based on. One register is produced per contract, as a spreadsheet-friendly CSV file and a JSON file with
the same content.

> **Research prototype.** Ratings use team-proposed risk weights (`v2-team-proposed`) that the advisor has not yet signed off.
> A register helps decide which contracts and clauses to read first. It is not legal advice and does not
> replace reading the contract.

## How to read a register

- Rows are sorted riskiest first (highest `composite_score`).
- A clause type can appear more than once, because the model reads the contract in overlapping
  sections and may find the same type in several places.
- Every quote is copied character for character from the contract, and its position is checked
  against the contract text when the register is written. ClauseTriage never paraphrases or
  reconstructs a quote.
- Rows marked `needs_manual_review = True` have no quote. See below.

## Column dictionary (CSV)

| Column | Meaning |
|---|---|
| `contract_id` | The contract's name (for CUAD contracts, the dataset's title). |
| `contract_rank` | The contract's place in the review order among the contracts scored together. 1 means open first. Contracts with the same triage score share a rank (1, 2, 2, 4). |
| `triage_score` | The contract's overall risk score, from 0 to 1. It combines the contract's single worst clause with the average of its three worst clause types. Repeated clauses of the same type count once. |
| `category` | The clause type, one of the 41 CUAD categories (for example *Uncapped Liability*, *Non-Compete*, *Governing Law*). |
| `rating` | `High`, `Medium` or `Low`, from `composite_score` and the two thresholds set by the weight version (in `v2-team-proposed`, Medium from 0.25 and High from 0.525). |
| `composite_score` | The clause's risk score, from 0 to 1: a weighted sum of the four signals below. |
| `category_severity` | How serious this clause type usually is (0 = harmless, such as the parties' names; 1 = critical, such as uncapped liability). Other clauses in the contract can lower it, for example a liability cap that softens liquidated damages. |
| `one_sidedness` | Whether the quoted wording gives one party unchecked power ("sole discretion", "at any time") rather than mutual, qualified terms ("each party", "reasonable"). 0 when there is no quote. |
| `magnitude` | How big the obligation is: broad-scope words in the quote ("perpetual", "irrevocable", "worldwide") or a duration beyond the limits set for the clause type (for example a three-year non-compete). 0 when there is no quote. |
| `contract_context` | Risk from what else the contract contains or lacks, for example uncapped liability with no insurance clause. |
| `probability` | The model's confidence that this clause type is present in this part of the contract (0 to 1). It is not a risk score: a confident detection is not a riskier clause. |
| `chunk_index` | Which section of the contract (as the model read it, numbered from 0) the clause was found in. Useful for tracing; reviewers can ignore it. |
| `page` | The page the quote starts on, when the document has pages (an uploaded PDF). Blank for plain text, such as the CUAD contracts, and for rows without a quote. |
| `start`, `end` | Character positions of the quote in the contract text (`end` is one past the last character). Blank when there is no quote. |
| `quote` | The sentence the rating is based on, copied exactly from the contract. Blank when there is no quote. |
| `has_evidence` | `True` when a quote was found and verified, `False` otherwise. |
| `needs_manual_review` | `True` when there is no quote. See the next section. |
| `review_reason` | Why there is no quote: `no_cue_match` (no sentence in that section used the wording expected for the clause type), `span_mismatch` (a quote was found but did not match the contract text exactly, so it was discarded), or `no_evidence` (the scoring step received no quote and no reason). Blank when there is a quote. |
| `model_name` | The clause-detection model that produced the row (for example `word_char_tfidf`). |
| `weight_version_id` | The version of the risk weights used. A new version is created whenever a weight changes, so a score can always be traced to the exact settings. |
| `weight_set_sha256` | A fingerprint of the full weight settings. Two registers with the same fingerprint were scored with identical settings. |

Blank cells mean "none". The CSV is UTF-8 with a header row.

## The JSON file

The JSON file holds the same information in two parts:

- `header`: one summary of the contract. It has `contract_id`, `triage_score`, `rank`, `ranked_out_of`
  (how many contracts were ranked together, or null), the rating counts `high_count`, `medium_count` and
  `low_count`, `unknown_count`, `clause_count`, `manual_review_count`, `model_name`, `model_seed`,
  `weight_version_id` and `weight_set_sha256`.
- `rows`: one object per clause, with the CSV's clause columns. The four signals are grouped under
  `signals`, and blanks are `null`.

The rating counts (`high_count`, `medium_count`, `low_count`) and `unknown_count` count **clause types**,
using each type's worst clause. This is how the triage score counts them, and it matches the demo
page's contract list. `clause_count` and `manual_review_count` count **rows**. For example, a contract
whose *Insurance* clause was found unquoted in three sections has three manual-review rows but adds at
most 1 to `unknown_count`.

Programs should read a register back with `src.register.export.read_json`. It validates the file, for
example that every quoted row has a consistent quote and position, that unquoted rows are flagged, and
that the header's counts agree with the rows.

## What "needs manual review" means

The model detected a clause type in part of the contract, but ClauseTriage could not point to the exact
sentence that supports it. This happens when no sentence in that section uses the wording expected for
the clause type, or when a candidate quote did not match the contract text exactly. ClauseTriage does not
guess in that case. It reports the detection without a quote and flags it.

For these rows:

- **The clause is still listed and still scored.** Dropping it would make the contract look safer than
  it might be. Without a quote, the wording-based signals (`one_sidedness`, `magnitude`) are 0, so the
  score reflects only how serious the clause type is and what else the contract contains.
- **The detection may be wrong.** With no supporting sentence, it is more likely to be a false alarm
  than a quoted row.
- **What to do:** use `chunk_index` (or search the contract for the clause type) to read that part of
  the contract, and confirm whether the clause is there and how risky it is.

## How to produce registers

**From the demo page.** Run `python -m src.ui.app`, upload or paste contracts, and download
**Clause register (CSV)**. That file holds every contract scored in the session in one table, using
the same columns with three renamed for the page: `contract` (`contract_id`), `risk_score`
(`composite_score`) and `model` (`model_name`). Page numbers are filled in for PDFs.

**For the CUAD validation contracts (batch).** From the repo root:

```bash
python -m scripts.export_registers                      # all 82 validation contracts
python -m scripts.export_registers --limit 5            # a quick run on the first 5
python -m scripts.export_registers --weights path/to/weights.json --output outputs/registers_v2
```

The script scores the validation contracts of `train_separate_questions.json` (the same split
`scripts/evaluate_triage.py` ranks; `test.json` is never read) with the demo model from
`src/registry/demo_model.py`. That model is downloaded from the GitHub Release, or built locally into
`models/` if the download fails, which takes a few minutes. The script ranks all contracts together and
writes to `outputs/registers/` by default (not committed):

- `<name>.csv` and `<name>.json` for each contract. `<name>` is the contract id with every run of
  characters other than letters and digits replaced by `-`, shortened to 60 characters, plus 8
  characters from a hash of the full id. It is safe on every operating system, the same on every run,
  and never shared by two contracts.
- `index.csv`: one row per contract in review order, with rank, triage score, counts and file name.
- `index.json`: the same rows, plus the model, weight version, split and chunking settings used.

A run on a laptop CPU takes under a minute once the model is available. Files left over from an earlier
run with different contracts are not deleted, so use a fresh `--output` folder when comparing runs.

**From code.**

```python
from src.register.export import build_register, build_registers, write_register

result = pipeline.predict_contract(contract_id, text)          # ContractInferencePipeline
register = build_register(result, text=text, page_starts=document_page_starts)
write_register(register, "outputs/registers")                  # writes <name>.csv and <name>.json

registers = build_registers(results, weight_set, texts=texts)  # several contracts, ranked together
```

Passing `text` re-checks every quote against it and raises `GroundingError` if a quote is not at its
recorded position. That would mean the text is not the one the contract was scored on.
