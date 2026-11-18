# Feedback and the review queue

`clausetriage/review/` is the human-in-the-loop layer on top of the artifact store. It turns
Unknown predictions into a queue a reviewer can work through, validates and records
reviewer verdicts, resolves repeated and conflicting verdicts, and exports resolved
feedback as labelled training examples without leaking held-out contracts into
training. It is plain Python: every function takes an `ArtifactStore` and plain
arguments and returns Pydantic models that serialise to JSON, so the API and the
reviewer UI can expose them directly. There is no HTTP or UI code in it.

| Module | Public API |
| --- | --- |
| `queue.py` | `build_review_queue`, `unknown_candidates`, `conflict_candidates`, `DEFAULT_SOURCES` |
| `feedback.py` | `submit_feedback`, `resolve_feedback`, `list_resolutions` |
| `export.py` | `build_training_manifest`, `export_training_examples`, `write_export`, `read_examples`, `to_partial_label_matrix`, `check_no_evaluation_overlap` |
| `schema.py` | `FeedbackSubmission`, `ReviewItem`, `ReviewQueuePage`, `Resolution`, `ReviewerVerdict`, `TrainingManifest`, `TrainingExample`, `FeedbackExport` |

CLI: `python -m scripts.export_feedback --store outputs/artifact_store.sqlite`.

## Verdicts

A verdict is about one (chunk, category) under the model version the reviewer
looked at. A detection is a stored prediction: grounded evidence or an Unknown.

| Verdict | Means | Allowed when | Label |
| --- | --- | --- | --- |
| `accept` | the category is in this chunk, as the model predicted | the model has a detection for the key | 1 |
| `reject` | the category is not in this chunk | the model has a detection, or a reviewer already said `missing` on the key | 0 |
| `missing` | the category is in this chunk, and the model did not predict it | the chunk was run under the model version and the model has no detection for the key | 1 |

`ArtifactStore.add_feedback` enforces these rules, so a caller that skips
`submit_feedback` can't bypass them. It also refuses unknown chunks, chunks never
run under the model version, non-canonical category names, unknown verdicts and
blank reviewers. Nothing is written when a verdict is refused.

- A second reviewer who agrees with someone's `missing` also says `missing`, and one
  who disagrees says `reject`. Without that rule nobody could contradict a claimed
  miss.
- A chunk whose probability cleared the threshold but whose quote was dropped as a
  duplicate of a neighbouring chunk's (the evidence stage keeps one quote per span)
  has no detection, so `missing` is the right verdict there.
- Reviewer names are stripped of surrounding whitespace and compared exactly, so
  "Ana" and "ana" are two reviewers. Reviewers aren't authenticated.
- Timestamps default to now. A given timestamp must carry a UTC offset
  (`FeedbackSubmission` refuses naive ones), and it is stored and returned as UTC.
  One reviewer's verdicts on a (chunk, category) can't go back in time.

## History and resolution

The feedback table is append-only. Changing your mind adds a row and never edits
one.

- **Current verdict:** a reviewer's latest verdict on a (chunk, category), in log
  order. Since timestamps can't go backwards per reviewer and key, log order and time
  order agree.
- **Resolved:** every reviewer's current verdict implies the same label (accept and
  missing both mean 1). The key resolves to that label.
- **Conflict:** current verdicts imply different labels. The key has no label and is
  reported with every reviewer's verdict. There is no majority vote and no senior
  reviewer: two accepts and one reject is still a conflict. It resolves when one of the
  disagreeing reviewers changes their verdict.

`list_resolutions(store, model_version=..., contract_id=..., status="conflict")`
lists the disagreements to settle. With a model version, only verdicts given under it
count. Without one (the export's default), verdicts on the same (chunk, category)
under different model versions resolve together, because the label is a fact about
the chunk text, not about the model.

## Review queue

`build_review_queue(store, model_version, contract_id=None, status=None, offset=0, limit=50)`
returns a `ReviewQueuePage`.

**Items.** By default the queue has two sources:

1. **`unknown`:** every Unknown, meaning a prediction above threshold whose evidence
   couldn't be grounded.
2. **`conflict`:** every (chunk, category) whose reviewers disagree under this model
   version. This covers grounded predictions and claimed misses (Unknowns are already
   listed by the first source).

A source is a function `(store, model_version, contract_id) -> list[QueueCandidate]`.
Uncertainty sampling (the stretch goal) would be one more such function passed in
`sources`. If two sources name the same key, the first source keeps it.

**What an item carries.** Each item has:

- contract id and upload file name;
- the chunk's index, absolute offsets and full text (contract text
  `[chunk_start:chunk_end] == chunk_text`);
- first and last page, when the contract has page offsets;
- category, probability and threshold;
- `reason` and a plain-language `reason_text` explaining why it is in the queue;
- the evidence quote and offsets, for a grounded conflict;
- `allowed_verdicts`;
- `status` (`pending`, `reviewed` or `conflict`), the resolved label and every
  reviewer's current verdict.

`item_id` is `f"{chunk_id}|{category}|{model_version}"`.

**Order** (stable for a given store content):

1. Status: `conflict`, then `pending`, then `reviewed`. Conflicts block the export and
   already hold reviewer work. Pending items are the backlog. Reviewed items need
   nothing.
2. Probability, highest first. These are the clauses the model is surest of but
   couldn't quote, so they are the most likely real clauses missing from the
   register.
3. Contract id, chunk index, then category column, as a fixed tiebreak.

**Paging.** `offset`/`limit` slice that order, with `limit` from 1 to 500. `total`
counts the items after the `status` filter, and `status_counts` counts every item.
Submitting a verdict moves an item from pending to reviewed, so offsets shift
between calls. A UI working through the backlog should ask for `status="pending"`
and reload page 0 after each verdict.

## Export as training data

`export_training_examples(store, manifest, model_version=None)` turns each
(chunk, category) with feedback into one `TrainingExample` or one `ExcludedFeedback`.
`write_export` writes two files:

- `examples.jsonl`: one example per line, with these fields:
  - `contract_id` and `contract_sha256`;
  - `chunk_id`, `chunk_index`, `start`, `end` and `chunk_text`, at absolute contract
    offsets;
  - `category`, `category_index` and `label` (0/1);
  - provenance: `verdicts`, `reviewers`, `model_versions`, `first_reviewed_at`,
    `last_reviewed_at`, and `votes` (each current verdict with its feedback id, model
    version, timestamp and note).
- `report.json`: counts, every exclusion with its reason and detail, the manifest's
  source, split, seed and SHA-256, `test_data_used: false`, and the SHA-256 of
  `examples.jsonl`.

The output is deterministic. Rows are sorted by contract, chunk index and category
column, and the JSON uses sorted keys with no export time. The same store and
manifest give byte-identical files.

**Shape for training.** Feedback labels one category of a chunk at a time, so the
examples are partial labels. `to_partial_label_matrix(examples)` returns
`(chunk_ids, texts, labels, mask)`:

- `texts` and `labels` (int8, `(chunks, 41)`) have the same shape as
  `clausetriage.pipeline.to_model_inputs`;
- `mask` marks the labels feedback actually gave.

The other 40 entries of a row are unknown, not negative, so a loss must ignore them
(for example by multiplying a per-element BCE by `mask`). The examples can't become
`LabeledChunk`s, which need all 41 labels. Nothing is wired into training.

**Exclusions**, checked in this order for each key:

| Reason | When |
| --- | --- |
| `offset_mismatch` | the stored contract text no longer hashes to its stored SHA-256, or the chunk text isn't the contract text at its offsets |
| `evaluation_contract` | the contract id, or its exact text under any id, is in the manifest's held-out set |
| `not_in_training_manifest` | the contract's exact text isn't in the manifest's training set (uploads, test contracts, renamed or edited texts) |
| `conflict` | reviewers disagree |

Leakage is checked before conflicts, so a held-out contract is never reported as a
conflict waiting to be settled for training.

## Leakage rule

Feedback collected on contracts used for evaluation must not flow into training, and
this code can't read `test.json` to find out which contracts those are. So the export
uses an allowlist, not a blocklist.

- **A `TrainingManifest` is required.** It lists the contracts that may train and the
  contracts held out, each as `contract_id -> SHA-256 of the exact text`. A contract's
  identity is its text, not its name.
- **The default manifest** is `build_training_manifest("data/cuad/train_separate_questions.json")`:
  the training partition of `split_by_contract(val_fraction=0.2, seed=42)`. That is the
  project split that the demo model, `evaluate_triage.py` and `export_registers.py`
  use. It has 326 training contracts and 82 held-out ones.
- **Default-deny.** A contract whose text isn't in the training set is excluded. That
  covers the validation contracts, any upload, and any contract from `test.json`,
  since none of their texts are in the training partition of the training file.
- **Name check.** `build_training_manifest` refuses files named `test.json` or
  `CUADv1.json` (which contains the test contracts). Nothing in `clausetriage/review/` or the
  CLI reads either file.
- **Recorded provenance.** Every export records the manifest's source file, its
  SHA-256, the split method, seed and fraction, and the manifest's own SHA-256. Every
  example carries `contract_sha256`.
- **Overlap check.** `check_no_evaluation_overlap(examples, evaluation_texts)` lets a
  training run, which does know its evaluation contracts, refuse any example whose
  contract text is among them. Final evaluation may read `test.json`, so this is the
  check that covers the test set.
- **Custom manifests.** `--manifest` loads a custom manifest, for example to add
  uploaded contracts on purpose. A manifest that lists one contract (by id or by text)
  as both training and held out fails validation.

In practice the demo model scores the validation partition (`export_registers.py`),
so feedback given in the demo UI is all excluded as `evaluation_contract`. That is
the point: those contracts are what the model is evaluated on.

**What it does not protect against:**

- **Near-duplicates.** A test contract uploaded as a PDF has different text from its
  CUAD JSON, so its hash matches nothing. It is excluded by default, but nothing stops
  someone listing it in a custom manifest. CUAD amendments and re-filings of the same
  agreement aren't detected either.
- **A custom manifest is trusted.** Whoever writes one decides what may train.
- **A different split later.** If a training run uses another seed or fraction, a
  contract exported as training under the default split may be in its validation set.
  The manifest's seed and fraction are recorded, but only `check_no_evaluation_overlap`
  in the training run catches this.
- **Copied or renamed test files.** The name check only catches the two known file
  names.
- **Calibration slices.** Thresholds tuned on a calibration slice inside the training
  partition (as the nested TF-IDF demo model does) can end up with feedback examples
  from that slice in training.
- **Selection bias.** This isn't leakage, but `accept`/`reject` only exist where the
  model predicted, so the examples over-represent the model's positives.

## What is verified and what is not

**Verified by tests** (`tests/test_review_*.py`, `tests/test_export_feedback.py`, and
the feedback tests in `tests/test_artifact_store.py`):

- queue contents, ordering, status and paging;
- every verdict on what it may refer to, and each refusal;
- changed and conflicting verdicts;
- export contents, byte-identical output, tampered offsets, conflict exclusion;
- the leakage rule, by id and by text;
- the manifest builder refusing test files;
- the CLI.

**Run once by hand on real data.** Four validation contracts and two training-partition
contracts from `train_separate_questions.json` were scored with the demo TF-IDF model
into a scratch store. The run produced:

- 27 Unknowns, all `no_cue_match`;
- verdicts from two made-up reviewers;
- an export through the CLI, written twice to check that it is byte-identical. It held
  2 examples; one conflict and two validation-contract keys were excluded.

**Not verified:**

- No real reviewer has used the queue. The verdicts in the hand run were scripted to
  exercise each path, not real judgements.
- The ordering's claim that high-probability Unknowns are the most useful is a design
  choice, not measured.
- Exported examples have never been used to train, so it is untested whether partial
  labels help.
- Concurrency: verdicts from two processes at once rely on SQLite's locking. The
  timestamp check and the insert run in one transaction, but this wasn't load-tested.
