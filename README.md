# AI Studio Challenge Project Title

### PR #48 comparison on the saved development split

`scripts/compare_gwen_chunking.py` evaluates Gwen's exact prototype from PR #48
commit `94d6d3d8b332de91338625bde100259a1717728b` alongside the four existing
strategies. `src/chunking/gwen_adapter.py` loads the reviewed Git source blob
unchanged, verifies that both commits use the identical `Chunk` implementation,
and records the source SHA-256. It does not merge, copy-edit, or repair her algorithm.

Her explicit `window_tokens=480` counts **document content tokens**, excluding
special tokens; `overlap_tokens=120` is also a document-token target. Our
`max_tokens=512` includes special tokens, with overlap targeting 64 content tokens.
Every final source slice is independently measured including special tokens,
with truncation and padding disabled. Gwen's algorithm does not enforce that
post-slice budget itself; observed overflow is reported rather than corrected.

The saved `results/token_chunking_development/report.json` is the authoritative
manifest. The runner verifies its dataset hash, package versions, tokenizer revision,
category mapping, and evaluation settings. It selects the exact 326 saved development
IDs in saved order, checking that the 82 validation IDs are distinct and excluded.
No new split is generated. All five methods are rerun together so runtime uses
the same loaded tokenizer and wall-clock policy: generation, source-coverage audit,
any-overlap label projection, and scoring/token measurement are timed separately.
The four existing methods' span/cost results must equal their saved results before
the comparison is written. The original four-method artifacts are preserved.

With the existing project environment and pinned tokenizer cache, run:

```sh
git fetch origin 94d6d3d8b332de91338625bde100259a1717728b
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python -B -m pytest -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -B -m scripts.compare_gwen_chunking
```

No new dependencies are needed. Environment mismatches fail explicitly; consult
the saved report's version manifest before reproducing on another machine.
Compact reports and findings are under `results/token_chunking_comparison/`;
the detailed `chunks.csv` remains generated locally and ignored. Source or coverage
failures are written to `verification_failures.json` and stop the evaluation,
without dropping contracts or repairing outputs. Model-budget failures remain
visible as overflow counts. This comparison does not run validation or train a model.

### Issue #12: token-aware development experiment

The shared `chunk_contract` remains unchanged. Two isolated functions in
`src/chunking/token_chunker.py` accept explicit `tokenizer`, `max_tokens`, and
`overlap_tokens` arguments and return the existing `Chunk` model. They compare
plain token windows (A) and sentence-aware token windows (B) against the existing
800/200 and 1500/200 **character** baselines. A and B use 512 total input tokens
and a target of 64 overlapping **content** tokens. These are experimental settings.

Use Python 3.12 and install `requirements.txt` in the project environment. Download
only the pinned tokenizer once (no encoder weights or training dependencies):

```sh
.venv/bin/python -c 'from transformers import AutoTokenizer; from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION; AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True, cache_dir=".venv/hf-cache")'
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m scripts.evaluate_token_chunking
```

The tokenizer is `distilbert/distilroberta-base` at revision
`fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`. Its overhead is obtained from the
tokenizer (two special tokens in this revision). Every final original-text slice
is measured with special tokens, `truncation=False`, and `padding=False`;
token offsets only estimate boundaries. Repeated or whitespace-trimmed offsets
do not justify dropping source characters. Budget correction retreats to earlier
character endpoints; overlap is reduced if necessary to guarantee progress.
No text is decoded from token IDs, normalized, or reconstructed from sentences.

A targets the final 64 content tokens, maps them to advancing source offsets,
and reduces overlap if retokenizing the suffix exceeds 64 tokens. B uses NUPunkt
0.8.0's nonadaptive, layout-free `sent_spans` to pack complete sentences. It keeps
the longest advancing suffix of complete trailing sentences whose combined source
slice fits 64 content tokens and allows at least one new sentence to fit. If its
last sentence alone exceeds 64 tokens, overlap is zero. Oversized sentences use A
internally, with zero overlap at entry/exit; their offsets and chunk indices are
reported separately. Section-heading heuristics are deferred; headings stay in
the source and are never prepended. Sentence detection is imperfect, particularly
on citations, abbreviations, numbering, and formatted legal text.

The runner reuses `load_cuad`, `CATEGORIES`, and `project_labels` (any character overlap),
and the saved IDs originally produced by `split_by_contract(seed=42, val_fraction=0.2)`.
It evaluates only the **326
development contracts**, records the 82 validation IDs, and has no validation-run
option. Earlier exploration covered all 408 official training-file contracts, so
validation is not a pristine holdout. Do not use `test.json` for this experiment.
Annotations enter evaluation and label projection only, after boundaries are fixed.

Reports go to `results/token_chunking_development/`: `summary.csv`,
`per_category.csv`, `chunks.csv`, and `report.json`. They include unique annotation
containment/cut-without-containment, annotation counts, natural input-token
distributions, overflow, chunk count, generation/scoring runtimes, actual overlap,
zero-overlap adjacency percentages, and improved/worsened examples. Coverage and
exact slices are checked before duplicated-character overhead is calculated as
`100 * (emitted_chars - source_chars) / source_chars`. First chunks are excluded
from adjacency denominators; zero overlap means no shared source characters.
Overlap token counts exclude special tokens. Fallback-internal adjacency and
fallback transitions have separate statistics. Percentiles use linear interpolation.
Generation runtime includes sentence detection and internal budget checks, excludes
loading/downloads, and is a single-process wall-clock observation, not a benchmark.
The detailed `chunks.csv` is generated locally and excluded from version control;
aggregate CSVs, the JSON report, and development findings are retained.

Annotation counting deduplicates `(contract_id, category, start, end)`, retaining
identical spans assigned to different categories; percentages divide by unique
annotations. Missing categories have zero counts and undefined percentages.
Standalone annotation lengths include special tokens and are diagnostics: their
tokenization can differ when embedded in context. These are annotated answer spans,
not necessarily complete legal clauses. Original-chunk containment is distinct
from evidence retained after encoder truncation. Realized overlap differs between
A and B, so this comparison tests the **complete strategies**, not sentence
boundaries alone. Metadata fixes dataset hash, split IDs, tokenizer revision,
package versions, detector behavior, settings, and counting policy.

Research motivation: [CUAD's sliding-window processing](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/6ea9ab1baa0efb9e19094440c317e21b-Paper-round1.pdf),
[LexGLUE's discussion of structure and mixed CUAD answer lengths](https://aclanthology.org/2022.acl-long.297/),
[legal sentence segmentation challenges](https://arxiv.org/abs/2305.01211),
[NUPunkt's span API](https://github.com/alea-institute/nupunkt), and
[mixed retrieval evidence for embedding-based semantic chunking](https://aclanthology.org/2025.findings-naacl.114/).
These motivate measurement; they do not establish an optimal CUAD strategy.

### Fixed-window chunking evaluation

From the repository root, run `python3 -m scripts.evaluate_chunking` after installing
`requirements.txt`. This reuses `load_cuad` and the existing `chunk_contract`, reads
only `data/cuad/train_separate_questions.json`, and writes
`results/chunking_evaluation.csv` and `results/chunking_examples.json`.
Run `python3 -m pytest tests/test_evaluation.py` for hand-calculated metric checks.

The explicit `(window_chars, overlap_chars)` comparison is `(800, 200)` and
`(1500, 200)` as references, plus `(800, 0)`, `(1500, 0)`, and `(3000, 200)` to
explore overlap and window size. These are character settings, not token limits
or agreed production defaults. Override the comparison with repeated options,
for example `python3 -m scripts.evaluate_chunking --config 800 200 --config 1500 200`.

For annotation `[a, b)` and chunk `[s, e)`, a cut means `a < s < b` or
`a < e < b`; touching an endpoint is not a cut. Full containment means
`s <= a` and `b <= e`. The third metric counts annotations cut without full
containment in any chunk. Spans with `b - a > window_chars` cannot be contained;
their count and percentage are reported separately. A cut span can still be
contained by another overlapping chunk, so cut and containment percentages
are not complementary.

Each annotation counts at most once per metric. Primary metrics deduplicate by
`(contract_id, category, start, end)` across all loaded annotations, preserving
identical spans in different categories. The denominator is the unique annotation
count; each percentage is `100 * matching_annotations / unique_annotations`.
Raw counts, unique counts, and duplicates removed are included. With no
annotations, percentages are undefined (`None`, or blank CSV cells).

Examples retain the first annotation of each outcome in dataset order, with its
contract ID, category, annotation offsets, cutting/containing chunk offsets, and
original text around strictly interior boundaries. Annotation previews are
limited to 240 characters and marked when truncated; boundary excerpts carry
absolute source offsets. This evaluates annotated answer spans, not every
sentence or legal clause, and does not measure label projection or model quality.

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Taylor Nguyen    | @taylornguyen | Data exploration, visualization, overall project coordination            |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
