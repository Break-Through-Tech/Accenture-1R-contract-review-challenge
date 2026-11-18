# Model registry and artifact store

## Model/prediction contract

`ClauseModelBundle` is the stable inference boundary: a predictor accepts chunk text
and returns raw probabilities shaped `(chunks, 41)` in the exact canonical
`CATEGORIES` order. The bundle holds category order, calibrated per-category
thresholds, model name, configuration, and seed. Thresholding is downstream of raw
probabilities so rescoring does not require rerunning the classifier.

`src/registry/mlflow_registry.py` stores each serialized bundle as a numbered MLflow
registered-model version, using a SQLite tracking database and local artifact
folder. Every version logs its evaluation metrics, seed, category schema hash,
thresholds, config, and caller-supplied data/split metadata. Load a specific version
rather than implicitly using `latest` in reproducible scoring.

Example after fitting a model and calibrating thresholds:

```python
from src.labels.categories import CATEGORIES
from src.registry.bundle import ClauseModelBundle
from src.registry.mlflow_registry import register_bundle
from src.registry.predictors import VectorizedClassifier

bundle = ClauseModelBundle(
    predictor=VectorizedClassifier(fitted_vectorizer, fitted_classifier),
    categories=CATEGORIES,
    thresholds=calibrated_thresholds,
    model_name="word_char_tfidf",
    config={"c": 1.0, "word_ngrams": [1, 2], "char_wb_ngrams": [3, 5]},
    seed=42,
)
registered = register_bundle(
    bundle, experiment_name="clause-review", registry_name="clause-classifier",
    metrics={"evaluation_macro_f1": 0.53}, metadata={"split": "...", "data_sha256": "..."},
    tracking_uri="results/mlflow/tracking.db",
)
```

Do not use held-out evaluation metrics to choose thresholds or hyperparameters. Use
an agreed calibration set for thresholds and record the untouched evaluation
partition and protocol in `metadata`.

## Artifact store

`src/store/` is the persistence layer between the pipeline and its readers (the API,
the reviewer UI, the clause-register export). It exists so risk weights can be changed
and every contract rescored from cached probabilities without rerunning the classifier.

- `tables.py`: the SQLAlchemy tables.
- `records.py`: the Pydantic models the read methods return. Clause and contract
  scores come back as the risk stage's own `ClauseScore` and `ContractScore`.
- `artifact_store.py`: `ArtifactStore`, with the writers and readers.
- `rescoring.py`: `rescore_contracts`, which rescores under a new `WeightSet`.

### Tables

| Table | Key | What it holds |
| --- | --- | --- |
| `contracts` | `contract_id` | exact text and its SHA-256; optional upload metadata: `source_name`, `source_type` (e.g. `pdf`, `docx`, `txt`), `page_starts_json`, `ocr_used` |
| `chunks` | `chunk_id` | `contract_id`, `chunk_index`, `start`, `end`, `text` (always `contract.text[start:end]`) |
| `model_versions` | `model_version` | `model_name`, `seed`, `thresholds_json` (one threshold per CUAD category) |
| `chunk_probs` | `chunk_id`, `category`, `model_version` | raw probability for all 41 categories of every chunk |
| `evidence` | `evidence_id` (unique per chunk, category, model version) | grounded `quote`, absolute `start`/`end` |
| `unknown_predictions` | `chunk_id`, `category`, `model_version` | detections with no grounded evidence (`reason`: `no_cue_match` / `span_mismatch`) |
| `weight_versions` | `weight_version_id` | canonical `values_json`, `content_sha256`, `created_by`, `created_at` |
| `clause_scores` | `chunk_id`, `category`, `model_version`, `weight_version_id` | `contract_id`, `signals_json`, `composite_score`, `rating`, `has_evidence`, `weight_set_sha256` |
| `contract_scores` | `contract_id`, `model_version`, `weight_version_id` | `triage_score`, High/Medium/Low/Unknown counts, `top_clauses_json`, `weight_set_sha256` |
| `feedback` | `id` | `chunk_id`, `category`, `model_version`, `verdict` (`accept` / `reject` / `missing`), `reviewer`, `timestamp`, `note` |

Foreign keys are enforced (`PRAGMA foreign_keys=ON` on the store's own engine).
Evidence, Unknowns and clause scores point at their `chunk_probs` row, and scores
point at their weight version. A contract's rank isn't stored, because it depends on
which other contracts are scored. The read side computes it with the same rule as
`risk.contract.aggregate_contracts`: highest triage score first, ties broken by
`contract_id`, and tied scores share a rank (1, 2, 2, 4).

### Write rules

- **Idempotent:** writing the same content again is a no-op. Writing different
  content under an existing key raises `ValueError`, and the transaction rolls back.
- **Immutable versions:** a weight version or model version can't be reused for
  different content. Edits need a new `weight_version_id`. A stored `WeightSet` loads
  back with an identical `content_sha256`.
- **Grounded:** chunk text and evidence quotes must equal the stored
  contract text at their offsets.
- Use the registry-qualified model key (for example `clause-classifier:3`) as
  `model_version`, because separate models can each have a version `1`.
- One contract keeps one chunking. Saving a result with a different chunk set for a
  stored contract fails. The store derives chunk IDs as `f"{contract_id}:{chunk_index}"`.

### Saving a pipeline run

```python
from src.risk.schema import load_weight_set
from src.store.artifact_store import ArtifactStore

store = ArtifactStore("outputs/artifact_store.sqlite")
weights = load_weight_set("src/risk/weights/default_weights_v1.json")
pipeline = ContractInferencePipeline(bundle, tokenizer, cues, weight_set=weights)
result = pipeline.predict_contract(document.name, document.text)
store.save_inference_result(
    result, text=document.text, model_version=registered.model_key,
    thresholds=bundle.thresholds, weight_set=weights,
    source_name="lease.pdf", source_type="pdf",          # optional upload metadata
    page_starts=document.page_starts, ocr_used=document.ocr,
)
```

`save_inference_result` writes the contract, chunks, all probabilities, the model
version's thresholds, evidence, Unknowns, the weight version, clause scores and the
contract score in one transaction. The result dict doesn't carry the contract text,
so pass the exact text the pipeline ran on.

### Rescoring without the model

```python
from src.store.rescoring import rescore_contracts

new_weights = weights.model_copy(update={"weight_version_id": "v1.1", "triage_max_weight": 0.7})
ranked = rescore_contracts(store, registered.model_key, new_weights)             # writes scores
preview = rescore_contracts(store, registered.model_key, new_weights, save=False)  # writes nothing
```

`get_clause_inputs` rebuilds the `ClauseInput` rows the pipeline scored from stored
probabilities, evidence and Unknowns, in the pipeline's order. Rescoring reuses the
detections recorded at inference time rather than re-deriving them from probabilities
and thresholds. The evidence stage drops duplicate spans found in overlapping
chunks, and that needs the extractor and its cue words, not just stored numbers. The
stored thresholds are checked against every detection and reported on Unknown-queue
items. Tests check that rescoring from the store gives exactly the clause scores,
contract scores and ranks of a fresh pipeline run under the same weights.

### Reading

`list_contracts`, `get_contract` (text plus source metadata), `get_chunks`,
`get_evidence`, `list_unknowns` (the review queue: detections above threshold with no
grounded evidence), `get_clause_scores`, `get_contract_score`, `list_contract_scores`
(ranked), and `get_contract_detail`, which bundles all of them for one contract, model
version and weight version. Feedback has an `add_*` / `list_*`
pair. Weight and model versions have `put_*` / `get_*` / `list_*`. `get_chunk` and
`get_detection` (grounded, Unknown or none for a chunk, category and model version)
serve the review queue.

`add_feedback` refuses verdicts that don't fit what the model predicted: `accept`
needs a stored detection, `missing` needs none, and `reject` needs a detection or an
earlier `missing` on the same key. Timestamps are stored and returned as UTC, and one
reviewer's verdicts on a (chunk, category) can't go back in time. The review queue,
verdict resolution and the training-data export are in `src/review/`; see
[Feedback and the review queue](feedback-review-queue.md).

### Schema changes

There are no migrations (Alembic is deferred). Opening a database whose tables have
different columns from the code raises `RuntimeError`. After a schema change, delete
the local database file (for example `outputs/artifact_store.sqlite`) and rerun the
pipeline to recreate it. Any database written before this version of the store
(older `contracts`, `clause_scores` and `contract_scores` columns) fails to open and
must be recreated this way, including the smoke-test `outputs/artifact_store.sqlite`
mentioned below.

Install dependencies with `uv pip install --python .venv/bin/python -r requirements.txt`.
The local test suite exercises MLflow save/load and SQLite persistence in temporary
directories, including a registered-bundle-to-artifact-store prediction roundtrip.

The long-context Transformer candidate is registered locally as
`clause-classifier-long-context:1`; the category-query evidence candidate is registered
as `clause-classifier-category-query:2` (version 2 exposes attention through
`predict_with_evidence`). Both remain candidates, not replacements for the stronger
word+character baseline. Their held-out reports are in
`results/long_context_transformer_seed42/` and `results/category_query_seed42/`.
CPU-loaded registry bundles were smoke-tested through inference; 41 raw probabilities
were persisted to `outputs/artifact_store.sqlite` using registry-qualified model keys.
The registry database, artifact database, and model weights are local and ignored by
Git. Do not commit datasets or model weights.
