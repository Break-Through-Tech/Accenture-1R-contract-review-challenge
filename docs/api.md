# API server

`clausetriage/api/` is the HTTP layer of ClauseTriage: the one place clients read pipeline outputs from. It
uploads and scores contracts in the background, stores every result in the artifact store
(`clausetriage/store/`), and serves rankings, contract details, clause registers, weight versions, the
review queue and reviewer feedback. The reviewer page ([reviewer-ui.md](reviewer-ui.md)) runs in the
same process at `/ui/` and calls the same service.

> **No authentication. Do not expose this server on a network.** It is a local research prototype:
> anyone who can reach the port can upload files, read every stored contract, create weight
> versions and record verdicts under any name. It binds to `127.0.0.1` by default and prints a warning if `--host` is anything else.
> Scores use weights set by the team, not validated by a legal expert, and are not legal advice.

## Running it

From the repo root, after `pip install -r requirements.txt` (the server alone needs only
`requirements-server.txt`, see [development.md](development.md)). To run it in Docker instead, see
[docker.md](docker.md).

```bash
python -m clausetriage.api                                   # http://127.0.0.1:8000
python -m clausetriage.api --port 8080 --database outputs/api/try.sqlite
python -m clausetriage.api --bundle models/word_char_tfidf.joblib --weights path/to/weights.json
python -m clausetriage.api --registry-version 3              # a numbered version from the local MLflow registry
python -m clausetriage.api --no-ui                           # API only, no Gradio page
```

- Interactive API docs: `http://127.0.0.1:8000/docs` (OpenAPI schema at `/openapi.json`).
- The reviewer page: `http://127.0.0.1:8000/ui/` ([reviewer-ui.md](reviewer-ui.md)). It reads and
  writes the same store through the same `ClauseTriageService` methods as the endpoints. (The
  standalone in-memory demo, `python -m clausetriage.ui.app`, is a separate page and is not mounted here.)
- The model is the demo model from `clausetriage/registry/demo_model.py` (downloaded from the GitHub Release on
  first use, or built locally). The tokenizer comes from `.venv/hf-cache`; set `HF_HUB_OFFLINE=1` if
  it is already there.
- The database defaults to `outputs/api/clausetriage.sqlite` (ignored by git), created on first run.
  Run **one server process per database file**.

### Loading the CUAD development contracts

Reviewers pick contracts from the dataset as well as uploading their own. Score the CUAD validation
contracts into the same database, **with the server stopped**:

```bash
python -m scripts.load_dataset_contracts               # all 82 validation contracts
python -m scripts.load_dataset_contracts --limit 5     # the first 5
```

These are the contracts `scripts/export_registers.py` uses: the 20% validation split of
`train_separate_questions.json` (seed 42), which the demo model never trained on. `test.json` is never
opened. Each contract is stored under its CUAD title as `contract_id`, with `source_type` `cuad`, and is
scored by the same service, pipeline, chunking (512 / 64 tokens) and model version as an upload.
Contracts already stored are skipped. Pass the same `--database`, `--bundle` / `--registry-version`,
`--weights` and `--model-version` options as the server.

## Endpoints

Every read takes optional `model_version` and `weight_version_id` query parameters. They default to
the server's model version and the default weight version (the weight set passed with `--weights`,
`v2-team-proposed` unless changed).

| Method and path | What it does | Success |
|---|---|---|
| `GET /` | Links to the docs and the page | 200 |
| `GET /health` | Status, the server's model version and default weight version | 200 |
| `GET /models` | Model versions with stored results, and the server's | 200 |
| `POST /contracts/upload` | Multipart upload, field `file`: a `.pdf`, `.docx` or `.txt` contract. Read immediately, scored in the background | 202 with the job (200 if this exact file is already scored or queued) |
| `POST /contracts/text` | JSON `{"text": "...", "name": "optional display name"}`: pasted text, scored unchanged | 202 / 200 as above |
| `GET /contracts/{id}/status` | The scoring job: `queued`, `running`, `complete` or `failed` (with `error`) | 200 |
| `GET /jobs` | Every scoring job under the server's model version, newest first | 200 |
| `GET /contracts` | Every scored contract ranked for review (rank 1 = open first), with rating counts and source metadata | 200 |
| `GET /contracts/{id}` | Text, source metadata, chunks with all 41 probabilities, detections with thresholds, evidence, Unknowns, clause scores, contract score, rank and `ranked_out_of` | 200 |
| `GET /contracts/{id}/register.csv` | The clause register as CSV ([clause-register.md](clause-register.md)) | 200, download |
| `GET /contracts/{id}/register.json` | The same register as JSON; read it back with `clausetriage.register.export.read_json` | 200, download |
| `GET /weights` | Every weight version (id, `created_by`, `created_at`, description, hash) and the default | 200 |
| `GET /weights/{id}` | One weight version with its full `WeightSet` | 200 |
| `POST /weights` | Body: a `WeightSet` JSON (the same format as `clausetriage/risk/weights/*.json`). Creates a new version and scores every stored contract with it | 201 (200 if identical content already exists) |
| `POST /weights/preview` | Body `{"weight_set": {...}, "contract_ids": [...] or null, "model_version": null}`: the ranking under a candidate weight set. Saves nothing | 200, `saved: false` |
| `GET /review/queue` | The review queue: Unknowns and reviewer conflicts, each with chunk text, pages, reason, probability, threshold, allowed verdicts and status. Query: `contract_id`, `status` (`pending`, `reviewed`, `conflict`), `offset`, `limit` (1 to 500, default 50), `model_version` | 200, a `ReviewQueuePage` |
| `POST /feedback` | Body `{"chunk_id": ..., "category": ..., "verdict": "accept" \| "reject" \| "missing", "reviewer": "name", "note": null, "model_version": null}`: records a verdict, timestamped by the server | 201, the stored `FeedbackRecord` |
| `GET /feedback` | Recorded verdicts, oldest first. Query: `contract_id`, `chunk_id`, `category`, `model_version` | 200 |
| `GET /feedback/resolutions` | Each reviewed (chunk, category) with every reviewer's current verdict, resolved to a label or a conflict. Query: `contract_id`, `status` (`resolved`, `conflict`), `model_version` | 200 |

The last four come from `clausetriage/api/routers/review.py`, which `python -m clausetriage.api` passes to
`create_app(service, routers=[review.router])`; an app built with `create_app(service)` alone doesn't
have them. Verdict rules and queue order are in [feedback-review-queue.md](feedback-review-queue.md).
A verdict that doesn't fit what the model predicted (for example `missing` on a detected category) is
a 400 with the reason, an unknown chunk, contract or model version a 404; nothing is written either way.

Errors are JSON `{"detail": ...}`: 400 for a file or text that can't be read (the reader's message,
for example "This file is named .pdf but isn't a PDF"), 404 for an unknown contract, job, model
version or weight version, 409 for a weight version id that already exists with other values,
413 for an oversized body or file, 415 for an unsupported file type, 422 for an invalid request
body (a list of `loc` / `msg` / `type`, never echoing the input), and a plain 500 for anything
unexpected. No error response contains a stack trace; the server logs them.

A typical session with `curl`:

```bash
B=http://127.0.0.1:8000
curl -F "file=@data/demo_contracts/nasa_kbr_agreement.pdf" $B/contracts/upload    # -> {"contract_id": "nasa-kbr-agreement-8ca1473f042e", "status": "queued", ...}
curl $B/contracts/nasa-kbr-agreement-8ca1473f042e/status                         # poll until "complete"
curl $B/contracts                                                                 # ranking
curl -OJ $B/contracts/nasa-kbr-agreement-8ca1473f042e/register.csv
curl $B/weights/v2-team-proposed > v2.json                                        # edit .weight_set, give it a new weight_version_id
curl -X POST -H "content-type: application/json" --data @v3.json $B/weights
curl "$B/contracts?weight_version_id=v3"
curl "$B/review/queue?status=pending&limit=5"                                     # Unknowns to check
curl -X POST -H "content-type: application/json" -d '{"chunk_id": "<chunk_id from the queue>", "category": "Insurance", "verdict": "accept", "reviewer": "Ana"}' $B/feedback
```

## How it behaves

**Contract ids.** An upload's id is an ASCII slug of its file name (letters and digits, other runs
turned into `-`, at most 60 characters) plus the first 12 hex characters of the SHA-256 of its
extracted text, for example `nasa-kbr-agreement-8ca1473f042e`. Ids are safe in URLs and file names,
the same file uploaded twice gets the same id (and is not rescored), and a different text under the
same name gets a different id. The original file name, with any directory parts and control
characters removed, is kept as `source_name`. Pasted text uses its `name`, or `Pasted text`. CUAD
contracts keep their dataset title; URL-encode it in paths.

**Scoring status.** Uploads are read (and refused with a reason) during the request; scoring runs
afterwards as a FastAPI background task. Job status lives in an `api_jobs` table in the same SQLite
file, separate from the artifact store's tables. A job becomes `complete` only after the store has
committed the whole pipeline result in one transaction; any error marks it `failed`, and a failed
job leaves nothing in the store. When a server (or the dataset loader) starts, it settles jobs a
previous process left `queued` or `running`: `complete` if the store has the result, otherwise
`failed` with "Scoring was interrupted". Uploaded bytes are kept in memory only until their job runs,
so an interrupted upload must be uploaded again. Resubmitting a failed contract queues it again.

**Weight versions.** Versions are immutable. `created_by` comes from the posted `WeightSet` and is not
checked (there are no accounts); `created_at` is set by the store. New ids must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$`. Posting the same id with the same content is a no-op (200);
with different content it is a 409 and nothing changes. A POST does not change the default: uploads
are always scored with the server's `--weights`.

**Scores under a new version** are computed from cached probabilities and stored detections with
`rescore_contracts`, never by rerunning the model:

- `POST /weights` stores the version and the scores of every contract stored under the server's model
  version in one transaction.
- Any read that names a weight version first scores, the same way, any contract stored under that
  model version without a score under it yet (for example one uploaded after the version was
  posted), and saves those scores. So every ranking always covers every scored contract.

A contract's rank is computed when read, among every contract stored under the same model and weight
version, as in the artifact store.

**Preview** calls `rescore_contracts(..., save=False)`: it ranks the contracts (all, or
`contract_ids`) under the candidate weight set and writes nothing, not even the weight version. The
candidate's `weight_version_id` is only a label in the response.

**Registers** are built by `clausetriage.register.export.build_register` from the stored evidence, Unknowns,
clause scores and ranked contract score, with every quote re-checked against the stored text and
page numbers from the stored page starts. They are identical to what `build_register` gives for a
fresh pipeline result ranked among the same contracts.

**Model versions.** A bundle file is stored as `<model_name>@sha256:<first 12 hex of the file>`
(the released demo model is `word_char_tfidf@sha256:6fd84c28600f`), so a locally rebuilt model never
shares rows with the released one. A registry bundle is `clause-classifier-tfidf:<n>`;
`--registry-version` must be a number, not `latest`. `--model-version` overrides either.

## Limits

| Limit | Value | Where |
|---|---|---|
| Upload request body | 26 MB (the 25 MB file limit plus 1 MB of multipart envelope) | `clausetriage/api/limits.py` |
| Uploaded file | 25 MB | `clausetriage/ingest/documents.py` `MAX_FILE_BYTES` |
| Pasted-text request body | 12,000,000 bytes + 1 MB (2,000,000 characters at up to 6 bytes each once JSON-escaped) | `clausetriage/api/limits.py` |
| Pasted text | 2,000,000 characters | `MAX_TEXT_CHARS` |
| Any other request body | 1 MB | `clausetriage/api/limits.py` |
| PDF pages, `.docx` expansion | as in [document-ingestion.md](document-ingestion.md) | `clausetriage/ingest/documents.py` |
| Gradio page uploads | 25 MB, enforced by Gradio | `clausetriage/api/app.py` |

Body limits are checked before the body is read: a declared `Content-Length` over the limit is refused
at once, and a body sent without one (chunked) is counted as it arrives and cut off with 413 as soon as
it passes the limit. Multipart uploads are parsed by Starlette, which spools file parts over 1 MB to
a temporary file that is deleted after the request; the handler then reads at most 25 MB + 1 byte.

## Using it in-process

The follow-up UI work can call the same layer without HTTP. Every method returns the Pydantic model
the matching endpoint returns:

```python
from clausetriage.api.service import ClauseTriageService
from clausetriage.store.artifact_store import ArtifactStore

service = ClauseTriageService(ArtifactStore("outputs/api/clausetriage.sqlite"), pipeline, model_version)
job, queued = service.submit_upload(data, "lease.pdf")      # or submit_text(text, name), submit_document(...)
if queued:
    service.run_job(job.contract_id)                         # synchronous here; the API runs it in the background
service.list_ranked(weight_version_id="v2-team-proposed")
service.get_contract_detail(job.contract_id)
service.get_register(job.contract_id)                        # a ContractRegister
service.preview(candidate_weight_set)
service.review_queue(status="pending")                       # a ReviewQueuePage
service.submit_feedback({"chunk_id": ..., "category": ..., "verdict": "accept", "reviewer": "Ana"})
service.list_feedback(contract_id), service.list_resolutions(status="conflict")
```

New endpoints go in a new `APIRouter` whose handlers take `service: ServiceDep` (from
`clausetriage.api.routers`), passed to `create_app(service, routers=[router])`. No existing handler changes.

## Code

| Module | Contents |
|---|---|
| `clausetriage/api/service.py` | `ClauseTriageService` and its errors; `safe_filename`, `upload_contract_id` |
| `clausetriage/api/jobs.py` | `JobStore` and `JobRecord` (the `api_jobs` table) |
| `clausetriage/api/schemas.py` | Request and response models |
| `clausetriage/api/limits.py` | `BodySizeLimitMiddleware` and the body limits |
| `clausetriage/api/routers/` | `contracts.py`, `weights.py`, `system.py`; `review.py` (queue and feedback, added by `python -m clausetriage.api`); `get_service` / `ServiceDep` |
| `clausetriage/api/app.py` | `create_app` (routers, middleware, error handlers, the reviewer page mount) |
| `clausetriage/ui/reviewer.py` | The reviewer page mounted at `/ui/` ([reviewer-ui.md](reviewer-ui.md)) |
| `clausetriage/api/runtime.py` | Loads the model, tokenizer and weights; `build_service` |
| `clausetriage/api/__main__.py` | `python -m clausetriage.api` |
| `scripts/load_dataset_contracts.py` | Scores the CUAD validation contracts into the store |

Tests: `tests/test_api_service.py`, `test_api_app.py` (every endpoint through FastAPI's test client,
with a keyword model instead of the real one), `test_api_review.py` (the review and feedback
endpoints and service methods), `test_api_limits.py`, `test_api_jobs.py`, `test_api_runtime.py` and
`test_load_dataset_contracts.py`.

## What is not verified

- **Concurrency under load.** Writes are serialized by one in-process lock and SQLite's own locking.
  Many simultaneous uploads, or a second process writing the same database, have not been tested.
- **Time limits.** A crafted PDF can be slow to parse within the size limit, and parsing runs in the
  request with no timeout. Scoring has no timeout either. Neither can be cancelled.
- **Memory.** Queued uploads are held in memory until they run; there is no cap on how many are queued.
- **Scale.** Each detail read issues one query per chunk for probabilities, and rankings rebuild every
  contract score. This was exercised with about ten contracts, not hundreds.
- **Real OCR** (scanned PDFs) has not been run through the API; it needs optional packages
  (see [document-ingestion.md](document-ingestion.md)).
- **Browsers other than the test client and curl.** The reviewer page was driven through
  `gradio_client`, never rendered in a browser (see [reviewer-ui.md](reviewer-ui.md)).
- **Hostile clients beyond input handling:** there is no authentication, rate limiting, CORS policy or
  TLS. Keep it on `127.0.0.1`.
