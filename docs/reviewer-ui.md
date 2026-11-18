# Reviewer page

The reviewer page is where a reviewer works through the whole triage loop on stored contracts. It
ranks contracts, shows the clauses and signals behind each score, lets you try other risk weights,
and records your verdicts on what the model found. It is written for a legal professional; the
last sections are for the team.

> **Research prototype. Not legal advice.** Ratings come from rules with adjustable weights. The
> default weights (`v2-team-proposed`) are the team's proposal and **have not been reviewed by the
> advisor**. The page says so at the top and next to every contract.

## Starting it

The page is served by the API server, which needs the full install
(`pip install -r requirements.txt`, or `requirements-no-torch.txt`; see [development.md](development.md)).
From the repo root:

```bash
python -m scripts.load_dataset_contracts --limit 10   # optional: score CUAD validation contracts (server stopped)
python -m src.api                                     # then open http://127.0.0.1:8000/ui/
```

- The page and the API share one process and one database (`outputs/api/clausetriage.sqlite` by
  default; `--database` picks another). See [api.md](api.md) for every server option.
- The server listens on `127.0.0.1` only. There is no login: the names you type are taken as given.
- The quick demo, `python -m src.ui.app`, is a different page. It scores uploads in memory, saves
  nothing, and has no weights editor or review queue. Use it for a demo without the full install.

## At the top of the page

- **Your name.** Saved with every verdict and with any weight version you create. Nothing is
  recorded until you type it.
- **Weight version shown.** Every score, rating, rank and register on the page uses this version.
  Switching rescores the stored contracts from the model's saved probabilities; it never reruns the
  model and never changes a version. The line under it says who created the version and whether
  it is the team's unreviewed proposal.

## Contracts tab

**Add contracts.** Upload PDF, Word (`.docx`) or text files, or paste text (with an optional name),
and press **Score and add to the ranking**. Each input gets one line in the message:

- *queued for scoring*: it is scored in the background. The status table below it shows
  *Waiting to be scored*, *Scoring now*, *Done: in the ranking* or *Failed* with the reason. The
  ranking refreshes when a job finishes (the page checks every 2 seconds).
- *not accepted* with the reason, for example a damaged PDF or a password-protected Word file.
  Files of any other type are refused by the upload box itself. Nothing is stored for them.
- *already scored*: the same text is already stored, so it is not scored again.

Uploads are kept in the database. A file is identified by its text, so uploading it again under
another name is "already scored", while a changed text is a new contract.

**The ranking.** Every stored contract is ranked together under the version shown, whether it came
from the CUAD dataset or was uploaded. Rank 1 is the one to open first, and tied contracts share a
rank. **Show** narrows the list to the dataset or to uploads, but the ranks stay those of the whole
list. The *Verdict* column reads *Open first* when any clause type is rated High, *Review* when any
is Medium, and *Low priority* otherwise. *No quote* counts the clause types the model detected
without a sentence it could point to.

**One contract.** Click a row, or pick it under **Contract**, to see:

- the headline: verdict, triage score, rank, rating counts, source, page count, and the model and
  weight version used;
- **Risky clauses**: each detected clause type's worst clause, with its rating, risk score, page,
  quote and the four signal values behind the score (severity, one-sidedness, magnitude, contract
  context). Tick **Show harmless clause types** to include Parties and dates;
- the contract text with every quoted clause highlighted in its rating's colour;
- **Clause register (CSV / JSON)**: the same files `GET /contracts/{id}/register.csv` and `.json`
  return, byte for byte, under the version shown.

**Verdicts on this contract** (folded away under the text):

- Pick **a detected clause** to see its quote, or the passage it came from when there is no quote.
  Then press **Accept** (the clause is there) or **Reject** (it is not).
- **A clause the model missed:** pick its category, paste a few words from it (capitals and line
  breaks don't matter), and press **Flag as missing**. The page finds the words, takes the passage
  that holds them, and records *missing* for that category there. If the words occur more than once,
  the first occurrence is used, and the message says so.
- The table lists every verdict on the contract, newest first.

## Weights tab

The editor starts from the version shown at the top. You can change:

- the four **signal weights**, each between 0 and 1, adding up to 1;
- the **Medium** and **High** thresholds (Medium below High);
- **Clauses averaged** and **Weight of the worst clause**, which set how a contract's triage score
  mixes its worst clause with the average of its worst few clause types;
- the **default severity**, and the **severity** (0 to 1) and **reason** of each of the 41 categories.
  A category marked *(uses the default severity)* follows the default unless you change it.

Context rules, phrase lists and duration rules are kept as they are in the version shown. To change
those, post a weight set to `POST /weights` (see [api.md](api.md) and
[risk-weights-v2.md](risk-weights-v2.md)).

**Preview ranking (saves nothing).** This re-ranks every stored contract under your edits, using the
model's saved probabilities. It compares the result with the ranking under the version shown. The
table lists each contract's rank and triage score now and in the preview, and how it moved (*up 2*,
*down 1*, *no change*). The message gives how many contracts changed rank, the largest move, and
what you changed. Nothing is written to the database, not even a draft.

**Save as new version.** This stores your edits as a new version, with your name as its creator,
and scores every stored contract with it. The page then shows it. You must give it a new id: letters,
digits, `.`, `_` and `-`. Existing versions never change. If you leave the description empty, one is
written for you that names the base version, you, the date and what changed. The default version for
new uploads stays the one the server was started with.

Edits that can't be saved are refused with a message, and nothing is stored. For example:

- a weight outside 0 to 1;
- weights that don't add up to 1;
- the Medium threshold not below High;
- a severity that isn't a number;
- an id that is already used for other values;
- an id with spaces or slashes;
- no id;
- no name.

Saving the identical edit twice under the same id stores nothing the second time.

## Review tab

The queue lists two kinds of item:

- **Unknowns.** The model detected a clause type in a passage (its probability cleared the
  threshold), but no sentence in the passage could be quoted as evidence. Nothing was guessed, so a
  person has to read the passage.
- **Conflicts.** Two or more reviewers gave verdicts that disagree.

Filter by status (*Needs review*, *Reviewers disagree*, *Reviewed*, *All*) and by contract, with 25
items per page. Conflicts come first, then the items the model was surest about. The counts at the
top cover every item in the queue.

Click a row, or pick an **Item**, to see:

- the contract, pages and passage number;
- what the CUAD category means;
- why the item is in the queue;
- the model's probability against its threshold;
- the verdicts given so far;
- the passage itself.

An Unknown has no quote, so nothing in its passage is highlighted. A passage is about 2,000
characters. Use **Highlight in passage** to mark a word or phrase (a defined term, *renew*,
*insurance*) wherever it appears. A conflict on a quoted clause shows the model's quote highlighted.

Only the verdicts that apply to the item are offered:

| Verdict | Means | Offered for |
|---|---|---|
| **Accept** | the clause is in this passage | anything the model detected |
| **Reject** | the clause is not in this passage | anything detected, or flagged as missing by someone |
| **Missing** | the clause is in this passage and the model did not detect it | a conflict over a missed clause |

After a verdict the queue reloads and moves on to the next item. A verdict that doesn't fit is
refused, with a plain sentence and the store's own reason, and nothing is written. For example,
*Missing* on a category the model did detect is refused, as is any verdict without a name.

### What a verdict does, and what it doesn't

- It is saved in the server's database with your name, the time and your note. The log is never
  edited: changing your mind adds a new verdict, and your latest one counts.
- It does **not** change any score, rating or ranking, and it does **not** retrain the model.
- Reviewers who disagree are shown as a conflict. There is no majority vote and no senior reviewer.
- Verdicts are kept as a review record. The team's training export
  ([feedback-review-queue.md](feedback-review-queue.md)) **leaves out every verdict on the CUAD
  validation contracts on purpose**, because those are the contracts the model is evaluated on. It
  also leaves out uploads unless the team adds them deliberately. Verdicts given here do not train
  the model.

## For the team

**Code.** The page is `src/ui/reviewer.py`: `ReviewerPage` holds one handler per action and
`build_reviewer_app()` wires the handlers to components. Its views are pure helpers in
`src/ui/reviewer_views.py`. The clause table and highlighting reuse `src/ui/report.py`, so they match
the quick demo. `create_app()` mounts the page at `/ui/`. The page reads and writes only through
`ClauseTriageService` (`src/api/service.py`), the same methods the endpoints call. It never touches
`ArtifactStore` directly.

**Background scoring.** Uploads from the page are scored one at a time on a background thread that
the page owns. Uploads over HTTP use FastAPI background tasks. Both call `service.run_job`, which
claims each job once.

**Gradio API.** Gradio exposes the page's handlers under `/ui/gradio_api/`, so `gradio_client` can
drive the page. The supported API for other clients is the HTTP API in [api.md](api.md).

**Tests:**

- `tests/test_ui_reviewer.py` calls the handlers directly over a real store, with the keyword
  pipeline. It covers the shared ranking, uploads and failures, the detail and downloads, preview
  and save (hand-computed rank moves), every refused edit, and every verdict path and refusal.
- `tests/test_ui_reviewer_views.py` covers the pure helpers.
- `tests/test_api_review.py` covers the HTTP endpoints.

## What is verified and what is not

**Verified:**

- **By tests.** Everything listed under *For the team*.
- **Against a running server** (`python -m src.api` on a scratch database, the released TF-IDF
  model, and 4 CUAD validation contracts loaded with the dataset script), driven through
  `gradio_client`:
  - an upload of `data/demo_contracts/nasa_kbr_agreement.pdf` (16 pages), pasted text, and a damaged
    PDF that was refused;
  - progress polling, the shared ranking and its filters;
  - the contract detail with pages and quotes; the downloads were byte-identical to the API's;
  - accept, reject, missing and the refusals, from the detail and from the queue;
  - preview (nothing saved), four refused saves, and a save. The old version was unchanged over
    HTTP, and the default ranking was unchanged;
  - switching versions;
  - a conflict between two reviewers, seen in the page and over `GET /review/queue`, `/feedback` and
    `/feedback/resolutions`.

**Not verified:**

- **The page in a browser.** It has not been rendered or clicked in a browser: no layout, colour,
  row-click or timer behaviour has been seen. All the checks above called the page's handlers through
  Gradio's API. The chained refreshes (switching the version reloads the ranking, contract and
  editor) are wired, but they have only been exercised one handler at a time.
- **A real reviewer.** No legal professional has used the page. The wording and the layout of the
  tabs are untested with users.
- **Scale.** Exercised with 6 contracts. Opening a contract reads its probabilities with one query
  per passage, and every ranking rebuilds all contract scores, so with all 82 validation contracts
  expect each action to take noticeably longer. The queue shows 25 items per page.
- **Several reviewers at once** on one server: writes are serialised, but this has not been tested
  with concurrent sessions.
- **Rank changes in the real-data preview.** The two previews tried on the 6 stored contracts moved
  no ranks, because those contracts are far apart. The rank-change table was checked by tests with
  hand-computed moves, not on real data.
- **Weights quality.** Whether any edited weights are better is not something the page can tell; see
  [risk-weights-v2.md](risk-weights-v2.md) and [advisor-agreement.md](advisor-agreement.md).
