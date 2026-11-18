"""
Tests for clausetriage/ui/reviewer.py: the reviewer page's handlers called directly over a real store, with the
keyword pipeline of tests/api_fixtures.py and uploads scored synchronously.

Contracts used for the weight previews (one clause each, so a severity-only weight set makes each
triage score equal its one category's severity, by hand):
    A  "This Agreement is governed by Delaware law."                                  Governing Law
    B  "Either party may terminate this Agreement for convenience on 30 days notice."  Termination For Convenience
    C  "The supplier shall maintain adequate coverage."                                Insurance (an Unknown)
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from scripts.load_dataset_contracts import load_contracts
from clausetriage.api.app import create_app
from clausetriage.api.service import INTERNAL_ERROR
from clausetriage.labels.loader import Contract
from clausetriage.store import tables
from clausetriage.ui.reviewer import ReviewerPage, build_reviewer_app
from clausetriage.ui.reviewer_views import ALL_CONTRACTS, DATASET_ONLY, UPLOADS_ONLY
from tests.api_fixtures import TEXT_1, TEXT_2, FailingPredictor, add_text, make_service, weights_a, weights_b
from tests.pdf_fixtures import pdf_bytes
from tests.test_artifact_store import row_counts

TEXT_A = "This Agreement is governed by Delaware law."
TEXT_B = "Either party may terminate this Agreement for convenience on 30 days notice."
TEXT_C = "The supplier shall maintain adequate coverage."
SKIP = {"__type__": "update"}


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A service over a fresh store with the keyword pipeline and the v1 weights as default."""
    built = make_service(tmp_path / "ui.db", monkeypatch)
    yield built
    built.store.close()


@pytest.fixture
def page(service):
    """The reviewer page over service, scoring uploads synchronously."""
    return ReviewerPage(service, run_job=service.run_job)


@pytest.fixture
def abc(service):
    """Contracts A, B and C stored, plus 'sev-base' (severity only: A 0.9, B 0.5, C 0.1), giving ranks A, B, C."""
    ids = {name: add_text(service, text, name) for name, text in (("A", TEXT_A), ("B", TEXT_B), ("C", TEXT_C))}
    service.create_weights(severity_only("sev-base", {"Governing Law": 0.9, "Termination For Convenience": 0.5,
                                                      "Insurance": 0.1}))
    return ids


@pytest.fixture
def alpha_beta(page, tmp_path):
    """TEXT_1 uploaded as 'alpha.txt' and TEXT_2 pasted as 'Beta'; returns their contract ids."""
    path = tmp_path / "alpha.txt"
    path.write_text(TEXT_1, encoding="utf-8")
    page.submit_contracts([str(path)], TEXT_2, "Beta")
    ranked = {c.source_name: c.contract_id for c in page.service.list_ranked().contracts}
    return ranked["alpha.txt"], ranked["Beta"]


def severity_only(version_id: str, severities: dict[str, float]):
    """The v1 weights with every weight on category severity, the worst clause alone ranking, and severities set."""
    return weights_a().model_copy(update={
        "weight_version_id": version_id, "created_by": "tests", "triage_max_weight": 1.0,
        "signal_weights": {"category_severity": 1.0, "one_sidedness": 0.0, "magnitude": 0.0, "contract_context": 0.0},
        "category_severity": {**weights_a().category_severity, **severities}})


def editor_values(page: ReviewerPage, version_id: str, severities: dict | None = None, **changes) -> list:
    """The editor's values (signals, thresholds, triage, severity table) for version_id, with edits applied."""
    view = page.editor_view(version_id)
    values = dict(zip(["signals", "medium", "high", "default", "top_k", "max_weight", "table"],
                      [list(view[1:5]), *view[5:11]]))
    values.update(changes)
    table = values["table"].copy()
    for category, value in (severities or {}).items():
        table.loc[table["Category"] == category, "Severity (0 to 1)"] = value
    return [*values["signals"], values["medium"], values["high"], values["default"], values["top_k"],
            values["max_weight"], table]


def version_rows(store, version_id: str) -> list:
    """Every stored row of a weight version and of the scores under it, as plain tuples."""
    rows = []
    with store.Session() as session:
        for name in ("weight_versions", "clause_scores", "contract_scores"):
            table = tables.Base.metadata.tables[name]
            rows += [tuple(row) for row in session.execute(select(table).where(table.c.weight_version_id == version_id))]
    return rows


def queue_item(page: ReviewerPage, contract_id: str, category: str, status: str = "All") -> tuple[str, dict]:
    """The (item id, items by id) of a contract's queue item for category."""
    *_, items = page.queue_view(status, contract_id, 1, None)
    return next(item_id for item_id, item in items.items() if item["category"] == category), items


# ----- Header and honest labelling -----
def test_disclaimer_says_team_weights_are_unreviewed_and_not_legal_advice(page):
    """With the shipped weights as default, the page says no legal expert has validated them."""
    text = page.disclaimer()
    assert "not legal advice" in text and "`v1-placeholder`" in text
    assert "have not been validated by a legal expert" in text and "no accounts" in text


def test_disclaimer_for_weights_given_as_a_file_says_they_are_not_the_teams(tmp_path, monkeypatch):
    """A default the team did not ship is not described as set by the team."""
    other = make_service(tmp_path / "other.db", monkeypatch, weight_set=weights_b())
    text = ReviewerPage(other, run_job=other.run_job).disclaimer()
    assert "not one of the team's shipped weight sets" in text and "were set by the team" not in text
    other.store.close()


def test_start_lists_versions_and_shows_the_default(page, service):
    """Page load lists every stored version, marks and shows the default, and names its origin."""
    service.create_weights(weights_b())
    version, note, jobs, seen = page.start()
    assert version["value"] == "v1-placeholder"
    assert [value for _, value in version["choices"]] == ["v1-placeholder", "test-b"]
    assert version["choices"][0][0].endswith("· default")
    assert "set by the team, **not validated by a legal expert**" in note and jobs.empty and seen == []


# ----- Dataset contracts and uploads in one ranking -----
def test_dataset_contracts_and_uploads_share_one_ranking(page, service, tmp_path):
    """A CUAD contract loaded by the dataset script, a PDF upload and pasted text are ranked together."""
    load_contracts(service, [Contract(contract_id="ACME CORP - Supply Agreement", text=TEXT_2, spans=[])],
                   progress=lambda line: None)
    pdf = tmp_path / "lease.pdf"
    pdf.write_bytes(pdf_bytes([TEXT_A, TEXT_B]))
    message, *_ = page.submit_contracts([str(pdf)], TEXT_C, "Pasted C")
    assert message.count("queued for scoring") == 2

    note, table, picker = page.ranking_view(None, ALL_CONTRACTS, None)
    expected = [(c.rank, c.contract_id) for c in service.list_ranked().contracts]
    assert list(zip(table["Rank"], table["ID"])) == expected and len(expected) == 3
    assert sorted(table["Source"]) == ["CUAD dataset", "PDF upload", "Pasted text"]
    assert table.loc[table["Source"] == "PDF upload", "Pages"].item() == 2
    assert "**3 contract(s)** ranked under weights `v1-placeholder`" in note
    assert [value for _, value in picker["choices"]] == [cid for _, cid in expected]
    assert list(page.ranking_view(None, DATASET_ONLY, None)[1]["Contract"]) == ["ACME CORP - Supply Agreement"]
    assert len(page.ranking_view(None, UPLOADS_ONLY, None)[1]) == 2


def test_ranking_on_an_empty_store_says_how_to_add_contracts(page):
    """With nothing stored, the note explains the upload and the dataset loader."""
    note, table, picker = page.ranking_view(None, ALL_CONTRACTS, None)
    assert table.empty and picker["choices"] == [] and "scripts.load_dataset_contracts" in note


def test_submit_contracts_refuses_unreadable_inputs_with_reasons(page, service, tmp_path):
    """An unsupported type, an empty file and invalid pasted text are refused with reasons; nothing is stored."""
    (tmp_path / "virus.exe").write_bytes(b"MZ")
    (tmp_path / "empty.txt").write_bytes(b"")
    message, jobs, files, pasted = page.submit_contracts([str(tmp_path / "virus.exe"), str(tmp_path / "empty.txt")],
                                                         "bad \ud800 text", None)
    assert "- **virus.exe**: not accepted. Can't read a file type '.exe'" in message
    assert "- **empty.txt**: not accepted. The file is empty (0 bytes)." in message
    assert "- **Pasted text**: not accepted. The text contains characters that aren't valid Unicode." in message
    assert pasted == {"__type__": "update", "value": "bad \ud800 text"} and files["value"] is None
    assert jobs.empty and service.store.list_contracts() == []


def test_submit_contracts_with_nothing_says_so(page):
    """No files and blank pasted text produce a message, not an error."""
    assert page.submit_contracts([], "   ", None)[0] == "**Nothing to score.** Choose files, or paste contract text."


def test_submitting_the_same_text_again_says_it_is_already_ranked(page, alpha_beta):
    """A repeated paste is not rescored."""
    message = page.submit_contracts(None, TEXT_2, "Beta")[0]
    assert message == "- **Beta**: already scored (the same text is stored), so it is already in the ranking."


def test_background_scoring_shows_progress_then_joins_ranking(service):
    """A queued upload shows as waiting; once scored, the next poll shows it done and in the ranking."""
    queued = []
    page = ReviewerPage(service, run_job=queued.append)
    _, jobs, *_ = page.submit_contracts(None, TEXT_1, "Alpha")
    assert list(jobs["Status"]) == ["Waiting to be scored"] and len(queued) == 1
    jobs_shown, seen, *_ = page.poll_jobs(None, None, ALL_CONTRACTS, None)
    assert page.poll_jobs(seen, None, ALL_CONTRACTS, None) == (SKIP, seen, SKIP, SKIP, SKIP)

    service.run_job(queued[0])
    jobs_shown, seen, note, table, picker = page.poll_jobs(seen, None, ALL_CONTRACTS, None)
    assert list(jobs_shown["Status"]) == ["Done: in the ranking"]
    assert list(table["ID"]) == queued and picker["value"] == queued[0]


def test_failed_scoring_is_shown_with_its_reason_and_not_ranked(tmp_path, monkeypatch):
    """A model crash shows Failed with the generic reason, and the contract stays out of the ranking."""
    failing = make_service(tmp_path / "fail.db", monkeypatch, predictor=FailingPredictor())
    page = ReviewerPage(failing, run_job=failing.run_job)
    _, jobs, *_ = page.submit_contracts(None, TEXT_1, "Alpha")
    assert jobs[["Status", "Problem"]].values.tolist() == [["Failed", INTERNAL_ERROR]]
    assert page.ranking_view(None, ALL_CONTRACTS, None)[1].empty
    failing.store.close()


# ----- Contract detail -----
def test_contract_view_shows_rank_quotes_signals_and_weight_version(page, alpha_beta):
    """The detail names model and weight version, lists clauses with quotes and signals, and highlights quotes."""
    alpha, _ = alpha_beta
    summary, clauses, text, csv, json, detections, views, first, feedback = page.contract_view(alpha, None, False)
    assert "## alpha.txt" in summary and "rank 1 of 2" in summary
    assert "weights `v1-placeholder`" in summary and "Model `keyword-model`" in summary
    assert "have not been validated by a legal expert" in summary
    rows = clauses.set_index("Category")
    assert rows.loc["Governing Law", "Evidence"] == TEXT_A
    assert rows.loc["Insurance", "Evidence"] == "(no quote found: needs manual review)"
    assert rows.loc["Governing Law", "Why (signals)"].startswith("severity 0.25")
    assert {"token": TEXT_A, "class_or_confidence": "Low: Governing Law"} in text.value
    assert [label.split(" · ")[1:3] for label, _ in detections["choices"]] == [
        ["Insurance", "no quote"], ["Termination For Convenience", "quote"], ["Governing Law", "quote"]]
    assert first == views[detections["value"]] and "No quote could be grounded" in first
    assert feedback.empty


def test_contract_downloads_are_the_api_registers_byte_for_byte(page, service, alpha_beta):
    """The CSV and JSON the page offers equal what GET /contracts/{id}/register.csv and .json return."""
    alpha, _ = alpha_beta
    _, _, _, csv, json, *_ = page.contract_view(alpha, None, False)
    with TestClient(create_app(service, ui_path=None)) as client:
        for download, suffix in ((csv, "csv"), (json, "json")):
            with open(download["value"], "rb") as handle:
                assert handle.read() == client.get(f"/contracts/{alpha}/register.{suffix}").content


def test_contract_downloads_are_enabled_and_disabled_but_never_hidden(page, alpha_beta):
    """The download buttons toggle `interactive` and never `visible`: showing both at once hangs Gradio 6.17.3."""
    alpha, _ = alpha_beta
    shown = page.contract_view(alpha, None, False)[3:5]
    empty = page.contract_view(None, None, False)[3:5]
    assert [download["interactive"] for download in shown] == [True, True]
    assert [(download["value"], download["interactive"]) for download in empty] == [(None, False), (None, False)]
    assert all("visible" not in download for download in (*shown, *empty))


def test_contract_view_under_another_weight_version_names_it(page, service, alpha_beta):
    """Switching the version shown rescores from the store and the headline says which version."""
    service.create_weights(weights_b())
    summary, clauses, *_ = page.contract_view(alpha_beta[0], "test-b", False)
    assert "weights `test-b`" in summary
    assert clauses.set_index("Category").loc["Insurance", "Rating"] == "High"


def test_contract_view_of_an_unknown_contract_says_so(page):
    """An id with no stored result gives a message and empty views."""
    summary, clauses, *_ = page.contract_view("nope", None, False)
    assert summary.startswith("**Can't show this contract:** No scored contract 'nope'") and clauses.empty


# ----- Weights: preview and save -----
def test_editor_view_fills_from_the_version_shown(page, abc):
    """The editor holds the version's signal weights, thresholds, triage settings and severities."""
    view = page.editor_view("sev-base")
    assert "**`sev-base`**" in view[0]
    assert list(view[1:10]) == [1.0, 0.0, 0.0, 0.0, 0.35, 0.55, 0.5, 3, 1.0]
    severities = view[10].set_index("Category")["Severity (0 to 1)"]
    assert (severities["Governing Law"], severities["Insurance"]) == (0.9, 0.1)


def test_preview_reports_rank_changes_and_saves_nothing(page, service, abc):
    """Hand-computed: swapping A's and C's severities ranks C, B, A (C up 2, A down 2); no row is written."""
    before = (row_counts(service.store), service.list_weights())
    values = editor_values(page, "sev-base", {"Governing Law": 0.1, "Insurance": 0.9})
    message, table = page.preview_weights("sev-base", "", "", "", *values)
    assert table[["Contract", "Rank now", "Rank in preview", "Move"]].values.tolist() == [
        ["C", 3, 1, "up 2"], ["B", 2, 2, "no change"], ["A", 1, 3, "down 2"]]
    assert list(table["Score in preview"]) == [0.9, 0.5, 0.1] and list(table["Score now"]) == [0.1, 0.5, 0.9]
    assert "nothing was saved" in message and "2 changed rank" in message
    assert "Largest move: **C**, up 2 (rank 3 to 1)" in message
    assert "severity of Governing Law 0.9 to 0.1, Insurance 0.1 to 0.9" in message
    assert (row_counts(service.store), service.list_weights()) == before


def test_save_creates_a_new_version_and_leaves_the_old_one_unchanged(page, service, abc):
    """Saving stores 'sev-swap' by the reviewer, ranks C, B, A under it, and 'sev-base' keeps every row."""
    base_rows = version_rows(service.store, "sev-base")
    values = editor_values(page, "sev-base", {"Governing Law": 0.1, "Insurance": 0.9})
    message, version = page.save_weights("sev-base", "sev-swap", " Ana ", "", *values)
    assert message.startswith("Saved new weight version **`sev-swap`**, created by Ana. 3 stored contract(s)")
    assert version["value"] == "sev-swap" and "sev-swap" in [value for _, value in version["choices"]]
    saved = service.get_weights("sev-swap")
    assert saved.version.created_by == "Ana" and saved.weight_set.description.startswith("Derived from sev-base")
    assert [(c.source_name, c.rank) for c in service.list_ranked(weight_version_id="sev-swap").contracts] == [
        ("C", 1), ("B", 2), ("A", 3)]
    assert version_rows(service.store, "sev-base") == base_rows
    assert "created by **Ana** (a self-declared name)" in page.version_note("sev-swap")


def test_saving_the_same_edit_twice_changes_nothing(page, service, abc):
    """A second save of identical content under the same id stores nothing new and says so."""
    values = editor_values(page, "sev-base", medium=0.3)
    page.save_weights("sev-base", "sev-twice", "Ana", "Draft", *values)
    before = row_counts(service.store)
    message, version = page.save_weights("sev-base", "sev-twice", "Ana", "Draft", *values)
    assert message.startswith("Version `sev-twice` already exists with exactly these values, so nothing was saved.")
    assert version["value"] == "sev-twice" and row_counts(service.store) == before


@pytest.mark.parametrize(("change", "fragment"), [
    ({"signals": [1.5, 0.0, 0.0, 0.0]}, "The category severity weight must be between 0 and 1 (you entered 1.5)."),
    ({"signals": [0.6, 0.2, 0.2, 0.1]}, "The four signal weights must add up to 1 (they add up to 1.1)."),
    ({"medium": 0.6, "high": 0.5}, "The Medium threshold must be below the High threshold."),
    ({"severities": {"Insurance": 1.2}}, "The severity of Insurance must be between 0 and 1 (you entered 1.2)."),
    ({"new_id": "v1-placeholder", "medium": 0.3}, "Weight version 'v1-placeholder' already exists with different values."),
    ({"new_id": "has spaces"}, "weight_version_id must start with a letter or digit"),
    ({"new_id": "  "}, "Give the new version an id"),
    ({"reviewer": ""}, "Type your name in **Your name**"),
    ({"reviewer": "x" * 121}, "Your name is longer than 120 characters."),
])
def test_save_refuses_invalid_edits_with_a_message_and_writes_nothing(page, service, change, fragment):
    """Each refused edit gives a plain message, leaves the version picker alone and stores nothing."""
    change = dict(change)
    new_id, reviewer = change.pop("new_id", "v-new"), change.pop("reviewer", "Ana")
    severities = change.pop("severities", None)
    before = (row_counts(service.store), service.list_weights())
    message, version = page.save_weights("v1-placeholder", new_id, reviewer, "",
                                         *editor_values(page, "v1-placeholder", severities, **change))
    assert message.startswith("**Not saved:** ") and fragment in message
    assert version == SKIP and (row_counts(service.store), service.list_weights()) == before


def test_preview_refuses_invalid_edits_with_a_message(page):
    """An invalid edit gives a message and an empty table instead of a preview."""
    message, table = page.preview_weights("v1-placeholder", "", "", "",
                                          *editor_values(page, "v1-placeholder", medium=0.9))
    assert message == "**Can't preview:** The Medium threshold must be below the High threshold." and table.empty


# ----- Review queue and verdicts -----
def test_queue_lists_unknowns_with_their_passage_and_allowed_verdicts(page, alpha_beta):
    """Both Unknowns are pending; an item shows its category's meaning, reason and passage, with accept/reject."""
    alpha, beta = alpha_beta
    note, table, picker, items = page.queue_view("Needs review", "", 1, None)
    assert note.startswith("**2 need review · 0 where reviewers disagree · 0 reviewed**")
    assert list(table["Category"]) == ["Insurance", "Insurance"] and len(picker["choices"]) == 2
    item_id, items = queue_item(page, alpha, "Insurance")
    summary, passage, accept, reject, missing = page.item_view(item_id, items, "")
    assert "**What this category means (CUAD):** Is there a requirement for insurance" in summary
    assert "no sentence in the chunk contains the category's cue words" in summary
    assert [segment["token"] for segment in passage.value] == [items[item_id]["chunk_text"]]
    assert (accept["visible"], reject["visible"], missing["visible"]) == (True, True, False)
    assert len(page.queue_view("All", beta, 1, None)[1]) == 1


def test_accepting_an_item_moves_it_out_of_the_pending_queue(page, alpha_beta):
    """After accept, the counts show one reviewed, the picker moves to the next item and the note clears."""
    note, table, picker, items = page.queue_view("Needs review", "", 1, None)
    first, second = [value for _, value in picker["choices"]]
    message, note, table, picker, items, note_box = page.accept_item(first, items, "Ana", "seen it", "Needs review",
                                                                     "", 1)
    assert message.startswith("Recorded **accept** (the clause is in this passage) for *Insurance*")
    assert "by Ana" in message and "Scores and rankings do not change." in message
    assert note.startswith("**1 need review · 0 where reviewers disagree · 1 reviewed**")
    assert picker["value"] == second and note_box == ""
    (record,) = page.service.list_feedback()
    assert (record.verdict, record.reviewer, record.note) == ("accept", "Ana", "seen it")


def test_disagreeing_verdicts_become_a_conflict_in_the_queue(page, alpha_beta):
    """Ana accepts and Ben rejects the same item: it is listed as a conflict with both verdicts."""
    item_id, items = queue_item(page, alpha_beta[0], "Insurance")
    page.accept_item(item_id, items, "Ana", "", "All", "", 1)
    page.reject_item(item_id, items, "Ben", "", "All", "", 1)
    note, table, *_ = page.queue_view("Reviewers disagree", "", 1, None)
    assert "1 where reviewers disagree" in note
    assert table[["Status", "Verdicts so far"]].values.tolist() == [["Reviewers disagree", "Ana: accept; Ben: reject"]]


@pytest.mark.parametrize(("verdict", "reviewer", "pick", "fragment"), [
    ("accept", "  ", True, "Type your name in **Your name** at the top of the page first"),
    ("missing", "Ana", True, "already detected this category in this passage, so choose Accept or Reject instead. "
                             "Reason: 'missing' is for clauses the model did not predict"),
    ("accept", "Ana", False, "Pick an item in the queue first."),
])
def test_refused_item_verdicts_show_the_reason_and_write_nothing(page, alpha_beta, verdict, reviewer, pick, fragment):
    """A missing name, a verdict that doesn't fit the prediction, or no item is refused; the note is kept."""
    item_id, items = queue_item(page, alpha_beta[0], "Insurance")
    message, *_, note_box = page.item_verdict(verdict, item_id if pick else None, items, reviewer, "my note", "All",
                                              "", 1)
    assert message.startswith("**Not recorded.**") and fragment in message
    assert note_box == "my note" and page.service.list_feedback() == []


def test_clause_verdicts_from_the_contract_detail_are_recorded(page, alpha_beta):
    """Accept and reject on grounded clauses are stored and listed on the contract, newest first."""
    alpha, _ = alpha_beta
    choices = dict((label.split(" · ")[1], value) for label, value in page.contract_view(alpha, None, False)[5]["choices"])
    message, feedback, note = page.accept_clause(alpha, choices["Governing Law"], "Ana", "clear")
    assert message.startswith("Recorded **accept**") and note == ""
    page.reject_clause(alpha, choices["Termination For Convenience"], "Ana", "")
    message, feedback, note = page.reject_clause(alpha, choices["Termination For Convenience"], "", "keep")
    assert message.startswith("**Not recorded.** Type your name") and note == "keep"
    table = page.contract_view(alpha, None, False)[8]
    assert table[["Reviewer", "Verdict", "Category", "Passage"]].values.tolist() == [
        ["Ana", "reject", "Termination For Convenience", "1"], ["Ana", "accept", "Governing Law", "0"]]


def test_flagging_a_missing_clause_finds_its_passage_and_can_be_disputed(page, alpha_beta):
    """'30 days notice' is found in passage 1; a reject there makes a conflict that offers the missing verdict."""
    alpha, _ = alpha_beta
    message, feedback, note = page.flag_missing(alpha, "Audit Rights", "30 DAYS\nnotice", "Ana", "")
    assert message.startswith("Recorded **missing** (the clause is in this passage and the model missed it) for "
                              "*Audit Rights* in passage 1, by Ana")
    page.service.submit_feedback({"chunk_id": f"{alpha}:1", "category": "Audit Rights", "verdict": "reject",
                                  "reviewer": "Ben"})
    item_id, items = queue_item(page, alpha, "Audit Rights", "Reviewers disagree")
    _, _, accept, reject, missing = page.item_view(item_id, items, "")
    assert (accept["visible"], reject["visible"], missing["visible"]) == (False, True, True)
    message, *_ = page.missing_item(item_id, items, "Cara", "", "All", "", 1)
    assert message.startswith("Recorded **missing**")


@pytest.mark.parametrize(("category", "phrase", "fragment"), [
    ("Insurance", "adequate coverage", "predicted 'Insurance'"),
    ("Audit Rights", "not in this contract", "Those words are not in the contract text."),
    (None, "30 days notice", "Pick the category of the missing clause."),
    ("Audit Rights", " ", "Paste a few words from where the clause is"),
])
def test_flagging_a_missing_clause_is_refused_with_a_reason(page, alpha_beta, category, phrase, fragment):
    """A category the model did predict there, words not in the text, or missing inputs are refused."""
    message, feedback, note = page.flag_missing(alpha_beta[0], category, phrase, "Ana", "n")
    assert message.startswith("**Not recorded.**") and fragment in message and note == "n"
    assert page.service.list_feedback() == []


# ----- The page as served -----
def test_build_reviewer_app_mounts_and_serves_the_page(service, monkeypatch):
    """create_app mounts the reviewer page at /ui/; the page builds against an empty store."""
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    assert build_reviewer_app(service, run_job=service.run_job) is not None
    with TestClient(create_app(service)) as client:
        page = client.get("/ui/")
        assert page.status_code == 200 and "ClauseTriage reviewer" in page.text
