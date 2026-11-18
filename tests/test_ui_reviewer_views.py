"""Tests for clausetriage/ui/reviewer_views.py: the reviewer page's pure tables, rank comparison, weight edits and passages."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from clausetriage.api.jobs import JobRecord
from clausetriage.api.schemas import ContractRanking, RankedContract
from clausetriage.labels.categories import CATEGORIES
from clausetriage.review.schema import ReviewItem, ReviewQueuePage
from clausetriage.store.records import ChunkRecord
from clausetriage.ui.reviewer_views import (CHANGE_COLUMNS, DATASET_ONLY, QUOTE_LABEL, SEARCH_LABEL, UPLOADS_ONLY,
                                   USES_DEFAULT, WeightEditError, candidate_weight_set, category_descriptions,
                                   chunk_for_span, contract_choices, describe_changes, item_segments, jobs_table,
                                   locate_phrase, move_label, preview_summary, queue_table, rank_changes,
                                   ranked_verdict, ranking_table, severity_table)
from tests.test_artifact_store import weights_a

T0 = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


def ranked(contract_id: str, rank: int, score: float, source_type: str = "cuad", high: int = 0,
           medium: int = 0) -> RankedContract:
    """A ranked contract with the given rank, score, source and High/Medium counts."""
    return RankedContract(contract_id=contract_id, rank=rank, triage_score=score, high_count=high,
                          medium_count=medium, low_count=1, unknown_count=0, source_name=f"{contract_id}.pdf",
                          source_type=source_type)


def ranking(*contracts: RankedContract, version: str = "v-now") -> ContractRanking:
    """A ranking of the given contracts under weight version `version`."""
    return ContractRanking(model_version="m", weight_version_id=version, weight_set_sha256="0" * 64,
                           ranked_out_of=len(contracts), contracts=list(contracts))


def editor_args(base=None, **changes) -> dict:
    """candidate_weight_set keyword arguments with base's own values, overridden by changes."""
    base = base or weights_a()
    args = dict(weight_version_id="edit-1", created_by="Ana", description="", signal_weights=dict(base.signal_weights),
                medium_threshold=base.medium_threshold, high_threshold=base.high_threshold,
                default_severity=base.default_severity, triage_top_k=base.triage_top_k,
                triage_max_weight=base.triage_max_weight, severities=severity_table(base))
    args.update(changes)
    return args


def with_severity(table: pd.DataFrame, category: str, value) -> pd.DataFrame:
    """A copy of the severity table with one category's severity replaced."""
    edited = table.copy()
    edited.loc[edited["Category"] == category, "Severity (0 to 1)"] = value
    return edited


# ----- Ranking and the dataset/upload filter -----
@pytest.mark.parametrize(("high", "medium", "expected"), [(1, 3, "Open first"), (0, 2, "Review"), (0, 0, "Low priority")])
def test_ranked_verdict_follows_worst_rating(high, medium, expected):
    """Any High clause type means open first; any Medium means review; else low priority."""
    assert ranked_verdict(ranked("c", 1, 0.5, high=high, medium=medium)) == expected


def test_ranking_table_and_choices_filter_by_source_keeping_global_ranks():
    """Dataset contracts and uploads share one ranking; the filter hides rows but keeps their ranks."""
    shown = ranking(ranked("cuad-a", 1, 0.9), ranked("upload-b", 2, 0.5, "pdf"), ranked("cuad-c", 3, 0.1))
    assert list(ranking_table(shown)["ID"]) == ["cuad-a", "upload-b", "cuad-c"]
    assert list(ranking_table(shown)["Source"]) == ["CUAD dataset", "PDF upload", "CUAD dataset"]
    assert list(ranking_table(shown, DATASET_ONLY)["Rank"]) == [1, 3]
    assert contract_choices(shown, UPLOADS_ONLY) == [("#2 · upload-b.pdf", "upload-b")]


# ----- Comparing a preview with the ranking shown -----
@pytest.mark.parametrize(("now", "preview", "expected"), [
    (3, 1, "up 2"), (1, 3, "down 2"), (2, 2, "no change"), (None, 1, "new"),
])
def test_move_label_counts_places_toward_rank_one_as_up(now, preview, expected):
    """Hand-computed moves: rank 3 to 1 is up 2, 1 to 3 is down 2."""
    assert move_label(now, preview) == expected


def test_rank_changes_lists_preview_order_with_moves_and_scores():
    """Hand-computed: a and c swap places around b; rows follow the preview ranking."""
    now = ranking(ranked("a", 1, 0.9), ranked("b", 2, 0.5), ranked("c", 3, 0.1))
    preview = ranking(ranked("c", 1, 0.9), ranked("b", 2, 0.5), ranked("a", 3, 0.1), version="cand")
    table = rank_changes(now, preview)
    assert list(table.columns) == CHANGE_COLUMNS
    assert table[["ID", "Rank now", "Rank in preview", "Move"]].values.tolist() == [
        ["c", 3, 1, "up 2"], ["b", 2, 2, "no change"], ["a", 1, 3, "down 2"]]
    assert list(table["Score now"]) == [0.1, 0.5, 0.9]
    summary = preview_summary(now, preview, ["severity of Insurance 0.1 to 0.9"])
    assert "nothing was saved" in summary and "2 changed rank" in summary
    assert "Largest move: **c.pdf**, up 2 (rank 3 to 1)" in summary
    assert "severity of Insurance 0.1 to 0.9" in summary


def test_rank_changes_with_ties_shares_ranks():
    """Two contracts tied under the preview share rank 1; the one that was 2nd moves up 1."""
    now = ranking(ranked("a", 1, 0.9), ranked("b", 2, 0.5))
    preview = ranking(ranked("a", 1, 0.5), ranked("b", 1, 0.5))
    assert rank_changes(now, preview)[["ID", "Move"]].values.tolist() == [["a", "no change"], ["b", "up 1"]]
    assert "1 changed rank" in preview_summary(now, preview)


# ----- Jobs -----
def test_jobs_table_shows_uploads_newest_first_and_leaves_out_dataset_contracts():
    """Dataset jobs are hidden; a failed upload shows its reason."""
    def job(contract_id, status, source_type, error=None):
        """A job record at T0."""
        return JobRecord(contract_id=contract_id, model_version="m", weight_version_id="w", status=status,
                         error=error, source_name=f"{contract_id}.pdf", source_type=source_type, submitted_at=T0,
                         updated_at=T0)
    table = jobs_table([job("new", "failed", "pdf", "This PDF is damaged."), job("cuad", "complete", "cuad"),
                        job("old", "queued", "txt")])
    assert table[["Contract", "Status", "Problem"]].values.tolist() == [
        ["new.pdf", "Failed", "This PDF is damaged."], ["old.pdf", "Waiting to be scored", ""]]


# ----- Finding where a missing clause is -----
@pytest.mark.parametrize(("phrase", "expected"), [
    ("governed by", (18, 29, 1)),
    ("GOVERNED   BY\nDelaware", (18, 38, 1)),
    ("agreement", (5, 14, 2)),
    ("not there", None),
    ("   ", None),
])
def test_locate_phrase_ignores_case_and_whitespace_runs(phrase, expected):
    """Hand-computed offsets in a two-sentence text; any whitespace run matches any other."""
    text = "This Agreement is governed by\nDelaware law. This agreement ends."
    assert locate_phrase(text, phrase) == expected


@pytest.mark.parametrize(("start", "end", "expected"), [
    (10, 30, 0), (50, 70, 1), (70, 90, 2), (35, 95, 0), (95, 99, 2), (200, 201, None),
])
def test_chunk_for_span_prefers_a_chunk_holding_the_whole_span(start, end, expected):
    """Chunks [0:40], [30:80], [60:100]: the first holding the whole span, else the one holding its start."""
    chunks = [ChunkRecord(chunk_id=f"c:{i}", contract_id="c", chunk_index=i, start=s, end=e, text="x" * (e - s))
              for i, (s, e) in enumerate([(0, 40), (30, 80), (60, 100)])]
    found = chunk_for_span(chunks, start, end)
    assert (found.chunk_index if found else None) == expected


# ----- Weight edits -----
def test_severity_table_lists_all_categories_with_default_marked():
    """Every CUAD category appears once; v1 leaves most at the default severity, marked as such."""
    table = severity_table(weights_a())
    assert sorted(table["Category"]) == sorted(CATEGORIES)
    row = table[table["Category"] == "Audit Rights"].iloc[0]
    assert (row["Severity (0 to 1)"], row["Reason"]) == (weights_a().default_severity, USES_DEFAULT)


def test_candidate_keeps_unchanged_defaults_implicit_and_records_changes():
    """Only an edited category gets its own severity; the auto description names base, creator and changes."""
    base = weights_a()
    candidate = candidate_weight_set(base, **editor_args(
        severities=with_severity(severity_table(base), "Audit Rights", 0.75), medium_threshold=0.3))
    assert dict(candidate.category_severity) == {**base.category_severity, "Audit Rights": 0.75}
    assert (candidate.weight_version_id, candidate.created_by, candidate.medium_threshold) == ("edit-1", "Ana", 0.3)
    assert candidate.cooccurrence_rules == base.cooccurrence_rules
    assert candidate.description.startswith("Derived from v1-placeholder in the reviewer page by Ana")
    assert "Medium threshold 0.35 to 0.3" in candidate.description
    assert "severity of Audit Rights 0.5 to 0.75" in candidate.description
    assert describe_changes(base, candidate) == ["Medium threshold 0.35 to 0.3", "severity of Audit Rights 0.5 to 0.75"]


def test_candidate_with_typed_description_keeps_it():
    """A description the reviewer typed is used as given (stripped)."""
    assert candidate_weight_set(weights_a(), **editor_args(description="  Team draft  ")).description == "Team draft"


@pytest.mark.parametrize(("changes", "message"), [
    ({"signal_weights": {"category_severity": 1.5, "one_sidedness": 0, "magnitude": 0, "contract_context": 0}},
     "The category severity weight must be between 0 and 1 (you entered 1.5)."),
    ({"signal_weights": {"category_severity": 0.6, "one_sidedness": 0.2, "magnitude": 0.2, "contract_context": 0.1}},
     "The four signal weights must add up to 1 (they add up to 1.1)."),
    ({"medium_threshold": 0.6, "high_threshold": 0.5}, "The Medium threshold must be below the High threshold."),
    ({"high_threshold": None}, "The High threshold is empty; enter a number."),
    ({"triage_top_k": 0}, "Clauses averaged must be at least 1."),
    ({"triage_top_k": 2.5}, "Clauses averaged must be a whole number (you entered 2.5)."),
    ({"default_severity": "high"}, "The default severity must be a number (you entered 'high')."),
])
def test_candidate_refuses_invalid_values_in_plain_words(changes, message):
    """Each invalid edit is a WeightEditError whose message names the field and the rule."""
    with pytest.raises(WeightEditError) as raised:
        candidate_weight_set(weights_a(), **editor_args(**changes))
    assert str(raised.value) == message


@pytest.mark.parametrize(("value", "message"), [
    (1.2, "The severity of Insurance must be between 0 and 1 (you entered 1.2)."),
    ("abc", "The severity of Insurance must be a number (you entered 'abc')."),
])
def test_candidate_refuses_invalid_severity(value, message):
    """A bad value in the severity table names its category."""
    table = with_severity(severity_table(weights_a()).astype({"Severity (0 to 1)": object}), "Insurance", value)
    with pytest.raises(WeightEditError) as raised:
        candidate_weight_set(weights_a(), **editor_args(severities=table))
    assert str(raised.value) == message


def test_candidate_refuses_a_severity_table_missing_categories():
    """A table without all 41 categories can't be turned into a weight set."""
    with pytest.raises(WeightEditError, match="each of the 41 CUAD categories"):
        candidate_weight_set(weights_a(), **editor_args(severities=severity_table(weights_a()).iloc[:40]))


# ----- Review queue items -----
def item(**changes) -> ReviewItem:
    """A pending Unknown queue item over the passage 'The supplier shall maintain coverage.' at offset 100."""
    data = dict(source="unknown", contract_id="c", chunk_id="c:2", chunk_index=2, category="Insurance",
                probability=0.7, threshold=0.5, detection="unknown", reason="no_cue_match", reason_text="why",
                item_id="c:2|Insurance|m", model_version="m", chunk_start=100, chunk_end=137,
                chunk_text="The supplier shall maintain coverage.", allowed_verdicts=["accept", "reject"],
                status="pending")
    data.update(changes)
    return ReviewItem(**data)


def test_item_segments_marks_quote_and_search_matches_without_overlap():
    """The quote is labelled; search matches outside it are labelled; a match inside the quote is not."""
    grounded = item(evidence_quote="maintain coverage", evidence_start=119, evidence_end=136)
    assert item_segments(grounded, "supplier") == [
        ("The ", None), ("supplier", SEARCH_LABEL), (" shall ", None), ("maintain coverage", QUOTE_LABEL), (".", None)]
    assert item_segments(grounded, "coverage") == [
        ("The supplier shall ", None), ("maintain coverage", QUOTE_LABEL), (".", None)]


def test_item_segments_for_an_unknown_is_the_plain_passage_until_searched():
    """An Unknown has no quote: nothing is highlighted until the reviewer searches."""
    assert item_segments(item()) == [("The supplier shall maintain coverage.", None)]
    assert item_segments(item(), "SHALL") == [("The supplier ", None), ("shall", SEARCH_LABEL),
                                              (" maintain coverage.", None)]


def test_queue_table_numbers_rows_from_the_page_offset():
    """Rows are numbered from offset + 1 and show status, reason, pages and verdicts in words."""
    page = ReviewQueuePage(model_version="m", offset=25, limit=25, total=26, status_counts={}, items=[
        item(first_page=3, last_page=4)])
    row = queue_table(page).iloc[0]
    assert (row["#"], row["Status"], row["Why it is here"], row["Pages"], row["ID"]) == (
        26, "Needs review", "No quote found", "pp. 3-4", "c:2|Insurance|m")


def test_category_descriptions_cover_every_category():
    """Every canonical category has its CUAD description, without the CSV's 'Description:' prefix."""
    descriptions = category_descriptions()
    assert set(descriptions) == set(CATEGORIES)
    assert descriptions["Insurance"].startswith("Is there a requirement for insurance")
