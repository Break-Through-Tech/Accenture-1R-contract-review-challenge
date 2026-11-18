import pytest
from pydantic import ValidationError

from clausetriage.register.schema import ContractRegister, RegisterHeader, RegisterRow

SHA = "a" * 64
SIGNALS = {"category_severity": 1.0, "one_sidedness": 0.5, "magnitude": 0.0, "contract_context": 0.0}


def _row(**changes) -> dict:
    """A valid grounded Uncapped Liability row as a dict, with any fields replaced."""
    row = dict(contract_id="c1", category="Uncapped Liability", rating="High", composite_score=0.625,
               signals=SIGNALS, probability=0.9, chunk_index=0, page=1, start=0, end=4, quote="Text",
               has_evidence=True, needs_manual_review=False, review_reason=None, model_name="m",
               weight_version_id="w", weight_set_sha256=SHA)
    return {**row, **changes}


def _unquoted(**changes) -> dict:
    """A valid manual-review row (no evidence) as a dict, with any fields replaced."""
    return _row(**{"page": None, "start": None, "end": None, "quote": None, "has_evidence": False,
                   "needs_manual_review": True, "review_reason": "no_cue_match", **changes})


def _header(**changes) -> dict:
    """A header matching one grounded High row, with any fields replaced."""
    header = dict(contract_id="c1", triage_score=0.416666667, rank=1, ranked_out_of=2, high_count=1,
                  medium_count=0, low_count=0, unknown_count=0, clause_count=1, manual_review_count=0,
                  model_name="m", model_seed=42, weight_version_id="w", weight_set_sha256=SHA)
    return {**header, **changes}


# ----- RegisterRow -----
def test_register_row_grounded_and_unquoted_rows_are_valid():
    """A row with a consistent quote and a flagged row with no quote both validate."""
    assert RegisterRow(**_row()).quote == "Text"
    unquoted = RegisterRow(**_unquoted())
    assert unquoted.quote is None and unquoted.needs_manual_review


def test_register_row_orders_signals_canonically():
    """Signals come back in SIGNAL_NAMES order whatever order they were given in."""
    row = RegisterRow(**_row(signals=dict(reversed(list(SIGNALS.items())))))
    assert list(row.signals) == ["category_severity", "one_sidedness", "magnitude", "contract_context"]


@pytest.mark.parametrize("fields", [
    _row(quote=None),                                         # evidence without a quote
    _row(end=5),                                              # quote length != end - start
    _row(needs_manual_review=True),                           # a grounded row is not a review row
    _row(review_reason="no_cue_match"),                       # nor does it carry a review reason
    _unquoted(quote="Text", start=0, end=4),                  # an unquoted row with an invented quote
    _unquoted(page=1),                                        # a page without a quote
    _unquoted(needs_manual_review=False),                     # an unquoted row that isn't flagged
    _unquoted(review_reason=None),                            # flagged without saying why
    _row(category="Not A Category"),                          # unknown CUAD category
    _row(signals={"category_severity": 1.0}),                 # missing signals
    _row(composite_score=1.5),                                # score out of [0, 1]
    _row(weight_set_sha256="abc"),                            # not a SHA-256
    _row(extra_column="x"),                                   # unknown field
])
def test_register_row_inconsistent_fields_are_rejected(fields):
    """Each inconsistent combination of evidence, review flag or value raises a ValidationError."""
    with pytest.raises(ValidationError):
        RegisterRow(**fields)


# ----- RegisterHeader -----
def test_register_header_rank_beyond_ranked_out_of_is_rejected():
    """Rank 3 of 2 contracts is impossible."""
    with pytest.raises(ValidationError, match="ranked_out_of"):
        RegisterHeader(**_header(rank=3))


# ----- ContractRegister -----
def test_contract_register_valid_register_round_trips_through_json():
    """A register written to JSON validates back into an equal model."""
    register = ContractRegister(header=_header(), rows=[_row()])
    assert ContractRegister.model_validate_json(register.model_dump_json()) == register


@pytest.mark.parametrize(("header", "rows", "message"), [
    (_header(), [_row(contract_id="c2")], "doesn't match the header"),
    (_header(), [_row(weight_version_id="other")], "doesn't match the header"),
    (_header(clause_count=2), [_row()], "clause_count"),
    (_header(manual_review_count=1), [_row()], "manual_review_count"),
    (_header(high_count=0, medium_count=1), [_row()], "rating counts"),
])
def test_contract_register_header_disagreeing_with_rows_is_rejected(header, rows, message):
    """Rows from another contract or weight version, or counts that don't match the rows, fail validation."""
    with pytest.raises(ValidationError, match=message):
        ContractRegister(header=header, rows=rows)


def test_contract_register_rating_counts_use_worst_clause_per_category():
    """Two clauses of one category count once, with the worse clause's rating (as ContractScore counts)."""
    rows = [_row(), _row(chunk_index=1, composite_score=0.2, rating="Low")]
    register = ContractRegister(header=_header(clause_count=2), rows=rows)
    assert register.header.high_count == 1 and register.header.low_count == 0
