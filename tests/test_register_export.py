import json

import pytest

from src.evidence.extractor import GroundedEvidence, Unknown
from src.evidence.schema import Evidence
from src.register.export import (CSV_COLUMNS, INDEX_COLUMNS, GroundingError, build_register, build_registers,
                                 check_page_starts, index_frame, page_of, read_json, register_file_stem,
                                 register_frame, to_json, write_csv, write_json, write_register)
from src.risk.contract import aggregate_contracts
from src.risk.schema import ClauseScore, WeightSet

# An explicit weight set, so a change of the repo's default weights can't move these golden values
WEIGHTS = WeightSet(
    weight_version_id="test-register", created_by="tests", default_severity=0.5,
    signal_weights={"category_severity": 0.55, "one_sidedness": 0.15, "magnitude": 0.15, "contract_context": 0.15},
    medium_threshold=0.35, high_threshold=0.55, triage_top_k=3, triage_max_weight=0.5,
)
SHA = WEIGHTS.content_sha256

# Contract A: two pages starting at offsets 0 and 36; Anti-Assignment has no quote (Unknown queue)
ACME = "ACME Corp, Inc. - Supply Agreement"
ACME_TEXT = ("Supplier's liability is unlimited.\n\n"           # [0:34], page 1
             "This Agreement is governed by Delaware law.\n\n"  # [36:79], page 2
             "Neither party may assign.")                         # [81:106], no quote found
ACME_PAGES = (0, 36)
LIABILITY = Evidence(quote="Supplier's liability is unlimited.", start=0, end=34)
LAW_ON_PAGE_2 = Evidence(quote="This Agreement is governed by Delaware law.", start=36, end=79)
# Contract B: plain text with no pages
BETA = "beta-llc"
BETA_TEXT = "This Agreement is governed by Delaware law."
LAW = Evidence(quote=BETA_TEXT, start=0, end=43)


def _clause(contract_id: str, category: str, chunk: int, composite: float, rating: str, probability: float,
            signals: tuple[float, float, float, float], has_evidence: bool) -> ClauseScore:
    """A hand-built ClauseScore under WEIGHTS (composites are given, not recomputed)."""
    names = ("category_severity", "one_sidedness", "magnitude", "contract_context")
    return ClauseScore(contract_id=contract_id, chunk_index=chunk, category=category, probability=probability,
                       signals=dict(zip(names, signals)), composite_score=composite, rating=rating,
                       has_evidence=has_evidence, weight_version_id="test-register", weight_set_sha256=SHA)


def _result(contract_id: str, clauses: list[ClauseScore], grounded: list[GroundedEvidence],
            unknown: list[Unknown]) -> dict:
    """A predict_contract-shaped result; contract_score is the contract ranked on its own."""
    return {"contract_id": contract_id, "model_name": "test-model", "seed": 42, "chunks": [],
            "grounded": grounded, "unknown": unknown, "clause_scores": clauses,
            "contract_score": aggregate_contracts(clauses, WEIGHTS, [contract_id])[0]}


ACME_RESULT = _result(
    ACME,
    [_clause(ACME, "Governing Law", 1, 0.1375, "Low", 0.8, (0.25, 0.0, 0.0, 0.0), True),
     _clause(ACME, "Uncapped Liability", 0, 0.625, "High", 0.9, (1.0, 0.5, 0.0, 0.0), True),
     _clause(ACME, "Anti-Assignment", 1, 0.35, "Medium", 0.6, (0.5, 0.0, 0.0, 0.5), False)],
    [GroundedEvidence(contract_id=ACME, chunk_index=0, category="Uncapped Liability", evidence=LIABILITY),
     GroundedEvidence(contract_id=ACME, chunk_index=1, category="Governing Law", evidence=LAW_ON_PAGE_2)],
    [Unknown(contract_id=ACME, chunk_index=1, category="Anti-Assignment", reason="no_cue_match")],
)
BETA_RESULT = _result(
    BETA, [_clause(BETA, "Governing Law", 0, 0.1375, "Low", 0.7, (0.25, 0.0, 0.0, 0.0), True)],
    [GroundedEvidence(contract_id=BETA, chunk_index=0, category="Governing Law", evidence=LAW)], [],
)
EMPTY_RESULT = _result("empty", [], [], [])

# Hand-computed triage scores (k = 3, m = 0.5): 0.5 * worst + 0.5 * (sum of the 3 worst) / 3
ACME_TRIAGE = 0.497916667  # 0.5 * 0.625 + 0.5 * (0.625 + 0.35 + 0.1375) / 3 = 0.3125 + 0.18541666...
BETA_TRIAGE = 0.091666667  # 0.5 * 0.1375 + 0.5 * 0.1375 / 3 = 0.06875 + 0.02291666...

GOLDEN_ACME_CSV = (
    ",".join(CSV_COLUMNS) + "\n"
    f'"{ACME}",1,{ACME_TRIAGE},Uncapped Liability,High,0.625,1.0,0.5,0.0,0.0,0.9,0,1,0,34,'
    f"Supplier's liability is unlimited.,True,False,,test-model,test-register,{SHA}\n"
    f'"{ACME}",1,{ACME_TRIAGE},Anti-Assignment,Medium,0.35,0.5,0.0,0.0,0.5,0.6,1,,,,,False,True,no_cue_match,'
    f"test-model,test-register,{SHA}\n"
    f'"{ACME}",1,{ACME_TRIAGE},Governing Law,Low,0.1375,0.25,0.0,0.0,0.0,0.8,1,2,36,79,'
    f"This Agreement is governed by Delaware law.,True,False,,test-model,test-register,{SHA}\n"
)
GOLDEN_BETA_JSON = {
    "header": {
        "contract_id": BETA, "triage_score": BETA_TRIAGE, "rank": 2, "ranked_out_of": 3, "high_count": 0,
        "medium_count": 0, "low_count": 1, "unknown_count": 0, "clause_count": 1, "manual_review_count": 0,
        "model_name": "test-model", "model_seed": 42, "weight_version_id": "test-register", "weight_set_sha256": SHA,
    },
    "rows": [{
        "contract_id": BETA, "category": "Governing Law", "rating": "Low", "composite_score": 0.1375,
        "signals": {"category_severity": 0.25, "one_sidedness": 0.0, "magnitude": 0.0, "contract_context": 0.0},
        "probability": 0.7, "chunk_index": 0, "page": None, "start": 0, "end": 43, "quote": BETA_TEXT,
        "has_evidence": True, "needs_manual_review": False, "review_reason": None, "model_name": "test-model",
        "weight_version_id": "test-register", "weight_set_sha256": SHA,
    }],
}


def _ranked() -> list:
    """The three test contracts ranked together, with texts and ACME's page starts."""
    return build_registers([BETA_RESULT, EMPTY_RESULT, ACME_RESULT], WEIGHTS,
                           texts={ACME: ACME_TEXT, BETA: BETA_TEXT}, page_starts={ACME: ACME_PAGES})


# ----- Golden exports -----
def test_write_csv_matches_golden_register_with_pages_and_unknown_row(tmp_path):
    """ACME's CSV: riskiest first, pages 1 and 2, and the Unknown row kept with blanks and a review flag."""
    path = write_csv(_ranked()[0], tmp_path / "acme.csv")
    assert path.read_bytes().decode("utf-8") == GOLDEN_ACME_CSV


def test_to_json_matches_golden_register_without_pages():
    """BETA's JSON: header with rank 2 of 3 and one quoted row with page None (no pages)."""
    assert json.loads(to_json(_ranked()[1])) == GOLDEN_BETA_JSON


def test_write_json_round_trips_into_an_equal_register(tmp_path):
    """Every register read back from its JSON file equals the original model."""
    for register in _ranked():
        assert read_json(write_json(register, tmp_path / "register.json")) == register


# ----- Ranking several contracts -----
def test_build_registers_ranks_contracts_and_keeps_empty_ones():
    """Registers come back in rank order with hand-computed triage scores; the empty contract is last."""
    registers = _ranked()
    headers = [r.header for r in registers]
    assert [(h.contract_id, h.rank, h.ranked_out_of) for h in headers] == [(ACME, 1, 3), (BETA, 2, 3), ("empty", 3, 3)]
    assert [h.triage_score for h in headers] == [ACME_TRIAGE, BETA_TRIAGE, 0.0]
    assert (headers[0].high_count, headers[0].medium_count, headers[0].low_count, headers[0].unknown_count) == (1, 1, 1, 1)
    assert registers[2].rows == [] and headers[2].clause_count == 0


def test_build_registers_repeated_contract_id_is_rejected():
    """Two results for one contract can't be ranked."""
    with pytest.raises(ValueError, match="unique"):
        build_registers([BETA_RESULT, BETA_RESULT], WEIGHTS)


def test_index_frame_lists_contracts_in_rank_order_with_file_stems():
    """One index row per contract with its counts and register file stem."""
    index = index_frame(_ranked())
    assert list(index.columns) == INDEX_COLUMNS
    assert index.to_dict("records")[1] == {
        "rank": 2, "contract_id": BETA, "triage_score": BETA_TRIAGE, "high_count": 0, "medium_count": 0,
        "low_count": 1, "unknown_count": 0, "clause_count": 1, "manual_review_count": 0,
        "file_stem": register_file_stem(BETA),
    }


def test_register_frame_concatenates_registers_in_the_order_given():
    """All rows of all registers in one table: ACME's three rows, then BETA's one."""
    frame = register_frame(_ranked())
    assert list(frame.columns) == CSV_COLUMNS
    assert list(frame["contract_id"]) == [ACME, ACME, ACME, BETA]
    assert list(frame["contract_rank"]) == [1, 1, 1, 2]


# ----- Single register -----
def test_build_register_defaults_to_the_results_own_score():
    """Without a ContractScore the result's own is used: rank 1, ranked_out_of unknown."""
    header = build_register(ACME_RESULT).header
    assert (header.rank, header.ranked_out_of, header.triage_score) == (1, None, ACME_TRIAGE)


def test_build_register_unknown_row_without_unknown_record_says_no_evidence():
    """A clause scored without evidence and missing from result['unknown'] is still kept, reason no_evidence."""
    result = {**ACME_RESULT, "unknown": []}
    row = next(r for r in build_register(result).rows if r.category == "Anti-Assignment")
    assert row.needs_manual_review and row.review_reason == "no_evidence" and row.quote is None


def test_build_register_accepts_results_read_back_from_json():
    """Results holding plain dicts (e.g. loaded from JSON) give the same register as model objects."""
    plain = json.loads(json.dumps({key: [item.model_dump() for item in value] if isinstance(value, list) else
                                   value.model_dump() if hasattr(value, "model_dump") else value
                                   for key, value in ACME_RESULT.items()}))
    assert build_register(plain, text=ACME_TEXT) == build_register(ACME_RESULT, text=ACME_TEXT)


# ----- Grounding re-check -----
@pytest.mark.parametrize("text", [
    "x" + ACME_TEXT,                                         # shifted by one character
    ACME_TEXT.replace("unlimited", "UNLIMITED"),               # same offsets, different words
    ACME_TEXT[:30],                                          # too short to hold the quote
])
def test_build_register_wrong_text_fails_the_grounding_recheck(text):
    """A quote that isn't at text[start:end] of the supplied text raises GroundingError."""
    with pytest.raises(GroundingError):
        build_register(ACME_RESULT, text=text)


def test_build_register_without_text_skips_the_grounding_recheck():
    """With no text supplied the quotes are exported as the pipeline verified them."""
    assert build_register(ACME_RESULT).rows[0].quote == LIABILITY.quote


def test_build_register_evidence_flag_without_grounded_record_is_rejected():
    """A clause marked has_evidence whose quote is missing raises instead of exporting a blank quote."""
    with pytest.raises(ValueError, match="no grounded evidence"):
        build_register({**ACME_RESULT, "grounded": []})


@pytest.mark.parametrize("score", [
    BETA_RESULT["contract_score"],                                                      # another contract
    aggregate_contracts([], WEIGHTS.model_copy(update={"weight_version_id": "other", "triage_top_k": 2}),
                        [ACME])[0],                                                     # another weight version
])
def test_build_register_mismatched_contract_score_is_rejected(score):
    """A ContractScore for another contract or weight version can't head this contract's register."""
    with pytest.raises(ValueError):
        build_register(ACME_RESULT, score)


# ----- Pages -----
@pytest.mark.parametrize(("offset", "page_starts", "page"), [
    (0, (0, 36), 1),
    (35, (0, 36), 1),
    (36, (0, 36), 2),   # a page's first character is on that page
    (500, (0, 36), 2),
    (10, (), None),     # plain text has no pages
])
def test_page_of_maps_offsets_to_one_based_pages(offset, page_starts, page):
    """Offsets map to the page whose start is the last one at or before them."""
    assert page_of(offset, page_starts) == page


@pytest.mark.parametrize("page_starts", [(5, 10), (0, 10, 10), (0, 10, 5)])
def test_check_page_starts_rejects_bad_offsets(page_starts):
    """Page starts must begin at 0 and strictly increase."""
    with pytest.raises(ValueError):
        check_page_starts(page_starts)


# ----- File names (the 8-character suffixes are pinned SHA-256 prefixes of the ids) -----
@pytest.mark.parametrize(("contract_id", "stem"), [
    (ACME, "ACME-Corp-Inc-Supply-Agreement-7b13e313"),
    ("Café / Agreement", "Cafe-Agreement-ed482fe3"),
    ("!!!", "contract-e84c538e"),
])
def test_register_file_stem_is_safe_and_deterministic(contract_id, stem):
    """Letters and digits survive, everything else becomes '-', and a short hash is appended."""
    assert register_file_stem(contract_id) == stem


def test_register_file_stem_keeps_near_identical_ids_apart():
    """Ids differing only in punctuation or case get different stems (no overwritten files)."""
    stems = {register_file_stem(cid) for cid in ["a b", "a-b", "a_b", "A B"]}
    assert len(stems) == 4


def test_write_register_writes_csv_and_json_under_the_stem(tmp_path):
    """Both files land in the directory, named by the contract's stem, with '\\n' line endings."""
    csv_path, json_path = write_register(_ranked()[0], tmp_path / "out")
    assert csv_path.name == register_file_stem(ACME) + ".csv" and json_path.suffix == ".json"
    assert b"\r\n" not in csv_path.read_bytes() and b"\r\n" not in json_path.read_bytes()
