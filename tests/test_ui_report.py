import pandas as pd
import pytest

from src.evidence.extractor import GroundedEvidence
from src.evidence.schema import Evidence
from src.risk.contract import aggregate_contracts
from src.risk.schema import ClauseInput, load_weight_set
from src.risk.scoring import score_contract_clauses
from src.ui.documents import Document
from src.ui.report import (CLAUSE_COLUMNS, CONTRACT_COLUMNS, NO_QUOTE, ScoredContract, clause_register,
                           clause_table, contract_table, highlighted_text, rank_contracts, verdict)

WEIGHTS = load_weight_set()
TEXT = "This Agreement is governed by Delaware law.\n\nThe liability of Supplier shall be unlimited."
LAW = Evidence(quote="This Agreement is governed by Delaware law.", start=0, end=43)
LIABILITY = Evidence(quote="The liability of Supplier shall be unlimited.", start=45, end=len(TEXT))


def _scored(contract_id: str, clauses: list[tuple[str, int, Evidence | None]]) -> ScoredContract:
    """A two-page Document and a predict_contract-shaped result for (category, chunk, evidence) clauses."""
    inputs = [ClauseInput(contract_id=contract_id, chunk_index=chunk, category=category, probability=0.9,
                          evidence=evidence) for category, chunk, evidence in clauses]
    scores = score_contract_clauses(inputs, WEIGHTS)
    grounded = [GroundedEvidence(contract_id=contract_id, chunk_index=chunk, category=category, evidence=evidence)
                for category, chunk, evidence in clauses if evidence]
    return ScoredContract(Document(contract_id, TEXT, (0, 45)), {
        "contract_id": contract_id, "model_name": "test", "grounded": grounded, "clause_scores": scores,
        "contract_score": aggregate_contracts(scores, WEIGHTS, [contract_id])[0],
    })


RISKY = _scored("risky", [("Uncapped Liability", 0, LIABILITY), ("Governing Law", 0, LAW), ("Parties", 0, None)])
QUIET = _scored("quiet", [("Governing Law", 0, LAW)])
EMPTY = _scored("empty", [])


# ----- Ranking several contracts -----
def test_rank_contracts_ranks_all_together_including_empty():
    """Contracts are ranked against each other; one with nothing detected still appears, last."""
    ranked = rank_contracts([QUIET, EMPTY, RISKY], WEIGHTS)
    assert [c.contract_id for c in ranked] == ["risky", "quiet", "empty"]
    assert [c.rank for c in ranked] == [1, 2, 3]


def test_contract_table_shows_verdict_and_riskiest_clause():
    """One row per contract in rank order; the verdict follows the worst clause's rating."""
    ranked = rank_contracts([QUIET, EMPTY, RISKY], WEIGHTS)
    table = contract_table(ranked)
    assert list(table.columns) == CONTRACT_COLUMNS
    assert table.iloc[0]["Riskiest clause"].startswith("Uncapped Liability")
    assert table.iloc[2]["Riskiest clause"] == "none detected" and table.iloc[2]["Verdict"] == "Low priority"
    assert [verdict(c) for c in ranked] == [{"High": "Open first", "Medium": "Review", "Low": "Low priority"}[
        c.top_clauses[0].rating if c.top_clauses else "Low"] for c in ranked]


# ----- One contract's clauses -----
def test_clause_table_hides_harmless_types_and_shows_page_and_signals():
    """Zero-severity Parties is hidden by default; rows carry page, quote and the four signals."""
    table = clause_table(RISKY)
    assert list(table.columns) == CLAUSE_COLUMNS
    assert list(table["Category"]) == ["Uncapped Liability", "Governing Law"]
    assert table.iloc[0]["Page"] == 2 and table.iloc[1]["Page"] == 1
    assert table.iloc[0]["Evidence"] == LIABILITY.quote
    assert table.iloc[0]["Why (signals)"].startswith("severity 1.00 · one-sided")


def test_clause_table_show_harmless_adds_unquoted_parties():
    """With show_harmless, Parties appears with the manual-review note and no page."""
    table = clause_table(RISKY, show_harmless=True)
    parties = table[table["Category"] == "Parties"].iloc[0]
    assert parties["Evidence"] == NO_QUOTE and parties["Page"] == ""


def test_highlighted_text_rebuilds_text_and_labels_quotes():
    """Joining the segments gives back the text; each quote is labelled 'rating: category'."""
    segments = highlighted_text(RISKY)
    assert "".join(segment for segment, _ in segments) == TEXT
    labels = {segment: label for segment, label in segments if label}
    assert labels[LIABILITY.quote].endswith(": Uncapped Liability")
    assert labels[LAW.quote].endswith(": Governing Law")


def test_highlighted_text_keeps_riskier_of_overlapping_quotes():
    """Two categories quoting the same sentence: only the riskier label is kept."""
    item = _scored("overlap", [("Governing Law", 0, LIABILITY), ("Uncapped Liability", 0, LIABILITY)])
    labelled = [label for _, label in highlighted_text(item) if label]
    assert len(labelled) == 1 and labelled[0].endswith("Uncapped Liability")


# ----- Download -----
def test_clause_register_lists_every_clause_with_rank_and_page():
    """Every clause of every contract, riskiest contract first, with page and quote (blank when unquoted)."""
    scored = [QUIET, RISKY]
    register = clause_register(scored, rank_contracts(scored, WEIGHTS))
    assert len(register) == 4
    assert register.iloc[0]["contract"] == "risky" and register.iloc[0]["category"] == "Uncapped Liability"
    assert register.iloc[0]["page"] == 2 and register.iloc[0]["category_severity"] == pytest.approx(1.0)
    parties = register[register["category"] == "Parties"].iloc[0]
    assert pd.isna(parties["quote"]) and parties["weight_version_id"] == WEIGHTS.weight_version_id
