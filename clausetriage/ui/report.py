"""
Pure helpers that turn pipeline output into what the demo page shows: the ranked contract list,
one contract's clause table, its highlighted text, and the downloadable clause register.

How everything comes together:
ScoredContract: one uploaded Document with its ContractInferencePipeline.predict_contract result
rank_contracts(): ranks several scored contracts together with risk.contract.aggregate_contracts
contract_table(), clause_table(), highlighted_text(), clause_register(): the page's views
clause_register() delegates to clausetriage/register/export.py, the core register that scripts and the API share
"""
from dataclasses import dataclass

import pandas as pd

from clausetriage.register.export import build_register, register_frame
from clausetriage.risk.contract import ContractScore, aggregate_contracts
from clausetriage.risk.schema import SIGNAL_NAMES, ClauseScore, WeightSet
from clausetriage.ui.documents import Document

# The download's original column names, kept so existing CSV consumers still find them
REGISTER_RENAMES = {"contract_id": "contract", "composite_score": "risk_score", "model_name": "model"}
REGISTER_COLUMNS = ["contract", "contract_rank", "category", "rating", "risk_score", *SIGNAL_NAMES, "probability",
                    "chunk_index", "page", "start", "end", "quote", "model", "weight_version_id"]
CONTRACT_COLUMNS = ["Rank", "Contract", "Verdict", "Triage score", "High", "Medium", "Low", "No quote",
                    "Riskiest clause"]
CLAUSE_COLUMNS = ["Rating", "Risk score", "Category", "Page", "Evidence", "Why (signals)", "Model probability"]
NO_QUOTE = "(no quote found: needs manual review)"
SIGNAL_LABELS = {"category_severity": "severity", "one_sidedness": "one-sided",
                 "magnitude": "magnitude", "contract_context": "context"}
# Verdict from the worst clause: one High clause is reason enough to open a contract first
VERDICTS = {"High": "Open first", "Medium": "Review", "Low": "Low priority"}


@dataclass
class ScoredContract:
    """An uploaded document and its predict_contract result."""
    document: Document
    result: dict


def rank_contracts(scored: list[ScoredContract], weight_set: WeightSet) -> list[ContractScore]:
    """Rank all contracts together (rank 1 = open first); every contract appears, even with no clauses."""
    clause_scores = [s for item in scored for s in item.result["clause_scores"]]
    return aggregate_contracts(clause_scores, weight_set, [item.result["contract_id"] for item in scored])


def verdict(contract: ContractScore) -> str:
    """'Open first' with any High clause, 'Review' with any Medium, else 'Low priority'."""
    worst = contract.top_clauses[0].rating if contract.top_clauses else "Low"
    return VERDICTS[worst]


def contract_table(ranked: list[ContractScore]) -> pd.DataFrame:
    """One row per contract in rank order."""
    return pd.DataFrame([{
        "Rank": c.rank, "Contract": c.contract_id, "Verdict": verdict(c), "Triage score": round(c.triage_score, 2),
        "High": c.high_count, "Medium": c.medium_count, "Low": c.low_count, "No quote": c.unknown_count,
        "Riskiest clause": f"{c.top_clauses[0].category} ({c.top_clauses[0].composite_score:.2f})"
                           if c.top_clauses else "none detected",
    } for c in ranked], columns=CONTRACT_COLUMNS)


def _evidence_by_clause(result: dict) -> dict:
    """(chunk_index, category) -> Evidence for every grounded prediction."""
    return {(g.chunk_index, g.category): g.evidence for g in result["grounded"]}


def _why(clause: ClauseScore) -> str:
    """The four signal values in one short line."""
    return " · ".join(f"{SIGNAL_LABELS[name]} {value:.2f}" for name, value in clause.signals.items())


def _worst_per_category(result: dict) -> list[ClauseScore]:
    """Each category's highest-scoring clause, riskiest first (how the triage score counts them)."""
    worst: dict[str, ClauseScore] = {}
    for clause in result["clause_scores"]:
        if clause.category not in worst or clause.composite_score > worst[clause.category].composite_score:
            worst[clause.category] = clause
    return sorted(worst.values(), key=lambda c: (-c.composite_score, c.category))


def clause_table(item: ScoredContract, show_harmless: bool = False) -> pd.DataFrame:
    """
    One row per detected category (its worst clause), riskiest first, with page, quote and signal breakdown.
    Categories with zero severity (Parties, dates) are hidden unless show_harmless.
    """
    evidence = _evidence_by_clause(item.result)
    rows = []
    for clause in _worst_per_category(item.result):
        if not show_harmless and clause.signals["category_severity"] == 0:
            continue
        found = evidence.get((clause.chunk_index, clause.category))
        rows.append({
            "Rating": clause.rating, "Risk score": round(clause.composite_score, 2), "Category": clause.category,
            "Page": (item.document.page_of(found.start) if found else None) or "",
            "Evidence": found.quote if found else NO_QUOTE,
            "Why (signals)": _why(clause), "Model probability": round(clause.probability, 2),
        })
    return pd.DataFrame(rows, columns=CLAUSE_COLUMNS)


def highlighted_text(item: ScoredContract) -> list[tuple[str, str | None]]:
    """
    The contract text split into (segment, label) pairs for gr.HighlightedText: each quote is labelled
    with its rating and category; where quotes overlap, the riskier one wins.
    """
    evidence = _evidence_by_clause(item.result)
    spans = sorted(
        ((found.start, found.end, f"{clause.rating}: {clause.category}", clause.composite_score)
         for clause in item.result["clause_scores"]
         if (found := evidence.get((clause.chunk_index, clause.category)))),
        key=lambda span: (-span[3], span[0]),
    )
    # Keep the riskiest spans that don't overlap one already kept, then walk the text in order
    kept: list[tuple[int, int, str]] = []
    for start, end, label, _ in spans:
        if all(end <= s or start >= e for s, e, _ in kept):
            kept.append((start, end, label))
    text, segments, position = item.document.text, [], 0
    for start, end, label in sorted(kept):
        if start > position:
            segments.append((text[position:start], None))
        segments.append((text[start:end], label))
        position = end
    if position < len(text):
        segments.append((text[position:], None))
    return segments


def clause_register(scored: list[ScoredContract], ranked: list[ContractScore]) -> pd.DataFrame:
    """
    Every scored clause of every contract, for download: contract rank, clause, signals, page and quote.
    Built by the core clausetriage/register module (which re-checks each quote against the document text); its
    columns are renamed to the page's original names (contract, risk_score, model) and listed first,
    followed by the core register's extra columns (triage score, manual-review flag and reason, hash).
    """
    score_of = {c.contract_id: c for c in ranked}
    registers = [
        build_register(item.result, score_of[item.result["contract_id"]], text=item.document.text,
                       page_starts=item.document.page_starts, ranked_out_of=len(ranked))
        for item in scored
    ]
    registers.sort(key=lambda register: (register.header.rank, register.header.contract_id))
    frame = register_frame(registers).rename(columns=REGISTER_RENAMES)
    return frame[[*REGISTER_COLUMNS, *(column for column in frame.columns if column not in REGISTER_COLUMNS)]]
