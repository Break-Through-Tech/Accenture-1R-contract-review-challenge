"""
Rolls clause risk scores up into one ranked triage score per contract, so reviewers know which
contracts to open first.

How everything comes together:
aggregate_contracts(): keeps each contract's worst-scoring clause per category, combines them into
                       a triage_score, and ranks the contracts (rank 1 = open first)
sensitivity(): rescores the same clauses under several weight sets and returns each contract's rank
               under each one, so ranking stability can be checked without rerunning the model

triage_score = (worst clause composite + mean of the TOP_K worst composites) / 2
The worst clause counts for half, so one severe clause always matters; the mean of the top
TOP_K counts for the other half, so a contract with several risky clauses outranks one with a
single clause of the same score. Missing slots in the top TOP_K count as 0, and each category
counts once, so a long contract that repeats a clause in many chunks isn't ranked higher for it.

PLACEHOLDER: TOP_K and the half/half split are engineering choices pending the advisor. Once
they are settled they should move into the WeightSet so a weight version covers them too.

Unknown-queue clauses (no evidence) are kept in the score and counted in unknown_count: dropping
them would rank a contract lower just because its evidence couldn't be found.

Inputs from other modules:
'ClauseScore' rows from scoring.score_contract_clauses
'ClauseInput' rows from inputs.clauses_from_probabilities (for sensitivity)
'WeightSet' from schema.load_weight_set()
"""
from collections import defaultdict
from collections.abc import Iterable, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from src.risk.schema import ClauseInput, ClauseScore, UnitFloat, WeightSet
from src.risk.scoring import COMPOSITE_DECIMALS, score_contract_clauses

TOP_K = 3


class ContractScore(BaseModel):
    """
    One contract's triage score and rank under one weight version, with the clauses behind it.
    Counts are over distinct categories (the worst clause of each). Tied scores share a rank.
    """
    contract_id: str
    triage_score: UnitFloat
    rank: int = Field(ge=1)
    high_count: int = Field(ge=0)
    medium_count: int = Field(ge=0)
    low_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    top_clauses: list[ClauseScore]
    weight_version_id: str
    weight_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _triage_score(composites: Sequence[float]) -> float:
    """(max + mean of the TOP_K highest, padding with 0) / 2; composites must be sorted high to low."""
    if not composites:
        return 0.0
    top = list(composites[:TOP_K])
    return round((top[0] + sum(top) / TOP_K) / 2, COMPOSITE_DECIMALS)


def aggregate_contracts(
    scores: Iterable[ClauseScore],
    weight_set: WeightSet,
    contract_ids: Iterable[str] = (),
) -> list[ContractScore]:
    """
    Score and rank every contract, highest triage_score first (ties by contract_id).

    scores: clause scores, all produced with weight_set (a mixed version raises ValueError)
    weight_set: the weight set the scores were produced with
    contract_ids: contracts to include even if nothing was detected in them (they score 0)
    """
    # Keep the worst clause per (contract, category); every score must come from this weight set
    worst: dict[str, dict[str, ClauseScore]] = defaultdict(dict)
    for score in scores:
        if (score.weight_version_id, score.weight_set_sha256) != (weight_set.weight_version_id, weight_set.content_sha256):
            raise ValueError(
                f"clause score from weight version {score.weight_version_id!r} can't be aggregated "
                f"under {weight_set.weight_version_id!r}"
            )
        kept = worst[score.contract_id].get(score.category)
        if kept is None or score.composite_score > kept.composite_score:
            worst[score.contract_id][score.category] = score
    for contract_id in contract_ids:
        worst.setdefault(contract_id, {})

    # Build each contract's summary from its worst clause per category
    unranked = []
    for contract_id, by_category in worst.items():
        clauses = sorted(by_category.values(), key=lambda s: (-s.composite_score, s.category))
        ratings = [s.rating for s in clauses]
        unranked.append(dict(
            contract_id=contract_id,
            triage_score=_triage_score([s.composite_score for s in clauses]),
            high_count=ratings.count("High"),
            medium_count=ratings.count("Medium"),
            low_count=ratings.count("Low"),
            unknown_count=sum(not s.has_evidence for s in clauses),
            top_clauses=clauses[:TOP_K],
            weight_version_id=weight_set.weight_version_id,
            weight_set_sha256=weight_set.content_sha256,
        ))

    # Rank 1 is the riskiest; tied scores share the best rank (1, 2, 2, 4)
    unranked.sort(key=lambda c: (-c["triage_score"], c["contract_id"]))
    ranked, rank = [], 0
    for position, contract in enumerate(unranked, start=1):
        if position == 1 or contract["triage_score"] != unranked[position - 2]["triage_score"]:
            rank = position
        ranked.append(ContractScore(**contract, rank=rank))
    return ranked


def sensitivity(
    clauses: Iterable[ClauseInput],
    weight_sets: Sequence[WeightSet],
    contract_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Rank the same clauses under each weight set; no model is rerun.
    Returns one row per contract (sorted by contract_id) and one column of ranks per
    weight_version_id. Compare two rankings with result.corr(method="spearman").
    """
    versions = [ws.weight_version_id for ws in weight_sets]
    if len(set(versions)) != len(versions):
        raise ValueError(f"weight_version_id values must be unique, got {versions}")
    clauses, contract_ids = list(clauses), list(contract_ids)

    # Rescore and rerank once per weight set
    ranks = {}
    for ws in weight_sets:
        contracts = aggregate_contracts(score_contract_clauses(clauses, ws), ws, contract_ids)
        ranks[ws.weight_version_id] = {c.contract_id: c.rank for c in contracts}
    return pd.DataFrame(ranks).sort_index()
