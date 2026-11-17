"""
Rescores stored contracts under a different WeightSet without loading or rerunning the model.

How everything comes together:
rescore_contracts(): rebuilds each contract's ClauseInput rows from the store (stored raw
                     probabilities, the model version's thresholds, grounded evidence and the
                     Unknown queue), scores them with risk.scoring, ranks them with risk.contract,
                     and (unless save=False) writes the new clause and contract scores under the
                     new weight version

Rescoring reuses the detections recorded when the pipeline ran rather than re-deriving them
from probabilities, because the evidence stage drops duplicate spans across overlapping chunks
and that needs the extractor and its cue words, not just the stored numbers. The result is
exactly what ContractInferencePipeline would have produced under the same weight set.
"""
from __future__ import annotations

from collections.abc import Iterable

from src.risk.contract import ContractScore, aggregate_contracts
from src.risk.schema import WeightSet
from src.risk.scoring import score_contract_clauses
from src.store.artifact_store import ArtifactStore


def rescore_contracts(
    store: ArtifactStore,
    model_version: str,
    weight_set: WeightSet,
    contract_ids: Iterable[str] | None = None,
    *,
    save: bool = True,
) -> list[ContractScore]:
    """
    Score stored contracts under weight_set and return them ranked (rank 1 = riskiest, among the
    contracts rescored). No model is loaded.

    model_version: the model version whose stored probabilities and detections to rescore
    weight_set: the weights to score with; stored as a new weight version when save is True
    contract_ids: contracts to rescore (default: every contract stored under model_version);
                  KeyError if one has no stored results under model_version
    save: False previews the ranking without writing anything (the UI's "preview ranking")
    """
    ids = sorted(contract_ids) if contract_ids is not None else store.list_run_contract_ids(model_version)

    # Rebuild every contract's clauses, then score and rank them together
    clauses = [clause for contract_id in ids for clause in store.get_clause_inputs(contract_id, model_version)]
    clause_scores = score_contract_clauses(clauses, weight_set)
    contract_scores = aggregate_contracts(clause_scores, weight_set, ids)

    # Write the new scores under the new weight version in one transaction
    if save:
        store.save_scores(model_version, weight_set, clause_scores, contract_scores)
    return contract_scores
