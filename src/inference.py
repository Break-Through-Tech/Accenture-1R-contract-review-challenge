"""End-to-end contract inference: traceable chunks, grounded evidence and rule-based risk scores."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.chunking.token_chunker import chunk_sentence_windows
from src.evidence.extractor import GroundedEvidence, Unknown, extract_for_predictions
from src.registry.bundle import ClauseModelBundle
from src.risk.contract import aggregate_contracts
from src.risk.schema import ClauseInput, WeightSet, load_weight_set
from src.risk.scoring import score_contract_clauses
from src.thresholds import apply_thresholds


class ContractInferencePipeline:
    """Chunk, classify, threshold, ground evidence and score risk for one contract at a time.

    The bundle's predictor scores chunks in order. Returned evidence is verified against
    the exact input text; predictions without cue-grounded evidence are kept in `unknown`
    and are still risk-scored. Scores use `weight_set` (default: the v2 team-proposed weights).
    """

    def __init__(
        self,
        bundle: ClauseModelBundle,
        tokenizer: Any,
        category_cues: dict[str, list[str]],
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        weight_set: WeightSet | None = None,
    ) -> None:
        if not 0 < max_tokens <= 512:
            raise ValueError("max_tokens must be between 1 and 512")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be nonnegative and less than max_tokens")
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.category_cues = category_cues
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.weight_set = weight_set or load_weight_set()

    def predict_contract(self, contract_id: str, text: str) -> dict[str, Any]:
        """
        Return chunk scores, grounded/unknown category predictions, one ClauseScore per
        prediction (`clause_scores`) and the contract's triage ContractScore (`contract_score`).
        Its rank is 1 on its own; rank several contracts with risk.contract.aggregate_contracts
        over their combined clause_scores.
        """
        if not isinstance(contract_id, str) or not contract_id.strip():
            raise ValueError("contract_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        chunks = chunk_sentence_windows(
            contract_id, text, self.tokenizer, self.max_tokens, self.overlap_tokens,
        )
        probabilities = self.bundle.predict_proba([chunk.text for chunk in chunks])
        detected = np.argwhere(apply_thresholds(probabilities, self.bundle.categories, self.bundle.thresholds))
        predictions = [(chunks[row], self.bundle.categories[column]) for row, column in detected]
        grounded, unknown = extract_for_predictions(text, predictions, self.category_cues)

        # Risk-score every prediction (Unknowns too, with evidence=None), then roll up the contract
        row_of = {chunk.chunk_index: row for row, chunk in enumerate(chunks)}
        column_of = {category: column for column, category in enumerate(self.bundle.categories)}
        clauses = [
            ClauseInput(
                contract_id=contract_id, chunk_index=record.chunk_index, category=record.category,
                probability=float(probabilities[row_of[record.chunk_index], column_of[record.category]]),
                evidence=getattr(record, "evidence", None),
            )
            for record in [*grounded, *unknown]
        ]
        clause_scores = score_contract_clauses(clauses, self.weight_set)
        contract_score = aggregate_contracts(clause_scores, self.weight_set, [contract_id])[0]
        return {
            "contract_id": contract_id,
            "model_name": self.bundle.model_name,
            "seed": self.bundle.seed,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "probabilities": dict(zip(self.bundle.categories, probabilities[row].tolist())),
                }
                for row, chunk in enumerate(chunks)
            ],
            "grounded": grounded,
            "unknown": unknown,
            "clause_scores": clause_scores,
            "contract_score": contract_score,
        }
