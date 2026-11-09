"""End-to-end contract inference with traceable chunks and grounded evidence."""

from __future__ import annotations

from typing import Any

from src.chunking.token_chunker import chunk_sentence_windows
from src.evidence.extractor import GroundedEvidence, Unknown, extract_for_predictions
from src.registry.bundle import ClauseModelBundle


class ContractInferencePipeline:
    """Chunk, classify, threshold, and ground evidence for one contract at a time.

    The bundle's predictor scores chunks in order. Returned evidence is verified against
    the exact input text; predictions without cue-grounded evidence are kept in `unknown`.
    """

    def __init__(
        self,
        bundle: ClauseModelBundle,
        tokenizer: Any,
        category_cues: dict[str, list[str]],
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
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

    def predict_contract(self, contract_id: str, text: str) -> dict[str, Any]:
        """Return chunk scores and grounded/unknown category predictions for a contract."""
        if not isinstance(contract_id, str) or not contract_id.strip():
            raise ValueError("contract_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        chunks = chunk_sentence_windows(
            contract_id, text, self.tokenizer, self.max_tokens, self.overlap_tokens,
        )
        probabilities = self.bundle.predict_proba([chunk.text for chunk in chunks])
        predictions = [
            (chunk, category)
            for row, chunk in enumerate(chunks)
            for column, category in enumerate(self.bundle.categories)
            if probabilities[row, column] >= self.bundle.thresholds[category]
        ]
        grounded, unknown = extract_for_predictions(text, predictions, self.category_cues)
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
        }
