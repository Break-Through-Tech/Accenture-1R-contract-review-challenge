"""
Turns cached classifier probabilities into the ClauseInput rows that scoring reads, so a retune
only reloads saved numbers and never reruns the model.

How everything comes together:
load_cached_probabilities(): reads a saved .npy probability matrix
clauses_from_probabilities(): applies the per-class thresholds, grounds each detected clause with
                              evidence.extractor.extract_for_predictions, and returns ClauseInput
                              rows (evidence=None for the Unknown queue)

The matrix has one row per chunk, in the same order as the chunks passed in, and one column per
category, in CATEGORIES order. Rows from several contracts can sit in one matrix.

Typical use:
    probabilities = load_cached_probabilities("val_probs.npy")
    clauses = clauses_from_probabilities(contract_texts, chunks, probabilities, thresholds, cues)
    scores = score_contract_clauses(clauses, load_weight_set())

Inputs from other modules:
'Chunk' objects from the chunker, the same ones the classifier scored
'thresholds' from thresholds.find_best_thresholds (category name -> threshold)
'category_cues' from evidence.cues.load_category_cues
"""
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.chunking.chunk import Chunk
from src.evidence.extractor import extract_for_predictions
from src.labels.categories import CATEGORIES
from src.risk.schema import ClauseInput
from src.thresholds import apply_thresholds


def load_cached_probabilities(path: str | Path) -> np.ndarray:
    """Read a saved probability matrix (.npy). Pickled files are refused because loading one can run code."""
    return np.load(path, allow_pickle=False)


def _check_probabilities(probabilities: np.ndarray, chunk_count: int) -> None:
    """Reject a matrix that doesn't line up with the chunks, or holds values that aren't probabilities."""
    expected = (chunk_count, len(CATEGORIES))
    if probabilities.shape != expected:
        raise ValueError(f"probabilities must have shape {expected} (chunks, categories), got {probabilities.shape}")
    # NaN must fail here: it compares False against every threshold, so the clause would silently vanish
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("probabilities must all be between 0 and 1")


def clauses_from_probabilities(
    contract_texts: Mapping[str, str],
    chunks: Sequence[Chunk],
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
    category_cues: dict[str, list[str]],
) -> list[ClauseInput]:
    """
    Build one ClauseInput per detected clause. Only probabilities at or above their category's
    threshold count as detected. Within each contract, grounded clauses come first, then Unknowns.

    contract_texts: contract_id -> full contract text, used to verify each Evidence
    chunks: the chunks behind the matrix, where row i belongs to chunks[i]
    probabilities: the cached matrix, shape (len(chunks), len(CATEGORIES))
    thresholds: category name -> threshold; a missing category raises KeyError
    category_cues: output of build_category_cues() / load_category_cues()
    """
    _check_probabilities(probabilities, len(chunks))

    # Group detections by contract, because evidence is verified one contract at a time
    predictions: dict[str, list[tuple[Chunk, str]]] = defaultdict(list)
    probability_of: dict[tuple[str, int, str], float] = {}
    for row, column in np.argwhere(apply_thresholds(probabilities, CATEGORIES, thresholds)):
        chunk, category = chunks[row], CATEGORIES[column]
        predictions[chunk.contract_id].append((chunk, category))
        probability_of[chunk.contract_id, chunk.chunk_index, category] = float(probabilities[row, column])

    clauses = []
    for contract_id, predicted in predictions.items():
        grounded, unknown = extract_for_predictions(contract_texts[contract_id], predicted, category_cues)
        clauses += [
            ClauseInput(
                contract_id=record.contract_id,
                chunk_index=record.chunk_index,
                category=record.category,
                probability=probability_of[record.contract_id, record.chunk_index, record.category],
                evidence=getattr(record, "evidence", None),
            )
            for record in [*grounded, *unknown]
        ]
    return clauses
