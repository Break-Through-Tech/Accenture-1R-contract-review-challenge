"""
Turns cached classifier probabilities into the ClauseInput rows that scoring reads, so a retune
only reloads saved numbers and never reruns the model.

How everything comes together:
load_cached_probabilities(): reads a saved .npy probability matrix
clauses_from_probabilities(): applies the per-class thresholds, grounds each detected clause with
                              evidence.extractor.extract_for_predictions, and returns ClauseInput
                              rows (evidence=None for the Unknown queue)
gold_clauses(): builds the same rows from CUAD's annotated answers instead of a classifier, as a
                perfect-detection reference for checking how far predictions move the ranking

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
from src.evidence.schema import Evidence
from src.labels.categories import CATEGORIES
from src.labels.loader import Contract
from src.labels.projection import spans_overlap
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


def gold_clauses(contracts: Sequence[Contract], chunks: Sequence[Chunk]) -> list[ClauseInput]:
    """
    Build one ClauseInput per annotated answer span: probability 1.0, the gold answer as evidence,
    and the index of the first chunk that overlaps it. This is what scoring would see with a perfect
    classifier, so never use it on test.json before final evaluation.
    Note the gold quote is the whole annotated answer, often longer than the extractor's one sentence.
    """
    # The first chunk (in order) of each contract that overlaps each span
    chunks_by_contract: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_contract[chunk.contract_id].append(chunk)

    clauses = []
    for contract in contracts:
        for span in contract.spans:
            chunk = next((c for c in chunks_by_contract[contract.contract_id]
                          if spans_overlap(c.start, c.end, span.start, span.end)), None)
            if chunk is None:
                raise ValueError(f"No chunk of {contract.contract_id!r} covers its span at [{span.start}:{span.end}]")
            clauses.append(ClauseInput(
                contract_id=contract.contract_id, chunk_index=chunk.chunk_index, category=span.category,
                probability=1.0, evidence=Evidence(quote=span.text, start=span.start, end=span.end),
            ))
    return clauses
