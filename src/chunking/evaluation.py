"""Measure fixed-window boundaries against CUAD annotations without projecting labels."""

from pydantic import BaseModel, Field

from src.chunking.chunk import Chunk
from src.chunking.chunker import chunk_contract
from src.labels.categories import CATEGORIES
from src.labels.loader import Contract


class BoundaryExcerpt(BaseModel):
    """Original source text around a boundary, with absolute excerpt offsets."""
    boundary: int
    start: int
    end: int
    text: str


class SpanExample(BaseModel):
    """One annotation and the chunks that cut or contain it, for manual review."""
    kind: str
    contract_id: str
    category: str
    start: int
    end: int
    annotation_preview: str
    preview_truncated: bool
    cutting_chunks: list[tuple[int, int]]
    containing_chunks: list[tuple[int, int]]
    boundary_excerpts: list[BoundaryExcerpt]


class BoundaryMetrics(BaseModel):
    """Counts and percentages over unique contract/category/start/end annotations."""
    window_chars: int | None
    overlap_chars: int | None
    raw_annotations: int
    unique_annotations: int
    duplicate_annotations: int
    cut_annotations: int
    contained_annotations: int
    cut_not_contained_annotations: int
    longer_than_window_annotations: int | None
    cut_pct: float | None
    contained_pct: float | None
    cut_not_contained_pct: float | None
    longer_than_window_pct: float | None


class BoundaryEvaluation(BaseModel):
    """A configuration's aggregate metrics and deterministic illustrative examples."""
    metrics: BoundaryMetrics
    examples: list[SpanExample]
    per_category: dict[str, BoundaryMetrics] = Field(default_factory=dict)


def validate_chunks(contract: Contract, chunks: list[Chunk]) -> None:
    """Verify exact slices, ordered indices, advancing offsets, and full source coverage."""
    if not contract.text:
        if chunks:
            raise ValueError("Empty source must have no chunks")
        return
    if not chunks or chunks[0].start != 0 or chunks[-1].end != len(contract.text):
        raise ValueError("Chunks must cover source endpoints")
    for index, chunk in enumerate(chunks):
        if (chunk.contract_id != contract.contract_id or chunk.chunk_index != index or
                not 0 <= chunk.start < chunk.end <= len(contract.text) or
                contract.text[chunk.start:chunk.end] != chunk.text):
            raise ValueError("Chunk identity, index, or source slice is invalid")
    for previous, current in zip(chunks, chunks[1:]):
        if not previous.start < current.start <= previous.end or current.end <= previous.end:
            raise ValueError("Chunks must advance with no source gaps")


def evaluate_boundaries(
    contracts: list[Contract], window_chars: int, overlap_chars: int,
) -> BoundaryEvaluation:
    """Evaluate each unique annotation once using the existing chunk_contract.

    A cut is a chunk start or end strictly inside [start, end); containment
    includes endpoint equality. Deduplicate by contract ID, category, start,
    and end across all supplied records, retaining different categories.
    Percentages use unique annotations; an empty denominator yields None.
    Keep the first example of each outcome in input order. No labels are made.
    """
    # Validate parameters even when the input has no contracts or annotations.
    if window_chars <= 0 or not 0 <= overlap_chars < window_chars:
        raise ValueError("Require window_chars > 0 and 0 <= overlap_chars < window_chars")

    chunks = {c.contract_id: chunk_contract(c.contract_id, c.text, window_chars, overlap_chars)
              for c in contracts}
    return evaluate_chunks(contracts, chunks, window_chars, overlap_chars)


def evaluate_chunks(
    contracts: list[Contract], chunks_by_contract: dict[str, list[Chunk]],
    window_chars: int | None = None, overlap_chars: int | None = None,
) -> BoundaryEvaluation:
    """Score supplied original-text chunks, deduplicating annotations overall and by category.

    Character configuration is optional metadata for the existing wrapper. For
    token methods, longer_than_window is undefined because no character window
    exists. Annotations are used only here, after all boundaries are chosen.
    """
    seen: set[tuple[str, str, int, int]] = set()
    sources: dict[str, str] = {}
    # Check duplicate IDs before validating their shared chunk mapping.
    for contract in contracts:
        if contract.contract_id in sources and sources[contract.contract_id] != contract.text:
            raise ValueError(f"Conflicting text for contract ID {contract.contract_id!r}")
        sources[contract.contract_id] = contract.text
    examples: dict[str, SpanExample] = {}
    raw = cut_count = contained_count = uncontained_count = long_count = 0
    category_counts = {category: [0, 0, 0, 0, 0, 0] for category in CATEGORIES}
    for contract in contracts:
        chunks = chunks_by_contract[contract.contract_id]
        validate_chunks(contract, chunks)
        for span in contract.spans:
            raw += 1
            counts = category_counts.setdefault(span.category, [0, 0, 0, 0, 0, 0])
            counts[0] += 1
            key = (contract.contract_id, span.category, span.start, span.end)
            if key in seen:
                continue
            seen.add(key)

            # Each boolean contributes at most one count, regardless of chunk multiplicity.
            cutting = [c for c in chunks if
                       span.start < c.start < span.end or span.start < c.end < span.end]
            containing = [c for c in chunks if c.start <= span.start and span.end <= c.end]
            cut, contained = bool(cutting), bool(containing)
            longer = window_chars is not None and span.end - span.start > window_chars
            cut_count += cut
            contained_count += contained
            uncontained_count += cut and not contained
            long_count += longer
            for index, value in enumerate((1, cut, contained, cut and not contained, longer), 1):
                counts[index] += value

            # Separate oversized spans from shorter spans lost at a boundary.
            kind = ("longer_than_window" if longer else
                    "cut_not_contained" if cut and not contained else
                    "cut_and_contained" if cut else "uncut_and_contained")
            if kind not in examples:
                boundaries = sorted({b for c in cutting for b in (c.start, c.end)
                                     if span.start < b < span.end})
                excerpts = []
                for boundary in boundaries:
                    start = max(0, boundary - 60)
                    end = min(len(contract.text), boundary + 60)
                    excerpts.append(BoundaryExcerpt(
                        boundary=boundary, start=start, end=end,
                        text=contract.text[start:end],
                    ))
                examples[kind] = SpanExample(
                    kind=kind, contract_id=contract.contract_id, category=span.category,
                    start=span.start, end=span.end, annotation_preview=span.text[:240],
                    preview_truncated=len(span.text) > 240,
                    cutting_chunks=[(c.start, c.end) for c in cutting],
                    containing_chunks=[(c.start, c.end) for c in containing],
                    boundary_excerpts=excerpts,
                )

    # Undefined percentages remain explicit when there are no annotations.
    total = len(seen)
    metrics = BoundaryMetrics(
        window_chars=window_chars, overlap_chars=overlap_chars,
        raw_annotations=raw, unique_annotations=total, duplicate_annotations=raw - total,
        cut_annotations=cut_count, contained_annotations=contained_count,
        cut_not_contained_annotations=uncontained_count,
        longer_than_window_annotations=long_count if window_chars is not None else None,
        cut_pct=100 * cut_count / total if total else None,
        contained_pct=100 * contained_count / total if total else None,
        cut_not_contained_pct=100 * uncontained_count / total if total else None,
        longer_than_window_pct=100 * long_count / total if total and window_chars is not None else None,
    )
    per_category = {}
    for category, (raw_n, unique_n, cut_n, contained_n, lost_n, long_n) in category_counts.items():
        per_category[category] = BoundaryMetrics(
            window_chars=window_chars, overlap_chars=overlap_chars,
            raw_annotations=raw_n, unique_annotations=unique_n, duplicate_annotations=raw_n - unique_n,
            cut_annotations=cut_n, contained_annotations=contained_n,
            cut_not_contained_annotations=lost_n,
            longer_than_window_annotations=long_n if window_chars is not None else None,
            cut_pct=100 * cut_n / unique_n if unique_n else None,
            contained_pct=100 * contained_n / unique_n if unique_n else None,
            cut_not_contained_pct=100 * lost_n / unique_n if unique_n else None,
            longer_than_window_pct=100 * long_n / unique_n if unique_n and window_chars is not None else None,
        )
    return BoundaryEvaluation(metrics=metrics, examples=list(examples.values()), per_category=per_category)
