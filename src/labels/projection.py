from pydantic import BaseModel, model_validator
from typing_extensions import Self
from src.chunking.chunk import Chunk
from src.labels.loader import AnswerSpan
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX


def spans_overlap(chunk_start: int, chunk_end: int,
                  span_start: int, span_end: int) -> bool:
    """
    Return True when a chunk and an answer span share at least one character.
    Both ranges are end-exclusive, so ranges that only touch at an endpoint
    (chunk_end == span_start) do not overlap. This is the project's overlap rule.
    """
    return max(chunk_start, span_start) < min(chunk_end, span_end)


class LabeledChunk(BaseModel):
    """
    A chunk paired with its multi-hot label vector over CATEGORIES.
    labels[i] is 1 when the chunk overlaps an answer span for CATEGORIES[i], else 0.
    """
    chunk: Chunk
    labels: list[int]

    # Model Validator; ensures one 0/1 label per category
    @model_validator(mode="after")
    def check_label_consistency(self) -> Self:
        if len(self.labels) != len(CATEGORIES):
            raise ValueError(
                f"Expected {len(CATEGORIES)} labels (one per category), got {len(self.labels)}"
            )
        if not set(self.labels) <= {0, 1}:
            raise ValueError(
                f"Labels must be 0 or 1, got invalid values {sorted(set(self.labels) - {0, 1})}"
            )
        return self


def project_chunk_labels(chunk: Chunk, spans: list[AnswerSpan]) -> LabeledChunk:
    """
    Project answer spans onto one chunk as a multi-hot label vector.
    A category is labeled 1 if any of its spans overlaps the chunk (see
    spans_overlap). spans must belong to the same contract as the chunk.
    """
    # Start with no labels, then switch on each category with an overlapping span
    labels = [0] * len(CATEGORIES)
    for span in spans:
        if spans_overlap(chunk.start, chunk.end, span.start, span.end):
            labels[CATEGORY_TO_INDEX[span.category]] = 1
    return LabeledChunk(chunk=chunk, labels=labels)


def project_labels(chunks: list[Chunk], spans: list[AnswerSpan]) -> list[LabeledChunk]:
    """
    Project answer spans onto every chunk of one contract, preserving chunk order.
    spans must belong to the same contract as the chunks.
    """
    return [project_chunk_labels(chunk, spans) for chunk in chunks]
