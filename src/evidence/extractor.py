"""
For each (chunk, category) pair the classifier predicted, it picks the single sentence in the
chunk that best supports the category, wraps it in an 'Evidence' object with absolute offsets, and
verifies it against the full contract text.
If no sentence qualifies, or verification fails, the prediction goes to the Unknown queue instead of
being guessed at.

How everything comes together:
_sentence_spans(): splits chunk text into (start, end) offsets
extract_evidence(): scores sentences by cue words, returns the best as Evidence (or None)
extract_for_predictions(): entry point other pipeline stages call. Runs the above for every
                           prediction in a contract, validates each result, sorts outcomes
                           into 'grounded' and 'unknown', and drops duplicate evidence that
                           overlapping chunks produce for the same clause

Inputs from other modules:
'category_cues' from cues.build_category_cues() / cues.load_category_cues()
'Chunk' objects from the chunker (their start converts chunk offsets to contract offsets)

Limitations:
- Sentence splitting breaks on abbreviations like "Inc." and "No."
- A sentence cut by a chunk boundary is quoted only up to that boundary
- Only one sentence is returned per clause, so a clause spanning several sentences is only partially captured
- Cue quality is capped by the wording of the category descriptions
"""
import re
from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel

from src.chunking.chunk import Chunk
from src.evidence.cues import normalize_category
from src.evidence.schema import Evidence

# A sentence ends at "." or ";" followed by whitespace, or at a line break (headings, list items)
_SENTENCE_BREAK = re.compile(r"(?<=[.;])\s+|\s*\n\s*")


@lru_cache(maxsize=None)
def _cue_pattern(cue: str) -> re.Pattern:
    """
    Compile a whole-word pattern for a cue that also accepts simple inflections.
    "state" matches "states" but not "statement"; "cause" does not match "because".
    """
    return re.compile(rf"\b{re.escape(cue)}(?:s|es|d|ed|ing)?\b")


def _sentence_spans(text: str) -> Iterator[tuple[int, int]]:
    """
    Yield (start, end) offsets for each sentence in text.
    Offsets come from regex match positions; leading and trailing whitespace is excluded.
    """
    # Skip leading whitespace so the first sentence starts on a real character
    start = len(text) - len(text.lstrip())
    for m in _SENTENCE_BREAK.finditer(text, start):
        if m.start() > start:
            yield start, m.start()
        start = m.end()

    # Final sentence, ignoring trailing whitespace
    end = len(text.rstrip())
    if start < end:
        yield start, end


def extract_evidence(
    chunk_text: str,
    category: str,
    category_cues: dict[str, list[str]],
    chunk_start: int = 0,
    min_score: int = 1,
) -> Evidence | None:
    """
    Return the best-supporting sentence in 'chunk_text' as an Evidence, or None.

    chunk_text: text of the chunk
    category: predicted category name
    category_cues: output of build_category_cues()
    chunk_start: offset of chunk within full contract
    min_score: min number of cue words a sentence must contain

    Returns None when category has no cues or no sentence scores high enough.
    Cues match whole words (plus simple inflections). Ties go to the earliest sentence.
    """
    cues = category_cues.get(normalize_category(category), [])
    if not cues:
        return None

    # Score every sentence by how many distinct cue words it contains
    patterns = [_cue_pattern(cue) for cue in cues]
    best, best_score = None, 0
    for s, e in _sentence_spans(chunk_text):
        sentence = chunk_text[s:e].lower()
        score = sum(bool(p.search(sentence)) for p in patterns)
        if score > best_score:
            best, best_score = (s, e), score

    if best is None or best_score < min_score:
        return None

    # Convert chunk-relative offsets to contract-absolute offsets
    s, e = best
    return Evidence(quote=chunk_text[s:e], start=chunk_start + s, end=chunk_start + e)


class GroundedEvidence(BaseModel):
    """A predicted clause whose Evidence passed validation against the full contract text."""
    contract_id: str
    chunk_index: int
    category: str
    evidence: Evidence


def _drop_contained(grounded: list[GroundedEvidence]) -> list[GroundedEvidence]:
    """
    Drop evidence whose span is equal to or inside another span of the same category.
    Overlapping chunks often pick the same sentence; the earliest chunk's record is kept
    and the original order is preserved.
    """
    # Sort by (category, start, longest first) so a contained span always follows its container
    order = sorted(range(len(grounded)), key=lambda i: (
        grounded[i].category, grounded[i].evidence.start, -grounded[i].evidence.end, i))
    keep, last = set(), None
    for i in order:
        g = grounded[i]
        if last is not None and last.category == g.category and g.evidence.end <= last.evidence.end:
            continue
        keep.add(i)
        last = g
    return [g for i, g in enumerate(grounded) if i in keep]


class Unknown(BaseModel):
    """
    A prediction that couldn't be grounded in the source text; feeds the Unknown / human-in-the-loop queue.
    'reason' says why:
        "no_cue_match": no sentence in the chunk contained enough cue words
        "span_mismatch": an Evidence was built but failed validate_against()
    """
    contract_id: str
    chunk_index: int
    category: str
    reason: Literal["no_cue_match", "span_mismatch"]


def extract_for_predictions(
    contract_text: str,
    predictions: Iterable[tuple[Chunk, str]],
    category_cues: dict[str, list[str]],
) -> tuple[list[GroundedEvidence], list[Unknown]]:
    """
    Pipeline entry point: grounds every predicted clause in a contract.

    contract_text: the full contract text used to verify each Evidence
    predictions: iterable of (chunk, category), one per (chunk, predicted category) pair
    category_cues: output of build_category_cues()

    Returns (grounded, unknown):
        grounded: GroundedEvidence records, all verified against contract_text, with at most
                  one record per distinct span per category (contained duplicates are dropped)
        unknown: Unknown records for everything that could not be grounded, one per (chunk, category)
    """
    grounded, unknown = [], []
    for chunk, category in predictions:
        ids = {"contract_id": chunk.contract_id, "chunk_index": chunk.chunk_index, "category": category}
        ev = extract_evidence(chunk.text, category, category_cues, chunk.start)

        # Route ungroundable or mismatched evidence to Unknown; never pass a bad Evidence downstream
        if ev is None:
            unknown.append(Unknown(**ids, reason="no_cue_match"))
        elif not ev.validate_against(contract_text):
            unknown.append(Unknown(**ids, reason="span_mismatch"))
        else:
            grounded.append(GroundedEvidence(**ids, evidence=ev))
    return _drop_contained(grounded), unknown
