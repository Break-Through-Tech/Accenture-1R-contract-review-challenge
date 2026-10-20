import re
from dataclasses import dataclass

from src.evidence.cues import normalize_category
from src.evidence.schema import Evidence

# A sentence ends at "." or ";" followed by whitespace.
_SENTENCE_BREAK = re.compile(r"(?<=[.;])\s+")

def _sentence_spans(text: str):
    """
    Yield (start, end) offsets for each sentence in text
    Offsets come from regex match positions
    """
    start = 0
    for m in _SENTENCE_BREAK.finditer(text):
        if m.start() > start:
             yield start, m.start()
        start = m.end()
    end = len(text.rstrip())  # final sentence, ignoring trailing whitespace
    if start < end:
        yield start, end

def extract_evidence(chunk_text, category, category_cues, chunk_start=0, min_score=1):
    """
    Return the best-supporting sentence in 'chunk_text' as an Evidence, or None

    chunk_text: text of the chunk
    category: predicted category name
    category_cues: output of build_category_cues()
    chunk_start: offset of chunk within full contract
    min_score: min number of cue words a sentence must contain

    Returns None when category has no cues or no sentence scores high enough
    """
    cues = category_cues.get(normalize_category(category), [])
    if not cues: 
        return None

    best, best_score = None, 0
    for s, e in _sentence_spans(chunk_text):
        sentence = chunk_text[s:e].lower()
        score = sum(cue in sentence for cue in cues)
        if score > best_score:
            best, best_score = (s, e), score
    
    if best is None or best_score < min_score:
        return None
    
    s, e = best

    # Convert chunk-relative offsets to contract-absolute offsets.
    return Evidence(quote=chunk_text[s:e], start=chunk_start + s, end=chunk_start + e)

