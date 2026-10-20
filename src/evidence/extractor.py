import re
from dataclasses import dataclass

from src.evidence.cues import normalize_category
from src.evidence.schema import Evidence

"""
For each (chunk, category) pair the classifier predicted, it picks the single sentence in the 
chunk that best supports the category, wraps it in a 'Evidence' object with absolute offsets, and
verifies it against the full contract text.
If no sentence qualifies, or verification fails, the prediction goes to the Unknown queue instead of
beign guessed at. 

How everything comes togehter:
_sentence_spans(): splits chunk into (start, end) offsets
extract_eveidence(): scores sentences by cue words, returns the best as Evidence (or None)
extract_for_predictions(): entry point other pipeline stages calls. Runs the above for every
                           prediction in a contract, validates each result, and sorts outcome
                           into 'grounded' and 'unknown'

Inputs from other modules:
'category_cues' from cues.build_category_cues()
'chunk_start', which comes form the chunker's Chunk object

Limitations:
- Sentence splitting on abberviations like "Inc." and "No."
- Only one sentence is returned per clause, so a clause spanning several sentecnes is only partially captured
- Cue quality is capped by wording of the category descriptions
"""
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

@dataclass
class Unknown:
    """
    A prediction that couldn't be grounded in the source text

    Feed the Unknown / human-in-the-loop queue
    'reason' says why:
        "no_cue_match": no sentence in the chunk contained enough cue words
        "span_mismatch": an Evidence was build but failed validate_against()
    """
    contract_id: str
    chunk_idx: int
    category: str
    reason: str

def extract_for_predictions(contract_id, contract_text, predictions, category_cues):
    """
    Pipeline entry point: grounds every predicted clause in a contract.

    contract_id: identifier carried through to any Unknown records
    contract_text: the full contract text used to verify each Evidence
    predictions: iterable of (chunk_idx, chunk_text, chunk_start, category),
                    one per (chunk, predicted category) pair. The field names must
                    line up with the real Chunk object from the chunker.
    category_cues: output of build_category_cues()

    Returns (grounded, unknown):
        grounded: list of (chunk_idx, category, Evidence), all verified
        unknown: list of Unknown records for everything that could not be grounded
    """
    grounded, unknown = [], []
    for chunk_idx, chunk_text, chunk_start, category in predictions:
        ev = extract_evidence(chunk_text, category, category_cues, chunk_start)
        if ev is None:
            unknown.append(Unknown(contract_id, chunk_idx, category, "no_cue_match"))
        elif not ev.validate_against(contract_text):
            # Should not happen if offsets are correct, but a bad Evidence must never be passed downstream.
            unknown.append(Unknown(contract_id, chunk_idx, category, "span_mismatch"))
        else:
            grounded.append((chunk_idx, category, ev))
    return grounded, unknown