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