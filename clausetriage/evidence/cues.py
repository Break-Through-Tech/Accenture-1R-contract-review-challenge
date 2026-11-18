"""
Cues are words that tend to show up in sentences about a given clause category
EX: "terminate" for Termination For Convenience
The extractor scores each sentence by how many of the category's cues it contains,
so the quality of the cues directly caps the quality of the evidence

Cues are derived from the Description column of data/cuad/category_descriptions.csv

Other notes:
- The real CSV prefixes every value ("Category: Governing Law", "Description: ..."),
  so those prefixes are stripped before building cues
- Words are sorted, so the same CSV produces the same cues on every run and machine
- Generic words (like "contract" and "party") are dropped since they don't carry a signal
- Category names are normalized so "Ip Ownership" and "IP Ownership" have the same key
- Document Name and Agreement Date have descriptions too terse to yield cues, so they get
  hand-written cues from MANUAL_CUES instead of always going to the Unknown queue
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

CATEGORY_DESCRIPTIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "cuad" / "category_descriptions.csv"
CATEGORY_COL = "Category (incl. context and answer)"
DESCRIPTION_COL = "Description"

# Hand-written cues, keyed by normalized category, added after the derived ones
MANUAL_CUES: dict[str, list[str]] = {
    "documentname": ["agreement"],
    "agreementdate": ["dated", "entered", "made", "day of"],
}


def normalize_category(name: str) -> str:
    """Lowercase a category name and strip everything except letters and digits."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _strip_label(value: object, label: str) -> str:
    """Remove a leading "<label>:" prefix (as used in the CUAD CSV) and surrounding whitespace."""
    text = str(value).strip()
    prefix = f"{label}:"
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):]
    return text.strip()


def build_category_cues(
    df: pd.DataFrame,
    name_col: str = CATEGORY_COL,
    desc_col: str = DESCRIPTION_COL,
    cues_per_category: int = 10,
    max_df: int = 3,
    manual_cues: dict[str, list[str]] | None = MANUAL_CUES,
) -> dict[str, list[str]]:
    """
    Build {normalized_category_name: [cue words]} from the category descriptions CSV.

    df: dataframe loaded from category_descriptions.csv
    name_col, desc_col: column names (defaults match the real CUAD CSV)
    cues_per_category: max cue words kept per category
    max_df: drop words used by more than this many categories
    manual_cues: extra cues appended (deduplicated) for categories present in df; None disables
    """
    # Collect distinct 5+ letter words in each category's description, after stripping CSV prefixes
    words_by_cat = {
        normalize_category(_strip_label(row[name_col], "Category")): {
            w.lower() for w in re.findall(r"[A-Za-z]{5,}", _strip_label(row[desc_col], "Description"))
        }
        for _, row in df.iterrows()
    }

    # Count how many categories use each word (document frequency)
    df_count = Counter(w for words in words_by_cat.values() for w in words)

    # Per category, keep the rarest words first. Sorting on (frequency, word) breaks ties alphabetically
    cues = {}
    for cat, words in words_by_cat.items():
        distinctive = (w for w in words if df_count[w] <= max_df)
        cues[cat] = sorted(distinctive, key=lambda w: (df_count[w], w))[:cues_per_category]

    # Append hand-written cues for categories whose descriptions are too terse
    for cat, extra in (manual_cues or {}).items():
        if cat in cues:
            cues[cat] += [w for w in extra if w not in cues[cat]]
    return cues


def load_category_cues(path: str | Path = CATEGORY_DESCRIPTIONS_PATH, **kwargs) -> dict[str, list[str]]:
    """Read the category descriptions CSV (BOM-safe) and build cues; kwargs go to build_category_cues."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return build_category_cues(df, **kwargs)
