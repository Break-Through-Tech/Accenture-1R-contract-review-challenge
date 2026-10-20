from pathlib import Path

import pytest

from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import load_cuad

TRAIN_PATH = Path("data/cuad/train_separate_questions.json")


# ----- Categories: fixed list of 41 in alphabetical order -----
def test_categories_has_41_unique_entries():
    """Checks CATEGORIES holds exactly 41 names with no duplicates."""
    assert len(CATEGORIES) == 41
    assert len(set(CATEGORIES)) == 41


def test_categories_ordered_alphabetically():
    """Checks CATEGORIES is sorted, so an accidental reorder fails loudly."""
    assert CATEGORIES == sorted(CATEGORIES)


# ----- Categories: name -> column index lookup -----
def test_category_to_index_round_trip():
    """Checks every category maps to a unique column that points back to the same name."""
    assert len(CATEGORY_TO_INDEX) == 41
    for c in CATEGORIES:
        assert CATEGORIES[CATEGORY_TO_INDEX[c]] == c


# ----- Categories: consistency with the loader on real data -----
@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="CUAD train file not available")
def test_loader_categories_are_in_categories():
    """Checks every category load_cuad produces on the train file is in CATEGORIES."""
    contracts = load_cuad(TRAIN_PATH)
    loaded = {s.category for c in contracts for s in c.spans}
    assert loaded <= set(CATEGORIES)
