import pandas as pd
import pytest

from src.evidence.cues import (
    CATEGORY_DESCRIPTIONS_PATH,
    MANUAL_CUES,
    build_category_cues,
    load_category_cues,
    normalize_category,
)
from src.labels.categories import CATEGORIES


def make_df(prefixed: bool = True) -> pd.DataFrame:
    """Build a 3-row frame shaped like category_descriptions.csv, with or without value prefixes."""
    rows = [
        ("Governing Law", "Which state law governs interpretation"),
        ("Termination for Convenience", "Can a party terminate without cause"),
        ("Insurance", "Insurance requirements for party"),
    ]
    if prefixed:
        rows = [(f"Category: {n}", f"Description: {d}") for n, d in rows]
    return pd.DataFrame(rows, columns=["Category (incl. context and answer)", "Description"])


# ----- normalize_category -----
@pytest.mark.parametrize(
    "name, expected",
    [
        ("IP Ownership Assignment", "ipownershipassignment"),
        ("Ip Ownership Assignment", "ipownershipassignment"),
        ("Rofr/Rofo/Rofn", "rofrroforofn"),
        ("No-Solicit Of Customers", "nosolicitofcustomers"),
    ],
)
def test_normalize_category_case_and_punctuation_insensitive(name, expected):
    """Case, spaces, slashes and hyphens are all dropped from category keys."""
    assert normalize_category(name) == expected


# ----- build_category_cues golden values (worked out by hand from make_df) -----
def test_build_category_cues_prefixed_csv_golden():
    """Prefixed values are stripped; 'party' (in 2 categories) is dropped at max_df=1; rarest-then-alpha order."""
    cues = build_category_cues(make_df(), max_df=1)
    assert cues == {
        "governinglaw": ["governs", "interpretation", "state", "which"],
        "terminationforconvenience": ["cause", "terminate", "without"],
        "insurance": ["insurance", "requirements"],
    }


def test_build_category_cues_unprefixed_matches_prefixed():
    """Values without 'Category:'/'Description:' prefixes produce identical cues."""
    assert build_category_cues(make_df(prefixed=False)) == build_category_cues(make_df())


def test_build_category_cues_respects_cues_per_category():
    """cues_per_category truncates each list after sorting."""
    cues = build_category_cues(make_df(), cues_per_category=2, max_df=1)
    assert cues["governinglaw"] == ["governs", "interpretation"]


def test_build_category_cues_custom_columns():
    """Non-default column names work when passed explicitly."""
    df = make_df().rename(columns={"Category (incl. context and answer)": "Name", "Description": "Desc"})
    assert build_category_cues(df, name_col="Name", desc_col="Desc") == build_category_cues(make_df())


# ----- Manual cues -----
def test_build_category_cues_manual_cues_appended_without_duplicates():
    """Manual cues are appended after derived ones, skipping words already derived."""
    df = pd.DataFrame([("Category: Document Name", "Description: The name of the agreement")],
                      columns=["Category (incl. context and answer)", "Description"])
    manual = {"documentname": ["agreement", "titled"], "absent": ["ignored"]}
    assert build_category_cues(df, manual_cues=manual) == {"documentname": ["agreement", "titled"]}


# ----- Real CSV -----
@pytest.fixture(scope="module")
def real_cues():
    """Cues built from the real category_descriptions.csv, skipping if the file is missing."""
    if not CATEGORY_DESCRIPTIONS_PATH.exists():
        pytest.skip(f"Category CSV not found at {CATEGORY_DESCRIPTIONS_PATH}")
    return load_category_cues()


def test_load_category_cues_real_csv_covers_all_categories(real_cues):
    """Every one of the 41 CUAD categories (as used in question IDs) has a key."""
    assert set(real_cues) == {normalize_category(c) for c in CATEGORIES}


def test_load_category_cues_real_csv_golden_governing_law(real_cues):
    """Governing Law cues from the real CSV match the hand-checked list."""
    assert real_cues["governinglaw"] == ["country", "governs", "interpretation", "state", "which"]


def test_load_category_cues_real_csv_every_category_has_cues(real_cues):
    """With MANUAL_CUES, no category is left without cues."""
    assert all(real_cues.values())


def test_load_category_cues_real_csv_without_manual_cues_leaves_terse_categories_empty():
    """Disabling MANUAL_CUES shows exactly which descriptions are too terse to yield cues."""
    if not CATEGORY_DESCRIPTIONS_PATH.exists():
        pytest.skip(f"Category CSV not found at {CATEGORY_DESCRIPTIONS_PATH}")
    cues = load_category_cues(manual_cues=None)
    assert sorted(k for k, v in cues.items() if not v) == ["agreementdate", "documentname"]


def test_load_category_cues_real_csv_golden_agreement_date(real_cues):
    """Agreement Date gets exactly its hand-written cues."""
    assert real_cues["agreementdate"] == MANUAL_CUES["agreementdate"] == ["dated", "entered", "made", "day of"]
