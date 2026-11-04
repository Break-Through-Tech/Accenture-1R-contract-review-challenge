"""Load category descriptions aligned to the canonical CUAD label order."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from src.labels.categories import CATEGORIES


def _normalize(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]", value.casefold()))


def load_category_descriptions(path: str | Path) -> list[str]:
    """Return ``name + description`` prompts in exact ``CATEGORIES`` order."""
    descriptions: dict[str, str] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            category_field = row["Category (incl. context and answer)"]
            category_name = category_field.removeprefix("Category: ").strip()
            description = row["Description"].removeprefix("Description: ").strip()
            key = _normalize(category_name)
            if not key or not description or key in descriptions:
                raise ValueError(f"invalid or duplicate CUAD category description: {category_name!r}")
            descriptions[key] = description
    expected = {_normalize(category) for category in CATEGORIES}
    if set(descriptions) != expected:
        missing = sorted(expected - set(descriptions))
        extra = sorted(set(descriptions) - expected)
        raise ValueError(f"category-description schema mismatch; missing={missing}, extra={extra}")
    return [f"{category}. {descriptions[_normalize(category)]}" for category in CATEGORIES]
