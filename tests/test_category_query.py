from pathlib import Path

from src.labels.categories import CATEGORIES
from src.models.encoder.category_descriptions import load_category_descriptions


def test_category_description_prompts_match_canonical_label_order():
    root = Path(__file__).resolve().parents[1]
    prompts = load_category_descriptions(root / "data/cuad/category_descriptions.csv")

    assert len(prompts) == len(CATEGORIES) == 41
    assert all(prompt.startswith(f"{category}.") for prompt, category in zip(prompts, CATEGORIES))
    assert all(len(prompt) > len(category) + 2 for prompt, category in zip(prompts, CATEGORIES))
