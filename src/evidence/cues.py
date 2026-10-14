import re
from collections import Counter

"""
Cues are words that tend to show up in sentences about a given clause category
EX: "terminate" for Termination
The extractor scores each stenence by how many of the category's cues it contains,
so the quality of the cues directly caps the quality of the evidence

Cues are derived from the description column of category_desciption.csv

Other notes:
- Words are sorted, so the same CSV produces and runs the same cues on every run and machine
- Generic words (like "contract" and "party) are dropped since they don't carry a signal
- Category names are normalzied so "Ip ownership" and "IP Ownership" would have the same key.
"""

def normalize_category(name: str) -> str:
    # Lowercase and strip everything except letters/digits
    return re.sub(r"[^a-z0-9]", "", name.lower())

def build_category_cues(df, name_col="Category", desc_col="Description", cues_per_category=10, max_df=3):
    """
    Builds {normalized_category_name: [cue words]} fronm category CSV

    df: dataframe loaded from category_descriptions.csv
    name_col, desc_col: column names
    cues_per_category: max cue words kept per category
    max_df: drop words used by more than this many categories
    """

    # Collects distinct 5+ letter words in each category's description
    words_by_cat = {
            normalize_category(row[name_col]): {w.lower() for w in re.findall(r"[A-Za-z]{5,}", str(row[desc_col]))}
            for _, row in df.iterrows()
    }

    # Count how many categories use each word (doc frequency)
    df_count = Counter(w for words in words_by_cat.values() for w in words)

    # Per category, keep the rarest words first. Soring on (frequency, word) breaks ties alphabetically
    cues = {}
    for cat, words in words_by_cat.items():
        distinctive = (w for w in words if df_count[w] <= max_df)
        cues[cat] = sorted(distinctive, key=lambda w: (df_count[w], w))[:cues_per_category]
    return cues