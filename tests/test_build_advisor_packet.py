import random

import pytest

from scripts.build_advisor_packet import (
    INSTRUCTIONS_FILE,
    KEY_FILE,
    PACKET_COLUMNS,
    PACKET_FILE,
    Candidate,
    bucket_quotas,
    build_packet,
    collapse,
    grounded_candidates,
    jaccard,
    load_category_descriptions,
    quote_context,
    select_clauses,
    write_packet,
)
from src.evidence.schema import Evidence
from src.labels.categories import CATEGORIES
from src.risk.agreement import load_key, read_packet_csv, rescore_key
from src.risk.schema import SIGNAL_NAMES, ClauseInput, WeightSet
from src.risk.scoring import score_contract_clauses

# Composite = category severity, so the rating is set by the category alone:
# Uncapped Liability 0.8 (High), Exclusivity 0.6 (High), Audit Rights 0.4 (Medium), Insurance 0.4 (Medium),
# Governing Law 0.1 (Low), Parties 0.0 (Low), anything else 0.2 (Low)
WEIGHTS = WeightSet(
    weight_version_id="packet-test-v1",
    created_by="tests",
    signal_weights={name: 0.0 for name in SIGNAL_NAMES} | {"category_severity": 1.0},
    medium_threshold=0.35,
    high_threshold=0.55,
    category_severity={"Uncapped Liability": 0.8, "Exclusivity": 0.6, "Audit Rights": 0.4, "Insurance": 0.4,
                       "Governing Law": 0.1, "Parties": 0.0},
    default_severity=0.2,
)
MODEL_ONLY_WORDS = ("score", "rating", "signal", "probability", "composite", "weight")


def _quote(tag: str) -> str:
    """A ten-word quote of words unique to tag, so no two tags look like near-duplicates."""
    return " ".join(f"{tag}w{j}" for j in range(10))


def _candidates(spec: list[tuple[str, str]]) -> tuple[list[Candidate], dict[str, str]]:
    """Grounded candidates for (contract_id, category) pairs, each quote placed in its own contract text."""
    texts: dict[str, str] = {}
    clauses = []
    for i, (contract_id, category) in enumerate(spec):
        quote, prefix = _quote(f"q{i}"), texts.get(contract_id, "Preamble text. ")
        texts[contract_id] = prefix + quote + " "
        clauses.append(ClauseInput(contract_id=contract_id, chunk_index=i, category=category, probability=0.9,
                                   evidence=Evidence(quote=quote, start=len(prefix), end=len(prefix) + len(quote))))
    scores = score_contract_clauses(clauses, WEIGHTS)
    return [Candidate(c, s) for c, s in zip(clauses, scores)], texts


# Six clauses per bucket, each in its own contract, from two categories per bucket
POOL_SPEC = [(f"k{i}", category) for i, category in enumerate(
    ["Uncapped Liability", "Exclusivity"] * 3 + ["Audit Rights", "Insurance"] * 3 + ["Governing Law", "Parties"] * 3)]


# ----- Category descriptions -----
def test_load_category_descriptions_reads_every_category_without_prefixes():
    """All 41 categories, keyed by canonical name, with the CSV's 'Description:' prefix removed."""
    descriptions = load_category_descriptions()
    assert set(descriptions) == set(CATEGORIES)
    assert descriptions["Governing Law"] == "Which state/country's law governs the interpretation of the contract?"


# ----- Text helpers -----
TEXT = "one two three four five six seven"  # "four" is [14:18]


@pytest.mark.parametrize("start, end, chars, expected", [
    (14, 18, 6, ("...three", "five...")),           # windows end on spaces: whole words kept
    (14, 18, 9, ("...three", "five six...")),       # "two" cut to "wo" is dropped; window ends before a space
    (14, 18, 4, ("...", "...")),                    # only partial words in the windows
    (0, 3, 6, ("", "two...")),                      # quote at the start: nothing before, no ellipsis
    (28, 33, 6, ("...six", "")),                    # quote at the end
    (14, 18, 100, ("one two three", "five six seven")),
])
def test_quote_context_trims_to_whole_words(start, end, chars, expected):
    """Context keeps whole words only and marks a cut with '...'."""
    assert quote_context(TEXT, start, end, chars) == expected


def test_collapse_and_jaccard_golden_values():
    """Whitespace runs become one space; two of four distinct words shared gives 0.5."""
    assert collapse("  a\n\n b\t c ") == "a b c"
    assert jaccard(frozenset("abc"), frozenset("bcd")) == 0.5
    assert jaccard(frozenset(), frozenset()) == 1.0


# ----- Sampling -----
@pytest.mark.parametrize("n_items, expected", [
    (50, {"Low": 16, "Medium": 17, "High": 17}),
    (4, {"Low": 1, "Medium": 1, "High": 2}),
    (3, {"Low": 1, "Medium": 1, "High": 1}),
    (0, {"Low": 0, "Medium": 0, "High": 0}),
])
def test_bucket_quotas_split_evenly_with_remainder_to_high_then_medium(n_items, expected):
    """Equal shares per bucket; leftovers go to High first, then Medium."""
    assert bucket_quotas(n_items) == expected


def test_grounded_candidates_drops_unknowns_short_quotes_and_repeats():
    """Unknown-queue clauses, headings under min_words, and the same span seen from two chunks are removed."""
    long_quote, heading = _quote("a"), "LIMITATION OF LIABILITY"
    clauses = [
        ClauseInput(contract_id="c", chunk_index=0, category="Exclusivity", probability=0.9,
                    evidence=Evidence(quote=long_quote, start=0, end=len(long_quote))),
        ClauseInput(contract_id="c", chunk_index=1, category="Exclusivity", probability=0.8,
                    evidence=Evidence(quote=long_quote, start=0, end=len(long_quote))),
        ClauseInput(contract_id="c", chunk_index=2, category="Uncapped Liability", probability=0.9,
                    evidence=Evidence(quote=heading, start=0, end=len(heading))),
        ClauseInput(contract_id="c", chunk_index=3, category="Insurance", probability=0.9, evidence=None),
    ]
    kept = grounded_candidates(clauses, score_contract_clauses(clauses, WEIGHTS), min_words=8)
    assert [(c.clause.chunk_index, c.clause.category) for c in kept] == [(0, "Exclusivity")]


def test_select_clauses_stratifies_over_buckets():
    """With enough clauses everywhere, 9 items split 3 / 3 / 3 over High, Medium and Low."""
    candidates, _ = _candidates(POOL_SPEC)
    chosen = select_clauses(candidates, 9, seed=0)
    assert sorted(c.score.rating for c in chosen) == ["High"] * 3 + ["Low"] * 3 + ["Medium"] * 3


def test_select_clauses_is_seeded():
    """The same seed picks the same clauses in the same order."""
    candidates, _ = _candidates(POOL_SPEC)
    assert select_clauses(candidates, 6, seed=3) == select_clauses(candidates, 6, seed=3)


def test_select_clauses_includes_priority_categories():
    """A priority category with one clause is always picked, even when its bucket would rarely draw it."""
    candidates, _ = _candidates(POOL_SPEC + [("rare", "Non-Compete")])  # default severity: Low
    for seed in range(5):
        chosen = select_clauses(candidates, 3, seed=seed, priority_categories=["Non-Compete"])
        assert "Non-Compete" in {c.clause.category for c in chosen}


def test_select_clauses_skips_near_duplicate_quotes():
    """Two quotes with the same words count once, whatever contract they come from."""
    candidates, _ = _candidates([("a", "Exclusivity"), ("b", "Exclusivity")])
    clone = candidates[1].clause.model_copy(update={"evidence": candidates[0].clause.evidence})
    chosen = select_clauses([candidates[0], Candidate(clone, candidates[1].score)], 2, seed=0)
    assert len(chosen) == 1


def test_select_clauses_respects_the_contract_cap():
    """No contract contributes more than max_per_contract clauses."""
    candidates, _ = _candidates([("same", category) for _, category in POOL_SPEC])
    chosen = select_clauses(candidates, 9, seed=0, max_per_contract=2)
    assert len(chosen) == 2


def test_select_clauses_fills_from_other_buckets_when_one_runs_short():
    """With a single High clause, the other slots come from Medium and Low instead of staying empty."""
    candidates, _ = _candidates([("h", "Uncapped Liability")] + POOL_SPEC[6:])
    chosen = select_clauses(candidates, 6, seed=0)
    assert len(chosen) == 6 and [c.score.rating for c in chosen].count("High") == 1


def test_select_clauses_raises_the_category_cap_for_a_bucket_with_few_categories():
    """High has two categories and a quota of 6, so each may give 3 clauses even with max_per_category=1."""
    spec = [(f"h{i}", category) for i, category in enumerate(["Uncapped Liability", "Exclusivity"] * 4)]
    candidates, _ = _candidates(spec)
    chosen = select_clauses(candidates, 18, seed=0, max_per_category=1)
    assert len(chosen) == 6  # quota 6 / 2 categories = cap 3 per category; only High clauses exist


# ----- Packet and key -----
def _built(seed: int = 4):
    """A packet and key over all of POOL_SPEC."""
    candidates, texts = _candidates(POOL_SPEC)
    categories = {cid: sorted({c.clause.category for c in candidates if c.clause.contract_id == cid}) for cid in texts}
    packet, key = build_packet(candidates, texts, categories, {"Exclusivity": "Is there exclusivity?"}, WEIGHTS,
                               seed, context_chars=20, settings={"seed": seed})
    return candidates, packet, key


def test_packet_shows_no_model_output():
    """The packet has exactly PACKET_COLUMNS, none of them model output, and blank advisor columns."""
    _, packet, _ = _built()
    assert tuple(packet.columns) == PACKET_COLUMNS
    assert not [c for c in packet.columns if any(word in c for word in MODEL_ONLY_WORDS)]
    assert (packet[["advisor_bucket", "advisor_rank", "advisor_notes"]] == "").all().all()


def test_packet_rows_are_shuffled_with_the_seed_and_numbered_in_that_order():
    """Rows follow the seeded shuffle of the chosen clauses; ids run C01, C02, ... down the shuffled rows."""
    candidates, packet, key = _built(seed=4)
    expected = list(candidates)
    random.Random(4).shuffle(expected)
    assert list(packet["quote"]) == [c.clause.evidence.quote for c in expected]
    assert list(packet["item_id"]) == [f"C{i:02d}" for i in range(1, len(candidates) + 1)]
    assert [i.item_id for i in key.items] == list(packet["item_id"])


def test_key_holds_what_rescoring_needs_and_reproduces_the_scores():
    """Each key item carries its span, probability and contract categories, and rescoring gives its stored score."""
    _, packet, key = _built()
    first = key.items[0]
    assert packet["quote"][0] == first.quote and packet["contract"][0] == first.contract_id
    assert first.contract_categories == (first.category,) and first.probability == 0.9
    assert key.settings == {"seed": 4} and key.packet_id == packet["packet_id"][0]
    rescored = rescore_key(key, WEIGHTS)
    assert all(rescored[i.item_id].composite_score == i.composite_score for i in key.items)


def test_packet_carries_category_descriptions_and_context():
    """The description comes from the lookup (blank when missing) and context comes from the contract text."""
    _, packet, _ = _built()
    exclusivity = packet[packet["category"] == "Exclusivity"].iloc[0]
    assert exclusivity["category_description"] == "Is there exclusivity?"
    assert exclusivity["context_before"] == "Preamble text."
    assert (packet[packet["category"] == "Audit Rights"]["category_description"] == "").all()


def test_build_packet_rejects_an_empty_selection():
    """An empty packet is an error, not an empty file."""
    with pytest.raises(ValueError):
        build_packet([], {}, {}, {}, WEIGHTS, 0)


def test_write_packet_writes_three_files_that_load_back(tmp_path):
    """The packet CSV reads back unchanged, the key loads, and the instructions name the packet and its size."""
    _, packet, key = _built()
    write_packet(tmp_path, packet, key)
    assert read_packet_csv(tmp_path / PACKET_FILE).equals(packet)
    assert load_key(tmp_path / KEY_FILE) == key
    text = (tmp_path / INSTRUCTIONS_FILE).read_text(encoding="utf-8")
    assert key.packet_id in text and f"{len(key.items)} clauses" in text and "1 (riskiest)" in text
