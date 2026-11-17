import math

import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.stats import spearmanr

from src.evidence.schema import Evidence
from src.risk.agreement import (
    AdvisorRanking,
    PacketKey,
    PacketKeyItem,
    ReturnedPacket,
    adjacent_rate,
    bootstrap_estimate,
    bucket_for,
    confusion_matrix,
    evaluate_agreement,
    exact_rate,
    high_precision,
    high_recall,
    join_packet,
    load_key,
    read_packet_csv,
    rescore_key,
    spearman,
    threshold_grid,
    threshold_sweep,
    weighted_kappa,
    write_key,
)
from src.risk.schema import SIGNAL_NAMES, ClauseInput, WeightSet
from src.risk.scoring import rate, score_clause

# Composite = 0.5 * severity + 0.5 * contract_context, so every score below is hand-computable:
# Uncapped Liability without Insurance: 0.5 * 0.6 + 0.5 * 0.5 = 0.55 (High); with Insurance: 0.30 (Low)
# Exclusivity: 0.5 * 0.8 = 0.40 (Medium); Audit Rights (default 0.5): 0.25 (Low); Governing Law: 0.10 (Low)
WEIGHTS = WeightSet(
    weight_version_id="agreement-test-v1",
    created_by="tests",
    signal_weights={name: 0.0 for name in SIGNAL_NAMES} | {"category_severity": 0.5, "contract_context": 0.5},
    medium_threshold=0.35,
    high_threshold=0.55,
    category_severity={"Uncapped Liability": 0.6, "Exclusivity": 0.8, "Governing Law": 0.2},
    default_severity=0.5,
    cooccurrence_rules=[{"target_category": "Uncapped Liability", "other_category": "Insurance",
                         "when": "absent", "adjustment": 0.5}],
)
L, M, H = "Low", "Medium", "High"


def _item(item_id: str, category: str, others: tuple[str, ...] = (), start: int = 0,
          weights: WeightSet = WEIGHTS) -> PacketKeyItem:
    """A key item for a quote naming its category, scored under weights, in a contract with these other categories."""
    quote = f"The {category} clause of item {item_id} reads like this."
    clause = ClauseInput(contract_id=f"contract-{item_id}", chunk_index=1, category=category, probability=0.7,
                         evidence=Evidence(quote=quote, start=start, end=start + len(quote)))
    score = score_clause(clause, {category, *others}, weights)
    return PacketKeyItem(
        item_id=item_id, contract_id=clause.contract_id, chunk_index=1, category=category, start=start,
        end=start + len(quote), quote=quote, probability=0.7, contract_categories=(category, *others),
        composite_score=score.composite_score, rating=score.rating, signals=score.signals,
        weight_version_id=score.weight_version_id, weight_set_sha256=score.weight_set_sha256,
    )


def _key(items: tuple[PacketKeyItem, ...] | None = None, weights: WeightSet = WEIGHTS) -> PacketKey:
    """A key over four clauses scoring High (0.55), Medium (0.40), Low (0.25) and Low (0.10)."""
    items = items or (_item("C1", "Uncapped Liability"), _item("C2", "Exclusivity"),
                      _item("C3", "Audit Rights"), _item("C4", "Governing Law"))
    return PacketKey(packet_id="pkt-test", weight_version_id=weights.weight_version_id,
                     weight_set_sha256=weights.content_sha256, medium_threshold=weights.medium_threshold,
                     high_threshold=weights.high_threshold, items=items)


def _packet(key: PacketKey, buckets: list[str], ranks: list[str], notes: list[str] | None = None) -> pd.DataFrame:
    """The packet a returned CSV would hold: one row per key item, with the advisor's cells filled in."""
    return pd.DataFrame({
        "packet_id": key.packet_id, "item_id": [i.item_id for i in key.items],
        "category": [i.category for i in key.items], "quote": [i.quote for i in key.items],
        "advisor_bucket": buckets, "advisor_rank": ranks, "advisor_notes": notes or [""] * len(key.items),
    })


def _ranking(**overrides) -> dict:
    """Valid AdvisorRanking fields, with overrides applied."""
    fields = {"contract_id": "c1", "category": "Uncapped Liability", "start": 10, "end": 15, "quote": "abcde",
              "advisor_rank": 1, "advisor_bucket": "High", "reviewer": "advisor", "ranking_set": "pkt-test"}
    return fields | overrides


# ----- AdvisorRanking -----
def test_advisor_ranking_field_names_match_the_table():
    """The record's fields are exactly the advisor_rankings table columns, in order."""
    assert list(AdvisorRanking.model_fields) == [
        "contract_id", "category", "start", "end", "quote", "advisor_rank", "advisor_bucket", "reviewer", "ranking_set",
    ]


def test_advisor_ranking_canonicalizes_category_and_allows_one_judgement():
    """A lower-case category gets its canonical spelling, and a bucket alone or a rank alone is enough."""
    assert AdvisorRanking(**_ranking(category="uncapped liability")).category == "Uncapped Liability"
    assert AdvisorRanking(**_ranking(advisor_rank=None)).advisor_bucket == "High"
    assert AdvisorRanking(**_ranking(advisor_bucket=None)).advisor_rank == 1


@pytest.mark.parametrize("overrides", [
    {"end": 10}, {"quote": "abcdef"}, {"advisor_rank": 0}, {"advisor_bucket": "Severe"},
    {"advisor_rank": None, "advisor_bucket": None}, {"category": "Not A Category"}, {"reviewer": ""},
    {"ranking_set": ""}, {"start": -1}, {"notes": "extra field"},
])
def test_advisor_ranking_rejects_malformed_records(overrides):
    """Empty or mismatched spans, ranks below 1, unknown buckets or categories, no judgement and extra fields fail."""
    with pytest.raises(ValidationError):
        AdvisorRanking(**_ranking(**overrides))


# ----- Packet key -----
def test_key_item_rebuilds_its_clause_input():
    """to_clause_input carries the span as grounded Evidence and the stored probability."""
    item = _item("C1", "Exclusivity", start=5)
    clause = item.to_clause_input()
    assert (clause.category, clause.probability, clause.chunk_index) == ("Exclusivity", 0.7, 1)
    assert (clause.evidence.start, clause.evidence.end, clause.evidence.quote) == (5, item.end, item.quote)


def test_key_round_trips_through_json(tmp_path):
    """write_key then load_key gives back an equal key."""
    write_key(_key(), tmp_path / "key.json")
    assert load_key(tmp_path / "key.json") == _key()


def test_key_rejects_duplicate_item_ids():
    """Two items with one id can't be told apart in a returned packet."""
    with pytest.raises(ValidationError, match="unique"):
        _key((_item("C1", "Exclusivity"), _item("C1", "Audit Rights")))


def test_key_rejects_items_from_another_weight_set():
    """Every stored score must come from the key's own weight set."""
    other = WEIGHTS.model_copy(update={"weight_version_id": "other", "high_threshold": 0.6})
    with pytest.raises(ValidationError, match="different weight set"):
        _key((_item("C1", "Exclusivity"), _item("C2", "Audit Rights", weights=other)))


def test_key_item_requires_its_own_category_in_the_contract():
    """contract_categories must include the item's category, as score_contract_clauses would build it."""
    fields = _item("C1", "Exclusivity").model_dump() | {"contract_categories": ("Audit Rights",)}
    with pytest.raises(ValidationError, match="must include"):
        PacketKeyItem(**fields)


# ----- Rescoring -----
def test_rescore_key_reproduces_stored_scores_and_uses_contract_categories():
    """Rescoring rebuilds 0.55 High without Insurance and 0.30 Low with it, matching the stored values."""
    key = _key((_item("C1", "Uncapped Liability"), _item("C2", "Uncapped Liability", ("Insurance",))))
    rescored = rescore_key(key, WEIGHTS)
    assert (rescored["C1"].composite_score, rescored["C1"].rating) == (pytest.approx(0.55), H)
    assert (rescored["C2"].composite_score, rescored["C2"].rating) == (pytest.approx(0.30), L)
    assert all(rescored[i.item_id].composite_score == i.composite_score for i in key.items)


@pytest.mark.parametrize("score", [0.0, 0.3499, 0.35, 0.5, 0.55, 0.9, 1.0])
def test_bucket_for_matches_scoring_rate(score):
    """The sweep's bucketing gives the same rating as scoring.rate, including exactly on a threshold."""
    assert bucket_for(score, WEIGHTS.medium_threshold, WEIGHTS.high_threshold) == rate(score, WEIGHTS)


# ----- Spearman -----
@pytest.mark.parametrize("x, y, expected", [
    ([0.9, 0.7, 0.5, 0.3], [4, 3, 2, 1], 1.0),                # perfect
    ([0.9, 0.7, 0.5, 0.3], [1, 2, 3, 4], -1.0),               # inverted
    ([1, 2, 3, 4], [0, 0, 1, 1], 2 / math.sqrt(5)),           # ties on one side: ranks 1.5, 1.5, 3.5, 3.5
    ([1, 2, 2, 3], [0, 0, 1, 1], 1 / math.sqrt(2)),           # ties on both sides: 3 / sqrt(4.5 * 4)
])
def test_spearman_matches_hand_computed_values(x, y, expected):
    """Average ranks for ties, then Pearson on the ranks; agrees with scipy.stats.spearmanr."""
    assert spearman(x, y) == pytest.approx(expected)
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)


@pytest.mark.parametrize("x, y", [([1, 2, 3], [2, 2, 2]), ([0.5], [1])])
def test_spearman_is_undefined_for_constant_or_single_values(x, y):
    """A constant side or a single clause has no correlation: None, not NaN."""
    assert spearman(x, y) is None


def test_spearman_rejects_unequal_lengths():
    """Mismatched inputs are a bug, not a missing value."""
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2, 3])


# ----- Bucket agreement -----
ADVISOR, MODEL = [L, M, M, H], [L, M, H, H]


def test_bucket_metrics_match_hand_computed_values():
    """3 of 4 exact, all within one step, the confusion matrix, and High P = 1/2, R = 1/1."""
    assert exact_rate(ADVISOR, MODEL) == 0.75
    assert adjacent_rate(ADVISOR, MODEL) == 1.0
    assert confusion_matrix(ADVISOR, MODEL) == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    assert high_precision(ADVISOR, MODEL) == 0.5
    assert high_recall(ADVISOR, MODEL) == 1.0


@pytest.mark.parametrize("scheme, expected", [
    ("linear", 5 / 7),      # observed 3.5/4 = 0.875, chance 0.5625: (0.875 - 0.5625) / 0.4375
    ("quadratic", 0.8),     # observed 3.75/4 = 0.9375, chance 0.6875: 0.25 / 0.3125
])
def test_weighted_kappa_matches_hand_computed_values(scheme, expected):
    """Weighted kappa with partial credit for a one-step disagreement."""
    assert weighted_kappa(ADVISOR, MODEL, scheme) == pytest.approx(expected)


@pytest.mark.parametrize("advisor, model, linear, quadratic, exact, adjacent", [
    ([L, M, H], [L, M, H], 1.0, 1.0, 1.0, 1.0),               # perfect
    ([H, M, L], [L, M, H], -0.5, -1.0, 1 / 3, 1 / 3),         # inverted: chance 5/9 (linear), 2/3 (quadratic)
])
def test_bucket_metrics_on_perfect_and_inverted_buckets(advisor, model, linear, quadratic, exact, adjacent):
    """Perfect agreement scores 1 everywhere; the inverted order scores below chance."""
    assert weighted_kappa(advisor, model, "linear") == pytest.approx(linear)
    assert weighted_kappa(advisor, model, "quadratic") == pytest.approx(quadratic)
    assert exact_rate(advisor, model) == pytest.approx(exact)
    assert adjacent_rate(advisor, model) == pytest.approx(adjacent)


def test_undefined_bucket_metrics_return_none():
    """Kappa with one bucket on both sides, precision with no model High, recall with no advisor High."""
    assert weighted_kappa([L, L], [L, L]) is None
    assert high_precision([H, L], [L, L]) is None
    assert high_recall([L, L], [H, L]) is None
    assert exact_rate([], []) is None


def test_weighted_kappa_rejects_unknown_scheme():
    """Only linear and quadratic weights exist."""
    with pytest.raises(ValueError):
        weighted_kappa(ADVISOR, MODEL, "cubic")


# ----- Bootstrap -----
def test_bootstrap_of_a_perfect_ranking_is_a_point_interval():
    """Every defined resample of a perfect ranking has rho = 1, so the interval is [1, 1]."""
    estimate = bootstrap_estimate(spearman, [[0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5]], n_boot=200, seed=1)
    assert (estimate.value, estimate.ci_low, estimate.ci_high) == pytest.approx((1.0, 1.0, 1.0)) and estimate.n == 5
    assert 0 < estimate.n_boot_valid <= 200  # resamples of one repeated clause are undefined and dropped


def test_bootstrap_is_seeded_and_drops_undefined_resamples():
    """The same seed gives the same interval; resamples with no model High don't count for precision."""
    columns = [[H, L, L, M, L, M], [H, L, L, L, M, M]]
    first = bootstrap_estimate(high_precision, columns, n_boot=300, seed=7)
    assert first == bootstrap_estimate(high_precision, columns, n_boot=300, seed=7)
    assert first.value == 1.0 and first.n_boot_valid < 300


def test_bootstrap_of_a_constant_statistic_and_of_no_clauses():
    """A constant statistic gives a zero-width interval; no clauses gives no value and no interval."""
    constant = bootstrap_estimate(lambda a: 0.5, [[1, 2, 3]], n_boot=50, seed=0)
    assert (constant.ci_low, constant.ci_high, constant.n_boot_valid, constant.ci_width) == (0.5, 0.5, 50, 0.0)
    empty = bootstrap_estimate(exact_rate, [[], []], n_boot=50)
    assert (empty.value, empty.ci_low, empty.n) == (None, None, 0)


# ----- Threshold sweep -----
@pytest.mark.parametrize("base, step, half_width, expected", [
    (0.5, 0.1, 0.2, [0.3, 0.4, 0.5, 0.6, 0.7]),
    (0.05, 0.05, 0.1, [0.0, 0.05, 0.1, 0.15]),      # clipped at 0
    (0.95, 0.05, 0.1, [0.85, 0.9, 0.95, 1.0]),      # clipped at 1
])
def test_threshold_grid_is_centred_on_the_base_and_clipped(base, step, half_width, expected):
    """The base is always a grid point, and values outside [0, 1] are dropped."""
    assert threshold_grid(base, step, half_width) == pytest.approx(expected)


def test_threshold_sweep_matches_hand_computed_grid():
    """Scores 0.2/0.4/0.6/0.8 against advisor L/M/M/H on a 3 x 3 grid around (0.35, 0.55)."""
    sweep = threshold_sweep([0.2, 0.4, 0.6, 0.8], [L, M, M, H], 0.35, 0.55, step=0.1, half_width=0.1, tolerance=0.05)
    got = {(p.medium_threshold, p.high_threshold): (p.exact_rate, p.changed_vs_base) for p in sweep.points}
    assert got == {
        (0.25, 0.45): (0.75, 0), (0.25, 0.55): (0.75, 0), (0.25, 0.65): (1.0, 1),
        (0.35, 0.45): (0.75, 0), (0.35, 0.55): (0.75, 0), (0.35, 0.65): (1.0, 1),
        (0.45, 0.55): (0.5, 1), (0.45, 0.65): (0.75, 2),  # (0.45, 0.45) is skipped: medium must be below high
    }
    assert sweep.base_exact_rate == 0.75 and sweep.stable_share == 7 / 8
    assert sweep.medium_stable_range == (0.25, 0.35)  # medium 0.45 drops agreement to 0.5
    assert sweep.high_stable_range == (0.45, 0.65)
    assert sweep.best_exact_rate == 1.0 and sweep.best_points == [(0.25, 0.65), (0.35, 0.65)]


def test_threshold_sweep_counts_buckets_at_each_setting():
    """At the base setting the four scores fall into 1 Low, 1 Medium and 2 High."""
    sweep = threshold_sweep([0.2, 0.4, 0.6, 0.8], [L, M, M, H], 0.35, 0.55, step=0.1, half_width=0.0)
    (point,) = sweep.points
    assert (point.low_count, point.medium_count, point.high_count) == (1, 1, 2)


def test_threshold_sweep_rejects_empty_input():
    """No clauses, no sweep."""
    with pytest.raises(ValueError):
        threshold_sweep([], [], 0.35, 0.55)


# ----- Joining a returned packet with its key -----
def test_join_packet_builds_records_and_skips_blank_rows():
    """Buckets in any case, 'Med', '2.0' ranks and shuffled rows all join; a row with no answers is skipped."""
    key = _key()
    packet = _packet(key, ["high", "Med", "", " low "], ["1", "2.0", "", "3"], ["", "needs context", "", ""])
    returned = join_packet(packet.iloc[[3, 0, 2, 1]], key, reviewer="advisor")
    by_id = {j.item_id: j for j in returned.judged}
    assert returned.skipped == ["C3"] and set(by_id) == {"C1", "C2", "C4"}
    assert (by_id["C2"].ranking.advisor_bucket, by_id["C2"].ranking.advisor_rank, by_id["C2"].notes) == (
        M, 2, "needs context")
    assert by_id["C4"].ranking.advisor_bucket == L
    first = by_id["C1"].ranking
    assert (first.contract_id, first.reviewer, first.ranking_set, first.quote) == (
        "contract-C1", "advisor", "pkt-test", key.items[0].quote)


def test_join_packet_ignores_fully_blank_rows():
    """Empty rows a spreadsheet appends at the end are not read as an unknown item."""
    key = _key()
    packet = _packet(key, [H, M, L, L], ["1", "2", "3", "4"])
    padded = pd.concat([packet, pd.DataFrame([dict.fromkeys(packet.columns, "")] * 2)], ignore_index=True)
    assert len(join_packet(padded, key, "advisor").judged) == 4


def test_join_packet_accepts_a_quote_with_collapsed_whitespace():
    """A spreadsheet that turned the quote's line breaks into spaces still matches its key item."""
    item = _item("C1", "Exclusivity")
    quote = item.quote.replace(" clause ", "\n  clause\t")
    item = PacketKeyItem(**item.model_dump() | {"quote": quote, "end": item.start + len(quote)})
    key = _key((item,))
    packet = _packet(key, [H], ["1"]).assign(quote=" ".join(quote.split()))
    assert join_packet(packet, key, "advisor").judged[0].ranking.quote == quote


@pytest.mark.parametrize("edit, message", [
    (lambda p: p.drop(columns="advisor_rank"), "missing columns"),
    (lambda p: p.assign(packet_id="pkt-other"), "packet_id"),
    (lambda p: pd.concat([p, p.iloc[[0]]]), "repeats item ids"),
    (lambda p: p.assign(item_id=["C1", "C2", "C3", "C9"]), "not in the key"),
    (lambda p: p.iloc[:3], "missing items"),
    (lambda p: p.assign(advisor_bucket=["High", "Severe", "", ""]), "C2: advisor_bucket"),
    (lambda p: p.assign(advisor_rank=["0", "", "", ""]), "C1: advisor_rank"),
    (lambda p: p.assign(advisor_rank=["2.5", "", "", ""]), "C1: advisor_rank"),
    (lambda p: p.assign(advisor_rank=["first", "", "", ""]), "C1: advisor_rank"),
    (lambda p: p.assign(advisor_rank=["5", "", "", ""]), "from 1 to 4"),
    (lambda p: p.assign(quote=list(p["quote"][::-1])), "doesn't match the key"),
    (lambda p: p.assign(category=["Exclusivity"] * 4), "doesn't match the key"),
    (lambda p: p.assign(advisor_bucket="", advisor_rank=""), "no judged items"),
])
def test_join_packet_rejects_malformed_input(edit, message):
    """Each kind of malformed returned packet raises a ValueError that says what is wrong."""
    key = _key()
    packet = _packet(key, [H, M, L, L], ["1", "2", "3", "4"])
    with pytest.raises(ValueError, match=message):
        join_packet(edit(packet), key, "advisor")


def test_read_packet_csv_falls_back_to_windows_encoding(tmp_path):
    """A CSV Excel saved as Windows-1252 still loads, accents intact."""
    path = tmp_path / "returned.csv"
    path.write_bytes("item_id,advisor_notes\nC1,clause déjà vu\n".encode("cp1252"))
    assert read_packet_csv(path)["advisor_notes"][0] == "clause déjà vu"


# ----- Full evaluation -----
def _returned(key: PacketKey, buckets: list[str], ranks: list[str]) -> ReturnedPacket:
    """Join a packet with these answers."""
    return join_packet(_packet(key, buckets, ranks), key, "advisor")


def test_evaluate_agreement_on_a_perfect_advisor():
    """An advisor who matches the model exactly gets rho = 1, exact rate 1 and kappa 1, from the stored scores."""
    key = _key()
    report = evaluate_agreement(_returned(key, [H, M, L, L], ["1", "2", "3", "4"]), key, n_boot=100)
    assert report.spearman_rank.value == pytest.approx(1.0)
    assert report.exact_rate.value == 1.0 and report.weighted_kappa_linear.value == pytest.approx(1.0)
    assert report.confusion_matrix == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert (report.rescored, report.scores_reproduced, report.weight_version_id) == (False, None, "agreement-test-v1")
    assert report.model_bucket_counts == {L: 2, M: 1, H: 1} and report.n_with_rank == 4


def test_evaluate_agreement_on_an_inverted_advisor():
    """Ranks in the opposite order give rho = -1, and only the Medium clause's bucket agrees."""
    key = _key()
    report = evaluate_agreement(_returned(key, [L, M, H, H], ["4", "3", "2", "1"]), key, n_boot=100)
    assert report.spearman_rank.value == pytest.approx(-1.0)
    assert report.exact_rate.value == 0.25 and report.high_precision.value == 0.0
    assert any("opposite to the advisor" in line for line in report.caveats)


def test_evaluate_agreement_rescores_under_another_weight_set():
    """A higher High threshold moves the 0.55 clause to Medium without any model; the same set reproduces scores."""
    key = _key()
    returned = _returned(key, [H, M, L, L], ["", "", "", ""])
    stricter = WEIGHTS.model_copy(update={"weight_version_id": "agreement-test-v2", "high_threshold": 0.6})
    report = evaluate_agreement(returned, key, stricter, n_boot=50)
    assert report.rescored and report.scores_reproduced is None and report.high_threshold == 0.6
    assert [i.rating for i in report.items] == [M, M, L, L] and report.exact_rate.value == 0.75
    assert report.spearman_rank is None and report.spearman_bucket is not None
    assert evaluate_agreement(returned, key, WEIGHTS, n_boot=50).scores_reproduced is True


def test_evaluate_agreement_rejects_a_packet_from_another_key():
    """A returned packet joined with one key can't be evaluated against another."""
    key = _key()
    returned = _returned(key, [H, M, L, L], ["1", "2", "3", "4"]).model_copy(update={"packet_id": "pkt-other"})
    with pytest.raises(ValueError, match="doesn't belong"):
        evaluate_agreement(returned, key, n_boot=10)
