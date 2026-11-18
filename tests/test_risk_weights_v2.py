"""
Checks on the shipped v2-team-proposed weight set: it validates, covers all 41 categories with a
tier and a reason, and scores a handful of hand-computed clauses as documented in
docs/risk-weights-v2.md.

v2 arithmetic used by the golden values: composite = 0.6 * severity + 0.1 * one_sidedness
+ 0.1 * magnitude + 0.2 * contract_context; Medium at 0.25, High at 0.525.
"""
import pytest

from clausetriage.evidence.cues import normalize_category
from clausetriage.evidence.schema import Evidence
from clausetriage.labels.categories import CATEGORIES
from clausetriage.risk.schema import DEFAULT_WEIGHTS_PATH, SIGNAL_NAMES, V1_WEIGHTS_PATH, ClauseInput, load_weight_set
from clausetriage.risk.scoring import score_clause
from clausetriage.risk.signals import magnitude

V2 = load_weight_set(DEFAULT_WEIGHTS_PATH)
V1 = load_weight_set(V1_WEIGHTS_PATH)
TIER_VALUES = {"None": 0.0, "Low": 0.25, "Medium": 0.5, "High": 0.75, "Critical": 1.0}


def _clause(category: str, quote: str | None) -> ClauseInput:
    """Build a detected clause in contract c1, with an Evidence for quote when given."""
    ev = Evidence(quote=quote, start=0, end=len(quote)) if quote else None
    return ClauseInput(contract_id="c1", chunk_index=0, category=category, probability=0.9, evidence=ev)


# ----- The weight set itself -----
def test_v2_is_the_default_and_keeps_the_four_signals():
    """v2 is what DEFAULT_WEIGHTS_PATH loads, and it weights exactly the four existing signals."""
    assert V2.weight_version_id == "v2-team-proposed"
    assert set(V2.signal_weights) == set(SIGNAL_NAMES)
    assert dict(V2.signal_weights) == pytest.approx(
        {"category_severity": 0.6, "one_sidedness": 0.1, "magnitude": 0.1, "contract_context": 0.2}
    )


def test_v2_names_every_category_with_a_severity_and_a_reason():
    """All 41 categories have an explicit severity and a reason, so default_severity is never used."""
    names = {normalize_category(c) for c in CATEGORIES}
    assert {normalize_category(c) for c in V2.category_severity} == names
    assert {normalize_category(c) for c in V2.severity_rationale} == names
    assert len(V2.category_severity) == len(CATEGORIES) == 41


@pytest.mark.parametrize("category", sorted(V2.category_severity))
def test_v2_severity_is_a_tier_named_by_its_reason(category):
    """Every severity is one of the five tiers, and its reason starts with that tier's name."""
    severity = V2.category_severity[category]
    tier, _, reason = V2.severity_rationale[category].partition(": ")
    assert TIER_VALUES[tier] == severity
    assert len(reason) > 10


def test_v2_every_rule_has_a_reason():
    """Each cooccurrence and magnitude rule carries a plain-language reason for the reviewer."""
    assert all(rule.rationale for rule in (*V2.cooccurrence_rules, *V2.magnitude_rules))


def test_v2_and_v1_are_different_versions():
    """v2 is a new version (new id, new content hash); v1 keeps its own."""
    assert V1.weight_version_id == "v1-placeholder"
    assert V2.content_sha256 != V1.content_sha256


# ----- Hand-scored clauses -----
@pytest.mark.parametrize(
    "category, quote, contract, composite, rating",
    [
        # Critical, alone, no Insurance: 0.6*1.0 + 0.2*0.5 = 0.70
        ("Uncapped Liability", "Licensee's liability for breach of Section 9 shall not be limited.",
         {"Uncapped Liability"}, 0.70, "High"),
        # Critical carve-out next to a cap, Insurance present: 0.6*(1.0*0.8) = 0.48
        ("Uncapped Liability", "Licensee's liability for breach of Section 9 shall not be limited.",
         {"Uncapped Liability", "Cap On Liability", "Insurance"}, 0.48, "Medium"),
        # Same carve-out with no Insurance: 0.48 + 0.2*0.5 = 0.58
        ("Uncapped Liability", "Licensee's liability for breach of Section 9 shall not be limited.",
         {"Uncapped Liability", "Cap On Liability"}, 0.58, "High"),
        # High tier, two years: 0.6*0.75 + 0.1*(730-180)/(1095-180) = 0.45 + 0.0601093 = 0.5101093
        ("Non-Compete", "For two (2) years after termination, Distributor shall not sell competing products.",
         {"Non-Compete"}, 0.5101093, "Medium"),
        # High tier, three years hits the magnitude ceiling: 0.45 + 0.1 = 0.55
        ("Non-Compete", "For three (3) years after termination, Distributor shall not sell competing products.",
         {"Non-Compete"}, 0.55, "High"),
        # Same three years with a carve-out: 0.6*(0.75*0.8) + 0.1 = 0.36 + 0.1 = 0.46
        ("Non-Compete", "For three (3) years after termination, Distributor shall not sell competing products.",
         {"Non-Compete", "Competitive Restriction Exception"}, 0.46, "Medium"),
        # Medium tier: 2 aggravating vs 2 mitigating -> 0.5; 30 days' notice is the shortest -> 1.0
        # 0.6*0.5 + 0.1*0.5 + 0.1*1.0 = 0.30 + 0.05 + 0.10 = 0.45
        ("Termination For Convenience",
         "Either party may terminate this Agreement at any time without cause upon thirty (30) days' "
         "prior written notice.", {"Termination For Convenience"}, 0.45, "Medium"),
        # Low tier with one unmitigated aggravating phrase: 0.6*0.25 + 0.1*1.0 = 0.25
        ("Anti-Assignment",
         "Distributor may not assign this Agreement without consent, which Supplier may withhold in its sole discretion.",
         {"Anti-Assignment"}, 0.25, "Medium"),
        # Same clause with the usual reasonableness wording, no aggravating phrase: 0.15
        ("Anti-Assignment",
         "Distributor may not assign this Agreement without consent, not to be unreasonably withheld.",
         {"Anti-Assignment"}, 0.15, "Low"),
        # Low tier, 30-day warranty is under the 90-day floor: 0.15 + 0.1*1.0 = 0.25
        ("Warranty Duration", "Seller warrants the Products for thirty (30) days after delivery.",
         {"Warranty Duration"}, 0.25, "Medium"),
        # Low tier, twelve months = 360 days: 0.15 + 0.1*(1 - 270/275) = 0.1518182
        ("Warranty Duration", "Seller warrants the Products for twelve (12) months after delivery.",
         {"Warranty Duration"}, 0.1518182, "Low"),
        # High tier with Exclusivity and no Termination For Convenience: 0.45 + 0.2*(0.5+0.3) = 0.61
        ("Minimum Commitment", "Buyer shall purchase a minimum of 10,000 units per calendar quarter.",
         {"Minimum Commitment", "Exclusivity"}, 0.61, "High"),
        # Same minimum with an exit for convenience and no exclusivity: 0.45
        ("Minimum Commitment", "Buyer shall purchase a minimum of 10,000 units per calendar quarter.",
         {"Minimum Commitment", "Termination For Convenience"}, 0.45, "Medium"),
        # High tier with a cap present: 0.6*(0.75*0.8) = 0.36; without one: 0.45 + 0.2*0.5 = 0.55
        ("Liquidated Damages", "Distributor shall pay a termination fee of $50,000.",
         {"Liquidated Damages", "Cap On Liability"}, 0.36, "Medium"),
        ("Liquidated Damages", "Distributor shall pay a termination fee of $50,000.",
         {"Liquidated Damages"}, 0.55, "High"),
        # None tier: 0
        ("Parties", "Acme Corp and Beta LLC.", {"Parties"}, 0.0, "Low"),
    ],
)
def test_v2_hand_scored_clause_golden_values(category, quote, contract, composite, rating):
    """Hand-computed v2 composites and ratings for representative clauses (arithmetic in each comment)."""
    score = score_clause(_clause(category, quote), contract, V2)
    assert score.composite_score == pytest.approx(composite, abs=1e-6)
    assert score.rating == rating
    assert score.weight_version_id == "v2-team-proposed"


def test_v2_unknown_queue_clause_scores_on_severity_and_context():
    """An Uncapped Liability clause with no evidence keeps severity and context: 0.6 + 0.2*0.5 = 0.70."""
    score = score_clause(_clause("Uncapped Liability", None), {"Uncapped Liability"}, V2)
    assert (score.signals["one_sidedness"], score.signals["magnitude"]) == (0.0, 0.0)
    assert score.composite_score == pytest.approx(0.70)
    assert (score.rating, score.has_evidence) == ("High", False)


# ----- What changed from v1 -----
def test_v2_drops_the_notice_period_rule_that_scored_the_renewal_term():
    """v1 read the one-year renewal in a notice clause as a 365-day notice period (magnitude 1); v2 has no rule."""
    clause = _clause(
        "Notice Period To Terminate Renewal",
        "This Agreement renews for successive one (1) year terms unless either party gives ninety (90) days' notice.",
    )
    assert magnitude(clause, set(), V1) == 1.0
    assert magnitude(clause, set(), V2) == 0.0


@pytest.mark.parametrize("phrase", ["worldwide", "throughout the world"])
def test_v2_territory_alone_is_not_broad_scope(phrase):
    """A plain worldwide licence no longer counts as broad scope (it was in ~30% of License Grant answers)."""
    clause = _clause("License Grant", f"Licensor grants Licensee a non-exclusive license {phrase}.")
    assert magnitude(clause, set(), V1) == 1.0
    assert magnitude(clause, set(), V2) == 0.0
