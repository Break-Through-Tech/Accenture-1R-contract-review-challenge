import pytest
from pydantic import ValidationError

from src.evidence.schema import Evidence

SOURCE = "The laws of New York govern."


# ----- Construction-time checks -----
def test_evidence_valid_span_constructs():
    """A non-empty span whose quote length matches end - start builds fine."""
    ev = Evidence(quote="New York", start=12, end=20)
    assert (ev.quote, ev.start, ev.end) == ("New York", 12, 20)


@pytest.mark.parametrize(
    "quote, start, end",
    [
        ("x", 5, 5),        # empty span
        ("x", 6, 5),        # end before start
        ("x", -1, 0),       # negative start
        ("abc", 0, 2),      # quote longer than span
        ("a", 0, 2),        # quote shorter than span
    ],
)
def test_evidence_inconsistent_span_raises(quote, start, end):
    """Empty, reversed, negative or length-mismatched spans are rejected at construction."""
    with pytest.raises(ValidationError):
        Evidence(quote=quote, start=start, end=end)


# ----- Grounding against the source text -----
def test_validate_against_exact_slice_is_true():
    """A quote that sits exactly at SOURCE[12:20] validates."""
    assert Evidence(quote="New York", start=12, end=20).validate_against(SOURCE)


@pytest.mark.parametrize(
    "quote, start, end",
    [
        ("New Yorq", 12, 20),   # same span, different text
        ("New York", 13, 21),   # shifted right by one: real slice is "ew York "
        ("New York", 11, 19),   # shifted left by one: real slice is " New Yor"
        ("govern.!", 22, 30),   # runs past the end of SOURCE (len 28)
    ],
)
def test_validate_against_mismatch_is_false(quote, start, end):
    """Wrong text, shifted offsets, or spans past the end of the source fail validation."""
    assert not Evidence(quote=quote, start=start, end=end).validate_against(SOURCE)
