import pytest
from src.labels.loader import Contract
from src.labels.splits import split_by_contract


def make_contracts(n: int) -> list[Contract]:
    """Build n minimal valid contracts with IDs c0..c{n-1} and no spans."""
    return [Contract(contract_id=f"c{i}", text="x", spans=[]) for i in range(n)]


def ids(contracts: list[Contract]) -> list[str]:
    """Return the contract IDs of a list of contracts, in order."""
    return [c.contract_id for c in contracts]


# ----- Split: reproducibility -----
def test_split_by_contract_same_seed_gives_identical_split():
    """Checks two calls with the same seed return the same train and val contracts."""
    contracts = make_contracts(20)

    first = split_by_contract(contracts, seed=7)
    second = split_by_contract(contracts, seed=7)

    assert ids(first[0]) == ids(second[0])
    assert ids(first[1]) == ids(second[1])


def test_split_by_contract_different_seed_gives_different_split():
    """Checks changing the seed changes which contracts land in val."""
    contracts = make_contracts(20)

    _, val_a = split_by_contract(contracts, seed=1)
    _, val_b = split_by_contract(contracts, seed=2)

    assert ids(val_a) != ids(val_b)


# ----- Split: no contract leaks across the split -----
def test_split_by_contract_train_and_val_are_disjoint_and_cover_input():
    """Checks every contract lands in exactly one of train or val."""
    contracts = make_contracts(30)

    train, val = split_by_contract(contracts)

    assert set(ids(train)).isdisjoint(ids(val))
    assert sorted(ids(train) + ids(val)) == sorted(ids(contracts))


# ----- Split: sizes -----
# 10 * 0.2 = 2.0 -> 2 val; 10 * 0.25 = 2.5 -> rounds up to 3 val; 10 * 0.5 = 5 val
@pytest.mark.parametrize("n, val_fraction, expected_val, expected_train", [
    (10, 0.2, 2, 8),
    (10, 0.25, 3, 7),
    (10, 0.5, 5, 5),
])
def test_split_by_contract_val_fraction_gives_hand_computed_sizes(
        n, val_fraction, expected_val, expected_train):
    """Checks val size is ceil(n * val_fraction) and train gets the rest."""
    train, val = split_by_contract(make_contracts(n), val_fraction=val_fraction)

    assert len(val) == expected_val
    assert len(train) == expected_train


# ----- Split: inputs are not modified -----
def test_split_by_contract_does_not_modify_input_list():
    """Checks the caller's list keeps its original order and length."""
    contracts = make_contracts(10)
    before = ids(contracts)

    split_by_contract(contracts)

    assert ids(contracts) == before


# ----- Split: invalid fractions are rejected -----
@pytest.mark.parametrize("val_fraction", [0, 1, -0.1, 1.5])
def test_split_by_contract_invalid_val_fraction_raises_value_error(val_fraction):
    """Checks a val_fraction outside the open interval (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="val_fraction must be between 0 and 1"):
        split_by_contract(make_contracts(10), val_fraction=val_fraction)
