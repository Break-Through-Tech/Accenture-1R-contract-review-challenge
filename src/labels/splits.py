from sklearn.model_selection import train_test_split
from src.labels.loader import Contract


def split_by_contract(contracts: list[Contract], val_fraction: float = 0.2,
                      seed: int = 42) -> tuple[list[Contract], list[Contract]]:
    """
    Split whole contracts into (train, val) with a seeded shuffle, so no contract
    leaks across the split. Raises ValueError unless 0 < val_fraction < 1.
    """
    # Reject fractions that would leave train or val empty
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    # Shuffle and split whole contracts (copies, so the caller's list is untouched)
    train, val = train_test_split(contracts, test_size=val_fraction, random_state=seed)
    return train, val
