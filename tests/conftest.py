"""Shared pytest setup for the whole test suite."""
from pathlib import Path

from clausetriage.legacy import remove_stale_package_folder

ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session) -> None:
    """Clear a leftover bytecode-only `src/` folder from before the rename, which would make `import src` succeed."""
    remove_stale_package_folder(ROOT)
