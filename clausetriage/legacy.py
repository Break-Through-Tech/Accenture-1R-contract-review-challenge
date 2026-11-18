"""
Read files that name the package's old module path, `src.<module>`, from before the rename to
`clausetriage` (2026-11).

A pickle records the module path of every class it holds. The released model file,
models/word_char_tfidf.joblib, holds `src.registry.bundle.ClauseModelBundle` and
`src.registry.predictors.VectorizedClassifier`. It is the frozen artifact of the sealed test
evaluation and is pinned by SHA-256 (BUNDLE_SHA256 in clausetriage/registry/demo_model.py), so it is
never rebuilt or re-saved under the new name. The archived PR #48 source
(experiments/archive/pr48_token_chunking.py) is byte-pinned too and imports `src.chunking.chunk`.

How everything comes together:
legacy_src_imports(): a context manager; inside it, importing `src` or `src.<module>` gives the very
                      `clausetriage.<module>` object, so old pickles and sources resolve to today's
                      classes. Outside it `import src` fails as before, so a stale import in the code is
                      still an error
load_joblib(): joblib.load inside legacy_src_imports(); use it for every bundle or other project
               pickle, old or new
remove_stale_package_folder(): deletes a leftover `src/` folder that holds only bytecode, which a
                               checkout that ran the old code keeps after pulling the rename
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

LEGACY_PACKAGE, PACKAGE = "src", "clausetriage"


def current_name(legacy_name: str) -> str | None:
    """'src.registry.bundle' -> 'clausetriage.registry.bundle'; None for a name outside the old package."""
    if legacy_name == LEGACY_PACKAGE or legacy_name.startswith(LEGACY_PACKAGE + "."):
        return PACKAGE + legacy_name[len(LEGACY_PACKAGE):]
    return None


class _LegacyAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook that answers `src.<module>` with the already imported `clausetriage.<module>` object."""

    def __init__(self) -> None:
        self._real_specs: dict[str, Any] = {}

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        """A spec for an old name whose new module exists; None (not ours) otherwise."""
        name = current_name(fullname)
        if name is None or importlib.util.find_spec(name) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: Any) -> ModuleType:
        """Import the new module and hand back that same object, so classes keep their identity."""
        module = importlib.import_module(current_name(spec.name))
        # The import system overwrites __spec__ next; remember the real one to put it back
        self._real_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        """Nothing to run (the module is already imported); restore its own __spec__."""
        legacy_name = module.__spec__.name
        module.__spec__ = self._real_specs.pop(legacy_name)


_FINDER = _LegacyAliasFinder()
_LOCK = threading.Lock()
_depth = 0


def _remove_aliases() -> None:
    """Drop the `src` entries this hook put in sys.modules, leaving any unrelated `src` package alone."""
    for legacy_name in [name for name in sys.modules if current_name(name) is not None]:
        module = sys.modules[legacy_name]
        if sys.modules.get(current_name(legacy_name)) is module:
            del sys.modules[legacy_name]


@contextmanager
def legacy_src_imports() -> Iterator[None]:
    """
    Resolve `src.<module>` to `clausetriage.<module>` for the duration of the block. Nested and
    concurrent blocks share one hook, which is removed (with its aliases) when the last one exits.
    """
    global _depth
    with _LOCK:
        if _depth == 0:
            sys.meta_path.insert(0, _FINDER)
        _depth += 1
    try:
        yield
    finally:
        with _LOCK:
            _depth -= 1
            if _depth == 0:
                sys.meta_path.remove(_FINDER)
                _remove_aliases()


def remove_stale_package_folder(root: str | Path) -> bool:
    """
    Delete `<root>/src` if it holds nothing but `__pycache__` bytecode, and return whether it was
    deleted. Git removes the tracked files of the old package on pull but leaves its ignored bytecode
    folders, and those make `import src` succeed as an empty namespace package. A folder with any
    other file in it is someone's work and is left alone.
    """
    stale = Path(root) / LEGACY_PACKAGE
    if not stale.is_dir():
        return False
    files = [path for path in stale.rglob("*") if path.is_file()]
    if any(path.parent.name != "__pycache__" for path in files):
        return False
    shutil.rmtree(stale)
    return True


def load_joblib(path: str | Path) -> Any:
    """joblib.load(path), resolving the pre-rename `src.` module path. Loading a pickle can run code: trusted files only."""
    import joblib

    with legacy_src_imports():
        return joblib.load(path)
