"""Tests for clausetriage/legacy.py: old `src.` module paths resolve inside the context only, and old pickles load."""

import importlib
import sys
import threading
from types import ModuleType

import joblib
import pytest

import clausetriage.registry.bundle
import clausetriage.registry.predictors
from clausetriage.labels.categories import CATEGORIES
from clausetriage.legacy import current_name, legacy_src_imports, load_joblib
from clausetriage.registry.bundle import ClauseModelBundle
from clausetriage.registry.demo_model import BUNDLE_PATH, BUNDLE_SHA256, file_sha256
from clausetriage.registry.predictors import VectorizedClassifier


def _legacy_entries() -> list[str]:
    """Names in sys.modules that belong to the old `src` package."""
    return [name for name in sys.modules if name == "src" or name.startswith("src.")]


def _write_legacy_pickle(path, obj) -> None:
    """Pickle obj with protocol 2 and rewrite its module paths to the pre-rename `src.` names."""
    joblib.dump(obj, path, protocol=2)
    data = path.read_bytes()
    # Protocol 2 writes module paths as newline-terminated text, so a plain replacement keeps the pickle valid
    assert b"clausetriage.registry." in data
    path.write_bytes(data.replace(b"clausetriage.registry.", b"src.registry."))


def _small_bundle() -> ClauseModelBundle:
    """A bundle with a placeholder predictor and a threshold of 0.5 for every category."""
    return ClauseModelBundle(predictor=VectorizedClassifier("vectorizer", "classifier"), categories=list(CATEGORIES),
                             thresholds={category: 0.5 for category in CATEGORIES}, model_name="legacy-test",
                             config={"c": 1.0}, seed=42)


# ----- Name mapping -----
@pytest.mark.parametrize(("legacy_name", "expected"), [
    ("src", "clausetriage"),
    ("src.registry", "clausetriage.registry"),
    ("src.registry.bundle", "clausetriage.registry.bundle"),
    ("srcs", None),
    ("srcx.registry", None),
    ("other.src", None),
    ("clausetriage.registry", None),
])
def test_current_name_maps_only_the_old_package(legacy_name, expected):
    """Only `src` and its submodules map to a clausetriage name."""
    assert current_name(legacy_name) == expected


# ----- Import aliasing -----
def test_legacy_import_fails_outside_the_context():
    """Without the context a stale `src.` import is still an error."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.registry.bundle")


def test_legacy_import_returns_the_current_module_object():
    """Inside the context `src.<module>` is the very clausetriage module, with its own spec kept."""
    with legacy_src_imports():
        module = importlib.import_module("src.registry.bundle")
        package = importlib.import_module("src")
    assert module is clausetriage.registry.bundle
    assert module.ClauseModelBundle is ClauseModelBundle
    assert package is sys.modules["clausetriage"]
    assert module.__spec__.name == "clausetriage.registry.bundle"
    assert package.__spec__.name == "clausetriage"


def test_legacy_import_context_removes_its_aliases_on_exit():
    """After the block no `src` entry is left in sys.modules and the import fails again."""
    with legacy_src_imports():
        importlib.import_module("src.registry.predictors")
        assert "src.registry.predictors" in sys.modules
    assert _legacy_entries() == []
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.registry.predictors")


def test_legacy_import_unknown_module_is_not_found():
    """An old name with no new module is reported missing, not aliased to something else."""
    with legacy_src_imports(), pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.no_such_module")


def test_legacy_import_nested_contexts_keep_the_hook_until_the_outer_exits():
    """An inner block exiting doesn't remove the hook the outer block still needs."""
    with legacy_src_imports():
        with legacy_src_imports():
            pass
        assert importlib.import_module("src.registry.bundle") is clausetriage.registry.bundle
    assert _legacy_entries() == []


def test_legacy_import_concurrent_contexts_share_one_hook():
    """Blocks entered from several threads all resolve the old names and leave nothing behind."""
    results, barrier = [], threading.Barrier(4)

    def worker():
        """Enter the context, wait for the other threads, then import an old name."""
        with legacy_src_imports():
            barrier.wait()
            results.append(importlib.import_module("src.registry.predictors"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [clausetriage.registry.predictors] * 4
    assert _legacy_entries() == []


def test_legacy_import_context_leaves_an_unrelated_src_module_alone(monkeypatch):
    """A `src` module from some other project, already imported, survives the cleanup."""
    other = ModuleType("src")
    monkeypatch.setitem(sys.modules, "src", other)
    with legacy_src_imports():
        pass
    assert sys.modules["src"] is other


# ----- Loading pickles -----
def test_load_joblib_reads_a_pickle_with_old_module_paths(tmp_path):
    """A bundle pickled under `src.registry` loads as today's classes; plain joblib.load can't read it."""
    path = tmp_path / "legacy.joblib"
    _write_legacy_pickle(path, _small_bundle())
    with pytest.raises(ModuleNotFoundError):
        joblib.load(path)
    bundle = load_joblib(path)
    assert type(bundle) is ClauseModelBundle
    assert type(bundle.predictor) is VectorizedClassifier
    assert (bundle.model_name, bundle.seed, bundle.thresholds["Anti-Assignment"]) == ("legacy-test", 42, 0.5)
    assert _legacy_entries() == []


def test_load_joblib_reads_a_pickle_with_current_module_paths(tmp_path):
    """A bundle saved after the rename loads the same way."""
    path = tmp_path / "current.joblib"
    joblib.dump(_small_bundle(), path)
    assert load_joblib(path).model_name == "legacy-test"


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="the released model file is not in models/ (git-ignored)")
def test_load_joblib_reads_the_released_model_unchanged():
    """The released file, pickled before the rename, loads as a ClauseModelBundle and keeps its pinned SHA-256."""
    assert file_sha256(BUNDLE_PATH) == BUNDLE_SHA256
    bundle = load_joblib(BUNDLE_PATH)
    assert type(bundle) is ClauseModelBundle
    assert type(bundle.predictor) is VectorizedClassifier
    assert (bundle.model_name, bundle.categories) == ("word_char_tfidf", CATEGORIES)
    assert bundle.predict_proba(["The Agreement may not be assigned without consent."]).shape == (1, 41)
    assert file_sha256(BUNDLE_PATH) == BUNDLE_SHA256
