import warnings

import joblib
import pytest
from sklearn.exceptions import InconsistentVersionWarning

from clausetriage.registry.demo_model import fetch_bundle_file, file_sha256, load_demo_bundle, load_or_build_bundle
from clausetriage.registry.predictors import VectorizedClassifier


def _warn_like_old_sklearn() -> str:
    """Unpickled stand-in that warns the way scikit-learn does for a file from another version."""
    warnings.warn(InconsistentVersionWarning(estimator_name="Model", current_sklearn_version="2",
                                             original_sklearn_version="1"), stacklevel=2)
    return "stale model"


class StaleModel:
    """Pickles to a call of _warn_like_old_sklearn, so loading it raises the version warning."""
    def __reduce__(self):
        """Rebuild through the warning function."""
        return _warn_like_old_sklearn, ()


def _builder(calls: list):
    """A stand-in for build_demo_bundle that records the path it was asked to build."""
    def build(path):
        """Record the path and return a marker instead of training."""
        calls.append(path)
        return "built model"
    return build


# ----- Downloading the released model -----
def test_fetch_bundle_file_downloads_and_checks_hash(tmp_path):
    """A missing file is downloaded and kept when its SHA-256 matches."""
    source = tmp_path / "release.joblib"
    source.write_bytes(b"model bytes")
    target = tmp_path / "models" / "bundle.joblib"

    assert fetch_bundle_file(target, source.as_uri(), file_sha256(source)) == target
    assert target.read_bytes() == b"model bytes"
    assert list(target.parent.iterdir()) == [target]  # no partial download left behind


def test_fetch_bundle_file_refuses_wrong_hash(tmp_path):
    """A download with a different hash raises and leaves nothing on disk."""
    source = tmp_path / "release.joblib"
    source.write_bytes(b"tampered")
    target = tmp_path / "models" / "bundle.joblib"

    with pytest.raises(ValueError, match="SHA-256"):
        fetch_bundle_file(target, source.as_uri(), "0" * 64)
    assert list(target.parent.iterdir()) == []


def test_fetch_bundle_file_keeps_existing_file_without_downloading(tmp_path):
    """An existing file (e.g. one you built) is used as is; the URL is never touched."""
    target = tmp_path / "bundle.joblib"
    target.write_bytes(b"local build")
    assert fetch_bundle_file(target, "https://invalid.example/never", "0" * 64) == target


def test_fetch_bundle_file_explains_failed_download(tmp_path):
    """An unreachable URL raises FileNotFoundError pointing to the build script, leaving nothing behind."""
    target = tmp_path / "models" / "bundle.joblib"
    with pytest.raises(FileNotFoundError, match="build_model_bundle"):
        fetch_bundle_file(target, (tmp_path / "missing.joblib").as_uri(), "0" * 64)
    assert list(target.parent.iterdir()) == []


# ----- Falling back to a local build -----
def test_load_or_build_bundle_loads_a_good_file_without_building(tmp_path):
    """A file that loads cleanly is returned and nothing is built."""
    target, calls = tmp_path / "bundle.joblib", []
    joblib.dump({"model": 1}, target)
    assert load_or_build_bundle(target, "https://invalid.example/never", "0" * 64, _builder(calls)) == {"model": 1}
    assert calls == []


def test_load_or_build_bundle_loads_a_pre_rename_file_without_building(tmp_path):
    """A file pickled when the package was still `src` (like the released model) loads instead of being rebuilt."""
    target, calls = tmp_path / "bundle.joblib", []
    joblib.dump(VectorizedClassifier("vectorizer", "classifier"), target, protocol=2)
    # Protocol 2 stores module paths as text, so renaming them in place keeps the pickle valid
    target.write_bytes(target.read_bytes().replace(b"clausetriage.registry.", b"src.registry."))
    loaded = load_or_build_bundle(target, "https://invalid.example/never", "0" * 64, _builder(calls))
    assert type(loaded) is VectorizedClassifier
    assert (loaded.vectorizer, loaded.classifier, calls) == ("vectorizer", "classifier", [])
    assert type(load_demo_bundle(target)) is VectorizedClassifier


def test_load_or_build_bundle_builds_when_download_fails(tmp_path):
    """An unreachable release triggers a local build into the same path."""
    target, calls = tmp_path / "models" / "bundle.joblib", []
    url = (tmp_path / "missing.joblib").as_uri()
    assert load_or_build_bundle(target, url, "0" * 64, _builder(calls)) == "built model"
    assert calls == [target]


def test_load_or_build_bundle_rebuilds_a_file_from_other_sklearn_versions(tmp_path):
    """A file that loads with scikit-learn's version warning is rebuilt instead of being used."""
    target, calls = tmp_path / "bundle.joblib", []
    joblib.dump(StaleModel(), target)
    assert load_or_build_bundle(target, "https://invalid.example/never", "0" * 64, _builder(calls)) == "built model"
    assert calls == [target]
