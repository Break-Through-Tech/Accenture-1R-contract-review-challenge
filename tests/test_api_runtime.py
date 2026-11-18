"""Tests for src/api/runtime.py and src/api/__main__.py: model version keys, options and the localhost default."""
import pytest

from src.api.__main__ import DEFAULT_HOST, is_loopback, parse_args
from src.api.runtime import DEFAULT_DATABASE, file_model_version, load_pipeline
from src.risk.schema import DEFAULT_WEIGHTS_PATH


# ----- Model version keys -----
def test_file_model_version_uses_the_file_hash(tmp_path):
    """Hand-checked: SHA-256 of b'abc' starts ba7816bf8f01."""
    path = tmp_path / "bundle.joblib"
    path.write_bytes(b"abc")
    assert file_model_version("word_char_tfidf", path) == "word_char_tfidf@sha256:ba7816bf8f01"


@pytest.mark.parametrize("version", ["latest", "1.0", "", "two"])
def test_load_pipeline_refuses_registry_version_that_is_not_a_number(version):
    """Only a numbered registry version names one exact model; nothing is loaded otherwise."""
    with pytest.raises(ValueError, match="version number"):
        load_pipeline(registry_version=version)


# ----- Server options -----
def test_parse_args_defaults_bind_localhost():
    """By default the server binds 127.0.0.1:8000 with the Gradio page and the default weights and database."""
    args = parse_args([])
    assert (args.host, args.port, args.no_ui) == (DEFAULT_HOST, 8000, False)
    assert DEFAULT_HOST == "127.0.0.1"
    assert (args.database, args.weights, args.bundle, args.model_version) == \
        (DEFAULT_DATABASE, DEFAULT_WEIGHTS_PATH, None, None)
    assert DEFAULT_DATABASE.parts[-3:] == ("outputs", "api", "clausetriage.sqlite")


@pytest.mark.parametrize(("host", "expected"), [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.0.0.2", True),
    ("0.0.0.0", False), ("192.168.1.10", False), ("::", False), ("example.com", False),
])
def test_is_loopback(host, expected):
    """Only loopback addresses and 'localhost' count as local; anything else triggers the warning."""
    assert is_loopback(host) is expected
