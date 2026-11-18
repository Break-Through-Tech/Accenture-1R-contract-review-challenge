"""Tests for scripts/export_feedback.py: the command-line export over a stored database."""
import json

import pytest

from scripts import export_feedback
from src.review.export import text_sha256
from src.review.schema import TrainingManifest
from tests import review_fixtures as fixtures
from tests.review_fixtures import verdict
from tests.test_artifact_store import TEXT_1, TEXT_2

# Bind the shared fixtures under their own names (see tests/test_store_rescoring.py)
make_pipeline = fixtures.make_pipeline
store = fixtures.store
review_store = fixtures.review_store


def test_main_writes_examples_and_report(review_store, tmp_path, capsys):
    """With a manifest file the CLI exports c-1's verdict, excludes held-out c-2, and saves the manifest it used."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    verdict(review_store, "c-2:0", "Insurance", "accept", "ana")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(TrainingManifest(
        source="fixture", split_method="by hand", training={"c-1": text_sha256(TEXT_1)},
        evaluation={"c-2": text_sha256(TEXT_2)}).model_dump_json(), encoding="utf-8")
    database = review_store.engine.url.database

    export_feedback.main(["--store", database, "--output", str(tmp_path / "out"), "--manifest", str(manifest_path),
                          "--save-manifest", str(tmp_path / "used.json")])
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert (report["example_count"], report["excluded_counts"]) == (1, {"evaluation_contract": 1})
    assert len((tmp_path / "out" / "examples.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert TrainingManifest.model_validate_json((tmp_path / "used.json").read_text(encoding="utf-8")).training == \
        {"c-1": text_sha256(TEXT_1)}
    assert "1 examples (1 positive)" in capsys.readouterr().out


def test_main_missing_store_exits_without_creating_it(tmp_path):
    """A store path that doesn't exist is refused rather than created empty."""
    with pytest.raises(SystemExit, match="no artifact store"):
        export_feedback.main(["--store", str(tmp_path / "nope.sqlite"), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "nope.sqlite").exists()
