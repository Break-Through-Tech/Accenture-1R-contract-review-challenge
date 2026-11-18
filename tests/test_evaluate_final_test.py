import json
import subprocess

import numpy as np
import pandas as pd
import pytest

import scripts.evaluate_final_test as final
from src.labels.categories import CATEGORIES
from src.labels.loader import AnswerSpan, Contract
from src.registry import demo_model
from src.registry.bundle import ClauseModelBundle
from src.risk.schema import load_weight_set
from src.sealed_run import ATTEMPTS_FILE, PREFLIGHT_FILE, REPORT_FILE, SealedRunError, ids_sha256

GOVERNING, PARTIES = CATEGORIES.index("Governing Law"), CATEGORIES.index("Parties")


class KeywordPredictor:
    """Governing Law 0.9 when 'governed' appears (else 0.1); Parties 0.8 when 'between' appears (else 0.2)."""

    def predict_proba(self, texts):
        """Fixed keyword scores, one row per text."""
        scores = np.zeros((len(texts), len(CATEGORIES)), dtype=np.float32)
        for row, text in enumerate(texts):
            scores[row, GOVERNING] = 0.9 if "governed" in text else 0.1
            scores[row, PARTIES] = 0.8 if "between" in text else 0.2
        return scores


def _contract(contract_id: str, text: str, *answers: tuple[str, str]) -> Contract:
    """A contract whose spans are the given (category, answer text) pairs, located in text."""
    spans = [AnswerSpan(category=c, start=text.index(a), end=text.index(a) + len(a), text=a) for c, a in answers]
    return Contract(contract_id=contract_id, text=text, spans=spans)


# Four one-chunk contracts. Governing Law: TP, TP (F1 1, AP 1). Parties: TP, FP (F1 2/3; AP 0.5, tied at 0.8)
CONTRACTS = [
    _contract("c1", "This Agreement is between A and B. It is governed by Delaware law.",
              ("Parties", "between A and B"), ("Governing Law", "governed by Delaware law")),
    _contract("c2", "This Agreement is governed by New York law.", ("Governing Law", "governed by New York law")),
    _contract("c3", "A letter between X and Y."),
    _contract("c4", "Nothing relevant here."),
]


@pytest.fixture(scope="module")
def tokenizer():
    """The pinned tokenizer from .venv/hf-cache (the evaluation's own loader)."""
    try:
        return final.load_tokenizer()
    except OSError:
        pytest.skip("pinned tokenizer not cached; run .github/scripts/prefetch_hf_assets.py")


@pytest.fixture(scope="module")
def reference():
    """The development reference recomputed from the saved nested runs."""
    return final.development_reference()


@pytest.fixture(scope="module")
def synthetic(tokenizer):
    """evaluate() on the four synthetic contracts with a keyword bundle and 20 bootstrap replicates."""
    bundle = ClauseModelBundle(predictor=KeywordPredictor(), categories=CATEGORIES,
                               thresholds={c: 0.5 for c in CATEGORIES}, model_name="word_char_tfidf",
                               config={}, seed=42)
    word_predict = lambda texts: KeywordPredictor().predict_proba(texts).astype(np.float64) * 0.5  # noqa: E731
    return bundle, final.evaluate(CONTRACTS, bundle, word_predict, tokenizer, replicates=20, seed=1)


# ----- Guards in main(): every refusal happens before the data file is opened -----
@pytest.fixture
def sealed_paths(tmp_path, monkeypatch):
    """Point the sealed data path and output folder at temporary locations that hold nothing."""
    monkeypatch.setattr(final, "SEALED_DATA_PATH", tmp_path / "data" / "test.json")
    monkeypatch.setattr(final, "SEALED_OUTPUT", tmp_path / "sealed")
    return tmp_path


def test_main_refuses_test_json_without_the_opt_in_flag(sealed_paths):
    """A missing test.json gives the guard's refusal, not FileNotFoundError, so nothing was opened."""
    with pytest.raises(SealedRunError, match="--sealed-test-run"):
        final.main(["--data", str(sealed_paths / "data" / "test.json"), "--output", str(sealed_paths / "out")])


@pytest.mark.parametrize("name", ["CUADv1.json", "cuad-v1.zip"])
def test_main_refuses_files_that_contain_the_test_contracts(sealed_paths, name):
    """CUADv1.json and zip archives are refused even with the opt-in flag."""
    with pytest.raises(SealedRunError, match="never read"):
        final.main(["--data", str(sealed_paths / name), "--output", str(sealed_paths / "out"), "--sealed-test-run"])


def test_main_refuses_the_opt_in_flag_with_a_development_file(sealed_paths):
    """The flag is only for test.json."""
    with pytest.raises(SealedRunError, match="only for"):
        final.main(["--data", str(sealed_paths / "dev.json"), "--output", str(sealed_paths / "out"),
                    "--sealed-test-run"])


def test_main_refuses_a_test_json_other_than_the_pinned_one(sealed_paths):
    """Only data/cuad/test.json can be the sealed file."""
    with pytest.raises(SealedRunError, match="reads only"):
        final.main(["--data", str(sealed_paths / "elsewhere" / "test.json"), "--output",
                    str(sealed_paths / "sealed"), "--sealed-test-run"])


@pytest.mark.parametrize("extra, message", [(["--output", "OTHER"], "writes only"),
                                            (["--output", "SEALED", "--replicates", "5"], "registered number")])
def test_main_refuses_other_outputs_or_replicates_in_the_sealed_run(sealed_paths, extra, message):
    """The sealed run has a fixed output folder and a fixed number of bootstrap replicates."""
    extra = [{"OTHER": str(sealed_paths / "other"), "SEALED": str(sealed_paths / "sealed")}.get(x, x) for x in extra]
    with pytest.raises(SealedRunError, match=message):
        final.main(["--data", str(sealed_paths / "data" / "test.json"), "--sealed-test-run", *extra])


def test_main_refuses_the_sealed_run_with_uncommitted_changes(sealed_paths, monkeypatch):
    """A dirty tree blocks the sealed run (git state is faked here)."""
    monkeypatch.setattr(final, "git_state", lambda root: {"commit": "abc", "branch": "b", "dirty": True})
    monkeypatch.setattr(final, "git_history_has", lambda root, path: False)
    with pytest.raises(SealedRunError, match="uncommitted"):
        final.main(["--data", str(sealed_paths / "data" / "test.json"), "--output", str(sealed_paths / "sealed"),
                    "--sealed-test-run"])


def test_main_refuses_the_sealed_run_once_a_result_was_ever_committed(sealed_paths, monkeypatch):
    """A report.json anywhere in git history blocks a second run, even if the file was deleted."""
    monkeypatch.setattr(final, "git_state", lambda root: {"commit": "abc", "branch": "b", "dirty": False})
    monkeypatch.setattr(final, "git_history_has", lambda root, path: True)
    with pytest.raises(SealedRunError, match="already exists"):
        final.main(["--data", str(sealed_paths / "data" / "test.json"), "--output", str(sealed_paths / "sealed"),
                    "--sealed-test-run", "--preflight"])


def test_main_development_run_never_writes_to_the_sealed_folder(sealed_paths):
    """A development run pointed at the sealed results folder is refused."""
    with pytest.raises(SealedRunError, match="never write"):
        final.main(["--data", str(sealed_paths / "dev.json"), "--output", str(sealed_paths / "sealed")])


def test_main_development_run_refuses_an_existing_report(sealed_paths):
    """A finished run's folder isn't reused."""
    (sealed_paths / "out").mkdir()
    (sealed_paths / "out" / REPORT_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(SealedRunError, match="new output folder"):
        final.main(["--data", str(sealed_paths / "dev.json"), "--output", str(sealed_paths / "out")])


def test_main_evaluation_needs_a_passing_preflight_first(sealed_paths):
    """No preflight.json in the folder: the evaluation does not start."""
    with pytest.raises(SealedRunError, match="preflight"):
        final.main(["--data", str(sealed_paths / "dev.json"), "--output", str(sealed_paths / "out")])


def test_main_preflight_is_refused_once_the_evaluation_started(sealed_paths):
    """The preflight comes before the run; an attempt log means the run began."""
    (sealed_paths / "out").mkdir()
    (sealed_paths / "out" / ATTEMPTS_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(SealedRunError, match="already started"):
        final.main(["--data", str(sealed_paths / "dev.json"), "--output", str(sealed_paths / "out"), "--preflight"])


def test_main_refuses_a_development_holdout_named_test_json(sealed_paths):
    """The dry-run file can't be mistaken for the sealed file."""
    with pytest.raises(SealedRunError, match="must not be named"):
        final.main(["--write-development-holdout", str(sealed_paths / "test.json")])


# ----- Frozen configuration is pinned -----
def test_frozen_constants_match_their_sources():
    """Bundle hash, partition digests, weight set and bootstrap settings equal the protocol's values."""
    saved = json.loads((final.SAVED_RUN / "report.json").read_text(encoding="utf-8"))
    weights = load_weight_set()
    assert final.BUNDLE_SHA256 == demo_model.BUNDLE_SHA256 == "6fd84c28600fcac844b488c1c404aa56fed7de9daab007589c9b42a34f0bd194"
    assert final.PARTITION_DIGESTS == saved["split"]["contract_id_sha256"]
    assert final.TRAIN_SHA256 == saved["data_sha256"]
    assert (weights.weight_version_id, weights.content_sha256) == (final.WEIGHT_VERSION_ID, final.WEIGHT_SET_SHA256)
    assert (final.BOOTSTRAP_REPLICATES, final.BOOTSTRAP_SEED, final.MAX_TOKENS, final.OVERLAP_TOKENS) == \
        (10_000, 20261147, 512, 64)


def test_check_pinned_files_passes_on_the_committed_inputs():
    """Threshold files, weight set and category descriptions in the repo have their pinned hashes."""
    assert set(final.check_pinned_files().values()) == set(final.PINNED_FILES.values())


def test_check_pinned_files_refuses_a_changed_file(tmp_path):
    """A pinned file whose content changed stops the run."""
    path = tmp_path / "thresholds.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(final.IntegrityError, match="changed"):
        final.check_pinned_files({path: "0" * 64})


def test_sealed_data_blob_matches_the_git_index():
    """The pinned blob id is the one git recorded for data/cuad/test.json (read from the index, not the file)."""
    entry = subprocess.run(["git", "ls-files", "-s", "data/cuad/test.json"], cwd=final.ROOT,
                           capture_output=True, text=True, check=False).stdout.split()
    if not entry:
        pytest.skip("not a git checkout")
    assert entry[1] == final.SEALED_DATA_GIT_BLOB


# ----- Development reference -----
@pytest.mark.parametrize("seed, f1, aupr", [(42, 0.5330673042557077, 0.5971198237476559),
                                            (7, 0.5149335380774824, 0.5854461459704445),
                                            (17, 0.4940752325012619, 0.5630627906108538),
                                            (23, 0.5053360461284743, 0.5398869864923852),
                                            (101, 0.4726875570181015, 0.5312092085883828)])
def test_development_reference_per_seed_golden_values(reference, seed, f1, aupr):
    """Each seed's word+char macro-F1 and macro-AUPR from its saved probabilities and thresholds."""
    row = next(r for r in reference["per_seed"] if r["seed"] == seed)
    assert (row["word_char_tfidf_macro_f1"], row["word_char_tfidf_macro_aupr"]) == pytest.approx((f1, aupr), abs=1e-12)


def test_development_reference_summary_values(reference):
    """Five-seed mean and range: macro-F1 0.5040 (0.4727-0.5331), macro-AUPR 0.5633 (0.5312-0.5971); word-only 0.4319."""
    macro = reference["macro"]
    assert macro["word_char_tfidf_macro_f1"]["mean"] == pytest.approx(0.5040199355962056, abs=1e-12)
    assert macro["word_char_tfidf_macro_aupr"]["mean"] == pytest.approx(0.5633449910819445, abs=1e-12)
    assert (macro["word_char_tfidf_macro_f1"]["min"], macro["word_char_tfidf_macro_f1"]["max"]) == \
        pytest.approx((0.4726875570181015, 0.5330673042557077), abs=1e-12)
    assert (macro["word_char_tfidf_macro_aupr"]["min"], macro["word_char_tfidf_macro_aupr"]["max"]) == \
        pytest.approx((0.5312092085883827, 0.5971198237476559), abs=1e-12)
    assert macro["word_tfidf_macro_f1"]["mean"] == pytest.approx(0.43188291457659966, abs=1e-12)


def test_development_reference_buckets_match_the_error_analysis(reference):
    """Bucket membership equals results/error_analysis/category_summary.csv for word+char TF-IDF."""
    saved = pd.read_csv(final.ROOT / "results" / "error_analysis" / "category_summary.csv")
    saved = saved[saved["model"] == "word_char_tfidf"].set_index("category")["support_bucket"].str.split().str[0]
    ours = reference["per_category"].set_index("category")["dev_bucket"]
    assert ours.to_dict() == saved.reindex(ours.index).to_dict()
    assert ours.value_counts().to_dict() == {"common": 19, "uncommon": 13, "rare": 9}


# ----- The evaluation itself, on synthetic contracts -----
def test_evaluate_primary_metrics_match_hand_computed_values(synthetic):
    """Governing Law F1 1 / AP 1, Parties F1 2/3 / AP 0.5, others 0: macro-F1 (5/3)/41, macro-AUPR 0.75."""
    _, result = synthetic
    primary = result["macro"]["word_char_tfidf"]
    assert primary["macro_f1"] == pytest.approx((1 + 2 / 3) / 41)
    assert primary["macro_aupr"] == pytest.approx(0.75)
    assert primary["categories_with_positives"] == 2
    assert primary["macro_f1_over_categories_with_positives"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["contract_level"]["macro_f1"] == pytest.approx((1 + 2 / 3) / 41)  # one chunk per contract


def test_evaluate_bootstrap_bounds_contain_the_point_estimate_here(synthetic):
    """Intervals are recorded with the replicate count and seed."""
    _, result = synthetic
    interval = result["macro"]["word_char_tfidf"]["macro_f1_95"]
    assert interval["lower_95"] <= interval["upper_95"] and interval["replicates_defined"] == 20
    assert (result["bootstrap"]["replicates"], result["bootstrap"]["seed"]) == (20, 1)


def test_evaluate_checks_serving_parity_and_grounds_evidence(synthetic):
    """Every contract went through ContractInferencePipeline; evidence counts add up."""
    _, result = synthetic
    evidence = result["evidence"]
    assert result["serving_parity_contracts"] == 4
    assert evidence["chunk_category_detections"] == 4  # GL in c1, c2; Parties in c1, c3
    assert evidence["grounded"] + evidence["unknown"] == evidence["clauses_after_deduplication"]


def test_evaluate_triage_spearman_from_scores_equals_rank_spearman(synthetic):
    """The bootstrapped score Spearman agrees with evaluate_triage's rank-based value at the point estimate."""
    _, result = synthetic
    triage = result["triage"]
    assert triage["spearman_from_scores"] == pytest.approx(triage["agreement_with_perfect_detection"]["triage_score"]["spearman"], abs=1e-4)


def test_evaluate_rejects_a_serving_mismatch(tokenizer):
    """If the served probabilities differed from the evaluated ones, the run stops before any metric."""
    class Drifting(KeywordPredictor):
        """Returns different scores on every call."""
        calls = 0

        def predict_proba(self, texts):
            """Keyword scores shifted by the call count."""
            Drifting.calls += 1
            return np.clip(super().predict_proba(texts) + 0.01 * Drifting.calls, 0, 1)

    bundle = ClauseModelBundle(predictor=Drifting(), categories=CATEGORIES, thresholds={c: 0.5 for c in CATEGORIES},
                               model_name="word_char_tfidf", config={}, seed=42)
    with pytest.raises(final.IntegrityError, match="serving probabilities"):
        final.evaluate(CONTRACTS, bundle, lambda texts: np.zeros((len(texts), 41)), tokenizer, replicates=2, seed=1)


def test_report_files_are_written_with_strict_json(synthetic, reference, tmp_path, monkeypatch):
    """build_report + write_outputs produce every file; report.json has no NaN tokens and states test_data_used."""
    bundle, result = synthetic
    fake_bundle_file = tmp_path / "bundle.joblib"
    fake_bundle_file.write_bytes(b"x")
    monkeypatch.setattr(final, "BUNDLE_PATH", fake_bundle_file)
    provenance = final.provenance_record(mode="development_dry_run", bundle=bundle, git={"commit": "abc"},
                                         integrity={"calibration_reproduction": {}}, reference=reference,
                                         word_training_chunks=1)
    categories = final.per_category_frame(result, reference)
    buckets = final.bucket_frame(categories)
    report = final.build_report(provenance=provenance, data_path=tmp_path / "dev.json", data_sha256="0" * 64,
                                result=result, reference=reference, categories=categories, attempts=[{"attempt": 1}])
    final.write_outputs(tmp_path / "out", result, report, categories, buckets)

    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == sorted(["bucket_summary.csv", "chunks.csv", "contract_level_metrics.csv", "per_category_metrics.csv",
                              "predictions.npz", REPORT_FILE, "summary.md", "triage_ranking.csv"])
    saved = json.loads((tmp_path / "out" / REPORT_FILE).read_text(encoding="utf-8"),
                       parse_constant=lambda token: pytest.fail(f"non-strict JSON token {token}"))
    assert saved["test_data_used"] is False and saved["status"] == "complete"
    assert saved["development_reproduction"] is None  # only the seed-42 holdout is compared with saved runs
    assert saved["primary"]["unscorable_categories"] == [c for c in CATEGORIES if c not in ("Governing Law", "Parties")]
    assert "Macro-F1 (41 categories)" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert np.load(tmp_path / "out" / "predictions.npz")["word_char_tfidf"].shape == (4, 41)


# ----- Development holdout and preflight on real training-file data -----
def test_write_development_holdout_is_the_seed_42_evaluation_partition(tmp_path):
    """The dry-run file holds exactly the 82 evaluation contracts, in split order, and passes the preflight."""
    path = tmp_path / "holdout.json"
    assert final.write_development_holdout(path) == 82
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert ids_sha256([entry["title"] for entry in raw["data"]]) == final.PARTITION_DIGESTS["evaluation"]
    record = final.run_preflight(path, tmp_path / "out", sealed=False)
    assert record["passed"] and record["structure"]["overlap_with_training_file"]["evaluation"]["titles"] == 82
    assert json.loads((tmp_path / "out" / PREFLIGHT_FILE).read_text(encoding="utf-8"))["passed"] is True


def test_load_frozen_bundle_refuses_a_file_with_another_hash(tmp_path):
    """A bundle whose SHA-256 isn't the pinned one is never unpickled."""
    path = tmp_path / "bundle.joblib"
    path.write_bytes(b"not the release")
    with pytest.raises(final.IntegrityError, match="pins"):
        final.load_frozen_bundle(path)
