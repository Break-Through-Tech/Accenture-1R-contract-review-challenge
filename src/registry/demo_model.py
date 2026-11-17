"""
Where the demo's clause model lives and how to get it, so every teammate loads the same file.

How everything comes together:
fetch_bundle_file(): returns models/word_char_tfidf.joblib, downloading it from the GitHub Release
                     first if it's missing; a download whose SHA-256 doesn't match is refused
build_demo_bundle(): trains the same model from the CUAD training file in the repo (a few minutes)
load_or_build_bundle(): loads that file, building it locally instead if it can't be downloaded or
                        doesn't load with the installed library versions
load_demo_bundle(): what the UI calls: a given file, a local MLflow registry version, or the above

The released file was built by build_demo_bundle() with the scikit-learn/NumPy/SciPy versions pinned
in requirements.txt. A bundle is a joblib (pickle) file and loading one can run code, so a
downloaded file is only kept when its hash matches BUNDLE_SHA256. A file you built yourself is
trusted as is. To publish a new model, upload it to a new release, then update RELEASE_TAG and
BUNDLE_SHA256 together.
"""
import hashlib
import json
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Callable

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from src.registry.bundle import ClauseModelBundle

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "models" / "word_char_tfidf.joblib"
RELEASE_TAG = "model-word-char-tfidf-v1"
RELEASE_URL = ("https://github.com/Break-Through-Tech/Accenture-1R-contract-review-challenge/releases/download/"
               f"{RELEASE_TAG}/word_char_tfidf.joblib")
BUNDLE_SHA256 = "6fd84c28600fcac844b488c1c404aa56fed7de9daab007589c9b42a34f0bd194"

# Local MLflow registry used by scripts/build_model_bundle.py --register (one per machine, not shared)
TRACKING_URI = ROOT / "outputs" / "mlflow" / "tracking.db"
EXPERIMENT_NAME, REGISTRY_NAME = "clause-review-tfidf", "clause-classifier-tfidf"
# Where the README setup step saves the pinned tokenizer (downloaded there if missing)
TOKENIZER_CACHE = ROOT / ".venv" / "hf-cache"

# How the released model was trained; must match scripts/evaluate_word_char_tfidf.py, or the saved
# thresholds won't fit the model. Reads only train_separate_questions.json; test.json is never touched.
TRAIN_PATH = ROOT / "data" / "cuad" / "train_separate_questions.json"
THRESHOLDS_PATH = ROOT / "results" / "word_char_tfidf_nested" / "word_char_tfidf_thresholds.json"
SEED, EVALUATION_FRACTION, CALIBRATION_FRACTION, MAX_TOKENS, OVERLAP_TOKENS, C = 42, 0.2, 0.2, 512, 64, 1.0


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_bundle_file(path: Path = BUNDLE_PATH, url: str = RELEASE_URL, sha256: str = BUNDLE_SHA256) -> Path:
    """Return `path`, downloading it from `url` first if it doesn't exist; raises ValueError on a hash mismatch."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Download next to the target, then move into place only once the hash checks out
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
        partial = Path(handle.name)
    try:
        try:
            urllib.request.urlretrieve(url, partial)
        except OSError as error:  # URLError/HTTPError (offline, release not published) are OSErrors
            raise FileNotFoundError(f"Couldn't download the model from {url} ({error}). "
                                    "Build it locally with: python -m scripts.build_model_bundle") from error
        actual = file_sha256(partial)
        if actual != sha256:
            raise ValueError(f"Downloaded model has SHA-256 {actual}, expected {sha256}; refusing to load it")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def build_demo_bundle(output: Path = BUNDLE_PATH, thresholds: Path = THRESHOLDS_PATH) -> ClauseModelBundle:
    """
    Train the word + character TF-IDF model on the nested split's training contracts (seed 42; 20%
    evaluation held out, 20% of the rest for calibration), bundle it with the calibrated thresholds,
    and save it to `output`.
    """
    # Training code is only needed when building, so it's imported here
    from transformers import AutoTokenizer

    from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
    from src.labels.categories import CATEGORIES
    from src.labels.loader import load_cuad
    from src.labels.splits import split_by_contract
    from src.models.naive.word_tfidf import fit_word_char_tfidf
    from src.pipeline import prepare_contracts, to_model_inputs
    from src.registry.predictors import VectorizedClassifier

    # The same nested split as the cached evaluation; train on the training contracts only
    development, _ = split_by_contract(load_cuad(TRAIN_PATH), val_fraction=EVALUATION_FRACTION, seed=SEED)
    training, _ = split_by_contract(development, val_fraction=CALIBRATION_FRACTION, seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
                                              cache_dir=TOKENIZER_CACHE)
    texts, labels = to_model_inputs(prepare_contracts(training, tokenizer, max_tokens=MAX_TOKENS,
                                                      overlap_tokens=OVERLAP_TOKENS))
    features, classifier = fit_word_char_tfidf(texts, labels, c=C)

    # Bundle the model with its calibrated thresholds (ClauseModelBundle validates both)
    bundle = ClauseModelBundle(
        predictor=VectorizedClassifier(features, classifier),
        categories=CATEGORIES,
        thresholds=json.loads(thresholds.read_text(encoding="utf-8")),
        model_name="word_char_tfidf",
        config={"thresholds": thresholds.name, "split_seed": SEED, "training_contracts": len(training),
                "max_tokens": MAX_TOKENS, "overlap_tokens": OVERLAP_TOKENS, "c": C,
                "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION},
        seed=SEED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return bundle


def load_or_build_bundle(path: Path = BUNDLE_PATH, url: str = RELEASE_URL, sha256: str = BUNDLE_SHA256,
                         build: Callable[[Path], ClauseModelBundle] = build_demo_bundle) -> ClauseModelBundle:
    """Load `path` (downloading it if missing); if that fails, build the model locally into `path` instead."""
    try:
        bundle_file = fetch_bundle_file(path, url, sha256)
        # A pickle from other scikit-learn versions may load but predict wrongly, so treat it as a failure
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            return joblib.load(bundle_file)
    except Exception as error:  # offline, hash mismatch, or unloadable here: the file can always be rebuilt
        print(f"Couldn't use the released model ({error}).\nBuilding it locally instead (a few minutes)...")
        return build(path)


def load_demo_bundle(path: Path | None = None, registry_version: str | None = None) -> ClauseModelBundle:
    """
    Load the demo model: from the MLflow registry if registry_version is given ("latest" or a number),
    from `path` if given (no fallback), else load_or_build_bundle().
    """
    if registry_version is not None:
        from src.registry.mlflow_registry import load_bundle  # MLflow is only needed for this route
        return load_bundle(REGISTRY_NAME, registry_version, tracking_uri=TRACKING_URI)
    return joblib.load(path) if path is not None else load_or_build_bundle()
