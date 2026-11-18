"""
Train the word + character TF-IDF clause model and save it with its thresholds as one file
(a ClauseModelBundle) that the inference pipeline and the UI can load without MLflow.

It reproduces the model behind results/word_char_tfidf_nested/: the same contract split
(seed 42; 20% evaluation held out, 20% of the rest for calibration), trained on the training
contracts only, with the thresholds that were tuned on the calibration contracts.
Reads only train_separate_questions.json; test.json is never touched. The training itself is
clausetriage.registry.demo_model.build_demo_bundle, which the UI also runs when it can't download the model.

Run from the repo root (takes several minutes):
    python -m scripts.build_model_bundle
    python -m scripts.build_model_bundle --register    # also log it to the local MLflow registry

--register logs the bundle, its held-out macro F1 and its settings as a new version of
REGISTRY_NAME in the local SQLite tracking store (outputs/mlflow/tracking.db, as the encoder
scripts use). Browse it with:
    mlflow ui --backend-store-uri sqlite:///outputs/mlflow/tracking.db

The output is a joblib (pickle) file: only load bundles you built or trust, since loading
one can run code.
"""
import argparse
import json
from pathlib import Path

from clausetriage.registry.demo_model import (BUNDLE_PATH, EXPERIMENT_NAME, REGISTRY_NAME, THRESHOLDS_PATH, TRACKING_URI,
                                     build_demo_bundle, file_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    parser.add_argument("--output", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--register", action="store_true", help="Also register the bundle in the local MLflow registry")
    args = parser.parse_args()

    bundle = build_demo_bundle(args.output, args.thresholds)
    print(f"Saved {args.output} ({bundle.config['training_contracts']} training contracts), "
          f"SHA-256 {file_sha256(args.output)}")

    # Optional: register it, with the held-out score from the evaluation that tuned its thresholds
    if args.register:
        from clausetriage.registry.mlflow_registry import register_bundle

        report = json.loads((args.thresholds.parent / "report.json").read_text(encoding="utf-8"))
        registered = register_bundle(
            bundle, experiment_name=EXPERIMENT_NAME, registry_name=REGISTRY_NAME,
            metrics={"evaluation_macro_f1": report["macro_f1_evaluation"]["word_char_tfidf"]},
            metadata={**bundle.config, "evaluation_report": args.thresholds.parent.name},
            tracking_uri=TRACKING_URI,
        )
        print(f"Registered {registered.model_key} (run {registered.run_id}) in {TRACKING_URI}")


if __name__ == "__main__":
    main()
