# Validation gate for the TF-IDF score-fusion candidate

The current score-fusion results are exploratory because the CUAD training-file evaluation
splits have already been inspected. This protocol describes what must be agreed before
claiming a real upgrade. It does not authorize use of `data/cuad/test.json`.

## Decisions needed from the advisor/team

1. **Evaluation source:** approve either new, independently annotated contracts with the
   41 CUAD categories mapped under documented guidelines, or explicitly authorize one
   sealed use of the official CUAD test set. Confirm data use, license, storage, and access
   restrictions before obtaining or opening any evaluation data.
2. **Success criteria:** agree on primary metrics and practical minimum improvement. A
   proposed statistical gate is a paired contract-bootstrap 95% lower bound above zero for
   both macro-F1 and macro-AUPR, plus predeclared tolerances for per-category regressions
   and inference cost. The advisor/team must approve or revise this gate before evaluation.
3. **Scope:** decide whether target-domain calibration is allowed. If it is, provide a
   separate approved calibration partition that is not part of the final holdout.

## Freeze before opening the holdout

- Pin dataset version/hash, contract partition manifest, tokenizer/model revisions, chunker
  settings, classifier configs, category order, candidate K, fusion formula, alpha-selection
  procedure, random seeds, and software dependencies.
- Select fusion alpha using training-only out-of-fold predictions, not the final holdout.
  Fit final category thresholds on the separate calibration partition only.
- Freeze the TF-IDF control and fusion candidate as a paired comparison. No architecture,
  threshold, or category-specific tuning is allowed after inspecting holdout metrics.
- Commit the configuration and protocol hash before the evaluation run; restrict the raw
  holdout data and predictions to the designated evaluator if feasible.

## One-time evaluation report

- Confirm identical contract IDs, chunking, labels, and category order for both systems.
- Report per-category precision, recall, F1, support, and AUPR; macro-F1 and macro-AUPR;
  paired contract-bootstrap intervals; and calibration/threshold provenance.
- Report candidate recall, CPU/GPU inference cost, memory, and the fraction of scores
  changed by reranking. Include examples of both gains and regressions.
- Categories absent from the evaluation source are marked unscorable, not interpreted as
  successful or failed. Report their count and support explicitly.
- Save reports and hashes, but do not persist or publish restricted source text.

Until these decisions are recorded and the protocol is frozen, score fusion remains a
research prototype. The existing exploratory splits must not be represented as a clean
confirmatory test. `data/cuad/test.json` remains untouched unless the team/advisor explicitly
authorizes its use under this one-time protocol.
