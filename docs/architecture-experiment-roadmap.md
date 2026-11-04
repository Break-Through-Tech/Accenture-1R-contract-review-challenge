# Next neural architecture experiments

This roadmap records five architecture directions before implementation. Each experiment
must have its own config, training/evaluation entry point, tests, and result report. Use
contract-level train/calibration/evaluation splits, tune checkpoints and thresholds on
calibration only, report once on held-out evaluation, and never load `data/cuad/test.json`.
Compare to the same-split word+character TF-IDF control (seed-42 macro-F1 0.533,
macro-AUPR 0.597) and the existing contract-context Transformer (0.496 / 0.507).
Do not promote a candidate based on attention visualizations alone.

## 1. High-recall retrieval followed by category-conditioned reranking

**Question:** Can sparse lexical retrieval cheaply shortlist likely chunk/category pairs,
then a pretrained cross-encoder disambiguate the candidate clauses using the category
description and local contract context?

Build the retriever on training chunks only. For each contract and category, retain a
bounded top-K candidate set and measure gold candidate recall before reranking. Train a
binary cross-encoder on positive annotations and hard negatives surfaced by the
training-only retriever. Report candidate recall@K, final chunk-level macro-F1/AUPR,
latency, and the number of pairs scored. This is explicitly a two-stage system; the
retriever's missed positives set an upper bound on reranker recall.

## 2. Contract graph neural network

Represent each chunk as a node. Add directed/undirected edges for immediate neighbors,
shared section membership where available, and high-similarity nonlocal chunk pairs
built from training-fitted features. Run message passing over this graph before the
41-label head. Ablate edge families and compare to the sequence Transformer; report
both classification metrics and graph size/runtime. Graph construction must be
annotation-blind and deterministic at inference.

## 3. Supervised contrastive legal-clause embeddings

Fine-tune a chunk encoder with a multi-label supervised-contrastive objective: chunks
sharing clause labels are positives, and retrieved near-miss clauses are hard negatives.
Then classify with calibrated category prototypes or a small learned head. Split
contracts before pair construction so no contract contributes across partitions. Report
retrieval recall@K and macro classification metrics; keep prototype construction train-only.

## 4. Document-level set prediction of clause instances

Predict a variable-size set of category-tagged clause spans per contract, rather than
independent labels for overlapping chunks. Use learned instance slots and bipartite
matching against CUAD answer spans; include a no-object class and evaluate category
presence plus span overlap/IoU. Preserve exact source offsets and explicitly measure
annotation coverage. Do not conflate CUAD answer spans with complete legally sufficient
clauses.

## 5. Long-document sparse-attention encoder

Evaluate a pretrained sparse-attention model (for example, Longformer) on substantially
longer contract windows with overlapping windows and overlap-aware aggregation. Preserve
source offsets and account for answers beyond the model's context limit. Compare coverage,
GPU memory, throughput, and classification quality against chunk-encoder plus
inter-chunk context; do not assume longer input alone improves quality.

## Completion checklist

For each direction: (1) implement and unit-test the model/data path; (2) run a small
sanity/overfit test; (3) run the predeclared nested evaluation; (4) save config, split
hashes, thresholds, raw predictions, per-category metrics, and limits; (5) make a
separate focused implementation/evaluation commit. Leave model weights and registry DBs
in ignored local `outputs/`; do not push or register a candidate as the default without
explicit approval.

## Progress

1. **Retrieval + reranking:** implemented and evaluated on seed 42. It improved macro-F1
   from 0.533 to 0.557 while macro-AUPR changed from 0.597 to 0.595; multi-seed
   robustness remains outstanding. Details: `retrieval-reranker-architecture.md`.
2. **Contract graph neural network:** queued.
3. **Supervised contrastive embeddings:** queued.
4. **Document-level set prediction:** queued.
5. **Long-document sparse attention:** queued.
