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

Predict a variable-size set of category-tagged answer-span intervals per contract, rather than
independent labels for overlapping chunks. Use learned instance slots and bipartite
matching against CUAD answer spans; include a no-object class and evaluate category
presence plus span overlap/IoU. For the first implementation, predict start/end **chunk
indices** and map them back to exact source character offsets; explicitly report the
chunk-boundary resolution limit and annotation coverage. Do not conflate CUAD answer spans
with complete legally sufficient clauses.

## 5. Long-document sparse-attention encoder

Evaluate Longformer-style sparse attention at the chunk-sequence level: a frozen pinned
encoder produces every overlapping chunk representation, then local windows plus a
global contract token contextualize the complete sequence. This is not raw Longformer
self-attention over contract tokens. Preserve source offsets; aggregate overlapping-window
scores by source interval. Compare full-text coverage, sparse-vs-dense attention links,
GPU memory, throughput, and classification quality against chunk encoder plus dense
inter-chunk context; do not assume longer input alone improves quality.

## Completion checklist

For each direction: (1) implement and unit-test the model/data path; (2) run a small
sanity/overfit test; (3) run the predeclared nested evaluation; (4) save config, split
hashes, thresholds, raw predictions, per-category metrics, and limits; (5) make a
separate focused implementation/evaluation commit. Leave model weights and registry DBs
in ignored local `outputs/`; do not push or register a candidate as the default without
explicit approval.

## Progress

1. **Retrieval + reranking:** implemented and evaluated on five predeclared seeds. The
   reranker improves mean macro-F1 by 0.016 and wins four of five splits, but mean
   macro-AUPR is 0.006 lower and every per-seed paired contract-bootstrap interval
   includes zero. It does not yet meet the robustness bar for replacing the control.
   Details and per-category tradeoffs: `retrieval-reranker-architecture.md`.
2. **Contract graph neural network:** implemented and screened on seed 42. The initial
   GraphSAGE candidate scored 0.390 macro-F1 / 0.426 macro-AUPR, below the TF-IDF control;
   its calibration curve was still rising at the three-epoch limit. See
   `contract-graph-architecture.md`.
3. **Supervised contrastive embeddings:** implemented and screened on seed 42. The
   train-only prototype classifier scored 0.365 macro-F1 / 0.372 macro-AUPR; prototype
   recall@16 was 0.950, but ranking scores were poorly calibrated. See
   `contrastive-architecture.md`.
4. **Document-level set prediction:** implemented and evaluated on seed 42 with 8
   Hungarian-matched slots/category. Chunk macro-F1/AUPR were 0.099/0.088; character
   span IoU-0.5 micro-F1 was 0.0011 despite full annotation coverage. The chunk-boundary
   pointer design is too coarse in this run. See `set-prediction-architecture.md`.
5. **Long-document sparse attention:** implemented and screened with local/global sparse
   attention over complete chunk sequences (not raw-token Longformer). It covered 100% of
   source text and used 46.6% of the equivalent dense chunk-attention links on this split,
   but scored 0.142/0.131 chunk macro-F1/AUPR versus the TF-IDF control's 0.533/0.597.
   See `sparse-long-context-architecture.md`.
