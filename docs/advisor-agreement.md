# Advisor agreement: protocol and metrics

The risk layer has no ground-truth labels. Its success criterion is agreement with an expert: a
strong Spearman correlation and good bucket agreement between the model's clause risk scores and
the advisor's hand-ranked clauses, plus evidence that the High/Medium boundary is stable. This
page describes how that agreement is collected and measured, and how to read the numbers.

Until an advisor returns a packet, the only agreement number in the repo is the one in
`results/contract_triage/report.json`, which compares predictions with a reference built from
CUAD annotations. That measures detection error. It says nothing about whether a human expert
agrees with the risk scores.

## Files

| File | Role |
|---|---|
| `scripts/build_advisor_packet.py` | Samples scored clauses and writes the packet, its key and the instructions |
| `src/risk/agreement.py` | `AdvisorRanking` record, key models, loader, metrics, threshold sweep, `evaluate_agreement` |
| `scripts/evaluate_advisor_agreement.py` | Joins a returned packet with its key and writes the report and tables |

## Protocol

1. **Build the packet** from the cached validation predictions. No model is run, and `test.json` is never read:

   ```
   python -m scripts.build_advisor_packet --output results/advisor_packet
   ```

   This writes three files:
   - `advisor_packet.csv` goes to the advisor.
   - `advisor_instructions.md` goes to the advisor.
   - `advisor_packet_key.json` stays with the team. Never send it to the advisor.

   `--weights` picks the weight set whose scores the key records, and defaults to `DEFAULT_WEIGHTS_PATH`. Later evaluations can rescore under any other weight set, so this choice only fixes the reference point. Rebuild the packet only before it is sent. A new seed or weight set gives a different sample and a new `packet_id`.
2. **Send the advisor only the packet CSV and the instructions.** The advisor gives each clause a bucket (Low, Medium or High), optionally a rank from 1 (riskiest) to N, and notes. A note starting `WRONG CATEGORY` marks a clause that isn't really of its stated category. The instructions estimate about 35 minutes for buckets and another 15 minutes for ranking, for 50 clauses.
3. **Evaluate** the returned CSV:

   ```
   python -m scripts.evaluate_advisor_agreement --key results/advisor_packet/advisor_packet_key.json \
       --returned <returned.csv> --reviewer "<advisor name>"
   ```

   Add `--weights <file>` to rescore every clause under another weight version. This reads only the key, so no model, probability cache or contract text is needed. Add `--exclude-wrong-category` to drop clauses the advisor flagged as detection errors.

The loader rejects a returned file that cannot be trusted, and its error names the item. It rejects:
- missing columns
- a `packet_id` from another packet
- duplicate, unknown or missing item ids
- a category or quote that no longer matches the key, which happens when rows are misaligned or edited
- a bucket that isn't Low, Medium or High (any case, and `Med` is accepted)
- a rank that isn't a whole number from 1 to N

A row with neither a bucket nor a rank counts as "couldn't judge". It is skipped and listed in the report, not rejected. Sorting the rows is fine, because the join uses `item_id`. The loader reads UTF-8 and falls back to Windows-1252 for files Excel saved.

## How the sample is drawn

The candidates are the predicted clauses on the 82 validation contracts that passed their per-class threshold, have grounded evidence, and have a quote of at least 8 words. Bare headings such as "LIMITATION OF LIABILITY" can't be judged.

The sample is stratified so the evaluation can actually test the High/Medium boundary:
- The three model buckets get equal shares where the data allows. For 50 clauses that is 17 High, 17 Medium and 16 Low.
- One clause comes first from every category with severity of at least 0.75 in the weight set.
- The rest are drawn round-robin over categories within each bucket.
- Caps per category within a bucket and per contract keep the spread.
- A quote that shares 70% or more of its words with a quote already chosen is skipped as a near-duplicate.
- Everything uses a fixed seed. The rows are shuffled and numbered after shuffling, so neither the order nor the ids carry information.

The sample is therefore **not representative** of all clauses. Under v1, Low makes up about 80% of the candidates but only a third of the packet. The rates describe agreement on this sample. They are not the share of all clauses the model buckets correctly.

## Why the advisor must not see model output

The packet contains no score, rating, signal value or probability, and its row order is random. An advisor who sees the model's opinion first is anchored by it. Their judgement then partly copies the model, and the agreement number measures that copying rather than independent agreement. That bias inflates agreement, and nothing in the data reveals it afterwards.

The same applies to anything else that shows model output: the demo UI, the clause register, `results/` and the key. Run the ranking session before the advisor sees any of them.

The packet still shows the predicted category, because the advisor needs to know what kind of clause they are judging. A category with high severity can hint at the model's view. That is unavoidable, and it is one reason `WRONG CATEGORY` notes are collected.

## What each metric means

Every metric comes with a 95% percentile bootstrap interval over clauses: 2,000 resamples of the judged clauses with replacement, using a fixed seed. A resample where a metric is undefined, for example one with no model-High clause when computing High precision, is dropped, and `n_boot_valid` records how many resamples were kept.

| Metric | Plain language |
|---|---|
| `spearman_rank` | Do the model's scores put the clauses in the same order as the advisor's ranks? 1 means the same order, 0 means no relation and -1 means reversed. Ties get average ranks. The model has many tied scores, and the advisor may tie ranks. |
| `spearman_bucket` | The same check against the advisor's buckets (Low < Medium < High). It works when the advisor gave buckets only. Buckets are heavily tied, which caps how high this can go. |
| `exact_rate` | The share of clauses where the model and the advisor chose the same bucket. |
| `adjacent_rate` | The share within one bucket of each other. With three buckets, only Low against High counts as a miss. |
| `confusion_matrix` | Counts with the advisor's bucket as rows and the model's as columns, in Low, Medium, High order. Off-diagonal cells show which way the model errs. |
| `weighted_kappa_linear` / `_quadratic` | Bucket agreement corrected for chance. 1 means perfect, 0 means what random buckets with the same frequencies would give, and below 0 means worse than chance. A one-step miss gets partial credit: 1/2 with linear weights, 3/4 with quadratic weights. Linear is the main figure. |
| `high_precision` | Of the clauses the model calls High, the share the advisor also calls High. Low precision means reviewers waste time on false alarms. |
| `high_recall` | Of the clauses the advisor calls High, the share the model also calls High. Low recall means risky clauses are missed, which is the more costly error for triage. |

### Threshold sweep (High/Medium boundary)

Thresholds change only how composite scores are bucketed, never the scores themselves, so the sweep needs no rescoring. The sweep:
- covers every (Medium, High) pair on a grid of ±0.15 around the weight set's values, in steps of 0.025, keeping only pairs where Medium < High
- reports, for each pair, exact agreement, weighted kappa, bucket counts and how many clauses changed bucket compared with the base setting
- counts a setting as **stable** when its exact rate is within 0.05 of the base rate

The summary gives the share of stable settings and, for each threshold, the unbroken range of values around the base that stays stable while the other threshold is held fixed. A wide `high_stable_range` is the evidence that "the High/Medium boundary is stable". A narrow one means a small change to `high_threshold` would change which clauses reviewers see as High.

`best_points` lists the settings with the highest agreement. Do not adopt them as new thresholds from one packet: with about 50 clauses, one clause moves the exact rate by 0.02, so the best setting mostly fits noise. Propose a threshold change only if the advisor also agrees, the change is supported by a second packet or by team rankings, and it goes into a new weight version.

## What sample size does to the conclusions

About 50 clauses is what a busy advisor can rank in one sitting, and it is not many. For a rate near 0.7, the 95% interval is roughly ±0.13 wide. High precision and recall rest only on the clauses in the High bucket, about 15 to 17, so their intervals are often ±0.2 or wider. The report never presents a point estimate as a conclusion. `caveats` turns each interval into a plain-language statement:

- **Spearman:** the statement depends on where the interval falls.
  - The whole interval is at or above 0.7: strong agreement is supported.
  - The interval is above 0 but reaches below 0.7: positive, but strength not confirmed.
  - The interval includes 0: no association can't be ruled out.
  - The whole interval is below 0: the model ranks opposite to the advisor.
- **Rates and kappa:** an interval wider than 0.3 is flagged as too wide to support a conclusion on its own.
- **Sample:** two further statements say that the sample is stratified and that the sweep is not a tuning result.

Report results as "Spearman 0.62 (95% CI 0.41 to 0.77, 48 clauses)", never as "0.62".

To narrow the intervals, collect more clauses: a second packet with a new `--seed`, or a second reviewer. A second reviewer also gives an inter-rater baseline. Two experts seldom agree perfectly, and the model shouldn't be expected to beat the agreement between them.

## Fallback if advisor rankings never arrive

If the advisor never returns a packet, the team can produce a substitute:
- Two or three team members each rank the **same** packet **independently**, before seeing any model output and without discussing it.
- Each person's file is evaluated separately with `--reviewer <name>`, and their agreement with each other is reported too, using the same metrics with one person's buckets in place of the model's.

These results must be labelled **team judgement**. They must never be reported as advisor agreement, or as meeting the advisor-agreement success criterion. Team members are not legal experts, and they have seen the scoring design, so they are not blind in the way the advisor is. The final report should state that the advisor-agreement criterion remains unmet and that the team-judgement numbers are a weaker substitute.

## Storage

`AdvisorRanking` fields match the `advisor_rankings` table: `contract_id`, `category`, `start`, `end`, `quote`, `advisor_rank` (1 is the riskiest, optional), `advisor_bucket` (Low, Medium or High, optional), `reviewer` and `ranking_set`. A record must have at least one of rank and bucket.

`ranking_set` defaults to the packet id. The evaluation writes these records to `advisor_rankings.csv`, ready to load into that table. A clause is identified by its contract, category and evidence span, not by `chunk_index`, so stored rankings stay valid when chunking changes. Agreement can then be recomputed for any weight version, as ARCHITECTURE.md requires.
