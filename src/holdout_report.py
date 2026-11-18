"""
Tables and text for the sealed evaluation's outputs (docs/final-test-evaluation.md). Pure functions
over the evaluation result and report; every number in summary.md comes from report.json or the
tables, so nothing is hand-edited.

How everything comes together:
per_category_frame(): the primary per-category table (test metrics, intervals, status, development means, word-only)
bucket_frame(): mean test and development F1/AUPR per development support bucket
render_summary(): summary.md from the report and the two tables
jsonable(): numpy scalars to Python numbers and NaN to None, so report.json is strict JSON
"""
import numpy as np
import pandas as pd

from src.holdout_metrics import support_status

BUCKET_ORDER = {"common": 0, "uncommon": 1, "rare": 2}


def per_category_frame(result: dict, reference: dict) -> pd.DataFrame:
    """
    One row per category in CATEGORIES order: status from test support, test contracts with the category,
    the bundle's threshold, test metrics with intervals, development means and bucket, and word-only metrics.
    """
    table = result["tables"]["word_char_tfidf"].copy()
    word = result["tables"]["word_tfidf"].set_index("category")
    table.insert(1, "status", table["support"].map(support_status))
    table.insert(2, "test_contracts_with_category", result["contract_presence"].sum(axis=0).astype(int))
    table.insert(3, "threshold", result["thresholds"]["word_char_tfidf"])
    table = table.merge(reference["per_category"], on="category", how="left")
    for metric in ("precision", "recall", "f1", "aupr"):
        table[f"word_tfidf_{metric}"] = table["category"].map(word[metric])
    return table


def bucket_frame(categories: pd.DataFrame) -> pd.DataFrame:
    """Mean test and development F1/AUPR per development support bucket (membership fixed before the test)."""
    grouped = categories.groupby("dev_bucket").agg(
        categories=("category", "size"), test_support=("support", "sum"),
        test_mean_f1=("f1", "mean"), dev_mean_f1=("dev_mean_f1", "mean"),
        test_mean_aupr=("aupr", "mean"), dev_mean_aupr=("dev_mean_aupr", "mean"),
        unscorable=("status", lambda s: int((s == "unscorable").sum())),
        insufficient_support=("status", lambda s: int((s == "insufficient support").sum())),
    ).reset_index()
    return grouped.sort_values("dev_bucket", key=lambda s: s.map(BUCKET_ORDER)).reset_index(drop=True)


def fmt(value: float | None, interval: dict | None = None) -> str:
    """'0.512' or '0.512 [0.470, 0.553]'; 'n/a' for None or NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if interval is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{fmt(interval['lower_95'])}, {fmt(interval['upper_95'])}]"


def render_summary(report: dict, categories: pd.DataFrame, buckets: pd.DataFrame) -> str:
    """summary.md: headline table with the pre-registered reading, buckets, per category, secondaries, run record."""
    primary, secondary = report["primary"], report["secondary"]
    lines = [f"# Final evaluation: {report['mode']}", "",
             f"Data `{report['data']['path']}` (SHA-256 `{report['data']['sha256'][:12]}...`), "
             f"{report['data_description']['contracts']} contracts, {report['data_description']['chunks']} chunks. "
             f"Model `{report['model']['path']}` (SHA-256 `{report['model']['sha256'][:12]}...`). "
             f"`test_data_used: {str(report['test_data_used']).lower()}`.", "",
             "## Primary: chunk-level detection, frozen bundle at its own thresholds", "",
             "Intervals: contract bootstrap, BCa 95% (percentile bounds are in report.json and the CSVs).", "",
             "| Metric | Result [95% BCa CI] | Development mean (range of 5 splits) | Seed 42 | Pre-registered reading |",
             "|---|---|---|---|---|"]
    for metric, label in (("macro_f1", "Macro-F1 (41 categories)"),
                          ("macro_aupr", "Macro-AUPR (categories with positives)")):
        reference = report["development_reference"]["macro"][f"word_char_tfidf_{metric}"]
        reading = primary["against_development"][metric]
        lines.append(f"| {label} | {fmt(primary[metric], primary[f'{metric}_95'])} | {fmt(reference['mean'])} "
                     f"({fmt(reference['min'])}-{fmt(reference['max'])}) | {fmt(reference['seed_42'])} | "
                     f"{reading['verdict']}, {reading['position']} |")
    lines += ["", f"Categories with positives: {primary['categories_with_positives']} of 41. "
              f"Unscorable (no positives): {', '.join(primary['unscorable_categories']) or 'none'}. "
              f"Insufficient support (<10 positives): {len(primary['insufficient_support_categories'])}.", "",
              "### By development support bucket", "",
              "| Bucket | Categories | Mean F1 | Dev mean F1 | Mean AUPR | Dev mean AUPR | Under 10 positives here |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in buckets.itertuples():
        lines.append(f"| {row.dev_bucket} | {row.categories} | {fmt(row.test_mean_f1)} | {fmt(row.dev_mean_f1)} | "
                     f"{fmt(row.test_mean_aupr)} | {fmt(row.dev_mean_aupr)} | {row.insufficient_support + row.unscorable} |")
    lines += ["", "### Per category (sorted by support)", "",
              "| Category | Support | Status | Precision | Recall | F1 [95% CI] | AUPR [95% CI] | Dev mean F1 |",
              "|---|---:|---|---:|---:|---|---|---:|"]
    for row in categories.sort_values(["support", "category"], ascending=[False, True]).itertuples():
        lines.append(f"| {row.category} | {row.support} | {row.status} | {fmt(row.precision)} | {fmt(row.recall)} | "
                     f"{fmt(row.f1, {'lower_95': row.f1_lower_95, 'upper_95': row.f1_upper_95})} | "
                     f"{fmt(row.aupr, {'lower_95': row.aupr_lower_95, 'upper_95': row.aupr_upper_95})} | "
                     f"{fmt(row.dev_mean_f1)} |")
    word = secondary["word_only_reference"]
    delta = word["word_char_minus_word"]
    triage = secondary["triage"]
    agreement = triage["agreement_with_perfect_detection"]["triage_score"]
    top_overlap = agreement[f"top_{triage['top_n']}_overlap"]
    evidence, contract = secondary["evidence"], secondary["contract_level"]
    lines += ["", "## Secondary analyses", "",
              f"- **S1 word-only TF-IDF reference:** macro-F1 {fmt(word['macro_f1'], word['macro_f1_95'])}, "
              f"macro-AUPR {fmt(word['macro_aupr'], word['macro_aupr_95'])}. Word+char minus word-only (paired): "
              f"macro-F1 {fmt(delta['macro_f1'], delta['macro_f1_95'])}, "
              f"macro-AUPR {fmt(delta['macro_aupr'], delta['macro_aupr_95'])}.",
              f"- **S2 contract-level presence:** macro-F1 {fmt(contract['macro_f1'], contract['macro_f1_95'])}, "
              f"macro-AUPR {fmt(contract['macro_aupr'], contract['macro_aupr_95'])} "
              f"({contract['categories_with_positives']} categories present in at least one contract).",
              f"- **S3 triage ranking vs perfect detection (`{triage['weight_version_id']}`):** Spearman "
              f"{fmt(agreement['spearman'], triage['triage_score_spearman_95'])}, largest rank move "
              f"{agreement['max_rank_move']}, top-{triage['top_n']} overlap {top_overlap} "
              f"over {triage['contracts']} contracts.",
              f"- **S4 evidence:** Unknown rate {fmt(evidence['unknown_rate'])} ({evidence['unknown']} of "
              f"{evidence['clauses_after_deduplication']} detected clauses); a grounded quote overlaps a gold answer "
              f"{fmt(evidence['grounded_quote_overlaps_gold'])} of the time overall and "
              f"{fmt(evidence['grounded_quote_overlaps_gold_on_true_positive_chunks'])} on correctly detected chunks.",
              "", "## Run record", "",
              f"- Commit `{report['git']['commit']}`; protocol SHA-256 `{report['protocol']['sha256'][:12]}...`.",
              f"- Attempts: {len(report['attempts'])}. Deviations: {len(report['deviations'])}.",
              f"- Serving parity checked on {report['integrity']['serving_parity_contracts']} contracts.", ""]
    return "\n".join(lines)


def jsonable(value):
    """Convert numpy scalars to Python numbers and NaN to None, recursively, so report.json is strict JSON."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
