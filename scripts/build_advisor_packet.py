"""
Build the blind packet the advisor ranks, and the key that turns their answers into agreement metrics.
Samples scored, evidence-grounded predicted clauses from the validation contracts (the same flow as
scripts/evaluate_triage.py: cached probabilities, no model rerun, test.json never touched).

Run from the repo root:
    python -m scripts.build_advisor_packet
    python -m scripts.build_advisor_packet --weights <weights.json> --n-items 50 --seed 42

Writes to --output:
    advisor_packet.csv: what the advisor sees and fills in. Item id, contract, category with its plain
                        description, the quote with surrounding context, and blank advisor_bucket,
                        advisor_rank and advisor_notes columns. No model score, rating or signal, and
                        rows in shuffled order, so nothing in the file hints at the model's opinion
    advisor_packet_key.json: item id -> everything needed to rescore the clause without a model, plus
                             the model's score, rating and signals at generation time. Keep it away
                             from the advisor
    advisor_instructions.md: what to do, in what order, and how long it takes

Sampling: grounded clauses with quotes of at least --min-quote-words words, stratified over the
model's three rating buckets (equal shares where there are enough clauses, so High and Medium aren't
swamped by Low), one clause first from every high-severity category in the weight set, then
round-robin over categories within each bucket. Per-category and per-contract caps keep the spread,
and a quote whose words overlap an already chosen quote by at least --max-similarity (Jaccard) is
skipped as a near-duplicate. The sample is deliberately not representative of all clauses.
"""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from scripts.evaluate_triage import (
    CACHE_DIR,
    MAX_TOKENS,
    OVERLAP_TOKENS,
    TRAIN_PATH,
    VAL_FRACTION,
    _load_cache,
    _shown,
)
from scripts.evaluate_triage import (
    SEED as SPLIT_SEED,
)
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.evidence.cues import (
    CATEGORY_DESCRIPTIONS_PATH,
    load_category_cues,
    normalize_category,
)
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.pipeline import prepare_contracts, to_model_inputs
from src.risk.agreement import BUCKETS, PacketKey, PacketKeyItem, write_key
from src.risk.inputs import clauses_from_probabilities
from src.risk.schema import (
    DEFAULT_WEIGHTS_PATH,
    ClauseInput,
    ClauseScore,
    WeightSet,
    load_weight_set,
)
from src.risk.scoring import score_contract_clauses

ROOT = Path(__file__).resolve().parents[1]
PACKET_FILE, KEY_FILE, INSTRUCTIONS_FILE = "advisor_packet.csv", "advisor_packet_key.json", "advisor_instructions.md"
PACKET_COLUMNS = (
    "packet_id", "item_id", "contract", "category", "category_description",
    "context_before", "quote", "context_after", "advisor_bucket", "advisor_rank", "advisor_notes",
)
# Categories at or above this severity in the weight set get at least one clause in the packet
HIGH_SEVERITY = 0.75
# Rough reading and judging time per clause, and for the ranking pass, used in the instructions
SECONDS_PER_ITEM, RANKING_MINUTES = 40, 15


@dataclass(frozen=True)
class Candidate:
    """One grounded predicted clause and its score under the packet's weight set."""
    clause: ClauseInput
    score: ClauseScore


# ----- Category descriptions -----
def load_category_descriptions(path: str | Path = CATEGORY_DESCRIPTIONS_PATH) -> dict[str, str]:
    """Read category_descriptions.csv into canonical category -> plain-language description."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    canonical = {normalize_category(c): c for c in CATEGORIES}
    descriptions = {}
    for _, row in df.iterrows():
        name = re.sub(r"^\s*Category:\s*", "", str(row.iloc[0])).strip()
        description = re.sub(r"^\s*Description:\s*", "", str(row["Description"])).strip()
        if normalize_category(name) in canonical:
            descriptions[canonical[normalize_category(name)]] = description
    return descriptions


# ----- Text helpers -----
def collapse(text: str) -> str:
    """Collapse runs of whitespace (newlines included) to one space, for a readable spreadsheet cell."""
    return re.sub(r"\s+", " ", text).strip()


def quote_context(text: str, start: int, end: int, chars: int = 300) -> tuple[str, str]:
    """
    Up to chars characters of text before and after [start:end], trimmed to whole words and
    whitespace-collapsed. "..." marks a side that was cut short of the contract's start or end.
    """
    low, high = max(0, start - chars), min(len(text), end + chars)
    before, after = text[low:start], text[end:high]

    # Drop a word the window cut in half, then mark the cut
    if low > 0:
        if not text[low - 1].isspace() and before[:1].strip():
            before = before.split(None, 1)[1] if len(before.split(None, 1)) > 1 else ""
        before = "..." + collapse(before)
    if high < len(text):
        if not text[high].isspace() and after[-1:].strip():
            after = after.rsplit(None, 1)[0] if len(after.rsplit(None, 1)) > 1 else ""
        after = collapse(after) + "..."
    return collapse(before), collapse(after)


def _words(quote: str) -> frozenset[str]:
    """Lowercase word set of a quote, for near-duplicate checks."""
    return frozenset(re.findall(r"[a-z0-9]+", quote.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Shared words over all words of two word sets (1.0 for two empty sets)."""
    return len(a & b) / len(a | b) if a | b else 1.0


# ----- Sampling -----
def bucket_quotas(n_items: int) -> dict[str, int]:
    """Split n_items evenly over the buckets; the remainder goes to High first, then Medium."""
    quotas = {bucket: n_items // len(BUCKETS) for bucket in BUCKETS}
    for bucket in ("High", "Medium", "Low")[:n_items % len(BUCKETS)]:
        quotas[bucket] += 1
    return quotas


def grounded_candidates(
    clauses: Sequence[ClauseInput], scores: Sequence[ClauseScore], min_words: int = 8,
) -> list[Candidate]:
    """
    Pair clauses with their scores, keep only those with evidence of at least min_words words (a bare
    heading such as "LIMITATION OF LIABILITY" can't be judged), and drop exact repeats (the same quote
    for the same category in two overlapping chunks), in a fixed order for seeded sampling.
    """
    seen, candidates = set(), []
    for clause, score in sorted(zip(clauses, scores, strict=True),
                                key=lambda cs: (cs[0].contract_id, cs[0].chunk_index, cs[0].category)):
        if clause.evidence is None or len(clause.evidence.quote.split()) < min_words:
            continue
        identity = (clause.contract_id, clause.category, clause.evidence.start, clause.evidence.end)
        if identity not in seen:
            seen.add(identity)
            candidates.append(Candidate(clause, score))
    return candidates


def select_clauses(
    candidates: Sequence[Candidate],
    n_items: int,
    seed: int,
    priority_categories: Sequence[str] = (),
    max_per_category: int = 3,
    max_per_contract: int = 3,
    max_similarity: float = 0.7,
) -> list[Candidate]:
    """
    Pick up to n_items candidates, stratified by rating bucket and spread over categories.

    1. one clause from each priority category (e.g. the high-severity ones), if any is eligible
    2. per bucket, up to its quota (bucket_quotas), one clause per category per round, categories in
       seeded random order
    3. any slots left (a bucket ran short) are filled the same way from every bucket
    A candidate is eligible only under the caps and when its quote's word overlap with every chosen
    quote is below max_similarity. The category cap applies per bucket and is raised to
    ceil(quota / categories in the bucket) when a bucket has too few categories to fill its quota
    (High clauses often come from 3 or 4 categories). May return fewer than n_items.
    """
    rng = random.Random(seed)
    quotas = bucket_quotas(n_items)
    category_cap = {
        bucket: max(max_per_category, math.ceil(quotas[bucket] / max(1, len(
            {c.clause.category for c in candidates if c.score.rating == bucket}))))
        for bucket in BUCKETS
    }
    chosen: list[Candidate] = []
    per_category, per_contract, per_bucket = Counter(), Counter(), Counter()
    chosen_words: list[frozenset[str]] = []

    def eligible(c: Candidate, use_quota: bool) -> bool:
        """Whether c can still be added under the caps, the similarity rule and (optionally) its bucket quota."""
        if c in chosen or per_category[c.score.rating, c.clause.category] >= category_cap[c.score.rating]:
            return False
        if per_contract[c.clause.contract_id] >= max_per_contract:
            return False
        if use_quota and per_bucket[c.score.rating] >= quotas[c.score.rating]:
            return False
        words = _words(c.clause.evidence.quote)
        return all(jaccard(words, other) < max_similarity for other in chosen_words)

    def add(c: Candidate) -> None:
        """Record a chosen candidate and update the counters."""
        chosen.append(c)
        chosen_words.append(_words(c.clause.evidence.quote))
        per_category[c.score.rating, c.clause.category] += 1
        per_contract[c.clause.contract_id] += 1
        per_bucket[c.score.rating] += 1

    def round_robin(pool: Sequence[Candidate], limit: int, use_quota: bool) -> None:
        """Add one eligible clause per category per round, until limit clauses are chosen or none is eligible."""
        categories = sorted({c.clause.category for c in pool})
        rng.shuffle(categories)
        progress = True
        while len(chosen) < limit and progress:
            progress = False
            for category in categories:
                options = [c for c in pool if c.clause.category == category and eligible(c, use_quota)]
                if options and len(chosen) < limit:
                    add(rng.choice(options))
                    progress = True

    # 1. Priority categories first, from any bucket that still has room
    for category in sorted(priority_categories):
        options = [c for c in candidates if c.clause.category == category and eligible(c, use_quota=True)]
        if options and len(chosen) < n_items:
            add(rng.choice(options))

    # 2. Each bucket up to its quota, then 3. any remaining slots from every bucket
    for bucket in ("High", "Medium", "Low"):
        round_robin([c for c in candidates if c.score.rating == bucket],
                    len(chosen) + quotas[bucket] - per_bucket[bucket], use_quota=True)
    round_robin(candidates, n_items, use_quota=False)
    return chosen


# ----- Packet and key -----
def make_packet_id(chosen: Sequence[Candidate], weight_set: WeightSet, seed: int) -> str:
    """A packet id from the weight version, the seed and a short hash of the chosen spans."""
    spans = sorted(f"{c.clause.contract_id}|{c.clause.category}|{c.clause.evidence.start}|{c.clause.evidence.end}"
                   for c in chosen)
    digest = hashlib.sha256("\n".join(spans).encode("utf-8")).hexdigest()[:8]
    return f"advisor-{weight_set.weight_version_id}-s{seed}-{digest}"


def build_packet(
    chosen: Sequence[Candidate],
    contract_texts: Mapping[str, str],
    contract_categories: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str],
    weight_set: WeightSet,
    seed: int,
    context_chars: int = 300,
    settings: Mapping | None = None,
) -> tuple[pd.DataFrame, PacketKey]:
    """
    Shuffle the chosen clauses (seeded), number them in that order, and return the blind packet
    (PACKET_COLUMNS, one row per item) and its key. contract_categories: every category detected in
    each contract, which the contract_context signal needs to rescore a clause later.
    """
    if not chosen:
        raise ValueError("no clauses to put in the packet")
    order = list(chosen)
    random.Random(seed).shuffle(order)
    packet_id = make_packet_id(order, weight_set, seed)
    width = max(2, len(str(len(order))))

    rows, items = [], []
    for number, c in enumerate(order, start=1):
        item_id, evidence = f"C{number:0{width}d}", c.clause.evidence
        before, after = quote_context(contract_texts[c.clause.contract_id], evidence.start, evidence.end, context_chars)
        # Advisor-facing row: no score, rating, signal or probability
        rows.append({
            "packet_id": packet_id, "item_id": item_id, "contract": c.clause.contract_id,
            "category": c.clause.category, "category_description": descriptions.get(c.clause.category, ""),
            "context_before": before, "quote": collapse(evidence.quote), "context_after": after,
            "advisor_bucket": "", "advisor_rank": "", "advisor_notes": "",
        })
        # Hidden key row: the inputs to rescore it, and the model's opinion at generation time
        items.append(PacketKeyItem(
            item_id=item_id, contract_id=c.clause.contract_id, chunk_index=c.clause.chunk_index,
            category=c.clause.category, start=evidence.start, end=evidence.end, quote=evidence.quote,
            probability=c.clause.probability, contract_categories=tuple(contract_categories[c.clause.contract_id]),
            composite_score=c.score.composite_score, rating=c.score.rating, signals=dict(c.score.signals),
            weight_version_id=c.score.weight_version_id, weight_set_sha256=c.score.weight_set_sha256,
        ))
    key = PacketKey(
        packet_id=packet_id, weight_version_id=weight_set.weight_version_id,
        weight_set_sha256=weight_set.content_sha256, medium_threshold=weight_set.medium_threshold,
        high_threshold=weight_set.high_threshold, settings=dict(settings or {}), items=tuple(items),
    )
    return pd.DataFrame(rows, columns=list(PACKET_COLUMNS)), key


def instructions(packet_id: str, n_items: int) -> str:
    """The advisor's instructions: what to do, in what order, and roughly how long it takes."""
    bucket_minutes = round(n_items * SECONDS_PER_ITEM / 60 / 5) * 5 or 5
    return f"""# Clause risk ranking: instructions

Packet `{packet_id}`, {n_items} clauses. About {bucket_minutes} minutes for steps 1 to 3, plus about
{RANKING_MINUTES} minutes for step 4.

Each row of `{PACKET_FILE}` is one clause taken from a commercial contract. We want your own judgement
of how risky each clause is, so the file deliberately shows nothing about how our tool scored it.
Please don't look at any of our tool's output before you finish.

## What to do

1. Open `{PACKET_FILE}` in Excel or Google Sheets. Don't sort or delete rows yet, and don't edit
   the columns `packet_id`, `item_id`, `category` or `quote`: they are how we match your answers back.
2. For each row, read the `quote` (the clause). `context_before` and `context_after` show the text
   around it; `category` and `category_description` say what kind of clause our tool thinks it is.
3. In `advisor_bucket`, write **Low**, **Medium** or **High**: how much attention this clause should
   get from a reviewer checking the contract, judged as a neutral reviewer (not for one particular
   side). Go with your first professional judgement; there is no answer key.
   - High: should be read carefully and probably raised with the client or negotiated.
   - Medium: worth a look, but usually acceptable or standard with minor caveats.
   - Low: routine; a reviewer can skim it.
4. In `advisor_rank`, rank the clauses from **1 (riskiest)** to **{n_items} (least risky)**. The easiest way:
   sort the sheet by `advisor_bucket` (High first), then order the clauses within each bucket.
   Ties are fine: give two clauses the same number if you can't separate them. If you are short on
   time, skip this step: buckets alone are still useful.
5. Use `advisor_notes` for anything worth telling us. If the quote is **not** really a clause of the
   stated category, start the note with `WRONG CATEGORY` and still rate the text as it stands.
   Leave both `advisor_bucket` and `advisor_rank` blank for a clause you can't judge.
6. Save as CSV (UTF-8 if offered) and send the file back. Sorting the rows is fine at this point.

Thank you. Your rankings are the only reference we have for whether the tool's risk scores match
an expert's judgement, so please answer independently of anything you have seen from the tool.
"""


def write_packet(output: Path, packet: pd.DataFrame, key: PacketKey) -> None:
    """Write the packet CSV (UTF-8 with BOM so Excel reads accents), the key JSON and the instructions."""
    output.mkdir(parents=True, exist_ok=True)
    packet.to_csv(output / PACKET_FILE, index=False, encoding="utf-8-sig")
    write_key(key, output / KEY_FILE)
    (output / INSTRUCTIONS_FILE).write_text(instructions(key.packet_id, len(key.items)), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Rebuild the validation chunks, score the cached predictions and write the packet, key and instructions."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--probabilities", type=Path, default=CACHE_DIR / "word_char_tfidf_evaluation.npz",
                        help=".npz with y_prob (and y_true, checked against the rebuilt chunks)")
    parser.add_argument("--thresholds", type=Path, default=CACHE_DIR / "word_char_tfidf_thresholds.json")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH,
                        help="weight set the key records scores under (rescoring later can use any other)")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "advisor_packet")
    parser.add_argument("--n-items", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="sampling and shuffling seed")
    parser.add_argument("--max-per-category", type=int, default=3,
                        help="clauses per category within one bucket (raised for a bucket with few categories)")
    parser.add_argument("--max-per-contract", type=int, default=3)
    parser.add_argument("--max-similarity", type=float, default=0.7,
                        help="skip a quote whose word Jaccard with a chosen quote is at least this")
    parser.add_argument("--context-chars", type=int, default=300)
    parser.add_argument("--min-quote-words", type=int, default=8, help="skip shorter quotes (bare headings)")
    args = parser.parse_args(argv)

    # Rebuild the validation chunks exactly as the cache was produced, and check they line up
    _, val = split_by_contract(load_cuad(TRAIN_PATH), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)
    labeled = prepare_contracts(val, tokenizer, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    probabilities, saved_labels = _load_cache(args.probabilities)
    if saved_labels is not None and (saved_labels != to_model_inputs(labeled)[1]).any():
        raise ValueError("Cache labels don't match the rebuilt chunks: wrong split, chunker or settings")

    # Predicted clauses, scored under the chosen weight set
    ws = load_weight_set(args.weights)
    texts = {c.contract_id: c.text for c in val}
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    clauses = clauses_from_probabilities(texts, [lc.chunk for lc in labeled], probabilities, thresholds,
                                         load_category_cues())
    scores = score_contract_clauses(clauses, ws)
    detected: dict[str, set[str]] = {}
    for clause in clauses:
        detected.setdefault(clause.contract_id, set()).add(clause.category)

    # Sample, then write the packet, key and instructions
    candidates = grounded_candidates(clauses, scores, args.min_quote_words)
    priority = [c for c, severity in ws.category_severity.items() if severity >= HIGH_SEVERITY]
    chosen = select_clauses(candidates, args.n_items, args.seed, priority, args.max_per_category,
                            args.max_per_contract, args.max_similarity)
    settings = {
        "probabilities": _shown(args.probabilities), "thresholds": _shown(args.thresholds),
        "weights": _shown(args.weights), "seed": args.seed, "n_items_requested": args.n_items,
        "max_per_category": args.max_per_category, "max_per_contract": args.max_per_contract,
        "max_similarity": args.max_similarity, "context_chars": args.context_chars,
        "min_quote_words": args.min_quote_words,
        "split": {"method": "split_by_contract", "seed": SPLIT_SEED, "val_fraction": VAL_FRACTION},
        "chunking": {"method": "chunk_sentence_windows", "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": MAX_TOKENS,
                     "overlap_tokens": OVERLAP_TOKENS},
        "population": {"contracts": len(val), "predicted_clauses": len(clauses),
                       "grounded_candidates": len(candidates),
                       "candidate_ratings": dict(Counter(c.score.rating for c in candidates))},
        "priority_categories": sorted(priority), "test_data_used": False,
    }
    packet, key = build_packet(chosen, texts, {cid: sorted(cats) for cid, cats in detected.items()},
                               load_category_descriptions(), ws, args.seed, args.context_chars, settings)
    write_packet(args.output, packet, key)

    # Summary for the person running it (the packet itself shows none of this)
    print(json.dumps({
        "packet_id": key.packet_id, "items": len(key.items), "output": str(args.output),
        "population": settings["population"],
        "packet_ratings": dict(Counter(i.rating for i in key.items)),
        "packet_categories": dict(Counter(i.category for i in key.items).most_common()),
        "contracts": len({i.contract_id for i in key.items}),
    }, indent=2))


if __name__ == "__main__":
    main()
