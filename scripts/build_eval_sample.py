"""Draw a human-labeling sample for evaluating the v2 emotion classifier.

Why this exists: v2's training labels came from a zero-shot `bart-large-mnli`
teacher (see `enrichment/zero_shot_labels.py`), which was the only tractable
option at 3.5M documents. The consequence is that no human-labeled data exists
anywhere in the pipeline, so a held-out split would measure how faithfully the
student reproduces the teacher - not whether either one is right. The only way
to get an accuracy number is to label documents by hand, and this builds the set
to label.

Three design decisions worth stating, because each one is a way the result could
otherwise be quietly wrong:

**The window is time-disjoint from training.** The training export was taken
2026-07-22 and scoring ran 07-27, so documents *created* after that were never
seen. Filtering on `created_utc` (not `cleaned_at`) is the conservative choice:
backfills insert old documents long after the fact, and an ingest-time filter
would let them through.

**Two strata, because one sample can't answer both questions.** Sampling evenly
across predicted labels is the only way to say anything about `sadness` (0.8% of
the window - a 600-document random draw would contain about five). But an even
draw is not the corpus, so its raw accuracy is not the corpus accuracy. A second
prevalence-representative sample is drawn alongside it to carry the headline
number, and `weight` on each stratified row reweights it back to true prevalence
when the two are pooled.

**The labeling file does not contain the model's prediction.** Anchoring is
real, and a labeler shown "mockery or sarcasm" will agree with it more often
than they should - which inflates exactly the number being measured. The
prediction lives in a separate key file, joined back after labeling is done.
"""
import argparse
import csv
import random
import re
from pathlib import Path

from ingest.storage import get_connection

# The v2 scoring run. Documents created after this were not in the training
# export, so they are genuinely unseen by the model.
TRAIN_CUTOFF = "2026-07-27"

LABELS = [
    "excitement or hype",
    "anger or frustration",
    "disappointment",
    "pessimism or resignation",
    "hope or optimism",
    "pride",
    "mockery or sarcasm",
    "sadness",
    "neutral analysis or discussion",
]

# Enough per class for a per-class precision estimate with a usable interval;
# at n=50 a point estimate near 0.7 carries roughly +/-13pp at 95%. Going much
# higher multiplies labeling time across nine classes for a tighter interval
# than the underlying human disagreement justifies.
PER_LABEL = 50

# Carries the prevalence-weighted headline number and the only unbiased view of
# what the model does to a typical document.
RANDOM_N = 150

# Labeled by a second person to establish how much two NBA fans agree with each
# other. This is the ceiling: a model scoring near the human-human agreement
# rate is not "70% accurate", it is at the limit of what the label set can
# express. Without this the headline number has no denominator.
OVERLAP_N = 50


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def fetch_window(conn):
    """Every scored fan document created after the training cutoff."""
    rows = conn.execute(
        """
        SELECT d.id, d.team, d.created_utc, d.url, s.top_emotion, s.top_score, d.text
        FROM sentiment_docs_v2 s
        JOIN clean_docs d ON d.id = s.id
        WHERE d.source != 'article'
          AND d.created_utc > ?
          AND s.top_emotion IS NOT NULL
        """,
        (TRAIN_CUTOFF,),
    ).fetchall()

    # Deduplicate on normalized text. Game threads are full of documents whose
    # entire content is "LETS GO" or a player's name, and the same string
    # recurring 40 times would eat 40 labeling slots to measure one decision.
    seen, out = set(), []
    for doc_id, team, ts, url, label, score, text in rows:
        clean = normalize(text)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "doc_id": doc_id,
                "team": team,
                "created_utc": ts,
                "url": url,
                "v2_label": label,
                "v2_score": round(float(score), 4),
                "text": clean,
                "char_len": len(clean),
            }
        )
    return out


def build(docs, rng, per_label: int, random_n: int):
    by_label: dict[str, list] = {}
    for doc in docs:
        by_label.setdefault(doc["v2_label"], []).append(doc)

    sample, taken = [], set()

    for label in LABELS:
        pool = by_label.get(label, [])
        picked = rng.sample(pool, min(per_label, len(pool)))
        # Reweights this stratum back to corpus prevalence. Sampling 50 of 496
        # sadness documents and 50 of 16,983 mockery documents means each
        # sadness row stands for ~10 documents and each mockery row for ~340;
        # averaging them unweighted would overstate the rare classes enormously.
        weight = len(pool) / len(picked) if picked else 0.0
        for doc in picked:
            taken.add(doc["doc_id"])
            sample.append({**doc, "stratum": "by_label", "weight": round(weight, 2)})

    # Drawn from documents not already taken, so the two strata stay disjoint
    # and a row is never labeled twice under two different weights.
    remaining = [d for d in docs if d["doc_id"] not in taken]
    for doc in rng.sample(remaining, min(random_n, len(remaining))):
        sample.append({**doc, "stratum": "random", "weight": 1.0})

    # Shuffled so the labeler never sees a run of one predicted class. Labeling
    # nine sadness documents in a row primes the next judgment toward sadness,
    # and the ordering would otherwise leak the prediction the key file is
    # deliberately withholding.
    rng.shuffle(sample)
    for i, row in enumerate(sample, 1):
        row["sample_id"] = i
        # The overlap block is the leading rows rather than a random subset, so
        # a second labeler is told "do the first 50" and nothing more.
        row["overlap"] = 1 if i <= OVERLAP_N else 0
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-label", type=int, default=PER_LABEL)
    parser.add_argument("--random-n", type=int, default=RANDOM_N)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--outdir", type=Path, default=Path("data/eval"))
    args = parser.parse_args()

    conn = get_connection()
    try:
        docs = fetch_window(conn)
    finally:
        conn.close()

    rng = random.Random(args.seed)
    sample = build(docs, rng, args.per_label, args.random_n)

    args.outdir.mkdir(parents=True, exist_ok=True)
    blind_path = args.outdir / "label_sample.csv"
    key_path = args.outdir / "label_sample_key.csv"

    # What the labeler opens: the text and nowhere to see what the model said.
    with blind_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sample_id", "overlap", "text", "human_label", "notes"]
        )
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "overlap": row["overlap"],
                    "text": row["text"],
                    "human_label": "",
                    "notes": "",
                }
            )

    # Everything needed to score the labels, joined back on sample_id.
    with key_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id", "doc_id", "stratum", "weight", "v2_label",
                "v2_score", "char_len", "team", "created_utc", "url",
            ],
        )
        writer.writeheader()
        for row in sample:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    counts: dict[str, int] = {}
    for row in sample:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    print(f"window: {len(docs):,} deduped fan docs created after {TRAIN_CUTOFF}")
    print(f"sample: {len(sample)} rows " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"overlap block: first {OVERLAP_N} rows (second labeler does these)")
    print(f"  blind -> {blind_path}")
    print(f"  key   -> {key_path}")


if __name__ == "__main__":
    main()
