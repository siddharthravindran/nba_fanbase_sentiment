"""Score clean_docs rows not yet present in sentiment_docs_v2 and upsert them.

Run after re-ingesting/re-cleaning new data, instead of re-scoring the full
dataset from scratch. Batches on the GPU when one is available (Apple Silicon
MPS or CUDA), which is fast enough that a normal incremental backlog no longer
needs the Colab export/import round trip.
"""
import json
import sqlite3

from enrichment.sentiment_enrich import load_model, pick_device, score_batch

DB_PATH = "data/raw_docs.db"
BATCH_SIZE = 128  # tokenizer pads to the longest item, so keep batches modest
COMMIT_EVERY = 2000

conn = sqlite3.connect(DB_PATH, timeout=60)
cur = conn.cursor()

# NOT EXISTS rather than NOT IN: the latter can't use the index on a table
# this size and turns the lookup into a repeated scan.
rows = cur.execute(
    """
    SELECT id, team, text FROM clean_docs cd
    WHERE NOT EXISTS (SELECT 1 FROM sentiment_docs_v2 s WHERE s.id = cd.id)
    """
).fetchall()
print(f"Scoring {len(rows):,} unscored rows...")

tokenizer, model = load_model()
device = pick_device()
model = model.to(device)
print(f"Using device: {device}")

pending = []
total = 0
for start in range(0, len(rows), BATCH_SIZE):
    chunk = rows[start : start + BATCH_SIZE]
    results = score_batch([text for _, _, text in chunk], tokenizer, model, device)
    for (doc_id, team, _), result in zip(chunk, results):
        pending.append(
            (
                doc_id,
                team,
                result["top_emotion"],
                result["top_score"],
                json.dumps(result["scores"]),
            )
        )

    if len(pending) >= COMMIT_EVERY:
        cur.executemany(
            "INSERT OR REPLACE INTO sentiment_docs_v2 (id, team, top_emotion, top_score, all_scores) VALUES (?, ?, ?, ?, ?)",
            pending,
        )
        conn.commit()
        total += len(pending)
        pending = []
        print(f"  scored {total:,}/{len(rows):,}", flush=True)

if pending:
    cur.executemany(
        "INSERT OR REPLACE INTO sentiment_docs_v2 (id, team, top_emotion, top_score, all_scores) VALUES (?, ?, ?, ?, ?)",
        pending,
    )
    conn.commit()
    total += len(pending)

print(f"Done. Scored {total:,} new rows.")
print(f"sentiment_docs_v2 total: {cur.execute('SELECT COUNT(*) FROM sentiment_docs_v2').fetchone()[0]:,}")
conn.close()
