"""Score clean_docs rows not yet present in sentiment_docs_v2 and upsert them.

Run after re-ingesting/re-cleaning new data, instead of re-scoring the full
1.96M-row dataset from scratch.
"""
import json
import sqlite3

from enrichment.sentiment_enrich import load_model, score

DB_PATH = "data/raw_docs.db"
BATCH_SIZE = 500

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

rows = cur.execute(
    "SELECT id, team, text FROM clean_docs WHERE id NOT IN (SELECT id FROM sentiment_docs_v2)"
).fetchall()
print(f"Scoring {len(rows):,} unscored rows...")

tokenizer, model = load_model()

batch = []
total = 0
for doc_id, team, text in rows:
    result = score(text, tokenizer, model)
    batch.append((doc_id, team, result["top_emotion"], result["top_score"], json.dumps(result["scores"])))
    if len(batch) >= BATCH_SIZE:
        cur.executemany(
            "INSERT OR REPLACE INTO sentiment_docs_v2 (id, team, top_emotion, top_score, all_scores) VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        total += len(batch)
        print(f"  scored {total:,}/{len(rows):,}")
        batch = []

if batch:
    cur.executemany(
        "INSERT OR REPLACE INTO sentiment_docs_v2 (id, team, top_emotion, top_score, all_scores) VALUES (?, ?, ?, ?, ?)",
        batch,
    )
    conn.commit()
    total += len(batch)

print(f"Done. Scored {total:,} new rows.")
new_total = cur.execute("SELECT COUNT(*) FROM sentiment_docs_v2").fetchone()[0]
print(f"sentiment_docs_v2 total: {new_total:,}")
conn.close()
