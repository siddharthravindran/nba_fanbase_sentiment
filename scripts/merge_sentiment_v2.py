import json
import sqlite3

DB_PATH = "data/raw_docs.db"
JSONL_PATH = "data/sentiment_v2_scored.jsonl"
BATCH_SIZE = 5000

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS sentiment_docs_v2")
cur.execute("""
    CREATE TABLE sentiment_docs_v2 (
        id TEXT PRIMARY KEY,
        team TEXT,
        top_emotion TEXT,
        top_score REAL,
        all_scores TEXT
    )
""")

batch = []
total = 0

with open(JSONL_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        batch.append((
            record["id"],
            record["team"],
            record["top_emotion"],
            record["top_score"],
            record["all_scores"],
        ))
        if len(batch) >= BATCH_SIZE:
            cur.executemany(
                "INSERT OR REPLACE INTO sentiment_docs_v2 (id, team, top_emotion, top_score, all_scores) VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            total += len(batch)
            batch = []
            if total % 100000 == 0:
                conn.commit()
                print(f"Inserted {total:,} rows")

if batch:
    cur.executemany(
        "INSERT OR REPLACE INTO sentiment_docs_v2 (id, team, top_emotion, top_score, all_scores) VALUES (?, ?, ?, ?, ?)",
        batch,
    )
    total += len(batch)

conn.commit()

cur.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_v2_team ON sentiment_docs_v2(team)")
conn.commit()

print(f"Done. {total:,} rows inserted into sentiment_docs_v2")

cur.execute("SELECT top_emotion, COUNT(*) FROM sentiment_docs_v2 GROUP BY top_emotion ORDER BY COUNT(*) DESC")
print("\nLabel distribution:")
for label, count in cur.fetchall():
    print(f"  {label}: {count:,} ({count/total:.1%})")

conn.close()
