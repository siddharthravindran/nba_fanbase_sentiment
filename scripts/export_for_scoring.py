"""Export clean_docs rows not yet scored to JSONL for GPU sentiment scoring
on Colab (mirrors export_for_embedding.py's pattern for the same reason:
CPU single-example inference proved too slow for a 1M+ backlog - see
colab_score.py for the batched GPU scoring step)."""
import json
import sqlite3

DB_PATH = "data/raw_docs.db"
OUT_PATH = "data/score_export.jsonl"

QUERY = """
SELECT id, text FROM clean_docs
WHERE id NOT IN (SELECT id FROM sentiment_docs_v2)
"""

conn = sqlite3.connect(DB_PATH)
rows = conn.execute(QUERY).fetchall()
print(f"Exporting {len(rows):,} unscored docs to {OUT_PATH}...")

with open(OUT_PATH, "w") as f:
    for doc_id, text in rows:
        f.write(json.dumps({"id": doc_id, "text": text}) + "\n")

print(f"Done. Upload {OUT_PATH} to Google Drive for the Colab scoring step.")
conn.close()
