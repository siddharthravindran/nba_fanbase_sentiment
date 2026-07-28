"""Export clean_docs (+ sentiment) id/text pairs to JSONL for GPU embedding on Colab.

Chroma's default embedding function is sentence-transformers/all-MiniLM-L6-v2
with mean pooling + L2 normalization - Colab should replicate that exactly
(see the paired Colab embedding script) so vectors match what local
queries will compute at retrieval time.
"""
import json
import sqlite3

DB_PATH = "data/raw_docs.db"
OUT_PATH = "data/embed_export.jsonl"

QUERY = """
SELECT c.id, c.text
FROM clean_docs c
JOIN sentiment_docs_v2 s ON s.id = c.id
"""

conn = sqlite3.connect(DB_PATH)
rows = conn.execute(QUERY).fetchall()
print(f"Exporting {len(rows):,} docs to {OUT_PATH}...")

with open(OUT_PATH, "w") as f:
    for doc_id, text in rows:
        f.write(json.dumps({"id": doc_id, "text": text}) + "\n")

print(f"Done. Upload {OUT_PATH} to Google Drive for the Colab embedding step.")
conn.close()
