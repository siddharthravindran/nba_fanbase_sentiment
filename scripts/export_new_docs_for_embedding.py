"""Export only the docs newly added by the July 2025 backfill for GPU
embedding, instead of re-exporting all 3.4M docs (export_for_embedding.py's
full join would re-embed ~2M docs already in Chroma).

Scopes to exactly the ids present in the sentiment_shard_*.npz files just
loaded into sentiment_docs_v2 by load_sentiment_shards.py - these are, by
construction, the docs that weren't in clean_docs at the time of the last
Chroma population and so aren't embedded yet.
"""
import glob
import json
import sqlite3

import numpy as np

DB_PATH = "data/raw_docs.db"
SHARDS_GLOB = "data/sentiment_shard_*.npz"
OUT_PATH = "data/embed_export_new.jsonl"

shard_paths = sorted(glob.glob(SHARDS_GLOB))
if not shard_paths:
    raise FileNotFoundError(f"No shard files found matching {SHARDS_GLOB}")

all_ids = []
for path in shard_paths:
    data = np.load(path, allow_pickle=True)
    all_ids.append(data["ids"])
new_ids = np.concatenate(all_ids).tolist()
print(f"Found {len(new_ids):,} newly-scored doc ids")

conn = sqlite3.connect(DB_PATH)
rows = []
BATCH = 5000
for start in range(0, len(new_ids), BATCH):
    batch_ids = new_ids[start:start + BATCH]
    placeholders = ",".join("?" for _ in batch_ids)
    rows.extend(conn.execute(
        f"SELECT id, text FROM clean_docs WHERE id IN ({placeholders})", batch_ids
    ).fetchall())

print(f"Exporting {len(rows):,} docs to {OUT_PATH}...")
with open(OUT_PATH, "w") as f:
    for doc_id, text in rows:
        f.write(json.dumps({"id": doc_id, "text": text}) + "\n")

print(f"Done. Upload {OUT_PATH} to Google Drive for the Colab embedding step.")
conn.close()
