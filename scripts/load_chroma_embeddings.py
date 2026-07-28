"""Load GPU-computed embeddings (from colab_embed.py) into local Chroma,
attaching team/source/sentiment metadata from SQLite.

Reads all embeddings_shard_*.npz files in data/ (colab_embed.py saves in
resumable chunks) and skips the slow local embedding step entirely - just
batched DB writes.
"""
import glob
import sqlite3

import numpy as np

from retrieval.vector_store import get_collection, add_docs

DB_PATH = "data/raw_docs.db"
SHARDS_GLOB = "data/embeddings_shard_*.npz"
BATCH_SIZE = 2000


def load_all_shards():
    shard_paths = sorted(glob.glob(SHARDS_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found matching {SHARDS_GLOB}")
    print(f"Found {len(shard_paths)} shard files")

    all_ids, all_embeddings = [], []
    for path in shard_paths:
        data = np.load(path, allow_pickle=True)
        all_ids.append(data["ids"])
        all_embeddings.append(data["embeddings"])

    return np.concatenate(all_ids), np.concatenate(all_embeddings)


def run():
    ids, embeddings = load_all_shards()
    print(f"Loaded {len(ids):,} precomputed embeddings")

    conn = sqlite3.connect(DB_PATH)
    collection = get_collection()

    total = 0
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE].tolist()
        batch_embeddings = embeddings[start:start + BATCH_SIZE].tolist()

        placeholders = ",".join("?" for _ in batch_ids)
        rows = conn.execute(
            f"""
            SELECT c.id, c.source, c.team, c.text, c.created_utc, s.top_emotion, s.top_score
            FROM clean_docs c
            JOIN sentiment_docs_v2 s ON s.id = c.id
            WHERE c.id IN ({placeholders})
            """,
            batch_ids,
        ).fetchall()
        meta_by_id = {r[0]: r for r in rows}

        docs, ordered_embeddings = [], []
        for doc_id, emb in zip(batch_ids, batch_embeddings):
            row = meta_by_id.get(doc_id)
            if row is None:
                continue  # doc removed/changed since export
            _, source, team, text, created_utc, top_emotion, top_score = row
            docs.append({
                "id": doc_id,
                "text": text,
                "team": team,
                "source": source,
                "top_emotion": top_emotion,
                "top_score": top_score,
                "created_utc": created_utc,
            })
            ordered_embeddings.append(emb)

        if docs:
            add_docs(collection, docs, embeddings=ordered_embeddings)
            total += len(docs)

        if total % 20000 < BATCH_SIZE:
            print(f"  loaded {total:,}/{len(ids):,}")

    print(f"Done. Loaded {total:,} docs into Chroma.")
    print(f"Collection count: {collection.count()}")
    conn.close()


if __name__ == "__main__":
    run()
