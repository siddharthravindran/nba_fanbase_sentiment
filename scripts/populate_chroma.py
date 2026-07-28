"""Embed clean_docs (+ their sentiment labels) into the Chroma vector store.

Resumable: uses upsert, so re-running only re-embeds docs whose id/text
changed - safe to interrupt and restart. Supports LIMIT for a quick test run.
"""
import argparse
import sqlite3
import time

from retrieval.vector_store import get_collection, add_docs

DB_PATH = "data/raw_docs.db"
BATCH_SIZE = 200

QUERY = """
SELECT c.id, c.source, c.team, c.text, c.url, c.created_utc,
       s.top_emotion, s.top_score
FROM clean_docs c
JOIN sentiment_docs_v2 s ON s.id = c.id
"""


def run(limit: int | None = None):
    conn = sqlite3.connect(DB_PATH)
    query = QUERY + (f" LIMIT {limit}" if limit else "")
    rows = conn.execute(query).fetchall()
    print(f"Embedding {len(rows):,} docs...")

    collection = get_collection()

    batch = []
    total = 0
    start = time.time()
    for row in rows:
        doc_id, source, team, text, url, created_utc, top_emotion, top_score = row
        batch.append({
            "id": doc_id,
            "text": text,
            "team": team,
            "source": source,
            "top_emotion": top_emotion,
            "top_score": top_score,
            "created_utc": created_utc,
        })
        if len(batch) >= BATCH_SIZE:
            add_docs(collection, batch)
            total += len(batch)
            batch = []
            if total % 2000 == 0:
                elapsed = time.time() - start
                rate = total / elapsed
                print(f"  embedded {total:,}/{len(rows):,} ({rate:.1f} docs/sec)")

    if batch:
        add_docs(collection, batch)
        total += len(batch)

    elapsed = time.time() - start
    print(f"Done. Embedded {total:,} docs in {elapsed:.1f}s ({total/elapsed:.1f} docs/sec)")
    print(f"Collection count: {collection.count()}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(limit=args.limit)
