"""Nightly incremental Chroma sync: embed + upsert docs that are scored
(in sentiment_docs_v2) but not yet in the Chroma collection.

Unlike the backfill's Colab-GPU embedding pipeline (needed for ~1.45M docs),
nightly volume is a few thousand docs - small enough that Chroma's default
local CPU embedding function (all-MiniLM-L6-v2, same model used in the GPU
pipeline) handles it in seconds, no GPU detour needed.

Scoped to a recent window (not the full 3.4M-doc table) to keep the
existence check cheap, while still being wide enough to self-heal if a
run is missed for a few days.
"""
from datetime import datetime, timedelta, timezone

from ingest.storage import get_connection
from retrieval.vector_store import add_docs, get_collection

LOOKBACK_DAYS = 7
BATCH_SIZE = 200


def run():
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    conn = get_connection()
    collection = get_collection()

    rows = conn.execute(
        """
        SELECT c.id, c.source, c.team, c.text, c.url, c.created_utc,
               s.top_emotion, s.top_score
        FROM clean_docs c
        JOIN sentiment_docs_v2 s ON s.id = c.id
        WHERE c.created_utc >= ?
        """,
        (cutoff,),
    ).fetchall()
    print(f"Checking {len(rows):,} recently-scored docs against Chroma...")

    to_add = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        batch_ids = [r[0] for r in batch_rows]
        existing = set(collection.get(ids=batch_ids, include=[])["ids"])
        for row in batch_rows:
            if row[0] not in existing:
                to_add.append(row)

    print(f"Found {len(to_add):,} docs missing from Chroma")

    total = 0
    for start in range(0, len(to_add), BATCH_SIZE):
        chunk = to_add[start:start + BATCH_SIZE]
        docs = [
            {
                "id": doc_id,
                "text": text,
                "team": team,
                "source": source,
                "top_emotion": top_emotion,
                "top_score": top_score,
                "created_utc": created_utc,
            }
            for doc_id, source, team, text, url, created_utc, top_emotion, top_score in chunk
        ]
        add_docs(collection, docs)  # no embeddings kwarg -> Chroma computes locally on CPU
        total += len(docs)

    print(f"Done. Added {total:,} docs to Chroma.")
    print(f"Collection count: {collection.count()}")
    conn.close()


if __name__ == "__main__":
    run()
