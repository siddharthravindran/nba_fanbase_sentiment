"""Nightly incremental Chroma sync: embed + upsert docs that are scored
(in sentiment_docs_v2) but not yet in the Chroma collection.

Unlike the backfill's Colab-GPU embedding pipeline (needed for ~1.45M docs),
nightly volume is a few thousand docs - small enough that Chroma's default
local CPU embedding function (all-MiniLM-L6-v2, same model used in the GPU
pipeline) handles it in seconds, no GPU detour needed.

Scoped to a recent window by default (not the full 3.5M-doc table) to keep
the existence check cheap, while still being wide enough to self-heal if a
run is missed for a few days.

The window is on `cleaned_at` (when we ingested the doc), NOT `created_utc`
(when it was published). Those diverge badly for backfilled news: the GDELT
crawler walks historical windows, so an article published in May can land in
the table in August. Filtering on the publication date meant every
backfilled article was born already outside the window and could never be
picked up - which silently left ~30k articles unembedded. Pass 0 to force a
full id diff as a repair pass.
"""
import sys
from datetime import datetime, timedelta, timezone

from ingest.storage import get_connection
from retrieval.vector_store import _client, add_docs, get_team_collection

LOOKBACK_DAYS = 7
BATCH_SIZE = 200
# Chroma's id-existence lookup is an index probe, so a much larger batch than
# the embedding batch is fine here and cuts the number of round trips.
CHECK_BATCH = 5000


def run(lookback_days: int = LOOKBACK_DAYS):
    conn = get_connection()
    client = _client()

    # Diff on ids alone first. Pulling `text` for every candidate up front would
    # mean holding millions of document bodies in memory just to discard the
    # ones already embedded; hydrate only the misses instead.
    if lookback_days:
        # Match SQLite's CURRENT_TIMESTAMP format ("YYYY-MM-DD HH:MM:SS", UTC).
        # isoformat() would produce a "T" separator and a "+00:00" offset, and
        # since these are compared as strings, " " < "T" makes the comparison
        # quietly wrong rather than raising.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows = conn.execute(
            """
            SELECT c.id, c.team FROM clean_docs c
            JOIN sentiment_docs_v2 s ON s.id = c.id
            WHERE c.cleaned_at >= ?
            """,
            (cutoff,),
        ).fetchall()
        print(f"Checking {len(rows):,} docs ingested since {cutoff[:10]} against Chroma...")
    else:
        rows = conn.execute(
            """
            SELECT c.id, c.team FROM clean_docs c
            JOIN sentiment_docs_v2 s ON s.id = c.id
            """
        ).fetchall()
        print(f"Checking all {len(rows):,} scored docs against Chroma...")

    # Group by team first: each doc's home is its own team collection, so both
    # the existence check and the write have to be aimed at the right one.
    by_team: dict[str, list[str]] = {}
    for doc_id, team in rows:
        by_team.setdefault(team or "", []).append(doc_id)

    total = 0
    for team, ids in sorted(by_team.items()):
        collection = get_team_collection(client, team)

        missing = []
        for start in range(0, len(ids), CHECK_BATCH):
            batch_ids = ids[start:start + CHECK_BATCH]
            existing = set(collection.get(ids=batch_ids, include=[])["ids"])
            missing.extend(i for i in batch_ids if i not in existing)
        if not missing:
            continue
        print(f"  {team or '(unassigned)'}: {len(missing):,} missing", flush=True)

        for start in range(0, len(missing), BATCH_SIZE):
            batch_ids = missing[start:start + BATCH_SIZE]
            placeholders = ",".join("?" * len(batch_ids))
            chunk = conn.execute(
                f"""
                SELECT c.id, c.source, c.team, c.text, c.url, c.created_utc,
                       s.top_emotion, s.top_score
                FROM clean_docs c
                JOIN sentiment_docs_v2 s ON s.id = c.id
                WHERE c.id IN ({placeholders})
                """,
                batch_ids,
            ).fetchall()
            docs = [
                {
                    "id": doc_id,
                    "text": text,
                    "team": doc_team,
                    "source": source,
                    "top_emotion": top_emotion,
                    "top_score": top_score,
                    "created_utc": created_utc,
                }
                for doc_id, source, doc_team, text, url, created_utc, top_emotion, top_score
                in chunk
            ]
            # no embeddings kwarg -> Chroma computes locally on CPU
            add_docs(collection, docs)
            total += len(docs)

    print(f"Done. Added {total:,} docs to Chroma.")
    conn.close()


if __name__ == "__main__":
    # Widen the window from the command line to catch up after a gap in the
    # nightly run (0 = full diff), without loosening the default, which keeps
    # the nightly existence check cheap.
    run(int(sys.argv[1]) if len(sys.argv) > 1 else LOOKBACK_DAYS)
