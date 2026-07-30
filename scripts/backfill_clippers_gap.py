"""One-off fix: scripts/backfill_range.py's comment backfill hit a transient
DNS resolution error partway through LA Clippers, leaving a gap between
2025-11-26 and 2026-01-23 (the rest of the league was unaffected)."""
from datetime import datetime, timezone

import requests

from ingest.arctic_shift_comments_ingest import MIN_COMMENT_SCORE
from ingest.arctic_shift_comments_ingest import PAGE_LIMIT
from ingest.arctic_shift_comments_ingest import fetch_comments_in_range
from ingest.arctic_shift_comments_ingest import to_doc
from ingest.storage import count_docs, get_connection, upsert_docs

TEAM = "LA Clippers"
SUBREDDIT = "LAClippers"
START = datetime(2025, 11, 26, tzinfo=timezone.utc)
END = datetime(2026, 1, 24, tzinfo=timezone.utc)  # 1-day overlap on each side, harmless (idempotent upsert)

after, before = int(START.timestamp()), int(END.timestamp())
conn = get_connection()

total = 0
batch = []
try:
    for comment in fetch_comments_in_range(SUBREDDIT, after, before):
        if comment.get("score", 0) < MIN_COMMENT_SCORE:
            continue
        batch.append(to_doc(comment, TEAM))
        if len(batch) >= PAGE_LIMIT:
            upsert_docs(conn, batch)
            total += len(batch)
            batch = []
except requests.exceptions.RequestException as e:
    if batch:
        upsert_docs(conn, batch)
        total += len(batch)
    print(f"Stopped early after {total}: {e}")
else:
    if batch:
        upsert_docs(conn, batch)
        total += len(batch)

print(f"Gap-filled {total} comments for {TEAM}")
print(f"Total docs in DB: {count_docs(conn)}")
conn.close()
