"""One-off historical backfill for an explicit date range (posts + comments),
used to expand data coverage without re-fetching dates already ingested.

Reuses the same fetch/retry/to_doc logic as ingest/arctic_shift_ingest.py and
ingest/arctic_shift_comments_ingest.py, just parameterized with an explicit
[START, END) window instead of "N days back from now".

Usage: python -m scripts.backfill_range [posts|comments|both]
"""
import sys
from datetime import datetime, timezone

import requests

from ingest.arctic_shift_comments_ingest import MIN_COMMENT_SCORE
from ingest.arctic_shift_comments_ingest import PAGE_LIMIT as COMMENT_PAGE_LIMIT
from ingest.arctic_shift_comments_ingest import fetch_comments_in_range
from ingest.arctic_shift_comments_ingest import to_doc as comment_to_doc
from ingest.arctic_shift_ingest import PAGE_LIMIT as POST_PAGE_LIMIT
from ingest.arctic_shift_ingest import fetch_posts_in_range
from ingest.arctic_shift_ingest import to_doc as post_to_doc
from ingest.storage import count_docs, get_connection, upsert_docs
from ingest.teams import TEAM_SUBREDDITS

# July 2025 = start of the last NBA league year. End has a 1-day overlap with
# the existing earliest ingested date (2026-01-22) - harmless, upsert is
# idempotent (ON CONFLICT DO NOTHING).
START = datetime(2025, 7, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 23, tzinfo=timezone.utc)


def backfill_posts():
    after, before = int(START.timestamp()), int(END.timestamp())
    conn = get_connection()
    for team, subreddit in TEAM_SUBREDDITS.items():
        total = 0
        batch = []
        try:
            for post in fetch_posts_in_range(subreddit, after, before):
                batch.append(post_to_doc(post, team))
                if len(batch) >= POST_PAGE_LIMIT:
                    upsert_docs(conn, batch)
                    total += len(batch)
                    batch = []
        except requests.exceptions.RequestException as e:
            if batch:
                upsert_docs(conn, batch)
                total += len(batch)
            print(f"[{team}] r/{subreddit} posts stopped early after {total}: {e}")
            continue

        if batch:
            upsert_docs(conn, batch)
            total += len(batch)
        print(f"[{team}] r/{subreddit}: {total} posts ingested")

    print(f"Total docs in DB: {count_docs(conn)}")
    conn.close()


def backfill_comments():
    after, before = int(START.timestamp()), int(END.timestamp())
    conn = get_connection()
    for team, subreddit in TEAM_SUBREDDITS.items():
        total = 0
        batch = []
        try:
            for comment in fetch_comments_in_range(subreddit, after, before):
                if comment.get("score", 0) < MIN_COMMENT_SCORE:
                    continue
                batch.append(comment_to_doc(comment, team))
                if len(batch) >= COMMENT_PAGE_LIMIT:
                    upsert_docs(conn, batch)
                    total += len(batch)
                    batch = []
        except requests.exceptions.RequestException as e:
            if batch:
                upsert_docs(conn, batch)
                total += len(batch)
            print(f"[{team}] r/{subreddit} comments stopped early after {total}: {e}")
            continue

        if batch:
            upsert_docs(conn, batch)
            total += len(batch)
        print(f"[{team}] r/{subreddit}: {total} comments ingested (score >= {MIN_COMMENT_SCORE})")

    print(f"Total docs in DB: {count_docs(conn)}")
    conn.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("posts", "both"):
        print(f"=== Backfilling posts: {START.date()} to {END.date()} ===")
        backfill_posts()
    if which in ("comments", "both"):
        print(f"=== Backfilling comments: {START.date()} to {END.date()} ===")
        backfill_comments()
