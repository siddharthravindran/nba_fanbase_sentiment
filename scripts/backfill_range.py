"""One-off historical backfill for an explicit date range (posts + comments),
used to expand data coverage without re-fetching dates already ingested.

Reuses the same fetch/retry/to_doc logic as ingest/arctic_shift_ingest.py and
ingest/arctic_shift_comments_ingest.py, just parameterized with an explicit
[START, END) window instead of "N days back from now".

Usage: python -m scripts.backfill_range [posts|comments|both]
       python -m scripts.backfill_range both --start 2026-07-29 --end 2026-08-29

Defaults cover the original July-2025 season backfill. Pass --start/--end to
heal a specific gap - e.g. after the pipeline sits paused for weeks, the
nightly job's short rolling lookback can't reach back far enough on its own.
"""
import argparse
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


def backfill_posts(start: datetime = None, end: datetime = None):
    start, end = start or START, end or END
    after, before = int(start.timestamp()), int(end.timestamp())
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


def backfill_comments(start: datetime = None, end: datetime = None):
    start, end = start or START, end or END
    after, before = int(start.timestamp()), int(end.timestamp())
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


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="both", choices=["posts", "comments", "both"])
    parser.add_argument("--start", type=_parse_date, default=START, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", type=_parse_date, default=END, help="YYYY-MM-DD (exclusive)")
    args = parser.parse_args()

    if args.which in ("posts", "both"):
        print(f"=== Backfilling posts: {args.start.date()} to {args.end.date()} ===")
        backfill_posts(args.start, args.end)
    if args.which in ("comments", "both"):
        print(f"=== Backfilling comments: {args.start.date()} to {args.end.date()} ===")
        backfill_comments(args.start, args.end)
