"""Nightly incremental ingest: posts + comments, all 30 teams, rolling
lookback window (not "since last run").

A rolling window (not a hard cutoff at last-run time) matters for two
reasons:
1. arctic_shift_comments_ingest filters out comments below MIN_COMMENT_SCORE
   at fetch time with no later re-check - a comment posted last night won't
   have 3 upvotes yet, so a narrow "since yesterday" window would permanently
   miss it. A multi-day window gives comments time to accumulate score
   before we capture them.
2. upsert is idempotent (ON CONFLICT DO NOTHING), so re-covering the last
   few days on every run is free - if a night's run is skipped (machine
   asleep/offline), the next run's window still covers the gap and the
   pipeline self-heals with no manual backfill.
"""
import sys
from datetime import datetime, timedelta, timezone

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

LOOKBACK_DAYS = 3


def ingest_posts(after: int, before: int):
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
        print(f"[{team}] r/{subreddit}: {total} posts upserted")

    print(f"Total docs in raw_docs: {count_docs(conn)}")
    conn.close()


def ingest_comments(after: int, before: int):
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
        print(f"[{team}] r/{subreddit}: {total} comments upserted (score >= {MIN_COMMENT_SCORE})")

    print(f"Total docs in raw_docs: {count_docs(conn)}")
    conn.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    after = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp())
    before = int(datetime.now(timezone.utc).timestamp())

    if which in ("posts", "both"):
        print(f"=== Ingesting posts: last {LOOKBACK_DAYS} days ===")
        ingest_posts(after, before)
    if which in ("comments", "both"):
        print(f"=== Ingesting comments: last {LOOKBACK_DAYS} days ===")
        ingest_comments(after, before)
