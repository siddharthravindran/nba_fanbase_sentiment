"""Historical top-level Reddit comment backfill via Arctic Shift (no auth required).

Only fetches top-level comments (parent_id = the post itself, not another comment)
and drops anything below MIN_COMMENT_SCORE to keep volume manageable.
"""
import time
from datetime import datetime, timedelta, timezone

import requests

from ingest.storage import get_connection, upsert_docs, count_docs
from ingest.teams import TEAM_SUBREDDITS

BASE_URL = "https://arctic-shift.photon-reddit.com"
PAGE_LIMIT = 100
REQUEST_DELAY_SEC = 1.0
MIN_COMMENT_SCORE = 3


def _fetch_page(subreddit: str, after: int, before: int) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/api/comments/search",
        params={
            "subreddit": subreddit,
            "parent_id": "",  # top-level only (parent is the post, not another comment)
            "after": after,
            "before": before,
            "sort": "asc",
            "limit": PAGE_LIMIT,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_comments_in_range(subreddit: str, after: int, before: int):
    """Yields top-level comments for a subreddit between after/before (epoch seconds)."""
    cursor = after
    while True:
        page = _fetch_page(subreddit, cursor, before)
        if not page:
            return
        for comment in page:
            yield comment
        if len(page) < PAGE_LIMIT:
            return
        cursor = page[-1]["created_utc"] + 1
        time.sleep(REQUEST_DELAY_SEC)


def to_doc(comment: dict, team: str) -> dict:
    return {
        "id": comment["id"],
        "source": "reddit_comment",
        "team": team,
        "subreddit": comment.get("subreddit"),
        "text": comment.get("body", ""),
        "url": f"https://reddit.com{comment['permalink']}" if comment.get("permalink") else None,
        "created_utc": datetime.fromtimestamp(
            comment["created_utc"], tz=timezone.utc
        ).isoformat(),
    }


def backfill(days_back: int = 180, min_score: int = MIN_COMMENT_SCORE):
    after = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    before = int(datetime.now(timezone.utc).timestamp())
    conn = get_connection()

    for team, subreddit in TEAM_SUBREDDITS.items():
        docs = []
        try:
            for comment in fetch_comments_in_range(subreddit, after, before):
                if comment.get("score", 0) < min_score:
                    continue
                docs.append(to_doc(comment, team))
        except requests.HTTPError as e:
            print(f"[{team}] r/{subreddit} comments failed: {e}")
            continue

        upsert_docs(conn, docs)
        print(f"[{team}] r/{subreddit}: {len(docs)} comments ingested (score >= {min_score})")

    print(f"Total docs in DB: {count_docs(conn)}")
    conn.close()


if __name__ == "__main__":
    backfill(days_back=180)
