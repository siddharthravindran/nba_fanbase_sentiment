"""Historical Reddit post backfill via Arctic Shift (no auth required).

Docs: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
"""
import time
from datetime import datetime, timedelta, timezone

import requests

from ingest.storage import get_connection, upsert_docs, count_docs
from ingest.teams import TEAM_SUBREDDITS

BASE_URL = "https://arctic-shift.photon-reddit.com"
PAGE_LIMIT = 100
REQUEST_DELAY_SEC = 1.0  # be polite; no documented rate limit but avoid hammering


def _fetch_page(subreddit: str, after: int, before: int) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/api/posts/search",
        params={
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "sort": "asc",
            "limit": PAGE_LIMIT,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_posts_in_range(subreddit: str, after: int, before: int):
    """Yields posts for a subreddit between after/before (epoch seconds), paginating via created_utc cursor."""
    cursor = after
    while True:
        page = _fetch_page(subreddit, cursor, before)
        if not page:
            return
        for post in page:
            yield post
        if len(page) < PAGE_LIMIT:
            return
        # advance cursor 1 second past the last post's timestamp to avoid re-fetching it
        cursor = page[-1]["created_utc"] + 1
        time.sleep(REQUEST_DELAY_SEC)


def to_doc(post: dict, team: str) -> dict:
    return {
        "id": post["id"],
        "source": "reddit",
        "team": team,
        "subreddit": post.get("subreddit"),
        "text": f"{post.get('title', '')}\n{post.get('selftext', '') or ''}".strip(),
        "url": post.get("url") or f"https://reddit.com/r/{post.get('subreddit')}/comments/{post['id']}",
        "created_utc": datetime.fromtimestamp(
            post["created_utc"], tz=timezone.utc
        ).isoformat(),
    }


def backfill(days_back: int = 180):
    after = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    before = int(datetime.now(timezone.utc).timestamp())
    conn = get_connection()

    for team, subreddit in TEAM_SUBREDDITS.items():
        docs = []
        try:
            for post in fetch_posts_in_range(subreddit, after, before):
                docs.append(to_doc(post, team))
        except requests.HTTPError as e:
            print(f"[{team}] r/{subreddit} failed: {e}")
            continue

        upsert_docs(conn, docs)
        print(f"[{team}] r/{subreddit}: {len(docs)} posts ingested")

    print(f"Total docs in DB: {count_docs(conn)}")
    conn.close()


if __name__ == "__main__":
    backfill(days_back=180)
