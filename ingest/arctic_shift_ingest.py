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
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 5.0  # doubles each retry: 5s, 10s, 20s, 40s


def _fetch_page(subreddit: str, after: int, before: int) -> list[dict]:
    """Fetch one page, retrying on transient errors: Arctic Shift intermittently
    returns 422 under load even for valid, previously-working requests, and
    occasionally times out at the network level (ReadTimeout/ConnectionError)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
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
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
    return []


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
        total = 0
        batch = []
        try:
            for post in fetch_posts_in_range(subreddit, after, before):
                batch.append(to_doc(post, team))
                if len(batch) >= PAGE_LIMIT:
                    upsert_docs(conn, batch)
                    total += len(batch)
                    batch = []
        except requests.exceptions.RequestException as e:
            if batch:
                upsert_docs(conn, batch)
                total += len(batch)
            print(f"[{team}] r/{subreddit} stopped early after {total} posts: {e}")
            continue

        if batch:
            upsert_docs(conn, batch)
            total += len(batch)
        print(f"[{team}] r/{subreddit}: {total} posts ingested")

    print(f"Total docs in DB: {count_docs(conn)}")
    conn.close()


if __name__ == "__main__":
    backfill(days_back=180)
