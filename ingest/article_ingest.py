"""Pull NBA articles from RSS feeds, extract clean text via trafilatura, and
tag each article with the team(s) it's about (articles aren't scoped to a
single subreddit the way Reddit posts are, so team is detected from text)."""
import hashlib
import re
import time
from datetime import datetime, timezone

import feedparser
import requests
import trafilatura

from ingest.storage import get_connection, upsert_docs, count_docs
from ingest.teams import TEAM_ALIASES

RSS_FEEDS = [
    "https://www.espn.com/espn/rss/nba/news",
    "https://sports.yahoo.com/nba/rss/",
    "https://www.cbssports.com/rss/headlines/nba/",
    # TODO: add Bleacher Report / The Ringer / RealGM feeds if a working URL
    # is ever confirmed (couldn't verify a working path for any of them -
    # sites either 404'd or didn't expose an RSS <link> tag).
]

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 3.0  # doubles each retry: 3s, 6s, 12s

# Word-boundary, case-sensitive patterns (case-sensitive to cut down on false
# positives from common-word nicknames like "Heat", "Jazz", "Magic", "Kings").
_TEAM_PATTERNS = {
    team: [re.compile(rf"\b{re.escape(alias)}\b") for alias in aliases]
    for team, aliases in TEAM_ALIASES.items()
}


def detect_teams(text: str) -> list[str]:
    return [team for team, patterns in _TEAM_PATTERNS.items() if any(p.search(text) for p in patterns)]


def _fetch_text(url: str) -> str | None:
    """Download + extract article body, retrying on transient network errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            downloaded = trafilatura.fetch_url(url)
            return trafilatura.extract(downloaded) if downloaded else None
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
    return None


def _article_id(url: str, team: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    team_slug = team.lower().replace(" ", "_")
    return f"article_{digest}_{team_slug}"


def _parsed_time_to_iso(entry) -> str | None:
    struct = entry.get("published_parsed")
    if not struct:
        return None
    return datetime(*struct[:6], tzinfo=timezone.utc).isoformat()


def fetch_articles(feed_url: str, seen_urls: set[str]):
    parsed = feedparser.parse(feed_url)
    for entry in parsed.entries:
        url = entry.link
        if url in seen_urls:
            continue
        seen_urls.add(url)

        text_body = _fetch_text(url)
        if not text_body:
            continue

        title = entry.get("title", "")
        full_text = f"{title}\n{text_body}".strip()
        teams = detect_teams(full_text)
        if not teams:
            continue  # not clearly about any tracked team; skip rather than store untagged

        yield {
            "url": url,
            "text": full_text,
            "created_utc": _parsed_time_to_iso(entry),
            "teams": teams,
        }


def to_docs(article: dict) -> list[dict]:
    """One row per matched team so the article surfaces under each relevant team's stats/retrieval."""
    return [
        {
            "id": _article_id(article["url"], team),
            "source": "article",
            "team": team,
            "subreddit": None,
            "text": article["text"],
            "url": article["url"],
            "created_utc": article["created_utc"],
        }
        for team in article["teams"]
    ]


def ingest_all():
    conn = get_connection()
    existing_urls = {
        row[0] for row in conn.execute("SELECT DISTINCT url FROM raw_docs WHERE source = 'article'")
    }

    total_articles = 0
    total_docs = 0

    for feed_url in RSS_FEEDS:
        batch = []
        for article in fetch_articles(feed_url, existing_urls):
            batch.extend(to_docs(article))
            total_articles += 1

        if batch:
            upsert_docs(conn, batch)
            total_docs += len(batch)
        print(f"[{feed_url}] {total_articles} articles ingested, {len(batch)} team-tagged rows")

    print(f"Total article rows this run: {total_docs}")
    print(f"Total docs in DB: {count_docs(conn)}")
    conn.close()


if __name__ == "__main__":
    ingest_all()
