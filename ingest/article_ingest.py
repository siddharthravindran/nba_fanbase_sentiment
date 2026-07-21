"""Pull NBA articles from RSS feeds and extract clean text."""
import feedparser
import trafilatura

RSS_FEEDS = [
    "https://www.espn.com/espn/rss/nba/news",
    # TODO: add team beat writer feeds, Bleacher Report, etc.
]


def fetch_articles(feed_url: str):
    parsed = feedparser.parse(feed_url)
    for entry in parsed.entries:
        downloaded = trafilatura.fetch_url(entry.link)
        text = trafilatura.extract(downloaded) if downloaded else None
        if not text:
            continue
        yield {
            "source": "article",
            "url": entry.link,
            "title": entry.title,
            "text": text,
            "published": entry.get("published"),
        }


if __name__ == "__main__":
    for feed in RSS_FEEDS:
        for article in fetch_articles(feed):
            print(article["title"])
