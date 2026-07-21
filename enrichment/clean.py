"""Filter and clean raw_docs -> clean_docs before sentiment scoring / embedding.

Drops: [deleted]/[removed] posts, empty/too-short text, AutoModerator boilerplate.
Cleans: markdown quote markers, link syntax, excess whitespace.
"""
import re

from ingest.storage import get_connection, upsert_clean_docs, count_docs

MIN_LENGTH = 15

REMOVED_MARKERS = ("[deleted]", "[removed]")
BOT_SIGNATURES = (
    "i am a bot, and this action was performed automatically",
    "action was performed automatically",
)

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [label](url) -> label
QUOTE_MARKER_RE = re.compile(r"^>\s?", re.MULTILINE)
WHITESPACE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = QUOTE_MARKER_RE.sub("", text)
    text = WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


def is_garbage(text: str) -> bool:
    lowered = text.lower()
    if any(marker in text for marker in REMOVED_MARKERS):
        return True
    if any(sig in lowered for sig in BOT_SIGNATURES):
        return True
    if len(text.strip()) < MIN_LENGTH:
        return True
    return False


def clean_all():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, source, team, subreddit, text, url, created_utc FROM raw_docs"
    ).fetchall()
    columns = ["id", "source", "team", "subreddit", "text", "url", "created_utc"]

    kept, dropped = [], 0
    for row in rows:
        doc = dict(zip(columns, row))
        cleaned = clean_text(doc["text"])
        if is_garbage(cleaned):
            dropped += 1
            continue
        doc["text"] = cleaned
        kept.append(doc)

    upsert_clean_docs(conn, kept)
    print(f"Cleaned {len(rows)} raw docs -> kept {len(kept)}, dropped {dropped}")
    print(f"Total in clean_docs: {count_docs(conn, 'clean_docs')}")
    conn.close()


if __name__ == "__main__":
    clean_all()
