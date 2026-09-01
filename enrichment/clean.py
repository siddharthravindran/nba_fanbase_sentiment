"""Filter and clean raw_docs -> clean_docs before sentiment scoring / embedding.

Drops: [deleted]/[removed] posts, empty/too-short text, AutoModerator boilerplate.
Cleans: markdown quote markers, link syntax, excess whitespace.
"""
import hashlib
import re

from ingest.storage import (
    count_docs,
    get_connection,
    record_rejected_docs,
    upsert_clean_docs,
)

MIN_LENGTH = 15

REMOVED_MARKERS = ("[deleted]", "[removed]")
BOT_SIGNATURES = (
    "i am a bot, and this action was performed automatically",
    "action was performed automatically",
)

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [label](url) -> label
QUOTE_MARKER_RE = re.compile(r"^>\s?", re.MULTILINE)
WHITESPACE_RE = re.compile(r"\n{3,}")

# Identifies the rules a rejection was made under, so a previously-rejected doc
# is reconsidered when - and only when - those rules change. Derived from the
# parameters themselves rather than hand-incremented, because a version someone
# has to remember to bump is a version that silently goes stale. The cleaning
# regexes are included because rejection happens *after* cleaning: widening
# MARKDOWN_LINK_RE changes the post-clean length, and so changes what is too
# short to keep.
FILTER_VERSION = hashlib.sha256(
    repr(
        (
            MIN_LENGTH,
            REMOVED_MARKERS,
            BOT_SIGNATURES,
            MARKDOWN_LINK_RE.pattern,
            QUOTE_MARKER_RE.pattern,
            WHITESPACE_RE.pattern,
        )
    ).encode()
).hexdigest()[:12]


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
    # Skip rows already kept (in clean_docs) and rows already rejected under
    # these same filters. Kept alone is not enough: a reject leaves no trace in
    # clean_docs, so "not in clean_docs" silently means "new OR rejected", and
    # every reject in the corpus was being re-read and re-filtered nightly.
    rows = conn.execute(
        """
        SELECT r.id, r.source, r.team, r.subreddit, r.text, r.url, r.created_utc
        FROM raw_docs r
        LEFT JOIN clean_docs c ON r.id = c.id
        LEFT JOIN rejected_docs j ON r.id = j.id AND j.filter_version = ?
        WHERE c.id IS NULL AND j.id IS NULL
        """,
        (FILTER_VERSION,),
    ).fetchall()
    columns = ["id", "source", "team", "subreddit", "text", "url", "created_utc"]

    kept, rejected = [], []
    for row in rows:
        doc = dict(zip(columns, row))
        cleaned = clean_text(doc["text"])
        if is_garbage(cleaned):
            rejected.append(doc["id"])
            continue
        doc["text"] = cleaned
        kept.append(doc)

    upsert_clean_docs(conn, kept)
    record_rejected_docs(conn, rejected, FILTER_VERSION)

    # Reporting the skip count is what makes the next anomaly legible: a run
    # that suddenly examines 300k rows instead of 2k is either a filter change
    # or a regression, and the two are indistinguishable without this line.
    skipped = conn.execute(
        "SELECT COUNT(*) FROM rejected_docs WHERE filter_version = ?",
        (FILTER_VERSION,),
    ).fetchone()[0]
    print(f"Cleaned {len(rows)} new raw docs -> kept {len(kept)}, dropped {len(rejected)}")
    print(f"Skipped {skipped} previously rejected (filter version {FILTER_VERSION})")
    print(f"Total in clean_docs: {count_docs(conn, 'clean_docs')}")
    conn.close()


if __name__ == "__main__":
    clean_all()
