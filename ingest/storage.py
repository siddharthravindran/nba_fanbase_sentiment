"""SQLite storage for raw ingested docs, shared across ingest sources."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_docs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_docs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    team TEXT,
    subreddit TEXT,
    text TEXT NOT NULL,
    url TEXT,
    created_utc TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clean_docs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    team TEXT,
    subreddit TEXT,
    text TEXT NOT NULL,
    url TEXT,
    created_utc TEXT,
    cleaned_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.executescript(SCHEMA)
    return conn


def upsert_docs(conn: sqlite3.Connection, docs: list[dict]):
    conn.executemany(
        """
        INSERT INTO raw_docs (id, source, team, subreddit, text, url, created_utc)
        VALUES (:id, :source, :team, :subreddit, :text, :url, :created_utc)
        ON CONFLICT(id) DO NOTHING
        """,
        docs,
    )
    conn.commit()


def upsert_clean_docs(conn: sqlite3.Connection, docs: list[dict]):
    conn.executemany(
        """
        INSERT INTO clean_docs (id, source, team, subreddit, text, url, created_utc)
        VALUES (:id, :source, :team, :subreddit, :text, :url, :created_utc)
        ON CONFLICT(id) DO NOTHING
        """,
        docs,
    )
    conn.commit()


def count_docs(conn: sqlite3.Connection, table: str = "raw_docs") -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
