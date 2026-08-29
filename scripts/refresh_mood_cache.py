"""Rebuild the league-wide mood table the landing chart reads.

Runs nightly after scoring. The underlying query joins sentiment_docs_v2 to
clean_docs for created_utc, which defeats the (team, top_emotion) covering
index and costs ~21s over 3.5M rows - fine here, far too slow to sit in front
of a page load.
"""
import sqlite3

from retrieval.aggregate import DB_PATH, refresh_team_mood_cache


def run():
    conn = sqlite3.connect(DB_PATH, timeout=120)
    try:
        n = refresh_team_mood_cache(conn)
    finally:
        conn.close()
    print(f"team_mood_cache refreshed: {n} rows", flush=True)


if __name__ == "__main__":
    run()
