"""Sentiment aggregation for the chat layer's `aggregate_sentiment` tool.

Two modes:
- Team-only (no topic): exact aggregation over the full `sentiment_docs_v2`
  table via SQL - every scored doc for that team, no sampling.
- Team + topic: SQL can't do semantic matching, so instead we pull the
  top-N semantically similar docs for (team, topic) from Chroma and
  aggregate over their `top_emotion` metadata (already stored per-doc at
  embedding time, so no extra SQLite round-trip is needed).
"""
import sqlite3

DB_PATH = "data/raw_docs.db"


def aggregate_team_sentiment(team: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT top_emotion, COUNT(*) FROM sentiment_docs_v2 WHERE team = ? GROUP BY top_emotion ORDER BY COUNT(*) DESC",
        (team,),
    ).fetchall()
    conn.close()

    total = sum(count for _, count in rows)
    return {
        "team": team,
        "topic": None,
        "mode": "exact",
        "n_docs": total,
        "distribution": [
            {"emotion": emotion, "count": count, "pct": round(100 * count / total, 1) if total else 0.0}
            for emotion, count in rows
        ],
    }


def aggregate_topic_sentiment(collection, team: str, topic: str, n_results: int = 100) -> dict:
    results = collection.query(
        query_texts=[topic],
        n_results=n_results,
        where={"team": team},
    )
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    counts: dict[str, int] = {}
    for meta in metadatas:
        emotion = meta.get("top_emotion") or "unknown"
        counts[emotion] = counts.get(emotion, 0) + 1

    total = len(metadatas)
    distribution = sorted(
        ({"emotion": emotion, "count": count, "pct": round(100 * count / total, 1) if total else 0.0}
         for emotion, count in counts.items()),
        key=lambda d: d["count"],
        reverse=True,
    )
    return {
        "team": team,
        "topic": topic,
        "mode": "sampled",
        "n_docs": total,
        "distribution": distribution,
    }
