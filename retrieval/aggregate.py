"""Sentiment aggregation for the chat layer's `aggregate_sentiment` tool.

Two modes:
- Team-only (no topic): aggregation over the full `sentiment_docs_v2` table
  via SQL - every scored doc for that team, no sampling.
- Team + topic: SQL can't do semantic matching, so instead we pull the
  top-N semantically similar docs for (team, topic) from Chroma and
  aggregate over their `top_emotion` metadata (already stored per-doc at
  embedding time, so no extra SQLite round-trip is needed).

Both modes apply exponential time-decay (see retrieval/recency.py) unless the
caller pins an explicit date range. Without it, "how do fans feel?" would
average 14 months of history into a single number and read as stale.

News articles are excluded throughout - see SENTIMENT_SOURCE_CLAUSE.
"""
import sqlite3
from datetime import datetime, timezone

from retrieval.recency import (
    DEFAULT_HALF_LIFE_DAYS,
    decay_weight,
    fuse_relevance_recency,
    weighted_distribution,
)
from retrieval.vector_store import query as vector_query

DB_PATH = "data/raw_docs.db"

# This is a *fanbase* sentiment product, and a wire-service game recap carries
# no fan emotion to measure. The v2 model was fine-tuned on fan writing and is
# plainly out of distribution on journalist prose: mean confidence 0.63 vs 0.74
# on Reddit, 28.6% of articles scored below 0.5 confidence vs 14.9%, and
# labels like "Trump plans to attend Knicks Finals Game 3" -> pessimism (0.74)
# or a bare box-score roundup -> mockery. It also reports 0.0% neutral analysis
# on newswire copy, which is self-evidently wrong.
#
# Articles are ~1.9% of the corpus, so dropping them barely moves the totals -
# but it removes mislabeled rows from the distribution and, more importantly,
# stops a reporter's sentence being quoted back to the user as fan opinion.
# They stay ingested and scored; they're just not evidence of how fans feel.
SENTIMENT_SOURCE_CLAUSE = "c.source != 'article'"
EXCLUDED_SOURCES = {"article"}

# (team, top_emotion) covers the app's league-wide GROUP BY entirely, so it
# reads the index instead of scanning several million rows - 6.3s -> 0.4s.
INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_sentiment_v2_team_emotion "
    "ON sentiment_docs_v2(team, top_emotion)"
)

# The landing chart needs a decay weight per (team, emotion), but the weight
# depends on created_utc, which lives on clean_docs - so the query has to join
# and the (team, top_emotion) index stops covering it. That turns a 1.1s
# index-only scan into 38s of random primary-key lookups across 3.5M rows,
# paid on every cold start, for a number that is the same for every visitor.
# So it's materialized instead: 270 rows, refreshed by the nightly pipeline.
MOOD_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS team_mood_cache (
    team TEXT NOT NULL,
    top_emotion TEXT NOT NULL,
    n_docs INTEGER NOT NULL,
    weight REAL NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (team, top_emotion)
)
"""


def refresh_team_mood_cache(conn) -> int:
    """Recompute the league-wide decay-weighted mood table. Slow (~40s) by
    design - it runs nightly, not in the request path.

    Serving a day-old cache does NOT skew the chart. Every weight decays by the
    same factor exp(-ln2 * elapsed / half_life), and the chart only ever uses
    weight ratios (each emotion's share of its team's total), so a uniform
    factor cancels exactly in the normalization. The only thing staleness
    actually costs is documents ingested since the last refresh.
    """
    conn.execute(MOOD_CACHE_SQL)
    rows = conn.execute(
        f"""
        SELECT s.team, s.top_emotion, COUNT(*),
               SUM(exp(-ln(2) * MAX(0, julianday('now') - julianday(c.created_utc))
                   / {DEFAULT_HALF_LIFE_DAYS}.0))
        FROM sentiment_docs_v2 s
        JOIN clean_docs c ON c.id = s.id
        WHERE s.team IS NOT NULL AND s.team != ''
          AND {SENTIMENT_SOURCE_CLAUSE}
        GROUP BY s.team, s.top_emotion
        """
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM team_mood_cache")
    conn.executemany(
        "INSERT INTO team_mood_cache (team, top_emotion, n_docs, weight, computed_at)"
        " VALUES (?, ?, ?, ?, ?)",
        [(t, e, c, w or 0.0, now) for t, e, c, w in rows],
    )
    conn.commit()
    return len(rows)


def _date_clause(since: str | None, until: str | None) -> tuple[str, list]:
    clauses, params = [], []
    if since:
        clauses.append("c.created_utc >= ?")
        params.append(since)
    if until:
        # `until` is inclusive of the whole day. Callers pass a plain date
        # ("2026-08-01") but created_utc carries a full timestamp, so a plain
        # <= would drop everything published later that same day. "T99" sorts
        # after any real "T.." time, which keeps this a string comparison the
        # created_utc index can still use.
        clauses.append("c.created_utc < ?")
        params.append(until + "T99")
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def aggregate_team_sentiment(
    team: str, since: str | None = None, until: str | None = None
) -> dict:
    """Every scored doc for the team, aggregated in SQL.

    The weighting is done with SQLite's exp()/julianday() rather than in Python
    on purpose: a popular fanbase has ~390k scored docs, and materializing all
    of them just to sum weights takes ~18s, which is far too slow for a tool
    call inside a chat turn. Grouping in SQL returns nine rows instead.
    """
    where_extra, params = _date_clause(since, until)
    pinned = bool(since or until)

    # An explicit range is the caller saying "this window is the question", so
    # don't also decay inside it - that would quietly re-bias a deliberately
    # historical query back toward its most recent edge.
    weight_expr = (
        "COUNT(*)"
        if pinned
        else (
            "SUM(exp(-ln(2) * MAX(0, julianday('now') - julianday(c.created_utc))"
            f" / {DEFAULT_HALF_LIFE_DAYS}.0))"
        )
    )

    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        # The unpinned case is exactly what team_mood_cache already stores -
        # same 21-day decay, same GROUP BY, just precomputed for all 30 teams.
        # Recomputing it live costs ~8.5s inside a chat turn because the
        # created_utc join defeats the (team, top_emotion) covering index.
        rows = []
        if not pinned:
            conn.execute(MOOD_CACHE_SQL)
            rows = conn.execute(
                "SELECT top_emotion, n_docs, weight FROM team_mood_cache WHERE team = ?",
                [team],
            ).fetchall()
        if not rows:
            rows = conn.execute(
                f"""
                SELECT s.top_emotion, COUNT(*), {weight_expr}
                FROM sentiment_docs_v2 s
                JOIN clean_docs c ON c.id = s.id
                WHERE s.team = ? AND {SENTIMENT_SOURCE_CLAUSE}{where_extra}
                GROUP BY s.top_emotion
                """,
                [team, *params],
            ).fetchall()
    finally:
        conn.close()

    n_docs = sum(count for _, count, _ in rows)
    total_weight = sum(weight or 0.0 for _, _, weight in rows)
    distribution = sorted(
        (
            {
                "emotion": emotion or "unknown",
                "count": count,
                "pct": round(100 * (weight or 0.0) / total_weight, 1) if total_weight else 0.0,
            }
            for emotion, count, weight in rows
        ),
        key=lambda d: d["pct"],
        reverse=True,
    )

    return {
        "team": team,
        "topic": None,
        "mode": "exact" if pinned else "recency_weighted",
        "n_docs": n_docs,
        "date_range": {"since": since, "until": until} if pinned else None,
        "half_life_days": None if pinned else DEFAULT_HALF_LIFE_DAYS,
        "distribution": distribution,
    }


# Enough text to judge whether a label fits. Short fragments are the problem
# case - "we need him" is scored as hope but reads as nothing, so it cannot
# settle whether a bucket is real signal or the classifier misfiring.
EXAMPLE_MIN_CHARS = 100
EXAMPLE_MAX_CHARS = 200
EXAMPLES_PER_EMOTION = 2


def _attach_examples(distribution: list[dict], sampled: list[tuple[dict, str]]) -> None:
    """Hang a couple of real posts off each emotion bucket, in place.

    Without these the model is asked to narrate percentages it cannot inspect,
    and the honest move - saying a bucket looks mislabeled - is not available to
    it, because judging a label requires reading the text that got the label.
    The v2 classifier's known weak spot is sarcasm, and mockery is this corpus's
    largest class, so "the excitement bucket is actually sarcasm" is a live
    possibility that only the underlying posts can settle.
    """
    # Two passes, long posts first. A single pass with a length floor starves
    # exactly the buckets that need explaining: this corpus is mostly short
    # comments (median ~38 chars), so on the Jaylen Brown trade the 34-post
    # "excitement or hype" bucket produced no qualifying example at all while a
    # 7-post bucket did. A short post is weak evidence but it is far better than
    # leaving the largest bucket unillustrated, which puts the model right back
    # to narrating a number it cannot inspect.
    by_emotion: dict[str, list[str]] = {}

    def collect(min_chars: int) -> None:
        for meta, doc in sampled:
            text = " ".join((doc or "").split())
            if len(text) < min_chars:
                continue
            label = meta.get("top_emotion") or "unknown"
            bucket = by_emotion.setdefault(label, [])
            snippet = text[:EXAMPLE_MAX_CHARS] + (
                "..." if len(text) > EXAMPLE_MAX_CHARS else ""
            )
            if len(bucket) < EXAMPLES_PER_EMOTION and snippet not in bucket:
                bucket.append(snippet)

    collect(EXAMPLE_MIN_CHARS)
    collect(1)

    for entry in distribution:
        examples = by_emotion.get(entry["emotion"])
        if examples:
            entry["examples"] = examples


def aggregate_topic_sentiment(
    collection,
    team: str,
    topic: str,
    n_results: int = 100,
    candidate_pool: int = 300,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    """Semantic search alone can't tell rumor-phase buzz apart from reaction
    to a settled outcome (e.g. "excitement about maybe landing a player" vs.
    "how fans feel now that it didn't happen" both match the same query).

    This used to keep only the newest `n_results` of a wide candidate pool,
    which fixed that contamination but threw away strong old matches outright -
    and made genuinely historical questions unanswerable. Now the pool is
    decay-weighted instead, so recent posts dominate without older ones being
    discarded, and an explicit date range turns the weighting off."""
    results = vector_query(collection, topic, team=team, n_results=candidate_pool)
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results.get("distances") else []
    # Documents, not just metadata: the chart is built from this sample, but the
    # model writes its answer from retrieve_quotes' separate sample, so it has
    # never seen the posts behind these bars. That gap is why a 27% "excitement
    # or hype" bucket could go unmentioned in an answer describing a uniformly
    # grim fanbase - the model could not explain a bucket it could not read.
    documents = results["documents"][0] if results.get("documents") else []

    rows = [
        (m, d, doc)
        for m, d, doc in zip(
            metadatas,
            distances or [None] * len(metadatas),
            documents or [""] * len(metadatas),
        )
        if m.get("created_utc") and m.get("source") not in EXCLUDED_SOURCES
    ]
    if since:
        rows = [r for r in rows if r[0]["created_utc"] >= since]
    if until:
        rows = [r for r in rows if r[0]["created_utc"] < until + "T99"]

    pinned = bool(since or until)
    if pinned:
        sampled = [(m, doc) for m, _, doc in rows[:n_results]]
        items = [(m.get("top_emotion"), 1.0) for m, _ in sampled]
    else:
        # Pick the sample by fusing the relevance and recency rankings, then
        # weight each pick by its decay so the percentages still lean recent.
        # Selecting on recency alone (an earlier version of this) is just a
        # date sort in disguise and discards the best matches outright.
        pool_dists = [d for _, d, _ in rows]
        order = fuse_relevance_recency(
            [m.get("created_utc") for m, _, _ in rows],
            pool_dists if all(d is not None for d in pool_dists) else None,
        )
        sampled = [(rows[i][0], rows[i][2]) for i in order[:n_results]]
        items = [
            (m.get("top_emotion"), decay_weight(m.get("created_utc")))
            for m, _ in sampled
        ]

    distribution = weighted_distribution(items)
    _attach_examples(distribution, sampled)

    dates = sorted(m["created_utc"] for m, _ in sampled if m.get("created_utc"))
    return {
        "team": team,
        "topic": topic,
        "mode": "exact" if pinned else "recency_weighted",
        "n_docs": len(sampled),
        "date_range": {"earliest": dates[0], "latest": dates[-1]} if dates else None,
        "half_life_days": None if pinned else 21,
        "distribution": distribution,
    }
