"""Tool schemas + dispatch for the Claude tool-calling chat layer."""
from ingest.storage import get_connection
from ingest.teams import TEAM_SUBREDDITS
from retrieval.aggregate import (
    EXCLUDED_SOURCES,
    aggregate_team_sentiment,
    aggregate_topic_sentiment,
)
from retrieval.recency import select_two_tier
from retrieval.vector_store import query

TEAM_NAMES = sorted(TEAM_SUBREDDITS.keys())

# Shared by both tools. Left unset, retrieval decays older posts so answers
# describe the present; setting a range pins the question to a past window
# instead (and turns the decay off, so a historical query isn't dragged back
# toward its most recent edge).
DATE_PARAM_SINCE = {
    "type": "string",
    "description": (
        "Optional ISO date (YYYY-MM-DD) - only include posts from on or after "
        "this date. Use for questions about a specific past period, e.g. 'how "
        "did fans react when the trade happened'. Omit for 'how do fans feel now'."
    ),
}
DATE_PARAM_UNTIL = {
    "type": "string",
    "description": (
        "Optional ISO date (YYYY-MM-DD) - only include posts up to and "
        "including this date. Pair with `since` to ask about a window, e.g. "
        "sentiment before an event was resolved."
    ),
}

TOOLS = [
    {
        "name": "aggregate_sentiment",
        "description": (
            "Get the emotion/sentiment distribution for an NBA team's fanbase, as "
            "counts and percentages across emotion categories (e.g. excitement or "
            "hype, disappointment, anger or frustration, hope or optimism, etc). "
            "If `topic` is omitted, returns exact stats over every scored doc for "
            "that team. If `topic` is given, returns stats sampled from the posts "
            "most semantically relevant to that topic - use this whenever the "
            "question is about a specific player, trade, game, or event rather "
            "than the fanbase's sentiment in general. "
            "By default, recent posts are weighted more heavily than old ones, "
            "so the result reflects how fans feel NOW. Only set `since`/`until` "
            "when the question is explicitly about a past period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "enum": TEAM_NAMES,
                    "description": "Full NBA team name, e.g. 'Los Angeles Lakers'",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic to scope sentiment to, e.g. 'LeBron James signing' or 'the trade deadline'",
                },
                "since": DATE_PARAM_SINCE,
                "until": DATE_PARAM_UNTIL,
            },
            "required": ["team"],
        },
    },
    {
        "name": "retrieve_quotes",
        "description": (
            "Retrieve actual fan quotes (Reddit posts/comments or article "
            "excerpts) most semantically relevant to a topic, for a given team. "
            "Use this to ground your answer with real examples rather than "
            "speaking only in aggregate statistics. Recent quotes are favored "
            "unless you pin a date range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": TEAM_NAMES},
                "topic": {
                    "type": "string",
                    "description": "What to search for, e.g. 'reaction to the coaching change'",
                },
                "n_results": {
                    "type": "integer",
                    "description": (
                        "How many quotes to retrieve (default 12, max 25). "
                        "These quotes are the only fan writing you get to see, "
                        "so ask for more when the question is broad (a team's "
                        "roster, season, or direction) - a handful of quotes "
                        "about one player can't support a claim about a whole "
                        "fanbase. Fewer is fine for a narrow question about a "
                        "single player or event."
                    ),
                },
                "since": DATE_PARAM_SINCE,
                "until": DATE_PARAM_UNTIL,
            },
            "required": ["team", "topic"],
        },
    },
]


def _lookup_urls(doc_ids: list[str]) -> dict[str, str]:
    """Chroma metadata doesn't carry the source URL (and rewriting metadata for
    millions of embedded docs to add it would be expensive), so resolve it with
    a primary-key lookup against clean_docs instead. Lets the UI link each
    quote back to the actual Reddit thread or article."""
    if not doc_ids:
        return {}
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(doc_ids))
        rows = conn.execute(
            f"SELECT id, url FROM clean_docs WHERE id IN ({placeholders})", doc_ids
        ).fetchall()
    finally:
        conn.close()
    return {doc_id: url for doc_id, url in rows if url}


def call_tool(name: str, tool_input: dict, collection) -> dict:
    if name == "aggregate_sentiment":
        team = tool_input["team"]
        topic = tool_input.get("topic")
        since = tool_input.get("since")
        until = tool_input.get("until")
        if topic:
            return aggregate_topic_sentiment(
                collection, team, topic, since=since, until=until
            )
        return aggregate_team_sentiment(team, since=since, until=until)

    if name == "retrieve_quotes":
        team = tool_input["team"]
        topic = tool_input["topic"]
        # Capped: quotes are the model's entire view of the corpus, so more is
        # generally better, but they all land in the context window verbatim.
        n_results = max(1, min(int(tool_input.get("n_results", 12)), 25))
        since = tool_input.get("since")
        until = tool_input.get("until")

        # Short comments dominate raw similarity: a 38-character "We need him
        # on the roster" is a near-perfect embedding match for a short topic
        # query, so the top-k fills with fragments. Measured on a 400-doc Lakers
        # pool, the median doc was 38 chars and only 1.5% of docs under 80 chars
        # named a player, against 40% of docs over 250. Those fragments are also
        # useless as displayed evidence - they give the reader nothing to judge.
        # So require some substance, and only fall back to short docs if the
        # filter would otherwise leave us short of quotes.
        MIN_QUOTE_CHARS = 120

        # Over-fetch a wide semantically-relevant pool, then re-rank it by
        # relevance AND recency together (see fuse_relevance_recency) with a
        # share of slots reserved for recent docs (see select_two_tier).
        #
        # The pool is wide for two compounding reasons. Substantive docs are the
        # tail of the length distribution (~10% clear 120 chars), and recent
        # docs are a thin slice of a fanbase's history - at 400 candidates a
        # Lakers "new roster additions" query surfaced 33 recent docs, only 1 of
        # which named any of that summer's signings. At 2,000 it was 24. Ranking
        # can only reorder what was retrieved, so the fix has to happen here.
        #
        # Widening k stays on the HNSW graph; a created_ts range filter would
        # force Chroma's brute-force path instead and measured slower (2.07s vs
        # 0.75s on Lakers) for a narrower pool.
        candidate_pool = max(2000, n_results * 40)
        results = query(collection, topic, team=team, n_results=candidate_pool)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results["ids"] else []
        dists = results["distances"][0] if results.get("distances") else []

        # Drop news articles: this tool feeds "here's what fans are saying", and
        # a wire-service recap sentence attributed as a fan quote is a lie about
        # the source. See SENTIMENT_SOURCE_CLAUSE in retrieval/aggregate.py.
        dated = [
            (doc_id, doc, meta, dist)
            for doc_id, doc, meta, dist in zip(
                ids, docs, metas, dists or [None] * len(ids)
            )
            if meta.get("created_utc") and meta.get("source") not in EXCLUDED_SOURCES
        ]
        if since:
            dated = [t for t in dated if t[2]["created_utc"] >= since]
        if until:
            dated = [t for t in dated if t[2]["created_utc"] < until + "T99"]

        substantive = [row for row in dated if len(row[1] or "") >= MIN_QUOTE_CHARS]
        fallback = [row for row in dated if len(row[1] or "") < MIN_QUOTE_CHARS]

        def _pick(rows, want):
            if not rows or want <= 0:
                return []
            if since or until:
                # A pinned window is the question, so the two-tier split is
                # meaningless inside it - rank purely by recency.
                return sorted(
                    rows, key=lambda row: row[2]["created_utc"], reverse=True
                )[:want]
            pool_dists = [row[3] for row in rows]
            order = select_two_tier(
                [row[2].get("created_utc") for row in rows],
                pool_dists if all(d is not None for d in pool_dists) else None,
                want,
            )
            return [rows[i] for i in order]

        sampled = _pick(substantive, n_results)
        if len(sampled) < n_results:
            sampled += _pick(fallback, n_results - len(sampled))

        urls = _lookup_urls([row[0] for row in sampled])

        return {
            "quotes": [
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "top_emotion": meta.get("top_emotion"),
                    "created_utc": meta.get("created_utc"),
                    "url": urls.get(doc_id),
                }
                for doc_id, doc, meta, _ in sampled
            ]
        }

    raise ValueError(f"Unknown tool: {name}")
