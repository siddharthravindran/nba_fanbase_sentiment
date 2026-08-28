"""Tool schemas + dispatch for the Claude tool-calling chat layer."""
from ingest.storage import get_connection
from ingest.teams import TEAM_SUBREDDITS
from retrieval.aggregate import aggregate_team_sentiment, aggregate_topic_sentiment
from retrieval.vector_store import query

TEAM_NAMES = sorted(TEAM_SUBREDDITS.keys())

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
            "than the fanbase's sentiment in general."
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
            "speaking only in aggregate statistics."
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
                    "description": "How many quotes to retrieve (default 5)",
                },
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
        if topic:
            return aggregate_topic_sentiment(collection, team, topic)
        return aggregate_team_sentiment(team)

    if name == "retrieve_quotes":
        team = tool_input["team"]
        topic = tool_input["topic"]
        n_results = tool_input.get("n_results", 5)

        # Over-fetch a wider semantically-relevant pool, then keep only the
        # most recent n_results - same rationale as aggregate_topic_sentiment:
        # recent posts better reflect settled sentiment than early rumor buzz
        # that happens to share the same topic/entities.
        candidate_pool = max(50, n_results * 10)
        results = query(collection, topic, team=team, n_results=candidate_pool)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results["ids"] else []

        dated = [
            (doc_id, doc, meta)
            for doc_id, doc, meta in zip(ids, docs, metas)
            if meta.get("created_utc")
        ]
        dated.sort(key=lambda triple: triple[2]["created_utc"], reverse=True)
        sampled = dated[:n_results]

        urls = _lookup_urls([doc_id for doc_id, _, _ in sampled])

        return {
            "quotes": [
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "top_emotion": meta.get("top_emotion"),
                    "created_utc": meta.get("created_utc"),
                    "url": urls.get(doc_id),
                }
                for doc_id, doc, meta in sampled
            ]
        }

    raise ValueError(f"Unknown tool: {name}")
