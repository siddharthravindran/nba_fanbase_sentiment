"""Chroma-backed vector store for retrieving relevant fan posts/articles.

Set CHROMA_HOST to talk to a Chroma server (`chroma run --path data/chroma`)
instead of opening the on-disk index directly. PersistentClient is an embedded
store: each process memory-maps its own copy of the HNSW index and flushes it
back independently, so a long-running app and the nightly sync writing at the
same time can corrupt the index. Server mode gives the files a single owner.

The embedded path stays the default so the repo is clone-and-run.
"""
import os
import re
from datetime import datetime

import chromadb

COLLECTION_NAME = "nba_fan_docs"
TEAM_PREFIX = "nba_team_"
# Docs with no team tag still get a home, so counts reconcile against the
# source collection instead of silently going missing.
UNASSIGNED = f"{TEAM_PREFIX}_unassigned"
# The source collection was built with l2. create_collection defaults to cosine,
# so a new per-team collection must be told, or "nearest" quietly means
# something different there than everywhere else.
HNSW_SPACE = "l2"


def _client(persist_dir: str = "data/chroma"):
    host = os.getenv("CHROMA_HOST")
    if host:
        return chromadb.HttpClient(host=host, port=int(os.getenv("CHROMA_PORT", "8000")))
    return chromadb.PersistentClient(path=persist_dir)


def team_collection_name(team: str | None) -> str:
    if not team:
        return UNASSIGNED
    slug = re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")
    return f"{TEAM_PREFIX}{slug}"


def get_collection(persist_dir: str = "data/chroma"):
    return _client(persist_dir).get_or_create_collection(COLLECTION_NAME)


def get_team_collection(client, team: str | None):
    return client.get_or_create_collection(
        team_collection_name(team), configuration={"hnsw": {"space": HNSW_SPACE}}
    )


class TeamCollections:
    """Routes each query to that team's own collection.

    A `where={"team": ...}` filter can't use the HNSW graph - Chroma pre-filters
    to matching ids and brute-forces them, which is O(N_team). Measured on Utah
    Jazz (the smallest fanbase, so the old path's best case): 10.90s filtered on
    the 3.5M collection vs 0.38s unfiltered on its own collection, a 29x
    speedup, with identical top-10 ordering and distances.

    Falls back to the single pre-migration collection when a team's collection
    is missing, so the app still works mid-migration or on a fresh clone that
    has only ever run populate_chroma.
    """

    def __init__(self, persist_dir: str = "data/chroma"):
        self._client = _client(persist_dir)
        self._cache: dict[str, object] = {}
        self._available = {
            c.name for c in self._client.list_collections() if c.name.startswith(TEAM_PREFIX)
        }
        self._fallback = None

    def for_team(self, team: str | None):
        """Returns (collection, needs_team_filter). The flag matters: a shared
        fallback collection still has to filter by team, a per-team one must
        not (it would be a no-op cost on every query)."""
        name = team_collection_name(team) if team else None
        if name and name in self._available:
            if name not in self._cache:
                self._cache[name] = self._client.get_collection(name)
            return self._cache[name], False
        if self._fallback is None:
            # get_collection, NOT get_or_create_collection. This branch runs
            # only when a team's own collection is missing, i.e. something is
            # already wrong. get_or_create would answer that by manufacturing an
            # empty collection, and every query against it would return zero
            # results while reporting success - the app would render an empty
            # chart and the model would state that fans have said nothing about
            # the topic. A fabricated silence is worse than an outage, because
            # nothing about it looks broken. Raising is the correct failure.
            self._fallback = self._client.get_collection(COLLECTION_NAME)
        return self._fallback, True


def resolve(collection, team: str | None):
    """Accepts either a TeamCollections router or a plain Chroma collection and
    returns (collection, needs_team_filter), so callers holding one of the two
    don't each have to branch."""
    if isinstance(collection, TeamCollections):
        return collection.for_team(team)
    return collection, True


def to_timestamp(created_utc: str | None) -> float:
    """ISO string -> Unix seconds, for Chroma's $gte/$lt range operators, which
    reject strings. 0.0 for missing/malformed, which sorts before any real date
    and so is excluded by every recency filter - correct for an undated doc."""
    if not created_utc:
        return 0.0
    try:
        return datetime.fromisoformat(created_utc).timestamp()
    except ValueError:
        return 0.0


def add_docs(collection, docs: list[dict], embeddings: list[list[float]] | None = None):
    """docs: [{id, text, team, source, top_emotion, top_score, created_utc}, ...]
    Uses upsert (not add) so re-running is idempotent/resumable.

    Pass precomputed `embeddings` (e.g. from a GPU embedding run) to skip
    running the local (slow, CPU-bound) embedding function during bulk loads."""
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=embeddings,
        metadatas=[
            {
                "team": d["team"] or "",
                "source": d["source"],
                "top_emotion": d.get("top_emotion") or "",
                "top_score": d.get("top_score") or 0.0,
                "created_utc": d.get("created_utc") or "",
                "created_ts": to_timestamp(d.get("created_utc")),
            }
            for d in docs
        ],
    )


def query(collection, text: str, team: str | None = None, n_results: int = 5):
    col, needs_team_filter = resolve(collection, team)
    where = {"team": team} if (team and needs_team_filter) else None
    return col.query(query_texts=[text], n_results=n_results, where=where)
