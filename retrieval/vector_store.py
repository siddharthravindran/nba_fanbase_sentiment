"""Chroma-backed vector store for retrieving relevant fan posts/articles."""
import chromadb

COLLECTION_NAME = "nba_fan_docs"


def get_collection(persist_dir: str = "data/chroma"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(COLLECTION_NAME)


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
            }
            for d in docs
        ],
    )


def query(collection, text: str, team: str | None = None, n_results: int = 5):
    where = {"team": team} if team else None
    return collection.query(query_texts=[text], n_results=n_results, where=where)
