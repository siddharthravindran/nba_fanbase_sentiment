"""Chroma-backed vector store for retrieving relevant fan posts/articles."""
import chromadb

COLLECTION_NAME = "nba_fan_docs"


def get_collection(persist_dir: str = "data/chroma"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(COLLECTION_NAME)


def add_docs(collection, docs: list[dict]):
    """docs: [{id, text, team, source, sentiment_labels, created_utc}, ...]"""
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[
            {
                "team": d["team"],
                "source": d["source"],
                "sentiment_labels": ",".join(d.get("sentiment_labels", [])),
                "created_utc": d.get("created_utc", ""),
            }
            for d in docs
        ],
    )


def query(collection, text: str, team: str | None = None, n_results: int = 5):
    where = {"team": team} if team else None
    return collection.query(query_texts=[text], n_results=n_results, where=where)
