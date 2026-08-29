"""One-time migration: single Chroma collection -> one collection per team,
with a numeric `created_ts` added to every doc's metadata.

Why per-team collections
------------------------
Every query today carries `where={"team": ...}`. A metadata filter can't ride
the HNSW graph - the graph's edges span all 3.5M docs, so following them lands
on other teams constantly. Chroma's fallback is to pre-filter to matching ids
and brute-force them, which is O(N_team): measured 10.3s filtered vs 0.2s
unfiltered, and it scales with fanbase size (Lakers 6.5s, Jazz 1.1s). A
collection holding only one team's docs needs no filter at all, so search is
back to ~O(log N) graph traversal.

Why created_ts
--------------
Chroma's range operators ($gte/$lt) require int/float. `created_utc` is an ISO
string, so date filtering currently raises ValueError - which blocks two-tier
retrieval (a second, recency-pinned query whose results get fused with the
relevance query). Storing the same instant as a Unix float makes it filterable.
Verified on a scratch collection: 490 expected docs >= 2026-06-01, 490 returned.

Nothing is re-embedded. Vectors are copied as-is (384-dim, verified readable),
which is what makes this ~2h instead of a multi-day GPU job.

Safety
------
- The source collection is never modified or deleted. Roll back by pointing
  retrieval/vector_store.py back at it.
- Writes are upserts and progress is checkpointed, so an interrupted run
  resumes rather than restarting.
- Target collections copy the source's hnsw space ("l2") and default embedding
  function. Getting either wrong silently changes what "nearest" means.

Usage:
    python -m scripts.migrate_per_team_collections            # full run
    python -m scripts.migrate_per_team_collections --verify   # check only
    python -m scripts.migrate_per_team_collections --reset    # start over
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

import chromadb

from retrieval.vector_store import COLLECTION_NAME

PERSIST_DIR = "data/chroma"
CHECKPOINT = "data/migration_checkpoint.json"

# Chroma rejects batches over 5461.
WRITE_BATCH = 5000
# Read throughput is very batch-size sensitive: 2k pages ran at ~200 docs/s,
# 20k pages at ~1,670 docs/s.
READ_BATCH = 20000

TEAM_PREFIX = "nba_team_"
# Docs whose team tag is empty still get a home, so total counts reconcile
# exactly and nothing disappears silently.
UNASSIGNED = f"{TEAM_PREFIX}_unassigned"


def collection_name_for(team: str) -> str:
    if not team:
        return UNASSIGNED
    slug = re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")
    return f"{TEAM_PREFIX}{slug}"


def to_timestamp(created_utc: str) -> float:
    """ISO string -> Unix seconds. 0.0 for missing/malformed, which sorts before
    any real date and is therefore excluded by every `$gte` recency filter -
    the right behavior for a doc whose age is unknown."""
    if not created_utc:
        return 0.0
    try:
        return datetime.fromisoformat(created_utc).timestamp()
    except ValueError:
        return 0.0


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as fh:
            return json.load(fh)
    return {"offset": 0, "written": 0}


def save_checkpoint(state: dict):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, CHECKPOINT)  # atomic, so a crash mid-write can't corrupt it


def get_source_config(client):
    """Mirror the source's HNSW space and embedding function onto the targets.
    The source is l2, while Chroma's create_collection default is often cosine -
    silently switching would change every distance and therefore every ranking.
    """
    src = client.get_collection(COLLECTION_NAME)
    cfg = getattr(src, "configuration_json", None) or {}
    space = (cfg.get("hnsw") or {}).get("space", "l2")
    return src, space


def run(reset: bool = False):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    source, space = get_source_config(client)
    total = source.count()
    print(f"source: {COLLECTION_NAME}  {total:,} docs  (hnsw space={space})", flush=True)

    if reset and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("checkpoint cleared", flush=True)

    state = load_checkpoint()
    offset = state["offset"]
    written = state["written"]
    if offset:
        print(f"resuming from offset {offset:,} ({written:,} already written)", flush=True)

    targets: dict[str, object] = {}

    def target_for(team: str):
        name = collection_name_for(team)
        if name not in targets:
            targets[name] = client.get_or_create_collection(
                name, configuration={"hnsw": {"space": space}}
            )
        return targets[name]

    started = time.perf_counter()
    while offset < total:
        page = source.get(
            limit=READ_BATCH,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        ids = page["ids"]
        if not ids:
            break

        # Bucket the page by team, then flush each bucket in <=5000-doc writes.
        buckets: dict[str, dict] = {}
        for i, doc_id in enumerate(ids):
            meta = dict(page["metadatas"][i])
            meta["created_ts"] = to_timestamp(meta.get("created_utc", ""))
            b = buckets.setdefault(
                meta.get("team", ""), {"ids": [], "emb": [], "docs": [], "metas": []}
            )
            b["ids"].append(doc_id)
            b["emb"].append(page["embeddings"][i])
            b["docs"].append(page["documents"][i])
            b["metas"].append(meta)

        for team, b in buckets.items():
            col = target_for(team)
            for i in range(0, len(b["ids"]), WRITE_BATCH):
                sl = slice(i, i + WRITE_BATCH)
                col.upsert(
                    ids=b["ids"][sl],
                    embeddings=b["emb"][sl],
                    documents=b["docs"][sl],
                    metadatas=b["metas"][sl],
                )
        written += len(ids)
        offset += len(ids)
        save_checkpoint({"offset": offset, "written": written})

        elapsed = time.perf_counter() - started
        rate = written / elapsed if elapsed else 0
        remaining = (total - offset) / rate / 60 if rate else 0
        print(
            f"  {offset:,}/{total:,} ({100 * offset / total:.1f}%)  "
            f"{rate:,.0f} docs/s  ~{remaining:.0f} min left",
            flush=True,
        )

    print(f"\ncopied {written:,} docs in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    verify(client, total)


def verify(client=None, expected: int | None = None):
    client = client or chromadb.PersistentClient(path=PERSIST_DIR)
    if expected is None:
        expected = client.get_collection(COLLECTION_NAME).count()

    names = [c.name for c in client.list_collections() if c.name.startswith(TEAM_PREFIX)]
    total = 0
    print(f"\n{'collection':<40}{'docs':>12}")
    for name in sorted(names):
        n = client.get_collection(name).count()
        total += n
        print(f"  {name:<38}{n:>12,}")
    print(f"  {'TOTAL':<38}{total:>12,}")
    print(f"  {'source':<38}{expected:>12,}")
    ok = total == expected
    print(f"\nreconciles: {ok}" + ("" if ok else f"  MISSING {expected - total:,}"))

    if names:
        sample = client.get_collection(sorted(names)[0])
        got = sample.get(limit=1, include=["metadatas"])
        if got["ids"]:
            m = got["metadatas"][0]
            print(f"created_ts present: {'created_ts' in m}  sample={m.get('created_ts')}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check counts only")
    ap.add_argument("--reset", action="store_true", help="ignore checkpoint, start over")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        run(reset=args.reset)
