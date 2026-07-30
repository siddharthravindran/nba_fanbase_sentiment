"""Load GPU-computed sentiment scores (from colab_score.py) into
sentiment_docs_v2, replacing the slow local CPU single-example scoring loop.

Reads all sentiment_shard_*.npz files in data/ (colab_score.py saves in
resumable chunks) plus data/sentiment_labels.json for the label order.
"""
import glob
import json
import sqlite3

import numpy as np

DB_PATH = "data/raw_docs.db"
SHARDS_GLOB = "data/sentiment_shard_*.npz"
LABELS_PATH = "data/sentiment_labels.json"
BATCH_SIZE = 2000


def load_all_shards():
    shard_paths = sorted(glob.glob(SHARDS_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found matching {SHARDS_GLOB}")
    print(f"Found {len(shard_paths)} shard files")

    all_ids, all_probs = [], []
    for path in shard_paths:
        data = np.load(path, allow_pickle=True)
        all_ids.append(data["ids"])
        all_probs.append(data["probs"])

    return np.concatenate(all_ids), np.concatenate(all_probs)


def run():
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    ids, probs = load_all_shards()
    print(f"Loaded {len(ids):,} precomputed sentiment scores")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total = 0
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE].tolist()
        batch_probs = probs[start:start + BATCH_SIZE]

        batch = []
        for doc_id, row_probs in zip(batch_ids, batch_probs):
            scored = sorted(zip(labels, row_probs.tolist()), key=lambda x: x[1], reverse=True)
            top_label, top_score = scored[0]
            # team isn't known at scoring time - filled in from clean_docs via
            # the existing team column already stored there (join at query time
            # via aggregate_team_sentiment doesn't need it), but sentiment_docs_v2
            # schema stores team directly, so pull it here.
            batch.append((doc_id, top_label, float(top_score), json.dumps(dict(scored))))

        # team column: look up alongside in the same batch, cheaper as one query
        placeholders = ",".join("?" for _ in batch_ids)
        team_rows = conn.execute(
            f"SELECT id, team FROM clean_docs WHERE id IN ({placeholders})", batch_ids
        ).fetchall()
        team_by_id = dict(team_rows)

        final_batch = [
            (doc_id, team_by_id.get(doc_id), top_label, top_score, all_scores_json)
            for doc_id, top_label, top_score, all_scores_json in batch
        ]

        cur.executemany(
            "INSERT OR REPLACE INTO sentiment_docs_v2 (id, team, top_emotion, top_score, all_scores) VALUES (?, ?, ?, ?, ?)",
            final_batch,
        )
        conn.commit()
        total += len(final_batch)
        if total % 20000 < BATCH_SIZE:
            print(f"  loaded {total:,}/{len(ids):,}")

    print(f"Done. Loaded {total:,} scores into sentiment_docs_v2.")
    new_total = cur.execute("SELECT COUNT(*) FROM sentiment_docs_v2").fetchone()[0]
    print(f"sentiment_docs_v2 total: {new_total:,}")
    conn.close()


if __name__ == "__main__":
    run()
