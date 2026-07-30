"""Run in Colab (GPU runtime) to embed docs exported by export_for_embedding.py.

Uses the exact same model/pooling/normalization as Chroma's default
embedding function (sentence-transformers/all-MiniLM-L6-v2, mean pooling,
L2-normalized, max_seq_length=256) so vectors are compatible with local
Chroma similarity search.

Saves incrementally (one .npz shard every CHUNK_SIZE docs) so a Colab
disconnect only costs the current chunk, not the whole run. Re-running
skips shards that already exist on Drive, so it's resumable.

Steps:
1. Upload data/embed_export.jsonl to Drive (e.g. MyDrive/nba_fanbase_sentiment/).
2. Runtime -> Change runtime type -> GPU.
3. Paste this into a Colab cell and run.
4. Download all shard files (embeddings_shard_*.npz) from Drive back to data/.
"""
# !pip install -q sentence-transformers

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from google.colab import drive

drive.mount("/content/drive")

IN_PATH = "/content/drive/MyDrive/nba_fanbase_sentiment/embed_export_new.jsonl"
OUT_DIR = "/content/drive/MyDrive/nba_fanbase_sentiment/embed_new/"
os.makedirs(OUT_DIR, exist_ok=True)
CHUNK_SIZE = 50_000  # docs per saved shard

ids, texts = [], []
with open(IN_PATH) as f:
    for line in f:
        row = json.loads(line)
        ids.append(row["id"])
        texts.append(row["text"])

print(f"Loaded {len(texts):,} docs")

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
model.max_seq_length = 256

n_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE
for chunk_idx in range(n_chunks):
    shard_path = os.path.join(OUT_DIR, f"embeddings_shard_{chunk_idx:04d}.npz")
    if os.path.exists(shard_path):
        print(f"Shard {chunk_idx} already exists, skipping")
        continue

    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(texts))
    chunk_ids = ids[start:end]
    chunk_texts = texts[start:end]

    embeddings = model.encode(
        chunk_texts,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.savez(shard_path, ids=np.array(chunk_ids), embeddings=embeddings.astype(np.float32))
    print(f"Saved shard {chunk_idx}/{n_chunks - 1} ({start:,}-{end:,}) to {shard_path}")

print("All shards done.")
