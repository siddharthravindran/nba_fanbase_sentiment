"""Run in Colab (GPU runtime) to sentiment-score docs exported by
export_for_scoring.py, using the fine-tuned v2 model (Siddharthr30/emotion-model-v2).

Mirrors colab_embed.py: batched GPU inference (vs. local CPU's single-example
loop) and incremental sharded saves so a Colab disconnect only costs the
current chunk, not the whole run. Re-running skips shards that already exist
on Drive, so it's resumable.

Steps:
1. Upload data/score_export.jsonl to Drive (e.g. MyDrive/nba_fanbase_sentiment/).
2. Runtime -> Change runtime type -> GPU.
3. Paste this into a Colab cell and run.
4. Download all shard files (sentiment_shard_*.npz) from Drive back to data/.
"""
# !pip install -q transformers

import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.colab import drive

drive.mount("/content/drive")

MODEL_NAME = "Siddharthr30/emotion-model-v2"
IN_PATH = "/content/drive/MyDrive/nba_fanbase_sentiment/score_export.jsonl"
OUT_DIR = "/content/drive/MyDrive/nba_fanbase_sentiment/"
CHUNK_SIZE = 50_000  # docs per saved shard
BATCH_SIZE = 128
MAX_LENGTH = 64

ids, texts = [], []
with open(IN_PATH) as f:
    for line in f:
        row = json.loads(line)
        ids.append(row["id"])
        texts.append(row["text"])

print(f"Loaded {len(texts):,} docs")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to("cuda")
model.eval()

# Fixed label order from model.config.id2label - saved once so the loader
# script knows which column is which emotion.
NUM_LABELS = model.config.num_labels
id2label = [model.config.id2label[i] for i in range(NUM_LABELS)]
with open(os.path.join(OUT_DIR, "sentiment_labels.json"), "w") as f:
    json.dump(id2label, f)

n_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE
for chunk_idx in range(n_chunks):
    shard_path = os.path.join(OUT_DIR, f"sentiment_shard_{chunk_idx:04d}.npz")
    if os.path.exists(shard_path):
        print(f"Shard {chunk_idx} already exists, skipping")
        continue

    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(texts))
    chunk_ids = ids[start:end]
    chunk_texts = texts[start:end]

    all_probs = np.zeros((len(chunk_texts), NUM_LABELS), dtype=np.float32)
    for b_start in range(0, len(chunk_texts), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(chunk_texts))
        batch_texts = chunk_texts[b_start:b_end]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs[b_start:b_end] = probs
        if (b_start // BATCH_SIZE) % 20 == 0:
            print(f"  chunk {chunk_idx}: {b_end}/{len(chunk_texts)}")

    np.savez(shard_path, ids=np.array(chunk_ids), probs=all_probs)
    print(f"Saved shard {chunk_idx}/{n_chunks - 1} ({start:,}-{end:,}) to {shard_path}")

print("All shards done.")
