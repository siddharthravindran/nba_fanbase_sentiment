"""Zero-shot re-labeling experiment: test whether sports-fandom-specific
candidate labels fit Reddit text better than the therapy-domain RoBERTa
classes (physical, fatigue, engaged, etc. don't map well to sports talk).

Uses facebook/bart-large-mnli, which scores arbitrary candidate labels
against a text via natural language inference - no training required.
"""
import pandas as pd
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

CANDIDATE_LABELS = [
    "excitement or hype",
    "anger or frustration",
    "disappointment",
    "pessimism or resignation",
    "hope or optimism",
    "pride",
    "mockery or sarcasm",
    "sadness",
    "neutral analysis or discussion",
]

INPUT_CSV = "data/clean_docs_export.csv"
OUTPUT_CSV = "data/zeroshot_full_scored.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    df["text"] = df["text"].fillna("").astype(str)

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1,  # CPU; sample is small enough not to need GPU
    )

    top_labels = []
    top_scores = []
    all_scores = []

    for i, text in enumerate(df["text"]):
        result = classifier(text, CANDIDATE_LABELS, multi_label=False)
        top_labels.append(result["labels"][0])
        top_scores.append(result["scores"][0])
        all_scores.append(dict(zip(result["labels"], result["scores"])))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(df)} scored")

    df["zeroshot_label"] = top_labels
    df["zeroshot_score"] = top_scores
    df["zeroshot_all_scores"] = all_scores

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Wrote {len(df)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
