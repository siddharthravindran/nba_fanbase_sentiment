"""Orchestrates the nightly update pipeline end to end:
ingest (rolling window) -> clean -> score (incremental) -> sync to Chroma.

Run via cron/GitHub Actions on a self-hosted runner. Every step is
idempotent, so a partial failure just means the next run picks up where
this one left off - no manual cleanup needed.
"""
import subprocess
import sys

# sys.executable, not "python3": under cron the PATH is not your shell's, so a
# bare "python3" resolves to the system interpreter, which has none of this
# project's dependencies installed (torch, chromadb) and fails at import.
PY = sys.executable

STEPS = [
    ("Ingest (posts + comments, rolling window)", [PY, "-m", "scripts.nightly_ingest"]),
    # Articles were absent from this list until 2026-08-29, so they only ever
    # advanced when someone ran the GDELT backfill by hand - publication dates
    # had drifted 11 days behind Reddit. They no longer feed sentiment, but they
    # still date the corpus and are the basis for factual grounding later.
    ("Ingest articles (RSS)", [PY, "-m", "ingest.article_ingest"]),
    ("Clean", [PY, "-m", "enrichment.clean"]),
    ("Score (incremental)", [PY, "-m", "scripts.score_incremental"]),
    ("Sync to Chroma (incremental)", [PY, "-m", "scripts.nightly_sync_chroma"]),
    ("Refresh league mood cache", [PY, "-m", "scripts.refresh_mood_cache"]),
]


def run():
    for name, cmd in STEPS:
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {name} (exit code {result.returncode})", flush=True)
            sys.exit(result.returncode)
    print("\n=== Nightly pipeline complete ===", flush=True)


if __name__ == "__main__":
    run()
