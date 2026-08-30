"""Orchestrates the nightly update pipeline end to end:
ingest (rolling window) -> clean -> score (incremental) -> sync to Chroma.

Run via cron/GitHub Actions on a self-hosted runner. Every step is
idempotent, so a partial failure just means the next run picks up where
this one left off - no manual cleanup needed.
"""
import subprocess
import sys

from dotenv import load_dotenv

# Load .env in the orchestrator so every step inherits it - load_dotenv writes
# into os.environ, and subprocess passes that down. This is what points the
# Chroma sync at the server (CHROMA_HOST) instead of opening data/chroma
# directly. Without it the sync silently falls back to embedded mode and writes
# to the index underneath a running app, which is the exact concurrent-writer
# case the server exists to prevent - and it fails by corrupting a 27GB index
# rather than by raising.
load_dotenv()

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
    # RSS is a bounded feed, not an archive: it holds a fixed number of recent
    # entries, so any night this pipeline doesn't run is a day of articles RSS
    # can never return. GDELT indexes by date range, so re-scanning the last 5
    # days heals those gaps on the next successful run. The overlap is nearly
    # free because URL dedup happens before the article text is fetched, so a
    # steady-state night re-checks 30 cheap queries and downloads only what's
    # actually new.
    ("Ingest articles (GDELT, trailing 5 days)",
     [PY, "-m", "scripts.gdelt_article_backfill", "--days", "5"]),
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
