"""Orchestrates the nightly update pipeline end to end:
ingest (rolling window) -> clean -> score (incremental) -> sync to Chroma.

Run via cron/GitHub Actions on a self-hosted runner. Every step is
idempotent, so a partial failure just means the next run picks up where
this one left off - no manual cleanup needed.
"""
import subprocess
import sys

STEPS = [
    ("Ingest (posts + comments, rolling window)", ["python3", "-m", "scripts.nightly_ingest"]),
    ("Clean", ["python3", "-m", "enrichment.clean"]),
    ("Score (incremental)", ["python3", "-m", "scripts.score_incremental"]),
    ("Sync to Chroma (incremental)", ["python3", "-m", "scripts.nightly_sync_chroma"]),
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
