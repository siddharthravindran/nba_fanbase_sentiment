#!/usr/bin/env bash
# Run the Chroma index as a server, so the app and the nightly sync can both
# use it at once.
#
# Chroma's default PersistentClient is embedded: every process memory-maps its
# own copy of the HNSW index and flushes it back independently. That is fine for
# one process, but the app is long-running and the nightly job writes, so the
# two overlap - and the failure mode is a corrupted 27GB index that took two
# hours to build, not an error. A server gives the files a single owner.
#
# Must be started BEFORE the app, and nothing else may hold data/chroma while it
# runs. Clients opt in via CHROMA_HOST/CHROMA_PORT in .env.
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${CHROMA_HOST:-127.0.0.1}"
PORT="${CHROMA_PORT:-8000}"

# Bind to loopback by default: the server has no authentication, so exposing it
# on 0.0.0.0 would publish the whole corpus to the local network.
exec chroma run --path data/chroma --host "$HOST" --port "$PORT"
