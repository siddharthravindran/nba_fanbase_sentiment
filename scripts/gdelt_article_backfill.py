"""Historical NBA article backfill via GDELT's DOC 2.0 API.

article_ingest.py only sees whatever's currently in a handful of live RSS
feeds (ESPN/Yahoo/CBS) - RSS is not a historical archive, so it can't reach
back to July 2025 the way Arctic Shift does for Reddit. GDELT indexes news
across thousands of domains (including sites like Bleacher Report, The
Ringer, RealGM that don't expose a usable RSS feed) going back to 2017, and
returns article URLs + publish dates for a keyword+date-range query - those
URLs then reuse the exact same trafilatura extraction / team-tagging /
storage path as article_ingest.py.

GDELT enforces "one request every 5 seconds" - this script paces requests
accordingly and chunks each team's date range into weekly windows (DOC 2.0
returns at most ~250 records per query, so long ranges need to be split to
avoid silently missing articles beyond that cap).

Usage: python -m scripts.gdelt_article_backfill
       python -m scripts.gdelt_article_backfill --team Celtics  # single team, for testing
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests

from ingest.article_ingest import _fetch_text, detect_teams, to_docs
from ingest.storage import get_connection, upsert_docs, count_docs
from ingest.teams import TEAM_ALIASES

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_DELAY_SEC = 6.0  # GDELT asks for >=5s between requests
MAX_RECORDS = 250  # DOC 2.0's practical per-query cap
WINDOW_DAYS = 7  # split each team's range into weekly chunks to stay under the cap
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 10.0

# Tracks which teams have fully completed a run, so an interrupted (Ctrl+C,
# sleep, crash) multi-hour backfill can resume without re-scanning every
# weekly window for teams already done - URL dedup alone would still skip
# duplicate inserts, but re-scanning wastes ~57 GDELT requests (~6min) per
# already-finished team for zero new rows.
STATE_PATH = "data/gdelt_backfill_state.json"


def _load_completed_teams() -> set[str]:
    if not os.path.exists(STATE_PATH):
        return set()
    with open(STATE_PATH) as f:
        return set(json.load(f))


def _mark_team_complete(team: str) -> None:
    completed = _load_completed_teams()
    completed.add(team)
    with open(STATE_PATH, "w") as f:
        json.dump(sorted(completed), f)

START = datetime(2025, 7, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)

# Optional: narrow to specific outlets (e.g. "bleacherreport.com") instead of
# all domains GDELT indexes. Empty = no restriction.
DOMAIN_ALLOWLIST: list[str] = []


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _build_query(team_alias: str) -> str:
    # GDELT rejects quoted phrases of 3 chars or less ("too short"), so "NBA"
    # can't be used as a quoted co-filter term - "basketball" does the same
    # job of disambiguating from non-sports usage of the team alias.
    query = f'"{team_alias}" "basketball"'
    if DOMAIN_ALLOWLIST:
        domain_clause = " OR ".join(f"domain:{d}" for d in DOMAIN_ALLOWLIST)
        query = f"{query} ({domain_clause})"
    return query


def _fetch_window(
    query: str, after: datetime, before: datetime
) -> tuple[list[dict] | None, str]:
    """Returns (articles, reason). articles is a (possibly empty) list on
    success, or None if every attempt failed - `reason` then says why.

    The success/failure distinction matters: GDELT legitimately returns zero
    results for many windows, but a dropped connection or a throttle also
    yields nothing - conflating them silently marks a team "complete" with no
    data (and the resume state file would then skip it forever).

    Note GDELT signals several errors as HTTP 200 with a plain-text body
    instead of JSON (rate limiting, malformed queries). Those must be surfaced,
    not swallowed, or they look identical to a network outage."""
    reason = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                GDELT_URL,
                params={
                    "query": query,
                    "mode": "artlist",
                    "startdatetime": _fmt(after),
                    "enddatetime": _fmt(before),
                    "maxrecords": MAX_RECORDS,
                    "format": "json",
                },
                timeout=30,
            )
            if resp.status_code == 429:
                reason = "HTTP 429 rate limited"
                time.sleep(RETRY_BACKOFF_SEC * attempt)
                continue
            resp.raise_for_status()
            try:
                return resp.json().get("articles", []), "ok"
            except ValueError:
                reason = f"non-JSON body: {resp.text.strip()[:120]!r}"
                time.sleep(RETRY_BACKOFF_SEC * attempt)
                continue
        except requests.exceptions.RequestException as e:
            # Unwrap to the innermost cause - requests nests the real error
            # (SSL/socket) several layers deep behind "Max retries exceeded".
            root = e
            while getattr(root, "__cause__", None) or getattr(root, "__context__", None):
                root = root.__cause__ or root.__context__
            reason = f"{type(e).__name__}: {type(root).__name__}: {str(root)[:160]}"
            if attempt == MAX_RETRIES:
                return None, reason
            time.sleep(RETRY_BACKOFF_SEC * attempt)
    return None, reason


def _parse_seendate(seendate: str) -> str | None:
    # GDELT format: "20250701T180000Z"
    try:
        dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError):
        return None


def backfill(team_filter: str | None = None):
    conn = get_connection()
    existing_urls = {
        row[0] for row in conn.execute("SELECT DISTINCT url FROM raw_docs WHERE source = 'article'")
    }
    seen_urls = set(existing_urls)

    total_articles, total_docs = 0, 0

    teams = TEAM_ALIASES.items()
    if team_filter:
        teams = [(t, a) for t, a in teams if t == team_filter]
        if not teams:
            print(f"Unknown team: {team_filter!r}. Valid teams: {sorted(TEAM_ALIASES)}")
            return

    completed_teams = _load_completed_teams()
    total_windows = ((END - START).days // WINDOW_DAYS) + 1

    for team, aliases in teams:
        if team in completed_teams:
            print(f"[{team}] already completed in a prior run, skipping")
            continue

        alias = aliases[0]  # most distinctive nickname, avoids common-word false positives
        query = _build_query(alias)
        team_docs = 0
        window_num = 0
        failed_windows = 0
        empty_windows = 0
        consecutive_failures = 0

        window_start = START
        while window_start < END:
            window_num += 1
            window_end = min(window_start + timedelta(days=WINDOW_DAYS), END)
            print(
                f"[{team}] window {window_num}/{total_windows} "
                f"({window_start.date()} to {window_end.date()})...",
                end=" ",
                flush=True,
            )
            articles, reason = _fetch_window(query, window_start, window_end)
            if articles is None:
                failed_windows += 1
                consecutive_failures += 1
                print(f"FAILED - {reason}")
                articles = []
                # Once several windows fail back to back it's a sustained
                # problem (throttle or connection loss), not a blip. Hammering
                # a rate limiter keeps it angry, so ease off before continuing.
                if consecutive_failures % 5 == 0:
                    cooldown = min(60 * consecutive_failures / 5, 300)
                    print(f"  ...{consecutive_failures} failures in a row, cooling down {cooldown:.0f}s")
                    time.sleep(cooldown)
            else:
                consecutive_failures = 0
                if not articles:
                    empty_windows += 1
                print(f"{len(articles)} candidates")
            time.sleep(REQUEST_DELAY_SEC)

            batch = []
            for article in articles:
                url = article.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                text_body = _fetch_text(url)
                if not text_body:
                    continue

                title = article.get("title", "")
                full_text = f"{title}\n{text_body}".strip()
                teams = detect_teams(full_text)
                if not teams:
                    continue

                doc = {
                    "url": url,
                    "text": full_text,
                    "created_utc": _parse_seendate(article.get("seendate", "")),
                    "teams": teams,
                }
                batch.extend(to_docs(doc))
                total_articles += 1
                team_docs += len(to_docs(doc))

            if batch:
                upsert_docs(conn, batch)
                total_docs += len(batch)

            window_start = window_end

        # A team returning zero candidates in *every* window is not a real
        # result - every NBA team gets news coverage across a 13-month range.
        # It means GDELT was degraded (it answers 200 with a non-article body
        # when rate-limited or when a query is malformed), so don't let the
        # resume state file record it as done.
        if failed_windows or empty_windows == window_num:
            reason = (
                f"{failed_windows}/{total_windows} windows failed"
                if failed_windows
                else "every window returned 0 candidates (GDELT likely degraded)"
            )
            print(f"[{team}] INCOMPLETE - {team_docs} rows added, {reason}; re-run to retry this team")
        else:
            print(f"[{team}] done - {team_docs} article rows added")
            _mark_team_complete(team)

    print(f"\nTotal article rows added: {total_docs}")
    print(f"Total docs in DB: {count_docs(conn)}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", help="Run for a single team only, e.g. --team Celtics (for testing)")
    args = parser.parse_args()
    backfill(team_filter=args.team)
