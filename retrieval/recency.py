"""Exponential time-decay weighting for sentiment aggregation.

The corpus spans over a year, so an unweighted count answers "how has this
fanbase felt since July 2025", not "how do they feel now". Decay makes recent
posts dominate without discarding older ones outright, which matters because a
hard cutoff is brittle here: in the offseason the quietest fanbase produces
only ~64 docs in 7 days, and adding a topic filter on top of that can leave a
handful of posts (or none) to aggregate.

Weight is exp(-ln(2) * age_days / half_life_days), i.e. a post loses half its
influence every `half_life_days`. Applied to the *count* each doc contributes
to its emotion bucket - not to the model's `top_score`, which is classifier
confidence and means something unrelated to age.
"""
import math
from datetime import datetime, timezone

# ~3 weeks: long enough that a normal week of discussion still carries weight,
# short enough that last season doesn't drown out the current news cycle.
DEFAULT_HALF_LIFE_DAYS = 21


def parse_utc(value: str | None) -> datetime | None:
    """created_utc is ISO-8601 with an offset for Reddit/GDELT rows, but Chroma
    metadata can hand back an empty string for docs embedded before the field
    was populated. Returns None rather than raising so callers can skip."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def decay_weight(
    created_utc: str | None,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Weight in (0, 1]. Undated docs get the floor weight instead of 0 so they
    still count for something; dropping them would silently bias the mix toward
    whichever sources happen to carry timestamps."""
    parsed = parse_utc(created_utc)
    if parsed is None:
        return 0.05
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - parsed).total_seconds() / 86400)
    return math.exp(-math.log(2) * age_days / half_life_days)


# Standard RRF constant. Dampens the top of each ranking so a single #1 hit
# can't dominate the fused order on its own.
RRF_K = 60


def fuse_relevance_recency(
    created_utcs: list[str | None],
    distances: list[float] | None = None,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> list[int]:
    """Rank indices by relevance AND recency, best first, via Reciprocal Rank
    Fusion: score = 1/(k+rank_relevance) + 1/(k+rank_recency).

    Multiplying a similarity score by a decay weight does NOT work here - the
    two scales are incomparable. Embedding distances for a filtered pool span
    a narrow band (~0.56-0.86), while decay spans three orders of magnitude
    across a year, so the product is dominated by recency and the ordering
    collapses to a plain date sort. RRF compares positions instead of values,
    which is scale-free, so a strong older match can still outrank a weak
    recent one.
    """
    n = len(created_utcs)
    if not n:
        return []
    now = now or datetime.now(timezone.utc)

    # Chroma returns candidates already sorted by distance; fall back to that
    # order if distances weren't requested.
    if distances:
        relevance_rank = {
            idx: rank
            for rank, idx in enumerate(sorted(range(n), key=lambda i: distances[i]))
        }
    else:
        relevance_rank = {idx: idx for idx in range(n)}

    weights = [decay_weight(created_utcs[i], now=now, half_life_days=half_life_days) for i in range(n)]
    recency_rank = {
        idx: rank
        for rank, idx in enumerate(sorted(range(n), key=lambda i: weights[i], reverse=True))
    }

    return sorted(
        range(n),
        key=lambda i: 1 / (RRF_K + relevance_rank[i]) + 1 / (RRF_K + recency_rank[i]),
        reverse=True,
    )


# How recent a doc has to be to count for the reserved slots. Wide enough to
# cover an offseason news cycle - a week would be empty for most teams in July.
RECENT_WINDOW_DAYS = 45
# Share of the returned quotes reserved for that window.
RECENT_QUOTA = 0.5


def split_recent(
    created_utcs: list[str | None],
    now: datetime | None = None,
    window_days: float = RECENT_WINDOW_DAYS,
) -> tuple[list[int], list[int]]:
    """Partition indices into (recent, older) by absolute age.

    Separate from decay_weight on purpose: decay is a soft preference that a
    strong old match can outvote, which is the right behavior for ranking but
    means recency can lose every slot. This is the hard cut used to reserve
    slots, so the two aren't interchangeable.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - window_days * 86400
    recent, older = [], []
    for i, value in enumerate(created_utcs):
        parsed = parse_utc(value)
        (recent if parsed and parsed.timestamp() >= cutoff else older).append(i)
    return recent, older


def select_two_tier(
    created_utcs: list[str | None],
    distances: list[float] | None,
    n_results: int,
    now: datetime | None = None,
    window_days: float = RECENT_WINDOW_DAYS,
    quota: float = RECENT_QUOTA,
) -> list[int]:
    """Pick `n_results` indices, reserving a share of them for recent docs.

    Fused ranking alone can't fix a pool that has almost no recent docs in it.
    Recent posts are a thin slice of a fanbase's history - measured on a Lakers
    "new roster additions" query, a top-400 pool held 33 recent docs and only 1
    of them named any of that summer's signings. The offseason moves weren't
    ranked badly; they weren't in the pool. So callers widen the pool (a bigger
    k is still graph traversal, whereas a created_ts range filter forces
    Chroma's brute-force path and costs more), and this splits it by date and
    ranks each side independently so the recent tier can't be crowded out.

    Both tiers are ordered by the same relevance+recency fusion, so the reserved
    slots still go to the best recent matches rather than merely the newest.
    Unfilled quota falls through to the other tier - a team with no recent
    activity returns the same thing it would have before.
    """
    recent, older = split_recent(created_utcs, now=now, window_days=window_days)

    def rank(idxs: list[int]) -> list[int]:
        if not idxs:
            return []
        order = fuse_relevance_recency(
            [created_utcs[i] for i in idxs],
            [distances[i] for i in idxs] if distances else None,
            now=now,
        )
        return [idxs[i] for i in order]

    ranked_recent, ranked_older = rank(recent), rank(older)
    n_recent = min(len(ranked_recent), round(n_results * quota))
    picked = ranked_recent[:n_recent]
    picked += ranked_older[: n_results - len(picked)]
    # Older tier came up short (a topic only discussed recently) - backfill.
    if len(picked) < n_results:
        picked += ranked_recent[n_recent : n_recent + n_results - len(picked)]
    return picked


def weighted_distribution(items: list[tuple[str, float]]) -> list[dict]:
    """items: [(emotion, weight), ...] -> distribution sorted by weight desc.

    `count` is the raw number of docs behind each bucket and `pct` is the
    decay-weighted share, so the percentages reflect recency while the counts
    still show how much real evidence is underneath.
    """
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    for emotion, weight in items:
        label = emotion or "unknown"
        weights[label] = weights.get(label, 0.0) + weight
        counts[label] = counts.get(label, 0) + 1

    total = sum(weights.values())
    return sorted(
        (
            {
                "emotion": emotion,
                "count": counts[emotion],
                "pct": round(100 * weight / total, 1) if total else 0.0,
            }
            for emotion, weight in weights.items()
        ),
        key=lambda d: d["pct"],
        reverse=True,
    )
