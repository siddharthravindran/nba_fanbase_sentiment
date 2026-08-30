"""Look up recent news coverage of a named player, to ground roster/transaction
claims against journalism instead of against fan speculation.

The problem this exists for: every other retrieval path in this app returns fan
writing, and fan writing does not distinguish a rumor from a settled outcome. A
post reading "we're linked to Kelly Olynyk" and a post reacting to a completed
signing are the same shape, land in the same collection, and score the same way.
The model then states the rumor as fact - which is the single failure mode real
NBA fans catch every time and that a reviewer without NBA knowledge never will.

News articles are the only source in the corpus that reports outcomes rather
than wishes, and they were previously invisible to the chat layer: both existing
tools drop `source='article'` (see EXCLUDED_SOURCES) because a reporter's
sentence must never be quoted back as fan opinion. That exclusion is right for
sentiment and wrong for facts, so this module reaches for articles specifically
and only articles.

Deliberately a *lookup*, not a verifier. It returns what was published and lets
the model read it. It cannot confirm a negative: silence here means no indexed
article matched, which is not evidence that a move didn't happen.
"""
import re

from ingest.storage import get_connection

# Articles average 6.5KB and run to 183KB, so returning bodies would blow the
# context window for a handful of lookups. Two excerpts are returned instead.
#
# The headline plus lede carries the fact when the article is *about* the player
# ("Free agent wing Jonathan Kuminga agreed to a two-year, $12.4 million deal
# with the Minnesota Timberwolves"), because news copy is an inverted pyramid.
#
# But the lede alone is worse than useless when the player is mentioned in
# passing. Measured on Kelly Olynyk - the exact claim a fan caught this app
# fabricating - the three most recent matching articles are offseason roundups
# whose ledes are about LeBron James, James Harden and the Spurs' calendar.
# Handing the model a LeBron lede as evidence about Olynyk invites it to infer
# from an article it cannot actually read. So also return a window centered on
# the player's first mention, which is the sentence that says what he did.
LEDE_CHARS = 400
MENTION_WINDOW = 360

# Article rows are stored one per (article, team) - `detect_teams` tags a single
# ESPN story with every franchise it mentions, so the Kuminga signing exists as
# 6+ identical rows under Hawks, Bulls, Warriors, Lakers, Wolves and Knicks.
# Returning those raw would be actively harmful: the model would read six copies
# of one story as six independent sources corroborating each other. Dedup by URL
# is what makes one story count once.
DEFAULT_LIMIT = 5
MAX_LIMIT = 10

# Headline matches are returned separately from recent mentions, and this is the
# fix for the failure that motivated the split. Ranking purely by recency hides
# the transaction: Jonathan Kuminga's five newest articles are all from August
# 2026 and concern Minnesota, while the story establishing he was ever a Hawk
# ("Jonathan Kuminga dud in Warriors-Hawks game shows why trade was right") is
# from March and never surfaces. A player's defining move is usually months old
# and buried under recent chatter, so newest-first is close to the worst
# possible selection for a roster question.
#
# A name in the headline is a cheap, strong signal that the article is *about*
# him rather than mentioning him in passing - it separates "Kuminga joining
# Wolves on 2-year, $12.4M deal" from a LeBron roundup that lists him among
# available free agents. Those are returned oldest-first, because for a
# transaction the sequence is the fact: Warriors -> Hawks at the February
# deadline -> option declined -> free agent -> Wolves. A timeline makes the
# rumor/settled boundary visible; a scatter of articles invites the model to
# collapse it.
MAX_TIMELINE = 8


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _excerpt(url: str, ts: str, text: str, name: str) -> dict:
    text = text or ""
    # article_ingest stores "title\nbody", so the first line is the headline.
    headline, _, body = text.partition("\n")
    lede = _squash(text[:LEDE_CHARS])

    # Case-insensitive so "KUMINGA" in an all-caps headline still anchors.
    pos = text.lower().find(name.lower())
    mention = ""
    if pos != -1:
        start = max(0, pos - MENTION_WINDOW // 3)
        mention = _squash(text[start : pos + MENTION_WINDOW])

    article = {
        "published": ts,
        "url": url,
        "headline": _squash(headline),
        "lede": lede,
    }
    # Only include the mention window when it adds something the lede didn't
    # already cover - otherwise it's the same sentence twice, which wastes
    # context and reads as two separate pieces of evidence.
    if mention and mention not in lede:
        article["mention_context"] = mention
        # A player absent from his own article's opening is being mentioned in
        # passing, so the headline and lede are about someone else. Saying so
        # stops the model treating a LeBron headline as an Olynyk source.
        if name.lower() not in lede.lower():
            article["note"] = (
                f"{name} is not in this article's headline or opening - the "
                "article is primarily about something else and mentions him in "
                "passing. Judge only from mention_context, not the headline."
            )
    return article


def _dedup(rows, name, seen_headlines):
    """One entry per distinct headline.

    GROUP BY url is not enough. Wire copy is syndicated, so the same story lives
    at several URLs - "Jonathan Kuminga dud in Warriors-Hawks game shows why
    trade was right" appears three times under three domains. Passing all three
    to the model presents one story as three sources agreeing with each other,
    which is the same false-corroboration failure the per-team fan-out causes.
    """
    out = []
    for url, ts, text in rows:
        article = _excerpt(url, ts, text, name)
        key = article["headline"].lower()
        if key in seen_headlines:
            continue
        seen_headlines.add(key)
        out.append(article)
    return out


def lookup_player_news(player: str, limit: int = DEFAULT_LIMIT) -> dict:
    """News coverage of `player`, split into a timeline and recent mentions.

    `timeline` is articles with his name in the headline, oldest first - stories
    written *about* him, which is where transactions live. `recent_mentions` is
    the newest articles mentioning him at all, which answers "what is his status
    now" but is often passing references.
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    name = (player or "").strip()
    if not name:
        return {"player": player, "timeline": [], "recent_mentions": [],
                "note": "No player name given."}

    like = f"%{name}%"
    conn = get_connection()
    try:
        # instr(...) < instr(text, newline) == the name occurs before the first
        # line break, i.e. inside the headline, since article_ingest stores
        # "title\nbody". Lowercased both sides so an all-caps headline matches.
        headline_rows = conn.execute(
            """
            SELECT url, MAX(created_utc) AS ts, text
            FROM clean_docs
            WHERE source = 'article'
              AND instr(lower(text), lower(?)) > 0
              AND instr(lower(text), lower(?)) < instr(text, char(10))
            GROUP BY url
            ORDER BY ts ASC
            """,
            (name, name),
        ).fetchall()

        recent_rows = conn.execute(
            """
            SELECT url, MAX(created_utc) AS ts, text
            FROM clean_docs
            WHERE source = 'article' AND text LIKE ?
            GROUP BY url
            ORDER BY ts DESC
            LIMIT ?
            """,
            (like, limit * 3),
        ).fetchall()

        total, first_seen, last_seen = conn.execute(
            """
            SELECT COUNT(DISTINCT url), MIN(created_utc), MAX(created_utc)
            FROM clean_docs
            WHERE source = 'article' AND text LIKE ?
            """,
            (like,),
        ).fetchone()
    finally:
        conn.close()

    seen: set[str] = set()
    # Timeline first so it owns the dedup slots: if a story is both about him and
    # recent, it belongs in the timeline, and repeating it under recent_mentions
    # would double-count it.
    timeline = _dedup(headline_rows, name, seen)
    if len(timeline) > MAX_TIMELINE:
        # Thin evenly across the span rather than keeping the two ends. Keeping
        # the ends looks reasonable and measurably loses the answer: on Kuminga
        # it dropped "Jonathan Kuminga dud in Warriors-Hawks game shows why
        # trade was right" - the single article that establishes he was a Hawk -
        # because the trade fell in the middle of his coverage. Mid-career moves
        # are exactly the thing being looked up, so the middle cannot be the
        # part that gets discarded. First and last are pinned since they bracket
        # the range the model is reasoning over.
        step = (len(timeline) - 1) / (MAX_TIMELINE - 1)
        timeline = [timeline[round(i * step)] for i in range(MAX_TIMELINE)]
    recent = _dedup(recent_rows, name, seen)[:limit]

    # Coverage volume, so the model can tell "no news exists about him" apart
    # from "he is real but marginal". The corpus carries 700-1,600 articles every
    # month since July 2025 with no gaps, so thin coverage of a player is a fact
    # about the player, not about the index. A fringe free agent legitimately
    # shows up only in roundups and never in a headline, and that pattern is the
    # answer to "is he on this roster" - not a reason to hedge.
    coverage = {
        "articles_mentioning_him": total or 0,
        "articles_about_him": len(timeline),
        "earliest": (first_seen or "")[:10] or None,
        "latest": (last_seen or "")[:10] or None,
    }

    if not timeline and not recent:
        return {
            "player": name,
            "timeline": [],
            "recent_mentions": [],
            "coverage": coverage,
            # Spelled out because an empty result is the most dangerous output
            # here. The model must not read "no articles" as "the rumor is
            # false" OR as permission to fall back on its own recollection -
            # both turn a coverage gap into a confident factual claim.
            "note": (
                f"No indexed article mentions {name!r}. This means the news corpus "
                "has no coverage of him - NOT that a move did or didn't happen. "
                "Do not treat this as confirmation either way, and do not "
                "substitute your own knowledge; say his status is unconfirmed."
            ),
        }

    result = {
        "player": name,
        "timeline": timeline,
        "recent_mentions": recent,
        "coverage": coverage,
    }
    if not timeline:
        result["note"] = (
            f"No article headline names {name} - he is only mentioned inside "
            "articles about other subjects. That is typical of a fringe or "
            "unsigned player, and it is evidence he has not been the subject of "
            "a reported move. Do not assert he is on any roster."
        )
    return result
