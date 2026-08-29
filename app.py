import html
import time
from datetime import datetime

import altair as alt
import anthropic
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from chat.assistant import stream_chat
from ingest.storage import get_connection
from retrieval.aggregate import INDEX_SQL, MOOD_CACHE_SQL, refresh_team_mood_cache
from retrieval.vector_store import TeamCollections

load_dotenv(override=True)

MAX_TURNS_PER_SESSION = 15  # caps API spend per visitor session

# Picked by actual recent volume, not vibes - an example prompt that lands on a
# thin slice of the corpus produces a weak answer and reads as the whole app
# being weak. Counts are docs since 2026-06 matching the topic.
#
# These deliberately ask open questions instead of asserting a premise. An
# earlier set included "Are Knicks fans sold on Brunson as the guy?" - written
# without checking, and by then he'd won Finals MVP two months earlier. To an
# NBA fan that single line discredits the whole product before they've asked
# anything. A prompt that only asks "how do fans feel about X" cannot be
# out of date that way.
EXAMPLE_PROMPTS = [
    "How do Lakers fans feel about the new-look roster?",       # 2,888
    "What are Spurs fans saying about Wembanyama?",             # 7,689
    "How are Knicks fans feeling after their Finals run?",      # 6,419
    "How do Nuggets fans feel about Jokic's supporting cast?",  # 15,752
]

SOURCE_LABELS = {
    "reddit_post": "Reddit post",
    "reddit_comment": "Reddit comment",
    "article": "Article",
}

# Streamlit's default avatars are initials pulled from the OS username, which
# renders as a random fragment of the developer's name in a colored circle.
USER_AVATAR = "👤"
BOT_AVATAR = "🏀"

# Throttle repaints on wall-clock, not character count. Token delivery is
# bursty, so a character threshold fires in clumps - several repaints back to
# back, then a pause - which is exactly what reads as choppy. A time budget
# gives a steady cadence no matter how the tokens arrive.
#
# Each repaint re-sends the whole answer and re-parses it as markdown, so the
# cost grows as the answer does; at ~15/s a long answer spends real time
# re-rendering text the reader has already read. This is the ceiling, not the
# target - it repaints only when new text has actually arrived.
STREAM_FLUSH_SECONDS = 0.07

# Grouping the 9 model labels by valence lets the charts encode meaning in
# color (is this a happy fanbase or a miserable one?) instead of handing out
# arbitrary categorical colors, and gives the league leaderboard a single
# net-positive score to rank on.
POSITIVE = {"excitement or hype", "hope or optimism", "pride"}
NEGATIVE = {"pessimism or resignation", "disappointment", "anger or frustration", "sadness"}

EMOTION_COLORS = {
    "excitement or hype": "#2e9e5b",
    "hope or optimism": "#5fbf82",
    "pride": "#8fd3a6",
    "neutral analysis or discussion": "#9aa0a6",
    "mockery or sarcasm": "#d99b2e",
    "disappointment": "#e07a5f",
    "pessimism or resignation": "#d1573f",
    "anger or frustration": "#b3402a",
    "sadness": "#8c5a7d",
}

# One accent per team, used only as a chrome accent - never for the emotion
# bars themselves, whose colors encode valence and are load-bearing.
# Not official brand hex: roughly a third of the league's primary color is navy
# or black, which is invisible on a #0B0D12 background, so those teams fall back
# to their secondary (Nuggets/Lakers gold, Wolves green, Nets white).
TEAM_COLORS = {
    "Atlanta Hawks": "#E5575B",
    "Boston Celtics": "#00A15C",
    "Brooklyn Nets": "#E7EAEE",
    "Charlotte Hornets": "#00A2B8",
    "Chicago Bulls": "#F04772",
    "Cleveland Cavaliers": "#FDBB30",
    "Dallas Mavericks": "#118CDF",
    "Denver Nuggets": "#FEC524",
    "Detroit Pistons": "#E55566",
    "Golden State Warriors": "#FFC72C",
    "Houston Rockets": "#E75272",
    "Indiana Pacers": "#FDBB30",
    "LA Clippers": "#E55566",
    "Los Angeles Lakers": "#FDB927",
    "Memphis Grizzlies": "#7D97CC",
    "Miami Heat": "#EC4C6E",
    "Milwaukee Bucks": "#17A44A",
    "Minnesota Timberwolves": "#78BE20",
    "New Orleans Pelicans": "#EC4D66",
    "New York Knicks": "#F58426",
    "Oklahoma City Thunder": "#00A2E8",
    "Orlando Magic": "#0092E8",
    "Philadelphia 76ers": "#1D8FE0",
    "Phoenix Suns": "#E56020",
    "Portland Trail Blazers": "#E5575B",
    "Sacramento Kings": "#9E70DD",
    "San Antonio Spurs": "#C4CED4",
    "Toronto Raptors": "#E75272",
    "Utah Jazz": "#F9A01B",
    "Washington Wizards": "#EC4D66",
}
DEFAULT_ACCENT = "#8A93A3"


TIMING_LOG = "data/timings.log"


def _log_timing(msg: str):
    """File, not stdout: latency is ours to tune, not something a visitor should
    be shown, and stdout lands in whichever terminal happens to own the process
    (there can be more than one Streamlit instance), which makes it unreadable
    after the fact."""
    try:
        with open(TIMING_LOG, "a") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    except OSError:
        pass


@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic()


@st.cache_resource
def get_cached_collection():
    # A router, not a collection: it picks the per-team collection so queries
    # skip the O(N_team) metadata pre-filter. Falls back to the single
    # collection for any team that hasn't been migrated yet.
    return TeamCollections()


@st.cache_data(ttl=3600, show_spinner=False)
def get_corpus_stats() -> dict:
    """Headline numbers + the league-wide mood table, in one pass, cached for an
    hour. The GROUP BY is served by a (team, top_emotion) covering index; without
    it this scans several million rows and stalls first paint for ~6 seconds."""
    conn = get_connection()
    try:
        conn.execute(INDEX_SQL)
        scored = conn.execute("SELECT COUNT(*) FROM sentiment_docs_v2").fetchone()[0]
        # MAX(created_utc) over clean_docs would overstate this: ingesting a doc
        # doesn't make it answerable, being scored and embedded does. Walking the
        # created_utc index descending and stopping at the first scored row costs
        # about the same and reports a date the assistant can actually speak to.
        latest = conn.execute(
            """
            SELECT c.created_utc FROM clean_docs c
            WHERE EXISTS (SELECT 1 FROM sentiment_docs_v2 s WHERE s.id = c.id)
            ORDER BY c.created_utc DESC LIMIT 1
            """
        ).fetchone()
        latest = latest[0] if latest else None
        # Recency-weighted, same 21-day half-life the chat tools use. An
        # unweighted count here averages 14 months into one bar, so a team that
        # just had a good week still shows last season's mood - the chart would
        # contradict the answers the assistant gives about the same team.
        # Read from the materialized table; computing this live costs 38s
        # because the created_utc join defeats the covering index.
        conn.execute(MOOD_CACHE_SQL)
        rows = conn.execute(
            "SELECT team, top_emotion, n_docs, weight FROM team_mood_cache"
        ).fetchall()
        if not rows:
            # First run before the nightly job has ever populated it.
            refresh_team_mood_cache(conn)
            rows = conn.execute(
                "SELECT team, top_emotion, n_docs, weight FROM team_mood_cache"
            ).fetchall()
    finally:
        conn.close()

    totals: dict[str, float] = {}
    doc_counts: dict[str, int] = {}
    positive: dict[str, float] = {}
    negative: dict[str, float] = {}
    for team, emotion, count, weight in rows:
        weight = weight or 0.0
        totals[team] = totals.get(team, 0.0) + weight
        doc_counts[team] = doc_counts.get(team, 0) + count
        if emotion in POSITIVE:
            positive[team] = positive.get(team, 0.0) + weight
        elif emotion in NEGATIVE:
            negative[team] = negative.get(team, 0.0) + weight

    mood = [
        {
            "team": team,
            "net": round(100 * (positive.get(team, 0.0) - negative.get(team, 0.0)) / total, 1),
            # Raw doc count, not the weight - the tooltip should say how much
            # real evidence sits behind the bar, not the decayed float.
            "posts": doc_counts.get(team, 0),
        }
        for team, total in totals.items()
        if total
    ]
    mood.sort(key=lambda r: r["net"], reverse=True)

    return {
        "scored": scored,
        "latest": latest[:10] if latest else None,
        "teams": len(totals),
        "mood": mood,
    }


def _style(chart):
    """Vega's defaults (gridlines, axis domains, its own font) fight the page's
    dark palette, so strip the chrome and match the app's typography."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFont="Inter",
            titleFont="Inter",
            labelColor="#98A1B0",
            titleColor="#6F7889",
            labelFontSize=12,
            titleFontSize=11,
            domain=False,
            tickSize=0,
            gridColor="#1C222D",
        )
        .configure_axisY(labelPadding=8)
        # Numbers in mono, category names in Inter: the digits line up column-wise
        # across redraws instead of shifting as values change between questions.
        .configure_axisX(labelFont="JetBrains Mono", labelFontSize=11)
    )


def emotion_chart(dist: list[dict], examples: dict[str, str] | None = None):
    """Horizontal bars colored by emotional valence, so the shape of a fanbase's
    mood is readable at a glance without parsing nine label names."""
    df = pd.DataFrame(dist)
    df["example"] = df["emotion"].map(lambda e: (examples or {}).get(e, ""))
    order = df.sort_values("pct", ascending=False)["emotion"].tolist()
    known = [e for e in order if e in EMOTION_COLORS]

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=alt.RelativeBandSize(0.68))
        .encode(
            # Vega's default tick density puts a label every 2% here, which is
            # noise at this scale - the shape of the distribution is the point,
            # not reading an exact value off the axis.
            x=alt.X(
                "pct:Q",
                title="% of posts",
                axis=alt.Axis(format=".0f", grid=True, tickMinStep=5),
            ),
            y=alt.Y(
                "emotion:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=220, grid=False),
            ),
            color=alt.Color(
                "emotion:N",
                scale=alt.Scale(domain=known, range=[EMOTION_COLORS[e] for e in known]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("emotion:N", title="Emotion"),
                alt.Tooltip("pct:Q", title="% of posts", format=".1f"),
                alt.Tooltip("count:Q", title="Posts", format=","),
                alt.Tooltip("example:N", title="Example"),
            ],
        )
        .properties(height=30 * len(df))
    )
    return _style(chart)


def league_mood_chart(mood: list[dict]):
    """Every fanbase ranked by net positive sentiment. This is the thing a
    visitor should see before typing anything - it proves there's real scored
    data behind the chat box instead of asking them to take it on faith."""
    df = pd.DataFrame(mood)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, height=alt.RelativeBandSize(0.62))
        .encode(
            x=alt.X("net:Q", title="Net sentiment  (% positive − % negative)", axis=alt.Axis(grid=True)),
            y=alt.Y(
                "team:N",
                sort=df["team"].tolist(),
                title=None,
                # 30 bands is enough for Vega to start dropping labels to avoid
                # overlap; every team needs to stay named for this to be useful.
                axis=alt.Axis(labelOverlap=False, labelLimit=220, grid=False),
            ),
            color=alt.condition(alt.datum.net > 0, alt.value("#3FB27A"), alt.value("#F0603C")),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("net:Q", title="Net sentiment", format="+.1f"),
                alt.Tooltip("posts:Q", title="Posts scored", format=","),
            ],
        )
        .properties(height=24 * len(df))
    )
    return _style(chart)


def render_distribution(call: dict, examples: dict[str, str] | None = None):
    """Show the emotion breakdown the model actually saw. The written answer
    stays narrative; the numbers live here so the model doesn't burn prose
    reciting percentages."""
    result = call["result"]
    dist = result.get("distribution")
    if not dist:
        return

    top = dist[0]
    team = result.get("team")
    scope = result.get("topic") or "all posts"
    parts = [scope, f"{result.get('n_docs', 0):,} posts scored"]

    # The percentages are a recency-weighted share unless a date range was
    # pinned, so don't call the leader "most common" - with weighting applied
    # it need not be the largest raw count.
    half_life = result.get("half_life_days")
    date_range = result.get("date_range") or {}
    if half_life:
        parts.append(f"recency-weighted ({half_life}-day half-life)")
    elif date_range.get("since") or date_range.get("until"):
        window = f"{date_range.get('since') or 'start'} → {date_range.get('until') or 'now'}"
        parts.append(window)

    parts.append(f"leading: {top['emotion']} ({top['pct']}%)")

    accent = TEAM_COLORS.get(team, DEFAULT_ACCENT)
    st.markdown(
        f'<div class="team-head" style="border-left-color:{accent}">'
        f'<span class="team-name" style="color:{accent}">{html.escape(team or "")}</span>'
        f'<span class="team-meta">{html.escape(" · ".join(parts))}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.altair_chart(emotion_chart(dist, examples), width="stretch")


def render_quotes(quotes: list[dict]):
    """The retrieved evidence, linked back to the original thread or article so
    a reader can verify the answer instead of trusting the model."""
    seen, unique = set(), []
    for q in quotes:
        key = (q.get("text") or "")[:200]
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
    if not unique:
        return

    with st.expander(f"Sources ({len(unique)})"):
        for q in unique:
            text = (q.get("text") or "").strip().replace("\n", " ")
            if len(text) > 400:
                text = text[:400].rstrip() + "…"

            meta = [SOURCE_LABELS.get(q.get("source"), q.get("source") or "unknown")]
            if q.get("created_utc"):
                meta.append(q["created_utc"][:10])
            if q.get("top_emotion"):
                meta.append(q["top_emotion"])
            line = " · ".join(meta)
            if q.get("url"):
                line += f" · [open source]({q['url']})"

            st.markdown(f"> {text}")
            st.caption(line)


def describe_tool_call(call: dict) -> str:
    team = call["input"].get("team", "the league")
    topic = call["input"].get("topic")
    if call["name"] == "aggregate_sentiment":
        scope = f"on {topic}" if topic else "overall"
        return f"Scoring {team} fan sentiment {scope}"
    return f"Pulling {team} fan quotes about {topic}"


def summarize_evidence(tool_calls: list[dict], elapsed: float | None = None) -> str:
    posts = sum(
        c["result"].get("n_docs", 0) for c in tool_calls if c["name"] == "aggregate_sentiment"
    )
    quotes = sum(
        len(c["result"].get("quotes", [])) for c in tool_calls if c["name"] == "retrieve_quotes"
    )
    parts = []
    if posts:
        parts.append(f"{posts:,} scored posts")
    if quotes:
        parts.append(f"{quotes} quotes")
    label = f"Answered from {' and '.join(parts)}" if parts else "Answered"
    return f"{label} · {elapsed:.1f}s" if elapsed else label


def render_evidence(tool_calls: list[dict]):
    quotes = [
        q
        for call in tool_calls
        if call["name"] == "retrieve_quotes"
        for q in call["result"].get("quotes", [])
    ]

    # A bar labelled "mockery or sarcasm - 28%" asks the reader to trust a
    # classifier they can't see. Attaching a real post that was scored into that
    # bucket lets them check the label against the actual language.
    examples: dict[str, str] = {}
    for q in quotes:
        emotion, text = q.get("top_emotion"), " ".join((q.get("text") or "").split())
        if emotion and text and emotion not in examples:
            examples[emotion] = text[:220] + ("..." if len(text) > 220 else "")

    for call in tool_calls:
        if call["name"] == "aggregate_sentiment":
            render_distribution(call, examples)
    render_quotes(quotes)


st.set_page_config(page_title="NBA Fanbase Sentiment", page_icon="🏀", layout="centered")

# Streamlit anchors a page containing st.chat_input at the bottom, which is
# right mid-conversation but wrong on arrival - a first-time visitor lands
# staring at the input box with the league chart scrolled off above them.
# Only force the top on a fresh load; scrolling on every rerun would fight the
# user while they read a streaming answer.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Scope the font override to text elements only. A blanket `*` rule also
       hits Streamlit's Material Symbols spans, and since those render glyphs
       from ligatures, overriding their font shows the raw ligature name
       ("keyboard_arrow_right") on top of the label. */
    html, body, p, li, div, span, label, input, textarea, button,
    h1, h2, h3, h4, h5, h6 { font-family: 'Inter', system-ui, sans-serif; }
    [class*="material-symbols"], [data-testid="stIconMaterial"],
    [data-testid*="Icon"], .material-icons {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Archivo', system-ui, sans-serif; }

    /* Default blocks all snap into place at once, which reads as flat.
       Staggering a short fade makes the page assemble itself. */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: none; }
    }
    [data-testid="stMetric"],
    [data-testid="stChatMessage"],
    [data-testid="stVegaLiteChart"],
    [data-testid="stExpander"] { animation: fadeUp 0.55s cubic-bezier(.2,.7,.3,1) both; }
    [data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"] { animation-delay: 0.09s; }
    [data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"] { animation-delay: 0.18s; }
    [data-testid="stVegaLiteChart"] { animation-delay: 0.26s; }

    .hero { padding: 0.5rem 0 1.75rem 0; }
    .hero-eyebrow {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em;
        text-transform: uppercase; color: #F0603C; margin-bottom: 0.6rem;
    }
    .hero-eyebrow::before {
        content: ""; display: inline-block; width: 7px; height: 7px;
        border-radius: 50%; background: #F0603C; margin-right: 0.55rem;
        vertical-align: middle; animation: pulse 2.4s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
    .hero-title {
        font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 3.1rem;
        line-height: 1.02; letter-spacing: -0.035em; margin: 0 0 0.7rem 0;
        background: linear-gradient(92deg, #FFFFFF 20%, #F0603C 105%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: #97A0AE; font-size: 1.02rem; line-height: 1.55; max-width: 34rem; }

    [data-testid="stMetric"] {
        background: #131822;
        border: 1px solid #232936;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        transition: border-color 0.25s ease, transform 0.25s ease;
    }
    [data-testid="stMetric"]:hover { border-color: #F0603C; transform: translateY(-2px); }
    [data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important; letter-spacing: 0.09em;
        text-transform: uppercase; color: #8A93A3 !important;
    }
    /* Mono + tabular figures for the headline numbers. Gives the stat row an
       analytical feel, and keeps the digits from reflowing when the nightly
       run changes the counts. */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        font-size: 1.6rem !important; letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }

    /* Streamlit gives every markdown block the same margin, so a subheading in
       an answer reads as just another paragraph. Give headings room above so
       the answer has visible structure. */
    [data-testid="stChatMessage"] h3 { margin-top: 1.9rem !important; }
    [data-testid="stChatMessage"] h4 { margin-top: 1.5rem !important; }
    [data-testid="stChatMessage"] h3:first-child,
    [data-testid="stChatMessage"] h4:first-child { margin-top: 0.2rem !important; }

    .section-title {
        font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.32rem;
        letter-spacing: -0.02em; margin: 2.4rem 0 0.3rem 0;
    }
    .section-sub { color: #8A93A3; font-size: 0.88rem; margin-bottom: 1rem; }

    /* Team accent lives here, on the chrome - the bars stay valence-colored. */
    .team-head {
        border-left: 3px solid #8A93A3; padding: 0.1rem 0 0.1rem 0.65rem;
        margin: 0.2rem 0 0.55rem 0; line-height: 1.35;
    }
    .team-head .team-name {
        font-family: 'Archivo', sans-serif; font-weight: 700;
        font-size: 0.95rem; letter-spacing: -0.01em; display: block;
    }
    .team-head .team-meta { color: #8A93A3; font-size: 0.82rem; }

    [data-testid="stChatMessage"] { background: #10151E; border: 1px solid #1D2430; border-radius: 14px; }
    [data-testid="stExpander"] summary { font-weight: 600; }
    blockquote {
        border-left: 3px solid #F0603C !important;
        background: #10151E; border-radius: 0 8px 8px 0;
        padding: 0.65rem 0.9rem !important; margin-bottom: 0.15rem !important;
        color: #C6CDD8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Live fanbase sentiment</div>
      <div class="hero-title">What every NBA<br>fanbase is feeling</div>
      <div class="hero-sub">
        Ask how a fanbase feels about a player, a trade, or last night's game.
        Answers come from real Reddit threads and news coverage, scored by a
        RoBERTa emotion model fine-tuned on NBA fan writing.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = get_corpus_stats()

c1, c2, c3 = st.columns(3)
c1.metric("Posts scored", f"{stats['scored'] / 1_000_000:.2f}M")
c2.metric("Fanbases", stats["teams"])
c3.metric("Searchable through", stats["latest"] or "—")

with st.sidebar:
    st.subheader("How this works")
    st.markdown(
        "Reddit posts and comments from all 30 team subreddits, plus national "
        "news coverage, are ingested nightly, scored by a RoBERTa emotion "
        "classifier fine-tuned on NBA fan text, and embedded into a vector index."
        "\n\n"
        "Your question goes to Claude, which calls two tools against that index — "
        "one for the emotion distribution, one for the actual quotes — and writes "
        "its answer only from what comes back."
    )
    st.subheader("Why it isn't just an LLM")
    st.markdown(
        "A general model would guess at fan sentiment from memory. Every claim "
        "here traces to dated, linkable posts you can open and check yourself."
    )
    st.subheader("Tips")
    st.markdown(
        "- Name a team — sentiment is tracked per fanbase\n"
        "- Ask about a specific player, trade, or game for a sharper answer\n"
        "- Open **Sources** under any answer to read the raw posts"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    ):
        st.write(msg["content"])
        if msg.get("tool_calls"):
            render_evidence(msg["tool_calls"])

# Landing state: an empty chat box gives a first-time visitor no sense of what
# this can answer or what's behind it, so lead with the league-wide result and
# real example questions they can click.
pending_prompt = None
if not st.session_state.messages:
    st.markdown(
        '<div class="section-title">Where every fanbase stands</div>'
        '<div class="section-sub">Share of each fanbase\'s posts reading as hopeful, '
        "excited, or proud, minus those reading as disappointed, angry, resigned, "
        "or sad.</div>",
        unsafe_allow_html=True,
    )
    st.altair_chart(league_mood_chart(stats["mood"]), width="stretch")

    st.markdown(
        '<div class="section-title">Ask about any of them</div>'
        '<div class="section-sub">Or type your own question below.</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(example, key=f"example_{i}", width="stretch"):
            pending_prompt = example

user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")

if user_turns >= MAX_TURNS_PER_SESSION:
    st.info(
        f"This session has reached its {MAX_TURNS_PER_SESSION}-question limit. "
        "Refresh the page to start a new one."
    )
else:
    if typed := st.chat_input("e.g. How do Celtics fans feel about their bench depth?"):
        pending_prompt = typed

    if pending_prompt:
        st.session_state.messages.append({"role": "user", "content": pending_prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(pending_prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            # Stream the turn instead of hiding it behind one spinner: the tool
            # calls take a few seconds each, and narrating them ("checking the
            # sentiment breakdown", "pulling quotes") both fills that time and
            # shows the reader that the answer is being sourced, not recalled.
            status = st.status("Working through the data...", expanded=True)
            answer = st.empty()
            reply, tool_calls = "", []
            started = time.perf_counter()
            last_paint_at = 0.0
            try:
                for kind, payload in stream_chat(
                    st.session_state.messages,
                    client=get_anthropic_client(),
                    collection=get_cached_collection(),
                ):
                    if kind == "text":
                        reply += payload
                        now = time.perf_counter()
                        if now - last_paint_at >= STREAM_FLUSH_SECONDS:
                            last_paint_at = now
                            answer.markdown(reply + "▌")
                    elif kind == "tool_start":
                        tool_started = time.perf_counter()
                        status.write(describe_tool_call(payload))
                    elif kind == "tool_result":
                        tool_calls.append(payload)
                        _log_timing(
                            f"{payload['name']} {payload['input']} -> "
                            f"{time.perf_counter() - tool_started:.2f}s"
                        )
                answer.markdown(reply)
                _log_timing(f"TURN TOTAL {time.perf_counter() - started:.2f}s")
                status.update(
                    label=summarize_evidence(tool_calls), state="complete", expanded=False
                )
            except Exception as e:
                reply = f"Something went wrong answering that: {e}"
                answer.markdown(reply)
                status.update(label="Failed", state="error", expanded=False)
            render_evidence(tool_calls)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "tool_calls": tool_calls}
        )

if stats["latest"]:
    st.caption(f"Data current through {stats['latest']} · updated nightly")


# Streamlit anchors a page containing st.chat_input to the bottom, which is
# right mid-conversation but wrong on arrival - a first-time visitor lands
# staring at the input box with the league chart scrolled off above them.
#
# This has to be the last thing in the script. Streamlit executes top to
# bottom, so a scroll issued earlier runs against a page that hasn't been
# painted yet: it scrolls an empty document and is then overridden by the
# bottom anchor. It also keeps re-asserting for a beat rather than stopping
# once scrollTop hits 0, because 0 is also the value on a blank page - exiting
# there is indistinguishable from success and quits before the anchor lands.
# Only on a fresh load; scrolling on every rerun would fight the user while
# they read a streaming answer.
if "scrolled_top" not in st.session_state:
    st.session_state.scrolled_top = True
    components.html(
        """<script>
        const doc = window.parent.document;
        const deadline = Date.now() + 1500;
        const tick = setInterval(() => {
            // Which element actually scrolls has moved between Streamlit
            // versions, so pin every plausible container.
            for (const sel of ['[data-testid="stMain"]', 'section.main',
                               '[data-testid="stAppViewContainer"]']) {
                const el = doc.querySelector(sel);
                if (el) el.scrollTop = 0;
            }
            if (doc.scrollingElement) doc.scrollingElement.scrollTop = 0;
            if (Date.now() > deadline) clearInterval(tick);
        }, 50);
        </script>""",
        height=0,
    )
