import altair as alt
import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from chat.assistant import stream_chat
from ingest.storage import get_connection
from retrieval.vector_store import get_collection

load_dotenv(override=True)

MAX_TURNS_PER_SESSION = 15  # caps API spend per visitor session

EXAMPLE_PROMPTS = [
    "How do Lakers fans feel about the roster right now?",
    "What are Knicks fans saying about their coaching?",
    "Why are Mavericks fans so frustrated?",
    "Are Warriors fans optimistic about next season?",
]

SOURCE_LABELS = {
    "reddit_post": "Reddit post",
    "reddit_comment": "Reddit comment",
    "article": "Article",
}

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


@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic()


@st.cache_resource
def get_cached_collection():
    return get_collection()


@st.cache_data(ttl=3600)
def get_corpus_stats() -> dict:
    """Headline numbers + the league-wide mood table, in one pass. The GROUP BY
    is a full scan of a few million scored rows (~6s), so it's cached for an
    hour rather than run per interaction."""
    conn = get_connection()
    try:
        scored = conn.execute("SELECT COUNT(*) FROM sentiment_docs_v2").fetchone()[0]
        latest = conn.execute("SELECT MAX(created_utc) FROM clean_docs").fetchone()[0]
        rows = conn.execute(
            """
            SELECT team, top_emotion, COUNT(*)
            FROM sentiment_docs_v2
            WHERE team IS NOT NULL AND team != ''
            GROUP BY team, top_emotion
            """
        ).fetchall()
    finally:
        conn.close()

    totals: dict[str, int] = {}
    positive: dict[str, int] = {}
    negative: dict[str, int] = {}
    for team, emotion, count in rows:
        totals[team] = totals.get(team, 0) + count
        if emotion in POSITIVE:
            positive[team] = positive.get(team, 0) + count
        elif emotion in NEGATIVE:
            negative[team] = negative.get(team, 0) + count

    mood = [
        {
            "team": team,
            "net": round(100 * (positive.get(team, 0) - negative.get(team, 0)) / total, 1),
            "posts": total,
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


def emotion_chart(dist: list[dict]):
    """Horizontal bars colored by emotional valence, so the shape of a fanbase's
    mood is readable at a glance without parsing nine label names."""
    df = pd.DataFrame(dist)
    order = df.sort_values("pct", ascending=False)["emotion"].tolist()
    known = [e for e in order if e in EMOTION_COLORS]

    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("pct:Q", title="% of posts", axis=alt.Axis(format=".0f")),
            y=alt.Y("emotion:N", sort=order, title=None),
            color=alt.Color(
                "emotion:N",
                scale=alt.Scale(domain=known, range=[EMOTION_COLORS[e] for e in known]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("emotion:N", title="Emotion"),
                alt.Tooltip("pct:Q", title="% of posts", format=".1f"),
                alt.Tooltip("count:Q", title="Posts", format=","),
            ],
        )
        .properties(height=26 * len(df))
    )


def league_mood_chart(mood: list[dict]):
    """Every fanbase ranked by net positive sentiment. This is the thing a
    visitor should see before typing anything - it proves there's real scored
    data behind the chat box instead of asking them to take it on faith."""
    df = pd.DataFrame(mood)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("net:Q", title="Net positive sentiment (% positive − % negative)"),
            y=alt.Y("team:N", sort=df["team"].tolist(), title=None),
            color=alt.condition(alt.datum.net > 0, alt.value("#2e9e5b"), alt.value("#c9503a")),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("net:Q", title="Net sentiment", format="+.1f"),
                alt.Tooltip("posts:Q", title="Posts scored", format=","),
            ],
        )
        .properties(height=18 * len(df))
    )


def render_distribution(call: dict):
    """Show the emotion breakdown the model actually saw. The written answer
    stays narrative; the numbers live here so the model doesn't burn prose
    reciting percentages."""
    result = call["result"]
    dist = result.get("distribution")
    if not dist:
        return

    top = dist[0]
    scope = result.get("topic") or "all posts"
    st.caption(
        f"**{result.get('team')}** · {scope} · {result.get('n_docs', 0):,} posts scored · "
        f"most common: {top['emotion']} ({top['pct']}%)"
    )
    st.altair_chart(emotion_chart(dist), use_container_width=True)


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


def summarize_evidence(tool_calls: list[dict]) -> str:
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
    return f"Answered from {' and '.join(parts)}" if parts else "Answered"


def render_evidence(tool_calls: list[dict]):
    for call in tool_calls:
        if call["name"] == "aggregate_sentiment":
            render_distribution(call)

    quotes = [
        q
        for call in tool_calls
        if call["name"] == "retrieve_quotes"
        for q in call["result"].get("quotes", [])
    ]
    render_quotes(quotes)


st.set_page_config(page_title="NBA Fanbase Sentiment", page_icon="🏀", layout="centered")

# Default Streamlit blocks snap into place all at once, which reads as flat.
# Staggering a short fade makes the page assemble itself and gives the metric
# row and charts a beat of their own.
st.markdown(
    """
    <style>
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: none; }
    }
    [data-testid="stMetric"],
    [data-testid="stChatMessage"],
    [data-testid="stVegaLiteChart"],
    [data-testid="stExpander"] { animation: fadeUp 0.5s ease both; }
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(201,80,58,0.10), rgba(46,158,91,0.08));
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"] { animation-delay: 0.10s; }
    [data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"] { animation-delay: 0.20s; }
    [data-testid="stVegaLiteChart"] { animation-delay: 0.25s; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 NBA Fanbase Sentiment")
st.caption(
    "Ask how a fanbase feels about a player, trade, or game — answered from real "
    "Reddit posts and news coverage, scored by a fine-tuned RoBERTa emotion model."
)

stats = get_corpus_stats()

c1, c2, c3 = st.columns(3)
c1.metric("Posts scored", f"{stats['scored'] / 1_000_000:.2f}M")
c2.metric("Fanbases tracked", stats["teams"])
c3.metric("Data through", stats["latest"] or "—")

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
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tool_calls"):
            render_evidence(msg["tool_calls"])

# Landing state: an empty chat box gives a first-time visitor no sense of what
# this can answer or what's behind it, so lead with the league-wide result and
# real example questions they can click.
pending_prompt = None
if not st.session_state.messages:
    st.subheader("Where every fanbase stands")
    st.caption(
        "Share of each fanbase's posts reading as hopeful, excited, or proud, "
        "minus those reading as disappointed, angry, resigned, or sad."
    )
    st.altair_chart(league_mood_chart(stats["mood"]), use_container_width=True)

    st.subheader("Ask about any of them")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
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
        with st.chat_message("user"):
            st.write(pending_prompt)

        with st.chat_message("assistant"):
            # Stream the turn instead of hiding it behind one spinner: the tool
            # calls take a few seconds each, and narrating them ("checking the
            # sentiment breakdown", "pulling quotes") both fills that time and
            # shows the reader that the answer is being sourced, not recalled.
            status = st.status("Working through the data...", expanded=True)
            answer = st.empty()
            reply, tool_calls = "", []
            try:
                for kind, payload in stream_chat(
                    st.session_state.messages,
                    client=get_anthropic_client(),
                    collection=get_cached_collection(),
                ):
                    if kind == "text":
                        reply += payload
                        answer.markdown(reply + "▌")
                    elif kind == "tool_start":
                        status.write(describe_tool_call(payload))
                    elif kind == "tool_result":
                        tool_calls.append(payload)
                answer.markdown(reply)
                status.update(label=summarize_evidence(tool_calls), state="complete", expanded=False)
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
