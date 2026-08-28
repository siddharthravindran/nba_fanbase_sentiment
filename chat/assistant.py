"""Claude tool-calling chat loop: answers fanbase-sentiment questions grounded
in `aggregate_sentiment` (SQLite stats) and `retrieve_quotes` (Chroma semantic
search) tool results, instead of the model guessing from its own knowledge."""
import json
from collections.abc import Iterator

import anthropic

from chat.tools import TOOLS, call_tool
from retrieval.vector_store import get_collection

MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """You are an assistant that answers questions about how NBA \
fanbases feel about players, trades, games, and other topics, grounded in \
real Reddit posts/comments and news articles scored by a sentiment model.

Use the aggregate_sentiment tool to get emotion distributions (the numbers), \
and retrieve_quotes to pull real example quotes to support your answer. For \
questions about a specific topic, prefer calling both so you can cite the \
overall sentiment breakdown and back it up with real quotes.

Synthesize a concise, conversational answer - don't just dump raw tool \
output. If a question doesn't map to a tracked team, or a tool returns no \
relevant results, say so honestly rather than making something up.

Tell a story, don't file a report. A distribution of emotion labels is not an \
answer by itself - it's evidence. The answer is *what the fanbase actually \
thinks and why*. Read the retrieved quotes and identify the specific \
grievances, hopes, players, trades, or games driving the mood, then explain \
the sentiment in terms of those concrete things. "Fans are frustrated" is \
useless; "fans are frustrated because the front office spent the offseason \
chasing a star and then didn't land a backup option" is an answer. Where the \
quotes disagree with each other, say so and characterize the split rather \
than flattening it into one average mood.

Formatting: write like a knowledgeable person answering in a chat, not like \
a report or executive summary. Never use emojis. Lead with the actual answer \
in plain prose - not a bolded conclusion, not a stat block. Use bold sparingly \
(once or twice at most), and never as a label pattern like "**Emotion** - 40%". \
You may use a single short bullet list only when enumerating three or more \
distinct themes; otherwise write in paragraphs. Cite emotion percentages only \
where they materially support a point, woven into a sentence ("about four in \
ten posts read as pessimistic") rather than dumped as a labeled breakdown. \
Keep quotes short and inline; don't build long blockquote sections unless the \
user explicitly asks for supporting evidence. Don't interrupt yourself with \
asides like "more on this below".

Important: retrieve_quotes and aggregate_sentiment match posts by semantic \
similarity to the topic, which does NOT distinguish negation, hypotheticals, \
rumors, or resolved-vs-unresolved outcomes. A query about a deal that fell \
through will still surface posts written *while it was still a rumor* (often \
hype/excitement about the possibility), not just posts reacting to the \
actual outcome. Before writing your answer, read each retrieved quote's \
actual content and judge whether it reflects the situation described in the \
user's question or an earlier/different phase of the story (e.g. rumor \
excitement vs. post-outcome reaction). Explicitly reconcile any mismatch \
between an emotion label and what the quote itself says, rather than taking \
the label at face value."""


def stream_chat(
    messages: list[dict], client: anthropic.Anthropic | None = None, collection=None
) -> Iterator[tuple[str, object]]:
    """messages: full turn history as [{"role": "user"/"assistant", "content": str}, ...].

    Yields (kind, payload) events as the turn unfolds, so the UI can show work
    in progress rather than a spinner that hides several seconds of tool calls:
      ("text", chunk)              - a piece of the model's answer
      ("tool_start", {name, input}) - a tool call is about to run
      ("tool_result", {name, input, result}) - that tool's output, which the UI
                                     renders as the emotion chart and the
                                     source-linked quote list

    `client`/`collection` can be passed in (e.g. cached once per app process via
    st.cache_resource) to avoid re-opening the Anthropic client and the
    on-disk Chroma index + embedding model on every single call."""
    client = client or anthropic.Anthropic()
    collection = collection or get_collection()

    conversation = [{"role": m["role"], "content": m["content"]} for m in messages]

    for _ in range(MAX_TOOL_ROUNDS):
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation,
        ) as stream:
            for chunk in stream.text_stream:
                yield "text", chunk
            response = stream.get_final_message()

        if response.stop_reason != "tool_use":
            return

        conversation.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                yield "tool_start", {"name": block.name, "input": block.input}
                result = call_tool(block.name, block.input, collection)
                yield "tool_result", {"name": block.name, "input": block.input, "result": result}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        conversation.append({"role": "user", "content": tool_results})

    yield "text", (
        "\n\nSorry, I wasn't able to settle on an answer after several tool calls - "
        "try rephrasing the question."
    )
