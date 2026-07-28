"""Claude tool-calling chat loop: answers fanbase-sentiment questions grounded
in `aggregate_sentiment` (SQLite stats) and `retrieve_quotes` (Chroma semantic
search) tool results, instead of the model guessing from its own knowledge."""
import json

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


def run_chat(messages: list[dict], client: anthropic.Anthropic | None = None, collection=None) -> str:
    """messages: full turn history as [{"role": "user"/"assistant", "content": str}, ...].
    Returns the assistant's final text reply for the latest user turn.

    `client`/`collection` can be passed in (e.g. cached once per app process via
    st.cache_resource) to avoid re-opening the Anthropic client and the
    on-disk Chroma index + embedding model on every single call."""
    client = client or anthropic.Anthropic()
    collection = collection or get_collection()

    conversation = [{"role": m["role"], "content": m["content"]} for m in messages]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        conversation.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = call_tool(block.name, block.input, collection)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        conversation.append({"role": "user", "content": tool_results})

    return "Sorry, I wasn't able to settle on an answer after several tool calls - try rephrasing the question."
