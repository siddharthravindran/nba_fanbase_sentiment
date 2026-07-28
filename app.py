import anthropic
import streamlit as st
from dotenv import load_dotenv

from chat.assistant import run_chat
from retrieval.vector_store import get_collection

load_dotenv(override=True)

MAX_TURNS_PER_SESSION = 15  # caps API spend per visitor session


@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic()


@st.cache_resource
def get_cached_collection():
    return get_collection()


st.set_page_config(page_title="NBA Fanbase Sentiment", page_icon="🏀")

st.title("🏀 NBA Fanbase Sentiment")
st.caption("Ask how a fanbase feels about a topic — grounded in Reddit + articles, scored by a fine-tuned RoBERTa model")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")

if user_turns >= MAX_TURNS_PER_SESSION:
    st.info(f"This session has reached its {MAX_TURNS_PER_SESSION}-question limit. Refresh the page to start a new one.")
elif prompt := st.chat_input("e.g. How do Celtics fans feel about their bench depth?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = run_chat(
                    st.session_state.messages,
                    client=get_anthropic_client(),
                    collection=get_cached_collection(),
                )
            except Exception as e:
                reply = f"Something went wrong answering that: {e}"
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
