import streamlit as st

st.set_page_config(page_title="NBA Fanbase Sentiment", page_icon="🏀")

st.title("🏀 NBA Fanbase Sentiment")
st.caption("Ask how a fanbase feels about a topic — grounded in Reddit + articles, scored by a fine-tuned RoBERTa model")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("e.g. How do Lakers fans feel about LeBron's minutes?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # TODO: wire up retrieval.vector_store + enrichment aggregation + Claude tool calls
    with st.chat_message("assistant"):
        st.write("Pipeline not wired up yet.")
