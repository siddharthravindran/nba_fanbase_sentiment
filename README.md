# NBA Fanbase Sentiment

Chatbot that answers "how does [team] fanbase feel about [topic]" using team
subreddits + NBA articles, scored with a fine-tuned RoBERTa sentiment model
(9 consolidated emotion classes, `Siddharthr30/emotion-model`) and surfaced
via retrieval + aggregation.

## Pipeline
1. `ingest/` — pull posts/comments from team subreddits (PRAW) and articles
   (RSS + trafilatura)
2. `enrichment/` — run each doc through the sentiment model, tag team/topic
3. `retrieval/` — embed + index docs (Chroma) for semantic search, plus
   SQL-style aggregation over sentiment labels
4. `app.py` — Streamlit chat UI; LLM calls an aggregate_sentiment tool and a
   retrieve_quotes tool to ground its answers

## Setup
```
source ~/venvs/nba-fanbase-sentiment/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in Reddit + Anthropic creds
```
