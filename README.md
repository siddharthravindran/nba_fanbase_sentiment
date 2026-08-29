# NBA Fanbase Sentiment

Ask how a team's fanbase feels about anything — a trade, a rookie, a coach, the
season — and get an answer grounded in what fans actually wrote, with the quotes
attached.

> _Screenshot placeholder — replace with a shot of a Lakers answer showing the
> emotion distribution and the linked quotes below it._

The corpus is **3,560,910 documents** from all 30 team subreddits plus NBA news
coverage, spanning **2025-07-01 to 2026-08-28**. Every document is scored by a
RoBERTa emotion classifier fine-tuned on NBA fan writing
([`Siddharthr30/emotion-model-v2`](https://huggingface.co/Siddharthr30/emotion-model-v2)),
indexed in Chroma for semantic search, and served through a Claude tool-calling
chat layer.

| Source | Documents |
| --- | ---: |
| Reddit comments | 3,225,148 |
| Reddit posts | 266,486 |
| News articles | 69,276 |
| **Fan documents used for sentiment** | **3,491,634** |

---

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Reddit (PRAW) + Anthropic credentials
streamlit run app.py
```

The Chroma index is expected at `data/chroma` and the document store at
`data/raw_docs.db`. Neither is in the repo (see [Status](#status-and-known-limitations)).

---

## The model: what fans sound like

The first version of this classifier was fine-tuned on a public mental-health
check-in dataset. It ran, it produced confident labels, and the labels were
wrong in a way that was easy to miss — because the *vocabulary itself* didn't fit
the domain:

| v1 (mental-health check-ins) | v2 (NBA fan writing) |
| --- | --- |
| anger, anxiety, calm, confused, engaged, fatigue, joy, physical, sadness | anger or frustration, disappointment, excitement or hype, hope or optimism, mockery or sarcasm, neutral analysis or discussion, pessimism or resignation, pride, sadness |

`fatigue`, `physical` and `calm` have nearly no meaning in a game thread. More
importantly, v1 had no label for **mockery or sarcasm** — which turns out to be
the single most common register in NBA fan writing, 973,498 documents, 27% of
the corpus. A model without that class doesn't return "unknown"; it silently
redistributes a quarter of the corpus into whatever it does have. v2 was
retrained on NBA data with a consolidated 9-class scheme built for how fans
actually talk.

### The same lesson, one layer out: articles

With v2 working on Reddit, the obvious move was to point it at news articles
too. It is out of distribution there, and the evidence is consistent:

- Mean confidence **0.63** on articles vs **0.74** on Reddit
- **28.6%** of articles scored below 0.5 confidence, vs **14.9%** of fan posts
- **0.0%** of newswire copy labeled `neutral analysis or discussion` — which is
  self-evidently wrong for wire-service game recaps
- Concrete failures: a headline about Trump attending a Finals game →
  `pessimism` (0.74); a bare box-score roundup → `mockery`

Articles are only ~1.9% of the corpus, so excluding them barely moves any
aggregate. That isn't the reason to exclude them. The reason is that this is a
*fanbase* sentiment product, and quoting a reporter's sentence back to the user
as fan opinion is a lie about the source. Articles stay ingested and scored;
they simply aren't treated as evidence of how fans feel.

---

## Retrieval design

**Recency weighting.** "How do Lakers fans feel about Luka?" and "how did they
feel when the trade happened" are different questions over the same corpus.
Unweighted retrieval averages 14 months into one number and reads as stale.
Results are decayed on a **21-day half-life**; passing an explicit date range
turns decay off, since a pinned window *is* the question and re-decaying inside
it would bias a deliberately historical query back toward its recent edge.

**Rank fusion, not a date sort.** An earlier version kept the newest N of a wide
candidate pool. That fixed staleness and broke relevance — it threw away strong
old matches outright and made historical questions unanswerable. Relevance and
recency are now combined with Reciprocal Rank Fusion, which is scale-free and
doesn't require calibrating a distance against a timestamp.

**A length floor on quotes.** Short comments dominate raw vector similarity: a
38-character "We need him on the roster" is a near-perfect match for a short
topic query, so the top-k filled with fragments. Measured on a 400-document
pool, the median retrieved document was **38 characters**, and only **1.5%** of
documents under 80 characters named a player — against **40%** of documents over
250. Those fragments were also useless as displayed evidence. Quotes now require
120 characters, falling back to shorter ones only if the filter would leave the
answer under-supported.

---

## Two performance diagnoses

Both slowdowns in this project were gradual, unmeasured, and invisible until
they hurt. Both are worth writing down as diagnoses rather than as fixes.

### A metadata filter that couldn't use the index

Every vector query carried `where={"team": ...}`. A metadata filter can't ride
an HNSW graph — the graph's edges span all 3.5M documents, so following them
from a Lakers entry point lands on other teams constantly. Chroma's fallback is
to resolve the filter to a set of IDs and brute-force distance against every
one, which is **O(N_team)** and scales with fanbase size.

The tell was visible in the code from day one: a filter applied on *100% of
queries* isn't a filter, it's a partition — and a partition belongs in the
storage layout, not the query. Splitting into one collection per team removes
the filter entirely and restores ordinary graph traversal.

Measured on Utah Jazz, the smallest fanbase and therefore the old path's best
case:

| | Query time |
| --- | ---: |
| Filtered, shared 3.5M collection | 10.90s |
| Unfiltered, per-team collection | **0.38s** |

**29x**, with identical top-10 ordering and identical distances.

Note that a single early benchmark would *not* have caught this. Cost scales
with team size, so back when the corpus was a couple hundred thousand documents
a filtered query was comfortably sub-second. What catches this class of bug is
timing the same query as the corpus grows and watching the slope.

### A join that destroyed a covering index

Making the league-wide mood chart recency-weighted required `created_utc`, which
lives on a different table than the sentiment labels. Adding the join defeated
the `(team, top_emotion)` covering index and turned a **1.09s** index-only scan
into **38.28s** of random primary-key lookups across 3.5M rows — paid on every
cold start, for a number that is identical for every visitor.

It's now materialized nightly into a 270-row table. Serving a day-old cache does
not skew the chart: every weight decays by the same factor
`exp(-ln2 · elapsed / half_life)`, and the chart uses only each emotion's *share*
of its team's total, so a uniform factor cancels exactly in the normalization.
The only thing staleness costs is documents ingested since the last refresh.

This one was a regression introduced during development and caught by adding
per-turn timing instrumentation, not by a user report.

---

## Evaluation

Standard ML metrics cover half of this system. A held-out F1 on the emotion
classifier says nothing about whether "how do Knicks fans feel about Brunson"
returns an answer a Knicks fan recognizes as true — and that failure mode is
*silent*, because a reviewer without NBA context cannot detect a wrong NBA
answer. It reads as fluent and plausible either way.

So the evaluation plan has two halves:

1. **Model-level** — held-out performance on the v2 fine-tune, plus the
   out-of-distribution comparison against news articles documented above.
2. **Domain-level** — structured review by actual NBA fans. A fixed set of
   questions per team, answers rated on whether the sentiment breakdown matches
   the rater's read of their own fanbase and whether the surfaced quotes are
   representative rather than cherry-picked, with disagreements traced back to
   either retrieval or the classifier.

> **Status:** the out-of-distribution comparison is done; held-out classifier
> metrics and the fan review are not yet reported here. Results will be added,
> including the ones that don't flatter the system.

---

## Architecture

```
ingest/       Reddit (PRAW) + news article collection
enrichment/   cleaning, team tagging, emotion scoring (batched, MPS/CUDA/CPU)
retrieval/    Chroma vector search, recency weighting, SQL aggregation
chat/         Claude tool definitions + dispatch
app.py        Streamlit UI
scripts/      nightly pipeline, backfills, migrations
```

The chat layer exposes two tools. `aggregate_sentiment` returns an emotion
distribution — exact SQL over every scored document for a team, or a
semantically-scoped sample when a topic is given. `retrieve_quotes` returns real
fan posts with source links. The model decides which to call, and answers are
grounded in the results rather than in its own NBA knowledge.

Scoring runs batched on Apple Silicon (MPS) when available, ~5x faster than the
one-at-a-time CPU path with identical labels.

---

## Status and known limitations

Written out rather than quietly omitted:

- **Article team attribution is weak.** 15,345 unique articles expand to 69,276
  team-tagged rows (4.51x fan-out), and 74.7% are tagged to a team not named in
  the headline. This is why articles currently serve as corpus dating rather
  than as a grounding source. A lede-scoped heuristic tested at ~1.10x fan-out
  and ~85% accuracy is the intended fix.
- **The nightly pipeline has not yet run on a schedule.** Every step is
  idempotent and the workflow is committed, but no runner is registered — it has
  only been run by hand.
- **Chroma runs in embedded mode**, so bulk writes require app downtime to avoid
  index corruption. Server mode (`chroma run --path data/chroma` + `CHROMA_HOST`)
  is supported in the client factory but not yet deployed.
- **Not publicly hosted.** The document store is ~6.2GB and the vector index
  ~14GB, which rules out the free tiers. Hosting is unresolved.
