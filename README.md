# NBA Fanbase Sentiment

Ask how a team's fanbase feels about anything — a trade, a rookie, a coach, the
season — and get an answer grounded in what fans actually wrote, with the quotes
attached.

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

### Where v2's training labels came from

Hand-labeling 3.5M documents is not a thing one person does, so the taxonomy was
developed and applied with a zero-shot teacher: `facebook/bart-large-mnli` scores
the nine candidate labels against each document via natural language inference
([`enrichment/zero_shot_labels.py`](enrichment/zero_shot_labels.py)), and v2 is a
RoBERTa fine-tune distilled from those labels. Zero-shot was also what made the
taxonomy itself cheap to iterate — candidate label sets could be swapped and
compared on a sample without training anything.

The fine-tune saw ~84,500 teacher-labeled documents with 14,920 held out, and was
then applied to all 3.5M.

### Choosing the checkpoint on macro F1, not accuracy

| Epoch | Train loss | Val loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.995 | 0.949 | 0.647 | 0.592 |
| 2 | 0.785 | **0.888** | 0.674 | **0.631** |
| 3 | 0.659 | 0.913 | 0.678 | 0.631 |

Epoch 3 is overfitting: training loss keeps falling while validation loss turns
around and climbs. Epoch 2 is the shipped checkpoint, restored automatically via
`load_best_model_at_end=True` with `metric_for_best_model="f1_macro"`.

The reason that argument is `f1_macro` and not accuracy is visible in the last
row: **at epoch 3 accuracy improves while macro F1 does not.** The macro dip
itself is 0.0008 and means nothing on its own — the direction is the point. The
extra epoch bought gains on the frequent classes and returned nothing on the
rare ones, which is the shape class collapse takes before it becomes visible.

That divergence matters here because of how little accuracy can see. `sadness` is
0.65% of the data, so a model that dropped the class outright would give up well
under a point of accuracy while macro F1 fell by roughly seven. Had epoch 3 been
selected on accuracy — the higher number, and the tempting one — the choice would
have been made using the metric least able to detect the failure that matters
most downstream, where these labels become a fanbase's emotion breakdown.

Held-out performance of that checkpoint, 14,920 documents:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| excitement or hype | 0.743 | 0.726 | **0.735** | 3,199 |
| disappointment | 0.747 | 0.643 | 0.691 | 2,714 |
| sadness | 0.673 | 0.680 | 0.677 | 97 |
| mockery or sarcasm | 0.644 | 0.703 | 0.672 | 3,527 |
| hope or optimism | 0.676 | 0.657 | 0.666 | 1,608 |
| pessimism or resignation | 0.608 | 0.669 | 0.637 | 2,743 |
| anger or frustration | 0.633 | 0.583 | 0.607 | 604 |
| neutral analysis or discussion | 0.554 | 0.475 | 0.511 | 217 |
| pride | 0.522 | 0.455 | **0.486** | 211 |
| **Accuracy** | | | **0.674** | 14,920 |
| **Macro avg** | 0.644 | 0.621 | 0.631 | 14,920 |

**No class collapse**, which is the result worth leading with. `sadness` is 0.65%
of the test set — 97 documents — and lands at 0.677 F1, above the macro average.
Collapsing rare classes into the majority is the default outcome at a 36:1
imbalance, and it didn't happen.

The two weakest classes are `pride` (0.486) and `neutral analysis or discussion`
(0.511), and both are boundary problems rather than rare-class problems. Pride
and excitement are separated by *why* the fan is animated, not by vocabulary —
"this is why we drafted him" and "LFG" share every surface feature. Neutral is
worse than rare, it's a residual: it's defined by the absence of the other eight
rather than by anything of its own. The systematic confusion is elsewhere and
visible in the precision/recall asymmetry — `disappointment` has precision 0.747
against recall 0.643 while `pessimism` inverts it at 0.608/0.669, which is
disappointment leaking into pessimism. Being let down by one result and expecting
to be let down forever is a genuinely thin line in fan writing.

**What this table measures is agreement with the teacher.** The held-out labels
are bart-mnli's, so 0.674 is distillation fidelity, not human accuracy — and it
is not an upper bound on accuracy either. The teacher labels each document
independently at 0.30–0.51 confidence, so many of its labels are near coin-flips;
a fine-tune fits systematic signal and generalizes over that noise, meaning some
share of the missing 33% is the student disagreeing with the teacher and being
right.

The consequence that matters is specific. `mockery or sarcasm` scores 0.672 here,
*above* the macro average — so the student reproduces the teacher's sarcasm
judgments well. That is not the same as getting sarcasm right, and a real Celtics
fan caught the difference in a live answer: "can't wait to watch him fight over
the ball and implode" was scored `excitement or hype`, which is what happens when
a model reads inverted-meaning text literally. Sarcasm detection is exactly where
an NLI-based zero-shot teacher is weakest, so the most likely explanation is that
teacher and student are wrong *together* — a blind spot this table cannot show,
by construction, because agreement is all it can see.

That is the gap the evaluation below is built to close.

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

So the evaluation plan has three halves, which is one more than a plan should
have and is the point:

1. **Model-level** — the held-out fine-tune numbers above, plus the
   out-of-distribution comparison against news articles.
2. **Against humans** — the piece the held-out number structurally cannot
   provide. 600 documents drawn from a window the model never trained on, labeled
   by hand ([`scripts/build_eval_sample.py`](scripts/build_eval_sample.py),
   [`scripts/label_app.py`](scripts/label_app.py)).
3. **Domain-level** — structured review by actual NBA fans. A fixed set of
   questions per team, answers rated on whether the sentiment breakdown matches
   the rater's read of their own fanbase and whether the surfaced quotes are
   representative rather than cherry-picked, with disagreements traced back to
   either retrieval or the classifier.

### How the human-label set is built

Four decisions, each guarding against a specific way the resulting number would
otherwise be quietly wrong:

- **Time-disjoint window.** The training export was taken 2026-07-22 and scoring
  ran 07-27; the sample is drawn only from documents *created* after that —
  60,303 of them. Filtering on `created_utc` rather than ingest time is the
  conservative choice, since backfills insert old documents long after the fact.
- **Two strata.** 50 per predicted class gives per-class precision including for
  `sadness`, which is 0.8% of the window and would appear about five times in a
  600-document random draw. But an even draw is not the corpus, so a separate
  150-document prevalence-representative sample carries the headline number, and
  each stratified row stores the weight that reverts it to true prevalence.
- **Blind to the model.** The prediction is held in a separate key file and
  joined back afterward. A labeler shown the model's guess agrees with it more
  than they should, which inflates precisely the quantity being measured.
- **A human ceiling.** 50 rows are labeled by two different people. If two NBA
  fans agree with each other 78% of the time, then a model at 74% is near the
  limit of what the label set can express, and reporting "74%" without that
  denominator misdescribes it.

The same pass also yields a three-way comparison — human vs. teacher vs. student
on identical documents — which is what separates "the student inherited
bart-mnli's blind spots" from "the student introduced its own."

> **Status:** model-level numbers and the OOD comparison are reported above. The
> human-label set is built and unlabeled; the fan review is not yet run. Both
> will be added, including the parts that don't flatter the system.

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

The chat layer exposes three tools. `aggregate_sentiment` returns an emotion
distribution — exact SQL over every scored document for a team, or a
semantically-scoped sample when a topic is given. `retrieve_quotes` returns real
fan posts with source links. `check_player_news` looks a player up in the news
articles, which exist precisely because fan writing cannot distinguish a rumor
from a completed transaction: a post reading "we're linked to Kelly Olynyk" and a
post reacting to a signing are the same shape and score the same way. The model
decides which to call, and answers are grounded in the results rather than in its
own NBA knowledge.

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
- **The nightly pipeline runs on a `launchd` timer**, not on a machine that is
  reliably awake. Every step is idempotent, and a trailing pass re-fetches recent
  days so a missed night self-heals — but this is a laptop, and there are gaps in
  the article corpus where it was closed.
- **Chroma runs in embedded mode**, so bulk writes require app downtime to avoid
  index corruption. Server mode (`chroma run --path data/chroma` + `CHROMA_HOST`)
  is supported in the client factory but not yet deployed.
- **Not publicly hosted.** The document store is ~6.2GB and the vector index
  ~14GB, which rules out the free tiers. Hosting is unresolved.
