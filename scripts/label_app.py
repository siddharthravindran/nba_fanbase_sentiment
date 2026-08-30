"""Blind labeling UI for the classifier eval sample.

    streamlit run scripts/label_app.py

One document, nine buttons, one click per judgment. Built as a Streamlit page
rather than a terminal prompt because the people whose labels matter most here
are NBA fans, not Python users - "open this link and click the button that
matches" is a request you can make of a friend, and `python -m scripts.label`
is not.

The model's prediction is deliberately absent. `build_eval_sample.py` keeps it
in a separate key file for exactly this reason: a labeler shown the prediction
agrees with it more than they should, which inflates the agreement rate being
measured. Nothing in this file reads the key.

Each annotator writes their own `labels_<name>.csv`, so the first 50 rows can be
labeled twice by different people to measure how much two fans agree with each
other. Progress saves after every click and reloads on restart - 600 documents
is more than one sitting.
"""
import csv
import html
import re
from pathlib import Path

import streamlit as st

SAMPLE = Path("data/eval/label_sample.csv")
OUTDIR = Path("data/eval")

LABELS = [
    "excitement or hype",
    "hope or optimism",
    "pride",
    "anger or frustration",
    "disappointment",
    "pessimism or resignation",
    "sadness",
    "mockery or sarcasm",
    "neutral analysis or discussion",
]

# Offered because forcing a guess on a genuinely unreadable document manufactures
# disagreement that looks like model error. These rows are reported as a coverage
# figure and excluded from accuracy - if the rate is high, that is a finding about
# the label set rather than about the model.
UNCLEAR = "unclear / not enough context"

RUBRIC = """
**Judge the dominant register — what the writer is actually doing, not the words they used.**

Sarcasm is the case that matters most. *"can't wait to watch him fight over the
ball and implode"* uses excited words to mock, and it is **mockery or sarcasm**,
not excitement. If the literal reading and the intended reading disagree, label
the intended one.

| Label | What it looks like |
| --- | --- |
| excitement or hype | Energy about something coming or just happened. "LFG", "this is our year" |
| hope or optimism | Expects things to go well, forward-looking, calmer than hype |
| pride | Identity and ownership. "*our* guy", "this is why we drafted him" |
| anger or frustration | Directed anger — at a player, the front office, refs |
| disappointment | Let down by a specific outcome |
| pessimism or resignation | Expects it to go badly, defeated, "same as every year" |
| sadness | Genuine sorrow — injury, retirement, a favorite leaving |
| mockery or sarcasm | Joking, ironic, dunking. Includes self-deprecating fan humor |
| neutral analysis or discussion | Factual or analytical, no emotional charge |

Two calls worth making consistently: a hostile *joke* is **mockery**, not anger —
anger is sincere. And **pride** is about belonging where **excitement** is about
anticipation.
"""


def out_path(annotator: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", annotator.lower()).strip("_") or "anon"
    return OUTDIR / f"labels_{slug}.csv"


@st.cache_data
def load_sample():
    with SAMPLE.open() as fh:
        return list(csv.DictReader(fh))


def load_done(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return {r["sample_id"]: r for r in csv.DictReader(fh)}


def save(path: Path, done: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "human_label", "notes"])
        writer.writeheader()
        for sid in sorted(done, key=int):
            writer.writerow(done[sid])


st.set_page_config(page_title="Label", page_icon="📋", layout="centered")

if not SAMPLE.exists():
    st.error(f"No sample at {SAMPLE}. Run: python -m scripts.build_eval_sample")
    st.stop()

rows = load_sample()

with st.sidebar:
    st.header("Labeler")
    annotator = st.text_input("Your name", value=st.session_state.get("annotator", ""))
    st.caption("Each person gets their own file, so two people can label the same rows.")
    # The overlap block is what makes a human-agreement ceiling possible, and it
    # only works if the second labeler does those exact rows.
    overlap_only = st.checkbox(
        "Second labeler? Only do the shared first 50",
        value=False,
        help="Both people label these same 50 so we can measure how often two fans agree.",
    )

if not annotator.strip():
    st.info("Enter your name in the sidebar to start.")
    st.stop()

st.session_state["annotator"] = annotator
path = out_path(annotator)
done = load_done(path)

queue = [r for r in rows if r["overlap"] == "1"] if overlap_only else rows
todo = [r for r in queue if r["sample_id"] not in done]

st.progress(
    (len(queue) - len(todo)) / len(queue) if queue else 1.0,
    text=f"{len(queue) - len(todo)} / {len(queue)} labeled",
)

with st.expander("How to label (read once)", expanded=not done):
    st.markdown(RUBRIC)

if not todo:
    st.success(f"Done — all {len(queue)} labeled. Saved to `{path}`.")
    st.stop()

row = todo[0]
st.markdown("#### What is this fan doing?")
st.markdown(
    f"<div style='background:#f4f4f6;border-radius:10px;padding:1.1rem 1.3rem;"
    # Escaped: this is arbitrary Reddit text going into a raw-HTML block, and a
    # comment containing markup would otherwise render as markup or break the card.
    f"font-size:1.12rem;line-height:1.6;'>{html.escape(row['text'])}</div>",
    unsafe_allow_html=True,
)
st.write("")


def record(label: str) -> None:
    done[row["sample_id"]] = {
        "sample_id": row["sample_id"],
        "human_label": label,
        "notes": st.session_state.get("notes", ""),
    }
    save(path, done)
    st.session_state["notes"] = ""


# Three columns keeps every label on one screen without scrolling, which is what
# makes this fast enough to get through 600 of them.
for start in range(0, len(LABELS), 3):
    for col, label in zip(st.columns(3), LABELS[start : start + 3]):
        if col.button(label, key=f"b{label}", use_container_width=True):
            record(label)
            st.rerun()

st.write("")
left, right = st.columns([3, 1])
if left.button(UNCLEAR, use_container_width=True):
    record(UNCLEAR)
    st.rerun()
if right.button("↩ Undo last", use_container_width=True, disabled=not done):
    del done[max(done, key=int)]
    save(path, done)
    st.rerun()

st.text_input("Note (optional)", key="notes", placeholder="Only if the call was hard")
st.caption(f"Saved to `{path}` after every click. Close and come back anytime.")
