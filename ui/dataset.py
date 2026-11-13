import re

import altair as alt
import pandas as pd
import streamlit as st

from censorbot import SOURCES_DIR
from ui.common import ensure_ready, load_evaluation, load_model

SOURCE_NAMES = {
    "surge": "Surge AI Obscenity List",
    "wordfreq": "Common English words (wordfreq)",
    "censorbot": "Hand-curated additions",
    "feedback": "User feedback",
}
PROFANE, CLEAN = "Profane", "Clean"


def mask(word):
    """Keep the first character and hide the rest, e.g. f***."""
    word = str(word)
    return word[0] + re.sub(r"\S", "*", word[1:])


@st.cache_data
def load_categories():
    surge = pd.read_csv(SOURCES_DIR / "surge_profanity_en.csv")
    return dict(zip(surge["text"].str.strip().str.lower(), surge["category_1"]))


def build_table(data):
    categories = load_categories()
    table = pd.DataFrame({
        "word": data["text"].astype(str),
        "label": (data["severity_rating"] >= 1.0).map({True: PROFANE, False: CLEAN}),
        "severity": data["severity_rating"],
        "root word": data["group"].astype(str) if "group" in data else data["text"].astype(str),
        "category": data["text"].astype(str).map(categories).fillna("—"),
        "source": data["source"].map(lambda s: SOURCE_NAMES.get(s, s)) if "source" in data else "—",
    })
    return table


def severity_chart(table):
    profane = table[table["label"] == PROFANE]
    return alt.Chart(profane).mark_bar(color="#ef4444", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("severity:Q", bin=alt.Bin(step=0.1), title="Severity (1.0 mild → 2.0 severe)"),
        y=alt.Y("count():Q", title="Words"),
        tooltip=[alt.Tooltip("count():Q", title="Words")],
    ).properties(height=240)


def category_chart(table):
    counts = table[table["category"] != "—"]["category"].value_counts().rename_axis("category").reset_index(name="words")
    return alt.Chart(counts).mark_bar(color="#ef4444", cornerRadius=3).encode(
        x=alt.X("words:Q", title="Words"),
        y=alt.Y("category:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        tooltip=["category", "words"],
    ).properties(height=300)


def dataset_page():
    ensure_ready(evaluation=True)
    _, _, data = load_model()
    _, summary, _ = load_evaluation()
    table = build_table(data)

    st.title("Dataset")
    st.write(
        "What CensorBot learns from: about 3,200 English words and short phrases, half of them profane and half "
        "everyday English. Everything is in English, so CensorBot only works for English messages. Profane words are masked below unless you choose to show them."
    )

    cols = st.columns(4)
    cols[0].metric("Entries", f"{summary['rows']:,}")
    cols[1].metric("Profane", f"{summary['rude']:,}")
    cols[2].metric("Clean", f"{summary['polite']:,}")
    cols[3].metric("Multi-word phrases", summary["multi_word"])

    st.subheader("Where the words come from")
    sources = pd.DataFrame(
        [(SOURCE_NAMES.get(k, k), v) for k, v in summary["sources"].items()], columns=["Source", "Entries"]
    )
    st.dataframe(sources, hide_index=True, use_container_width=True)
    st.markdown(
        "- **[Surge AI Obscenity List](https://huggingface.co/datasets/mmathys/profanity)** (MIT): ~1,600 English "
        "profanities and their spelling variants (`a55`, `@ss`, …). Five human annotators rated each one from "
        "1 (mild) to 3 (severe); CensorBot maps that onto its own scale, where 1.0 is the profane threshold and "
        "2.0 the most severe. Each variant also lists its *root word*, used to keep variants together during evaluation.\n"
        "- **[wordfreq](https://github.com/rspeer/wordfreq)** (CC-BY-SA 4.0): the most frequent English words, used as "
        "clean examples. Stopwords, anything on the profanity list and a short list of mild insults (*stupid*, "
        "*loser*, …) are left out, and the list is cut so both classes are the same size.\n"
        "- **Hand-curated additions:** ~100 words picked by hand for CensorBot, kept where Surge doesn't cover a word (e.g. *idiot*, *moron*)."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**How severe are the profane words?**")
        st.altair_chart(severity_chart(table), use_container_width=True)
    with right:
        st.markdown("**What kind of profanity?** (Surge AI categories)")
        st.altair_chart(category_chart(table), use_container_width=True)

    st.subheader("Browse the data")
    filters = st.columns([1, 1.6, 1.4])
    label = filters[0].selectbox("Label", ["All", PROFANE, CLEAN])
    source = filters[1].multiselect("Source", sorted(table["source"].unique()), placeholder="All sources")
    search = filters[2].text_input("Search", placeholder="e.g. weekend")
    uncensored = st.toggle("Show profane words uncensored", value=False)

    view = table
    if label != "All":
        view = view[view["label"] == label]
    if source:
        view = view[view["source"].isin(source)]
    if search:
        needle = search.strip().lower()
        view = view[view["word"].str.contains(needle, regex=False) | view["root word"].str.contains(needle, regex=False)]
    if not uncensored:
        profane = view["label"] == PROFANE
        view = view.assign(
            word=view["word"].where(~profane, view["word"].map(mask)),
            **{"root word": view["root word"].where(~profane, view["root word"].map(mask))},
        )
    st.caption(f"{len(view):,} of {len(table):,} entries")
    st.dataframe(
        view,
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={"severity": st.column_config.NumberColumn(format="%.2f")},
    )

    st.subheader("How it's built")
    st.markdown(
        "The dataset is generated by `scripts/build_dataset.py`, so it can be rebuilt and audited from its sources:\n"
        "```\n"
        "pip install -r requirements-dev.txt\n"
        "python scripts/build_dataset.py\n"
        "```\n"
        "Full credits and licence texts are in `data/NOTICE.md`."
    )
    if summary["conflicts"]:
        st.caption("Labelled both profane and clean: " + ", ".join(summary["conflicts"]))
