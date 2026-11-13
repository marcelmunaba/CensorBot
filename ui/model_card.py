import altair as alt
import pandas as pd
import streamlit as st

from censorbot.insights import top_words
from ui.common import ensure_ready, load_evaluation, load_model
from ui.dataset import mask

# The leetspeak normaliser's recovery rate before symbols (@, $, !, |) were handled
LEETSPEAK_BEFORE = "47%"


def confusion_chart(matrix):
    labels = ["Clean", "Profane"]
    cells = pd.DataFrame(
        [(labels[a], labels[p], int(matrix[a][p])) for a in range(2) for p in range(2)],
        columns=["Actual", "Predicted", "Count"],
    )
    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0, labelFontSize=11)),
        y=alt.Y("Actual:N", sort=labels),
    )
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None)
    )
    text = base.mark_text(fontSize=18, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(alt.datum.Count > float(matrix.max()) / 2, alt.value("white"), alt.value("black")),
    )
    return (heat + text).properties(height=220, width=220)


def top_words_chart(classifier, vectorizer):
    profane, clean = top_words(classifier, vectorizer, n=8)
    words = pd.concat([profane, clean]).rename("score").rename_axis("word").reset_index()
    words["leans"] = words["score"].map(lambda s: "Profane" if s > 0 else "Clean")
    # Censor the profane words in the chart too - it's a censor bot after all
    words["label"] = [
        mask(w) if leans == "Profane" else w
        for w, leans in zip(words["word"], words["leans"])
    ]
    return alt.Chart(words).mark_bar(cornerRadius=3).encode(
        x=alt.X("score:Q", title="log P(word | profane) − log P(word | clean)"),
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color(
            "leans:N",
            scale=alt.Scale(domain=["Profane", "Clean"], range=["#ef4444", "#22c55e"]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[alt.Tooltip("label:N", title="word"), alt.Tooltip("score:Q", format="+.2f")],
    ).properties(height=380)


def model_card_page():
    ensure_ready(evaluation=True)
    classifier, vectorizer, data = load_model()
    metrics, summary, (leet_hits, leet_total) = load_evaluation()

    st.title("Model")
    st.write(
        "How well CensorBot performs, which words matter most, and where it falls short. "
        "For how it works step by step, see the *Pipeline* page."
    )

    st.subheader("Evaluation")
    st.caption(
        f"Stratified, grouped {metrics['folds']}-fold cross-validation on the training dataset. "
        "Every test word is held out from the model that scores it, and spelling variants of the same "
        "word (`a55`, `@ss`, …) always stay in the same fold."
    )
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.0%}")
    cols[1].metric("Precision", f"{metrics['precision']:.0%}", help="Of the words flagged as profane, how many really are")
    cols[2].metric("Recall", f"{metrics['recall']:.0%}", help="Of the profane words, how many were flagged")
    cols[3].metric("F1", f"{metrics['f1']:.2f}", help="Balance of precision and recall")
    st.caption("Precision, recall and F1 are for the profane class.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        st.altair_chart(confusion_chart(metrics["confusion"]), width="content")
    with right:
        st.markdown("**What these numbers mean**")
        st.write(
            "This is a deliberately hard test: every word is scored by a model that has never seen it, "
            "or any spelling of it. A bag-of-words model has no information about a brand-new word, so "
            "it mostly falls back to \"clean\" - hence the low recall. Without grouping, spelling "
            "variants would leak between folds and inflate the score. In the live app the model knows "
            "the whole list, so known profanity and its variants are caught reliably."
        )

    st.markdown("**Leetspeak normaliser**")
    st.write(
        f"Tested on {leet_total} single-word spelling variants from the Surge AI list (`a55`, `@ss`, …): "
        f"the normaliser recovers the real word for **{leet_hits / leet_total:.0%}** of them, up from "
        f"{LEETSPEAK_BEFORE} before symbols like `@` and `$` were handled. The rest are spelling changes "
        "a character-by-character mapping can't undo, like swapped vowels."
    )

    st.subheader("Most influential words")
    st.altair_chart(top_words_chart(classifier, vectorizer), width="stretch")

    st.subheader("Dataset")
    st.write(
        f"{summary['rows']:,} entries ({summary['rude']:,} profane, {summary['polite']:,} clean) from the Surge AI "
        "Obscenity List, common English words from wordfreq and a few hand-curated additions. "
        "The **Dataset** page has the full breakdown, charts and a searchable table."
    )
    st.markdown("[🗂️ Explore the dataset →](dataset)")

    st.subheader("Limitations & next steps")
    st.markdown(
        "- **Words, not sentences:** the model is trained on single words and short phrases, so it can't "
        "learn context. *\"you're killing it\"* and a real threat look alike. Next step: train on labelled "
        "sentences such as the Civil Comments dataset, then compare against a fine-tuned transformer.\n"
        "- **English only:** both the profanity list and the clean words are English, so swear words in "
        "other languages aren't detected. Next step: add "
        "profanity lists and common words for other languages.\n"
        "- **Unseen words are missed:** a brand-new insult has no effect. More data helps, context helps more.\n"
        "- **Leetspeak rules are hand-written:** `f4cking` becomes `facking` (then split into `fac king`), which the model doesn't recognise. "
        "Next step: fuzzy matching or character n-grams.\n"
        "- **Profanity, not toxicity:** mild insults like *stupid* aren't in the training data either way, and "
        "hateful messages without swear words pass as clean.\n"
        "- **Feedback stays local:** the CLI saves corrections to your local copy of the dataset, and the web app "
        "can't take feedback yet. Next step: move the dataset to an online database so feedback updates it for everyone."
    )
