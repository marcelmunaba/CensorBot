import pandas as pd
import streamlit as st

from censorbot.model import severity_weights
from censorbot.preprocessing import preprocess_steps
from ui.common import ensure_ready, load_evaluation, load_model
from ui.dataset import SOURCE_NAMES, mask

PIPELINE = [
    ("Lowercase", "`What the F*CK?!` → `what the f*ck?!`"),
    ("Normalise leetspeak", "Digits and symbols used as letters are mapped back: `sh1t` → `shit`, `@ss` → `ass`, `b!tch` → `bitch`. "
     "Plain numbers like `10` and trailing punctuation like `hi!` are left alone"),
    ("Strip punctuation", "`what the f*ck?!` → `what the fck`"),
    ("Tokenise", "NLTK `word_tokenize` splits the text into words"),
    ("Remove stopwords", "Common words like *the*, *is*, *a* carry no signal and are dropped"),
    ("Lemmatise", "WordNet lemmatiser reduces words to their base form: `jerks` → `jerk` (known profanity is left as is)"),
    ("Split joined words", "`wordninja` splits words typed together: `stupidjerk` → `stupid jerk` (known profanity like `fuckoff` is kept whole)"),
    ("Vectorise", "`CountVectorizer` with single words + word pairs"),
    ("Classify", "Multinomial Naive Bayes, trained with word counts weighted by each word's severity rating"),
]

EXAMPLE_WORDS = ["fuck", "dumbass", "idiot", "family", "weekend"]


def training_examples(data):
    """A handful of dataset rows with the label and training weight the model gives them."""
    weights = severity_weights(data)
    rows = data[data['text'].isin(EXAMPLE_WORDS)].drop_duplicates('text')
    examples = []
    for word, severity, source in zip(rows['text'], rows['severity_rating'], rows['source']):
        profane = severity >= 1.0
        examples.append({
            "Word": mask(word) if profane else word,
            "Source": SOURCE_NAMES.get(source, source),
            "Severity": severity,
            "Label": "Profane (1)" if profane else "Clean (0)",
            "Training weight": round(weights.get(word, 1.0), 2),
        })
    return pd.DataFrame(examples).sort_values("Severity", ascending=False)


def pipeline_page():
    ensure_ready(evaluation=True)
    _, _, data = load_model()
    _, summary, _ = load_evaluation()

    st.title("Pipeline")
    st.write(
        "How CensorBot learns from its word list, and what happens to your message step by step."
    )

    st.subheader("How it was trained")
    st.markdown(
        f"1. **Build the dataset.** {summary['rows']:,} entries: {summary['rude']:,} profanities from the "
        "Surge AI Obscenity List (plus a few hand-curated additions) and an equal number of "
        "common English words as clean examples. See the *Dataset* page.\n"
        "2. **Label it.** Severity **1.0 or higher → profane (1)**, anything lower → clean (0).\n"
        "3. **Clean every entry like a message.** Each training word goes through the same pipeline as your "
        "messages, so `a55` and `@ss` are learned as `ass`. Training on raw spellings while predicting on "
        "cleaned text would be a training/serving mismatch.\n"
        "4. **Vectorise.** `CountVectorizer` learns a vocabulary and turns each entry into word counts. "
        f"Multi-word profanities ({summary['multi_word']} of them) only contribute their word pairs, so "
        "*son of a …* teaches the phrase, not that *son* is profane.\n"
        "5. **Weight by severity.** Each profane word's counts are multiplied by its severity ÷ the "
        "highest severity, so harsher words count more than milder ones. Clean words keep a weight of 1.0. "
        "The weighting applies during training only.\n"
        "6. **Fit Multinomial Naive Bayes.** For every word it learns how likely it is to appear in profane vs. "
        "clean text. Those probabilities are what the word chips on the *Try it* page show."
    )
    st.dataframe(training_examples(data), hide_index=True, use_container_width=True)
    st.caption("A few rows from the dataset and what training does with them.")

    st.subheader("How a message is classified")
    for i, (step, detail) in enumerate(PIPELINE, start=1):
        st.markdown(f"**{i}. {step}** — {detail}")

    st.subheader("See it on your own text")
    text = st.text_input("Message", value="Stop being such a dumb4ss", label_visibility="collapsed")
    if text.strip():
        _, steps = preprocess_steps(text, data)
        st.dataframe(pd.DataFrame(steps, columns=["Step", "Text"]), hide_index=True, use_container_width=True)
        st.caption("The last row is what the model scores. Head to *Try it* for the verdict and censored text.")
