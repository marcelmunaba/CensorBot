import base64
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st

from censorbot import DATA_PATH
from censorbot.insights import cross_validate, dataset_summary, leetspeak_recovery
from censorbot.model import train_model

GITHUB_URL = "https://github.com/marcelmunaba/CensorBot"
ROBOT_PATH = Path(__file__).resolve().parent.parent / "assets" / "censorbot.svg"
_ROBOT_URI = "data:image/svg+xml;base64," + base64.b64encode(ROBOT_PATH.read_bytes()).decode()


def robot_img(size, css_class=""):
    """The CensorBot robot as an inline <img> tag."""
    return (
        f'<img src="{_ROBOT_URI}" width="{size}" height="{size}" alt="CensorBot robot" '
        f'class="{css_class}" style="display:inline-block; vertical-align:middle">'
    )


def header(title, size=64):
    """Page title with the robot beside it."""
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem">'
        f'{robot_img(size)}<h1 style="margin:0; padding:0">{title}</h1></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def download_nltk_data():
    # Needed by the preprocessing pipeline (tokeniser, stopwords, lemmatiser)
    for package in ('punkt_tab', 'stopwords', 'wordnet'):
        nltk.download(package, quiet=True)


@st.cache_resource(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH, encoding='utf-8')


@st.cache_resource(show_spinner=False)
def load_model():
    download_nltk_data()
    data = load_dataset()
    classifier, vectorizer = train_model(data)
    return classifier, vectorizer, data


@st.cache_data(show_spinner=False)
def load_evaluation():
    _, _, data = load_model()
    return cross_validate(data), dataset_summary(data), leetspeak_recovery()


# Startup steps that have finished in this server process. The model is trained once per
# server start (after a deploy, reboot or wake-up) and then shared by every visitor.
_ready = set()


def ensure_ready(evaluation=False):
    """Show a loading screen while the model is being trained for the first time."""
    steps = [
        ("nltk", "Downloading language data", download_nltk_data),
        ("data", "Loading the training words", load_dataset),
        ("model", "Training the Naive Bayes model", load_model),
    ]
    if evaluation:
        steps.append(("evaluation", "Evaluating the model with cross-validation", load_evaluation))
    pending = [step for step in steps if step[0] not in _ready]
    if not pending:
        return

    screen = st.empty()
    with screen.container():
        st.markdown(
            "<style>@keyframes cb-bob {0%, 100% {transform: translateY(0) rotate(-3deg)} "
            "50% {transform: translateY(-10px) rotate(3deg)}} .cb-bob {animation: cb-bob 1.2s ease-in-out infinite}</style>"
            "<div style='text-align:center; padding: 3rem 0 1rem 0'>"
            f"{robot_img(110, 'cb-bob')}"
            "<h2 style='margin-bottom:0.25rem'>Training CensorBot…</h2>"
            "<p style='opacity:0.7'>The model learns from its word list every time the app starts. "
            "This only takes a few seconds and happens once per restart.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        progress = st.progress(0.0)
        with st.status("Getting ready…", expanded=True) as status:
            for i, (key, label, step) in enumerate(steps):
                status.update(label=f"{label}…")
                st.write(f"{label}…")
                step()
                _ready.add(key)
                progress.progress((i + 1) / len(steps))
            status.update(label="Ready!", state="complete")
    screen.empty()


def sidebar():
    with st.sidebar:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem">'
            f'{robot_img(44)}<span style="font-size:1.3rem; font-weight:700">CensorBot</span></div>',
            unsafe_allow_html=True,
        )
        st.write(
            "A chat filter like the ones in online games, built from scratch: custom NLP preprocessing, "
            "leetspeak normalisation and a weighted Naive Bayes classifier, trained on "
            "~3,200 words from Surge AI's profanity list and common English."
        )
        st.markdown(
            "**Stack:** Python · scikit-learn · NLTK · pandas · Streamlit"
        )
        st.link_button("View source on GitHub", GITHUB_URL, use_container_width=True)
