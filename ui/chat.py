import html

import pandas as pd
import streamlit as st

from censorbot.insights import analyse
from ui.common import ROBOT_PATH, ensure_ready, header, load_model

EXAMPLES = [
    "Have a lovely weekend with your family!",
    "What the fuck is going on here?",
    "Stop being such a dumb4ss",
    "$hit, I missed the bus",
]

CARD_CSS = """
<style>
.cb-censored {
    font-size: 1.1rem; padding: 0.6rem 0.9rem; border-radius: 0.5rem;
    border-left: 4px solid var(--cb-accent); background: rgba(128, 128, 128, 0.08);
    margin: 0.25rem 0 0.75rem 0; overflow-wrap: anywhere;
}
.cb-token {
    display: inline-block; padding: 0.1rem 0.5rem; margin: 0.15rem; border-radius: 999px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem;
    border: 1px solid rgba(128, 128, 128, 0.3);
}
.cb-rude { background: rgba(239, 68, 68, 0.18); border-color: rgba(239, 68, 68, 0.6); }
.cb-polite { background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.55); }
.cb-unknown { opacity: 0.6; border-style: dashed; }
.cb-legend { font-size: 0.8rem; opacity: 0.75; }
</style>
"""


def token_chip(token, score):
    if score is None:
        css, tip = "cb-unknown", "Not in the model's vocabulary - no influence"
    elif score > 0:
        css, tip = "cb-rude", f"Pushes towards profane ({score:+.2f})"
    else:
        css, tip = "cb-polite", f"Pushes towards clean ({score:+.2f})"
    return f'<span class="cb-token {css}" title="{html.escape(tip)}">{html.escape(token)}</span>'


def render_result(result):
    accent = "#ef4444" if result["is_rude"] else "#22c55e"
    if result["is_rude"]:
        st.markdown(":red-background[**Profane**] &nbsp; Hey, that's a bit rude! Watch your language 😠")
    else:
        st.markdown(":green-background[**Clean**] &nbsp; All clean, thanks for keeping it friendly 😊")

    st.markdown(
        f'<div class="cb-censored" style="--cb-accent:{accent}">{html.escape(result["censored"])}</div>',
        unsafe_allow_html=True,
    )
    st.progress(result["confidence"], text=f"Model confidence: {result['confidence']:.0%}")

    chips = "".join(token_chip(t, s) for t, s in result["word_scores"]) or "<em>No words left after preprocessing</em>"
    st.markdown(
        f"<div><strong>What the model saw</strong></div><div>{chips}</div>"
        '<div class="cb-legend">Red pushes towards profane, green towards clean, '
        "dashed = unknown word. Hover a word for its score.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("How CensorBot read this"):
        steps = pd.DataFrame(result["steps"], columns=["Step", "Text"])
        st.dataframe(steps, hide_index=True, width="stretch")
        st.caption(
            "The final text is turned into unigram + bigram counts and scored by a "
            f"Multinomial Naive Bayes classifier: P(profane) = {result['prob_rude']:.2f}."
        )


def chat_page():
    ensure_ready(evaluation=False)
    classifier, vectorizer, data = load_model()
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    header("CensorBot")
    st.write(
        "Ever typed something in the chat of an online game, only to see it come out as `****`? "
        "CensorBot is my take on building that kind of chat filter from scratch with machine learning."
    )
    st.write(
        "Type a message and CensorBot will tell you whether it contains profanity, censor the "
        "offending words, and show you exactly which words drove the decision."
    )
    st.caption("🌐 English only: CensorBot is trained exclusively on English words, so other languages won't be detected.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle a new message (typed or from an example button) before drawing the page,
    # so the examples and the "Clear chat" button reflect it straight away
    prompt = st.chat_input("Type a message...") or st.session_state.pop("pending", None)
    if prompt:
        result = analyse(prompt, classifier, vectorizer, data)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "result": result})

    with st.sidebar:
        if st.button("Clear chat", width="stretch", disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()

    with st.chat_message("assistant", avatar=str(ROBOT_PATH)):
        st.markdown("Hi there, I'm CensorBot. What do you have in mind?")

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.text(message["content"])
        else:
            with st.chat_message("assistant", avatar=str(ROBOT_PATH)):
                render_result(message["result"])

    # Suggestions stay below the latest message, so there's always something to try next
    if st.session_state.messages:
        st.caption("Try another one:")
    else:
        st.caption("Not sure what to type? Try one of these:")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if columns[i % 2].button(example, key=f"example-{i}", width="stretch"):
            st.session_state.pending = example
            st.rerun()
