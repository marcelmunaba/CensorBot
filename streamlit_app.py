import streamlit as st

from ui.chat import chat_page
from ui.common import ROBOT_PATH, sidebar
from ui.dataset import dataset_page
from ui.model_card import model_card_page
from ui.pipeline import pipeline_page

# Streamlit web app - entry point. Run with: streamlit run streamlit_app.py
st.set_page_config(page_title="CensorBot", page_icon=str(ROBOT_PATH), layout="centered")

sidebar()
page = st.navigation([
    st.Page(chat_page, title="Try it", icon="💬", url_path="chat", default=True),
    st.Page(pipeline_page, title="Pipeline", icon="⚙️", url_path="pipeline"),
    st.Page(model_card_page, title="Model", icon="📊", url_path="model"),
    st.Page(dataset_page, title="Dataset", icon="🗂️", url_path="dataset"),
])
page.run()
