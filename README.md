# CensorBot
A simple profanity checker made using machine Learning that utilizes the Naive Bayes classifier from sklearn library.
It comes in two flavours: a Streamlit web app (`streamlit_app.py`) and a command line version (`cli.py`).

## Brief Background
Has it ever occured to you, that you typed something in the chat while playing online video games, and it gets censored? 
![demo_clean](https://github.com/marcelmunaba/CensorBot/assets/70313979/93a2d2e3-09bf-405d-b56c-630de9fed5d9)
![demo_censor](https://github.com/marcelmunaba/CensorBot/assets/70313979/56b83295-79ba-45ff-b057-5f7aeed29846)
Datasets are made personally by me and enlarged with help of ChatGPT. 

## Installation
Make sure you have Python installed, then install the dependencies:
```
pip install -r requirements.txt
```

## Usage
Web app (Streamlit):
```
streamlit run streamlit_app.py
```

Command line:
```
python cli.py
```
