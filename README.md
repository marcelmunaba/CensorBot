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
Web app (Streamlit) - live demo at https://censorbot.streamlit.app:
```
streamlit run streamlit_app.py
```
The web app has four pages: **Try it**, which classifies and censors your message and shows which words drove the decision; **Pipeline**, which explains training and every preprocessing step (and runs them on your own text); **Model**, with cross-validated metrics, the most influential words and known limitations; and **Dataset**, with the data sources, charts and a searchable table.

### Command line (CLI)
Prefer the terminal? `cli.py` runs the same model without a browser. It trains on startup, then classifies whatever you type and censors profane words:
```
python cli.py
```
```
Welcome to CensorBot :) (press Ctrl+C to quit)
Hello! What do you have in mind? : Stop being such a dumb4ss
Censored text:  Stop being such a *******
Hey that's a bit rude! Watch your language >:| (profane, 56% sure)
Do I get that right? (Y/n)
```
Unlike the web app, the CLI learns from feedback: answer `n` when it's wrong and it saves the message with the corrected label to your local copy of the dataset (`data/training_words.csv`), so the model learns from it the next time it starts.

*Planned:* once CensorBot's dataset moves to an online database, feedback will be able to update that shared dataset instead of just your local copy.

## How It Works
CensorBot has two parts: **training** (it learns from a word list once, when the app starts) and **classifying** (what happens every time you send a message).

### 1. Training: learning which words are profane
1. **Build the word list.** [`scripts/build_dataset.py`](scripts/build_dataset.py) combines ~1,600 profanities from the [Surge AI Obscenity List](https://huggingface.co/datasets/mmathys/profanity) with the same number of common English words from [wordfreq](https://github.com/rspeer/wordfreq) into [`data/training_words.csv`](data/training_words.csv). Each entry has a severity rating from 0 (harmless) to 2 (very offensive).
2. **Label each entry.** A severity of **1.0 or higher means profane**, anything lower means clean.
3. **Clean each entry like a message.** Training words go through the same steps as your messages (below), so `a55` and `@ss` are both learned as `ass`.
4. **Turn words into numbers.** A model can't read text, so each entry becomes a vector of counts: "this word appears once" (`CountVectorizer`). Multi-word profanities like *son of a …* only teach the model the word pairs, so *son* on its own stays clean.
5. **Give harsher words more weight.** Each profane word is weighted by how severe it is, so a strong swear word counts more than a mild one. This only applies during training.
6. **Train a Naive Bayes classifier.** For every word, it learns how likely that word is to show up in profane vs. clean text.

### 2. Classifying a message
Take the message `Stop being such a dumb4ss`:

| Step | Result | Why |
|---|---|---|
| Lowercase | `stop being such a dumb4ss` | `Ass` and `ASS` should count as the same word |
| Fix leetspeak | `stop being such a dumbass` | People swap letters for numbers and symbols to dodge filters (`4`→a, `1`→i, `3`→e, `0`→o, `5`→s, `@`→a, `$`→s, `!`→i) |
| Remove punctuation | `stop being such a dumbass` | `ass!` and `ass` should match. This runs *after* leetspeak, so `@ss` isn't lost |
| Split into words | `stop` `being` `such` `a` `dumbass` | The model looks at words one at a time |
| Drop filler words | `stop` `dumbass` | Words like *being*, *such* and *a* say nothing about profanity |
| Reduce to base form | `stop` `dumbass` | `jerks` → `jerk`, so plurals match the word list |
| Split glued words | `stop` `dumbass` | `stupidjerk` → `stupid jerk`; known swear words like `dumbass` are left whole |
| Score with the model | **Profane, 56%** | `dumbass` pushes towards profane, `stop` (a common clean word) pulls the other way |

Finally, every word in **your original message** that pushed towards profane is replaced with asterisks: `Stop being such a *******`.

### Things to keep in mind
- **English only.** The training data is exclusively English, so profanity in other languages isn't detected.
- **It only knows the words it was trained on.** A brand-new insult has no effect, and a message with only unknown words counts as clean.
- **It doesn't understand context.** It looks at words, not meaning, so *"you're killing it"* and a real threat look alike.
- **It detects profanity, not rudeness.** Mild insults like *stupid* aren't profanity, and hateful messages without swear words pass as clean.
- **The confidence is a balance of the words.** Clean words in a message pull the score down, so one swear word in a long sentence can score close to 50%.

The web app's **Pipeline** page walks through the same steps, the **Model** page has the evaluation metrics, and the **Try it** page shows each step for your own message.

## Data
| Source | Used for | Licence |
|---|---|---|
| [Surge AI Obscenity List](https://huggingface.co/datasets/mmathys/profanity) | ~1,600 profanities and spelling variants, rated 1-3 by human annotators | MIT |
| [wordfreq](https://github.com/rspeer/wordfreq) | The most common English words, as clean examples | CC-BY-SA 4.0 (data) |
| Hand-curated additions | ~100 words picked by hand for CensorBot: everyday words and insults like *idiot* and *moron* that the other sources miss | This project |

To rebuild the training data from its sources:
```
pip install -r requirements-dev.txt
python scripts/build_dataset.py
```
Full credits and licence texts are in [`data/NOTICE.md`](data/NOTICE.md).

## Project Structure
```
CensorBot/
├── streamlit_app.py        # Streamlit web app (entry point)
├── cli.py                  # Command line version (entry point)
├── requirements.txt
├── requirements-dev.txt    # Extra tools for rebuilding the dataset
├── censorbot/              # Core logic
│   ├── preprocessing.py    # Text cleaning, leetspeak handling, tokenizing
│   ├── model.py            # Naive Bayes training with weighted features
│   ├── predict.py          # Prediction, censoring and dataset updates
│   └── insights.py         # Word-level explanations and cross-validation
├── ui/                     # Streamlit pages
│   ├── chat.py             # "Try it" page with explained results
│   ├── pipeline.py         # Training process and preprocessing steps
│   ├── model_card.py       # Evaluation metrics, influential words and limitations
│   ├── dataset.py          # Dataset sources, charts and a searchable table
│   └── common.py           # Shared model loading and sidebar
├── assets/
│   └── censorbot.svg       # Robot mascot (logo, avatar and favicon)
├── scripts/
│   └── build_dataset.py    # Builds the training data from its sources
├── .streamlit/config.toml  # Theme
└── data/
    ├── training_words.csv  # Training dataset (generated)
    ├── NOTICE.md           # Data credits and licences
    └── sources/            # Surge AI list and the hand-curated additions
```
