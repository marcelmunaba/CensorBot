"""Build CensorBot's training dataset (data/training_words.csv) from its sources.

Sources (see data/NOTICE.md for licences):
  - data/sources/surge_profanity_en.csv  The Obscenity List by Surge AI - ~1,600 profanities with severity 1-3
  - data/sources/censorbot_original.csv  The original hand-made CensorBot word list
  - wordfreq                             The most common English words, used as the "clean" class

Run from the repo root:
    pip install -r requirements-dev.txt
    python scripts/build_dataset.py
"""
from pathlib import Path

import nltk
import pandas as pd
from wordfreq import top_n_list

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources"
OUTPUT = ROOT / "data" / "training_words.csv"

# How many of the most common English words to consider for the clean class
COMMON_WORDS = 5000

# Insults that are rude but not profanity. They are kept out of the clean class so the model
# doesn't learn them as "clean"; the ones in the original CensorBot list stay profane.
MILD_INSULTS = {
    "stupid", "dumb", "idiot", "idiotic", "moron", "loser", "jerk", "crap", "crappy", "damn",
    "damned", "hell", "suck", "sucks", "sucked", "ugly", "pathetic", "freak", "creep", "trash",
    "screw", "screwed", "piss", "pissed", "bloody", "bastard", "douche", "retard", "retarded",
    "butt",
}


def surge_rows():
    surge = pd.read_csv(SOURCES / "surge_profanity_en.csv")
    return pd.DataFrame({
        "text": surge["text"].str.strip().str.lower(),
        # Surge rates severity 1 (mild) to 3 (severe); map onto CensorBot's scale where
        # 1.0 is the "profane" threshold and 2.0 the most severe
        "severity_rating": (1.0 + (surge["severity_rating"] - 1.0) / 2.0).round(2),
        # Variants of the same word share a group, so cross-validation never splits them
        "group": surge["canonical_form_1"].str.strip().str.lower(),
        "source": "surge",
    })


def original_rows(surge):
    original = pd.read_csv(SOURCES / "censorbot_original.csv")
    original["text"] = original["text"].str.strip().str.lower()
    # "cockroach" is labelled both ways in the original list; it's an insect, so keep it clean
    original = original[~((original["text"] == "cockroach") & (original["severity_rating"] >= 1.0))]
    original = original.drop_duplicates("text")
    # Words Surge already covers come from Surge
    original = original[~original["text"].isin(surge["text"])]
    groups = dict(zip(surge["text"], surge["group"]))
    return pd.DataFrame({
        "text": original["text"],
        "severity_rating": original["severity_rating"],
        "group": original["text"].map(lambda w: groups.get(w, w)),
        "source": "censorbot",
    })


def clean_rows(excluded, needed):
    nltk.download("stopwords", quiet=True)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = []
    for word in top_n_list("en", COMMON_WORDS):
        if not word.isalpha() or len(word) < 2 or word in stop_words or word in excluded:
            continue
        words.append(word)
        if len(words) == needed:
            break
    return pd.DataFrame({"text": words, "severity_rating": 0.0, "group": words, "source": "wordfreq"})


def main():
    surge = surge_rows()
    original = original_rows(surge)
    profane = pd.concat([surge, original[original["severity_rating"] >= 1.0]])
    hand_clean = original[original["severity_rating"] < 1.0]

    # Never put a profane word, a variant root, or a mild insult into the clean class
    # Words that only appear inside profane phrases (e.g. "mother", "gay") stay eligible: the model
    # learns phrases as word pairs, so on their own these words should be clean
    excluded = set(profane["text"]) | set(profane["group"]) | MILD_INSULTS | set(hand_clean["text"])

    # Balance the classes: as many clean words as profane ones
    common = clean_rows(excluded, needed=len(profane) - len(hand_clean))
    dataset = pd.concat([profane, hand_clean, common], ignore_index=True)
    dataset.to_csv(OUTPUT, index=False)

    is_profane = dataset["severity_rating"] >= 1.0
    print(f"Wrote {len(dataset)} rows to {OUTPUT.relative_to(ROOT)}")
    print(f"  profane: {is_profane.sum()}  clean: {(~is_profane).sum()}")
    print(dataset.groupby(["source", is_profane.map({True: 'profane', False: 'clean'})]).size().to_string())


if __name__ == "__main__":
    main()
