"""Helpers that explain the model's decisions and evaluate it - used by the Streamlit UI."""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from .model import train_model
from . import SOURCES_DIR
from .preprocessing import normalise_leetspeak, preprocess_steps, preprocess_text

# A word as typed: letters/digits, optionally starting with @ or $ and with leetspeak symbols
# inside it (@ss, $hit, b!tch) - but trailing punctuation ("bitch!") is not part of the word
WORD_PATTERN = re.compile(r"[@$]?[A-Za-z0-9](?:[A-Za-z0-9'@$!|]*[A-Za-z0-9])?")


def word_log_ratios(classifier, vectorizer):
    """log P(word | rude) - log P(word | polite) for every word the model knows.

    Positive means the word pushes the prediction towards rude, negative towards polite.
    """
    names = vectorizer.get_feature_names_out()
    log_prob = classifier.feature_log_prob_
    ratios = log_prob[1] - log_prob[0]
    # Skip features with no training counts at all (single words of profane phrases): the model
    # has effectively never seen them on their own, so treat them as unknown
    seen = classifier.feature_count_.sum(axis=0) > 0
    return {name: ratio for name, ratio, ok in zip(names, ratios, seen) if ok}


def leetspeak_recovery():
    """How often the normaliser recovers the real word from Surge AI's spelling variants.

    Uses single-word variants containing digits or symbols (e.g. a55, @ss) and checks whether the
    root word Surge lists for it appears in the normalised text.
    """
    surge = pd.read_csv(SOURCES_DIR / "surge_profanity_en.csv")
    variants = surge[
        surge['text'].str.contains(r"[0-9@$!|]")
        & ~surge['text'].str.contains(" ")
        & ~surge['canonical_form_1'].str.contains(" ")
    ]
    recovered = [
        root in normalise_leetspeak(text)
        for text, root in zip(variants['text'].str.lower(), variants['canonical_form_1'].str.lower())
    ]
    return sum(recovered), len(recovered)


def analyse(text, classifier, vectorizer, data):
    """Classify a message and collect everything the UI needs to explain the result."""
    preprocessed, steps = preprocess_steps(text, data)
    features = vectorizer.transform([preprocessed])
    prob_polite, prob_rude = classifier.predict_proba(features)[0]
    is_rude = bool(classifier.predict(features)[0] == 1)

    ratios = word_log_ratios(classifier, vectorizer)
    # Score each preprocessed token; None means the model has never seen the word
    word_scores = [(token, ratios.get(token)) for token in preprocessed.split()]

    # Censor the words the user actually typed, not the preprocessed version
    flagged = []
    censored = text
    if is_rude:
        def mask(match):
            word = match.group(0)
            tokens = preprocess_text(word, data).split()
            if any((ratios.get(token) or 0) > 0 for token in tokens):
                flagged.append(word)
                return re.sub(r"[^']", "*", word)
            return word
        censored = WORD_PATTERN.sub(mask, text)

    return {
        "text": text,
        "is_rude": is_rude,
        "prob_rude": float(prob_rude),
        "confidence": float(prob_rude if is_rude else prob_polite),
        "preprocessed": preprocessed,
        "steps": steps,
        "word_scores": word_scores,
        "censored": censored,
        "flagged": flagged,
    }


def top_words(classifier, vectorizer, n=10):
    """The n words that push hardest towards each class."""
    ratios = pd.Series(word_log_ratios(classifier, vectorizer)).sort_values()
    return ratios.tail(n)[::-1], ratios.head(n)


def dataset_summary(data):
    labels = (data['severity_rating'] >= 1.0)
    words = data['text'].str.strip().str.lower()
    # Same word labelled both rude and polite somewhere in the dataset
    conflicts = sorted(w for w, group in labels.groupby(words) if group.nunique() > 1)
    return {
        "rows": len(data),
        "rude": int(labels.sum()),
        "polite": int((~labels).sum()),
        "duplicates": int(words.duplicated().sum()),
        "conflicts": conflicts,
        "multi_word": int((labels & words.str.contains(" ")).sum()),
        "sources": data['source'].value_counts().to_dict() if 'source' in data else {},
    }


def cross_validate(data, folds=5, seed=42):
    """Stratified, grouped k-fold evaluation using the same preprocessing + prediction path as the app.

    Each fold's model is trained only on that fold's training rows, so every test word is unseen.
    Spelling variants of the same word (e.g. a55 and @ss) share a group and always land in the
    same fold - otherwise the model would be tested on words it effectively saw during training.
    """
    y = (data['severity_rating'] >= 1.0).astype(int).to_numpy()
    groups = data['group'] if 'group' in data else data['text']
    y_true, y_pred = [], []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(data, y, groups):
        train = data.iloc[train_idx].copy()
        classifier, vectorizer = train_model(train)
        for i in test_idx:
            features = vectorizer.transform([preprocess_text(data['text'].iloc[i], train)])
            y_pred.append(int(classifier.predict(features)[0]))
            y_true.append(int(y[i]))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    return {
        "folds": folds,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # rows = actual (polite, rude), columns = predicted (polite, rude)
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }
