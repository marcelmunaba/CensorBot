import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .preprocessing import preprocess_text


def prepare_training_text(data):
    """Run every training row through the same preprocessing as user messages.

    Without this the model would learn raw spellings (a55, @ss) that never appear after
    a message has been normalised - a classic training/serving mismatch.
    """
    return data['text'].astype(str).map(lambda text: preprocess_text(text, data))


def severity_weights(data, texts=None):
    """Training weight for each profane word: its severity divided by the highest severity.

    Only words rated above 1.0 get a weight; every other word keeps the default weight of 1.0.
    `texts` are the (preprocessed) words to key the weights by; defaults to the raw text column.
    """
    texts = data['text'] if texts is None else texts
    severities = data['severity_rating'].groupby(texts.to_numpy()).max()  # variants can share a spelling
    profane = severities[severities > 1.0]  # Consider only profane words with severity > 1.0
    return (profane / profane.max()).to_dict()  # Normalize weights


def train_model(data):
    # Step 1: Clean the training words exactly like incoming messages
    texts = prepare_training_text(data)

    # Step 2: Feature extraction using CountVectorizer with unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2))  # Include unigrams and bigrams
    X = vectorizer.fit_transform(texts)

    # Profane phrases ("son of a b****") should teach the model the phrase, not that each
    # word in it ("son") is profane on its own - so they only contribute their word pairs
    is_unigram = np.array([' ' not in name for name in vectorizer.get_feature_names_out()])
    is_phrase = ((data['severity_rating'] >= 1.0) & (texts.str.split().str.len() > 1)).to_numpy()
    X = X - X.multiply(is_phrase[:, None]).multiply(is_unigram[None, :])

    # Step 3: Assign weights to profane words based on severity ratings
    word_weights = severity_weights(data, texts)

    # Step 4: Label the data based on severity
    y_label = (data['severity_rating'] >= 1.0).astype(int)

    # Step 5: Train Naive Bayes classifier with weighted features
    classifier = MultinomialNB()
    weighted_X = X.multiply([word_weights.get(word, 1.0) for word in vectorizer.get_feature_names_out()])
    classifier.fit(weighted_X.tocsr(), y_label)

    return classifier, vectorizer
