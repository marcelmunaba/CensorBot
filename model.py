from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(data):
    # Step 1: Feature extraction using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])

    # Step 2: Label the data based on severity
    y_label = (data['severity_rating'] >= 1.0).astype(int)

    # Step 3: Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, y_label)

    return classifier, vectorizer
