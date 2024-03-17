import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(data):
    # Feature Engineering
    data['text'] = data[['canonical_form_1', 'canonical_form_2', 'canonical_form_3']].apply(lambda x: ' '.join(x.dropna()), axis=1)

    # Model Training
    X = data['text']
    y = (data['text'].str.contains('fuck') | data['text'].str.contains('ass')).astype(int)  # Binary label for curse word detection

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Step 4: Evaluation
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return classifier,vectorizer