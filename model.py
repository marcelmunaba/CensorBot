import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the contains_curse_word function
def contains_curse_word(row, data):
    # Extract all curse words from the dataset
    curse_words = set(data[['canonical_form_1', 'canonical_form_2', 'canonical_form_3']].values.flatten())
    
    # Check if any word in the row matches any curse word
    return any(word in curse_words for word in row)

# Define the train_model function
def train_model(data):
    # Create the target variable (y) using contains_curse_word function
    y = data[['canonical_form_1', 'canonical_form_2', 'canonical_form_3']].apply(contains_curse_word, axis=1, data=data).astype(int)

    # Extract features using Bag-of-Words model
    X = data[['canonical_form_1', 'canonical_form_2', 'canonical_form_3']].fillna('').apply(lambda row: ' '.join(row), axis=1)
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.1, random_state=42)

    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = classifier.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return classifier, vectorizer