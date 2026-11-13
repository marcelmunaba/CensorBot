import pandas as pd
from preprocessing import preprocess_text
from model import train_model
import numpy as np
import csv

# Predict function after receiving input from the user
def predict_curse(classifier, vectorizer, text, data):
    new_text = preprocess_text(text, data)
    #print("Preprocessed text: ", new_text) #debugging
    new_text_transformed = vectorizer.transform([new_text])
    prediction = classifier.predict(new_text_transformed)
    return prediction, new_text

# Used to update the csv - future implementation: update the database
def append_to_csv(file_path, new_word, severity_rating):
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Round the severity rating to one decimal place
        severity_rating_rounded = round(severity_rating, 1)
        writer.writerow([new_word, severity_rating_rounded])  # Enclose both elements within a list or tuple

# Returns the censored text written by the user based on the prediction
def censor_text(predict, text, vectorizer):
    if predict == 1:
        original_text = text
        # Get the feature names (words) from the vectorizer
        feature_names = vectorizer.get_feature_names_out()
        # Iterate through each word in the preprocessed text
        for word in text.split():
            # Check if the word is not in the feature names (learned during training)
            if word not in feature_names:
                continue
            # Replace the word with asterisks in the original text
            original_text = original_text.replace(word, "*" * len(word))
        
        return original_text
    else:
        return text

# The command line loop lives in cli.py, the web app in streamlit_app.py
