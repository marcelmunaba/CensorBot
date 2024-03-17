import pandas as pd
from preprocessing import preprocess_text
from model import train_model
import numpy as np

#TODO: censor the curse word to **** and add a funny text :)
def predict_curse(classifier, vectorizer, text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = classifier.predict(vectorized_text)[0]
    if prediction == 1:
        return "Curse word detected."
    else:
        return "No curse word detected."

if __name__ == "__main__":
    data = pd.read_csv('./profanity_en.csv', encoding='utf-8')

    # Train the model
    classifier, vectorizer = train_model(data)

     # Test the model - Example : "What the heck is going on here?"
    text_input = input("Type your input here : ")
    print(predict_curse(classifier, vectorizer, text_input))