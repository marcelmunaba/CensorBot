import pandas as pd
from preprocessing import preprocess_text
from model import train_model
import numpy as np
import joblib

def predict_curse(classifier, vectorizer, text):
    new_text = preprocess_text(text)
    print("Preprocessed text: ", new_text)
    new_text_transformed = vectorizer.transform([new_text])
    prediction = classifier.predict(new_text_transformed)
    
    if prediction == 1:
        censored_text = text
        for word in new_text.split():
            censored_text = censored_text.replace(word, "*" * len(word))
        print("Censored text: ", censored_text)
        return 1
    else:
        return 0

if __name__ == "__main__":
    data = pd.read_csv('./profanity_sample.csv', encoding='utf-8')
    
    # Train the model
    classifier, vectorizer = train_model(data)
    
    while True:
        print("Welcome to CensorBot :)")
        # Test the model - Example : "What the heck is going on here?"
        text_input = input("What do have in your mind? : ")
        print("Testing the model with input: " + text_input)
        
        if (predict_curse(classifier, vectorizer, text_input)) == 1:
            print("Hey that's a bit rude! Watch your language >:|")
        else :
            print("I see. Congratulations on being polite :)")
        input("Press any key to restart")
        print("\033c", end="")