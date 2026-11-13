import random
import nltk
import pandas as pd
from censorbot import predict as pred
from censorbot import DATA_PATH

# Command line version of CensorBot. For the web version, run: streamlit run streamlit_app.py

# Download necessary NLTK data - important for preprocessing!
for package in ('punkt_tab', 'stopwords', 'wordnet'):
    nltk.download(package, quiet=True)

if __name__ == "__main__":
    #TODO: fetch from Database
    data = pd.read_csv(DATA_PATH, encoding='utf-8')

    # Train the model
    classifier, vectorizer = pred.train_model(data)

    print("Welcome to CensorBot :)")
    while True:
        # Test the model - Example : "What the heck is going on here?"
        text_input = input("Hello! What do have in your mind? : ")
        print("Testing the model with input: " + text_input)

        predict, preprocessedText = pred.predict_curse(classifier, vectorizer, text_input, data)
        if predict == 1:
            print("Censored text: ", pred.censor_text(predict, preprocessedText, vectorizer))
            print("Hey that's a bit rude! Watch your language >:|")
        else:
            print("I see. Congratulations on being polite :)")

        response = input("Do I get that right? (Y/n) ")
        new_resp = response.lower()
        if new_resp == 'n' and predict == 0:
            #write the new vocabulary to the csv
            pred.append_to_csv(DATA_PATH, preprocessedText, random.uniform(1.0, 2.0))
            print("Okay. I'll keep that in mind :)\n")
        elif new_resp == 'n' and predict == 1:
            #write the new vocabulary to the csv
            pred.append_to_csv(DATA_PATH, preprocessedText, random.uniform(0.0, 1.0))
            print("Okay. I'll keep that in mind :)\n")
        else:
            print("Thanks for your feedback!\n")
