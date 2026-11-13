import random
import nltk
import pandas as pd
from censorbot import predict as pred
from censorbot import DATA_PATH
from censorbot.insights import analyse

# Command line version of CensorBot. For the web version, run: streamlit run streamlit_app.py

# Download necessary NLTK data - important for preprocessing!
for package in ('punkt_tab', 'stopwords', 'wordnet'):
    nltk.download(package, quiet=True)

if __name__ == "__main__":
    #TODO: fetch from Database
    data = pd.read_csv(DATA_PATH, encoding='utf-8')

    # Train the model
    classifier, vectorizer = pred.train_model(data)

    print("Welcome to CensorBot :) (press Ctrl+C to quit)")
    while True:
        # Test the model - Example : "What the heck is going on here?"
        try:
            text_input = input("Hello! What do you have in mind? : ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye! Keep it friendly :)")
            break
        print("Testing the model with input: " + text_input)

        result = analyse(text_input, classifier, vectorizer, data)
        predict, preprocessedText = int(result["is_rude"]), result["preprocessed"]
        if predict == 1:
            print("Censored text: ", result["censored"])
            print(f"Hey that's a bit rude! Watch your language >:| (profane, {result['confidence']:.0%} sure)")
        else:
            print(f"I see. Congratulations on being polite :) (clean, {result['confidence']:.0%} sure)")

        response = input("Do I get that right? (Y/n) ")
        new_resp = response.lower()
        if new_resp == 'n' and predict == 0:
            #write the new vocabulary to the csv
            pred.append_to_csv(DATA_PATH, preprocessedText, random.uniform(1.0, 2.0))
            print("Okay. I'll keep that in mind :)\n")
        elif new_resp == 'n' and predict == 1:
            #write the new vocabulary to the csv
            pred.append_to_csv(DATA_PATH, preprocessedText, random.uniform(0.0, 0.9))
            print("Okay. I'll keep that in mind :)\n")
        else:
            print("Thanks for your feedback!\n")
