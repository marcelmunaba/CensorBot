import re #regex

def preprocess_text(text):
    text = text.lower() # Set text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text