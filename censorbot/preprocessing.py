import re
import string
from functools import lru_cache
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import wordninja


@lru_cache(maxsize=1)
def _stopwords():
    return frozenset(stopwords.words('english'))


@lru_cache(maxsize=1)
def _lemmatizer():
    return WordNetLemmatizer()


LEET_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g',
    '@': 'a', '$': 's', '!': 'i', '|': 'i',
}
LEET_SYMBOLS = '@$!|'


def _normalise_word(word):
    # Leave pure numbers and punctuation alone ("10 apples", "$5", "!!!")
    if not re.search(r'[a-z]', word):
        return word
    chars = []
    for i, char in enumerate(word):
        if char.isdigit():
            chars.append(LEET_MAP.get(char, char))
        elif char in LEET_SYMBOLS:
            followed_by_word = re.search(r'[a-z0-9]', word[i + 1:]) is not None
            # ! and | only count as letters inside a word (b!tch), @ and $ also at the start (@ss, $hit)
            # (checked against the already-converted text, so runs like @$$ work too)
            preceded_ok = (i > 0 and chars[-1].isalnum()) or (i == 0 and char in '@$')
            chars.append(LEET_MAP[char] if followed_by_word and preceded_ok else char)
        else:
            chars.append(char)
    return ''.join(chars)


def normalise_leetspeak(text):
    """Map digits and symbols used as letters back to letters, word by word."""
    return ' '.join(_normalise_word(word) for word in text.split(' '))


def profanity_dictionary(data):
    """Every individual word that appears in a rude (severity >= 1.0) entry."""
    # The DataFrame is reused for every message, so build the set once and keep it on the frame
    cached = data.attrs.get('_profanity_dictionary')
    if cached is not None and cached[0] == len(data):
        return cached[1]
    rude = data.loc[data['severity_rating'] >= 1.0, 'text'].astype(str).str.lower()
    # Include the de-leeted spelling too (a55 -> ass), so variants are recognised after normalisation
    normalised = rude.map(normalise_leetspeak).str.translate(str.maketrans('', '', string.punctuation))
    dictionary = set(rude.str.split().explode()) | set(normalised.str.split().explode())
    data.attrs['_profanity_dictionary'] = (len(data), dictionary)
    return dictionary


def preprocess_steps(text, data):
    """Run the preprocessing pipeline and record the text after every step.

    Returns (preprocessed_text, steps) where steps is a list of
    (step name, text after that step) pairs - used by the UI to explain the pipeline.
    """
    steps = [("Original input", text)]

    # Convert text to lowercase
    text = text.lower()
    steps.append(("Lowercase", text))

    # Handle digits and symbols used as letters (leetspeak), e.g. sh1t -> shit, @ss -> ass.
    # This runs before punctuation removal, otherwise symbols like @ and $ would be deleted.
    text = normalise_leetspeak(text)
    steps.append(("Normalise leetspeak", text))

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    steps.append(("Remove punctuation", text))

    custom_dictionary = profanity_dictionary(data)

    # Tokenize the text
    tokens = word_tokenize(text)
    steps.append(("Tokenise", ' '.join(tokens)))

    # Remove stopwords
    tokens = [word for word in tokens if word not in _stopwords()]
    steps.append(("Remove stopwords", ' '.join(tokens)))

    # Lemmatization - reduce words to their base form, but leave known profanity untouched
    tokens = [word if word in custom_dictionary else _lemmatizer().lemmatize(word) for word in tokens]
    steps.append(("Lemmatise", ' '.join(tokens)))

    # Split concatenated words (e.g. stupidjerk -> stupid jerk), again skipping known profanity
    split_tokens = []
    for word in tokens:
        split_tokens.extend([word] if word in custom_dictionary else wordninja.split(word))
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(split_tokens)
    steps.append(("Split joined words", preprocessed_text))

    return preprocessed_text, steps


def preprocess_text(text, data):
    return preprocess_steps(text, data)[0]
