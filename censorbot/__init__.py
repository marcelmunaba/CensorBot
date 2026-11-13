from pathlib import Path

# Path to the training dataset, independent of the folder the app is started from
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "training_words.csv"
SOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"
