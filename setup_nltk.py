"""Download required NLTK corpora into the project for serverless deploys."""

from pathlib import Path

import nltk

NLTK_DIR = Path(__file__).parent / "nltk_data"
NLTK_DIR.mkdir(exist_ok=True)

CORPORA = ["punkt", "punkt_tab", "brown", "wordnet", "averaged_perceptron_tagger"]

for corpus in CORPORA:
    try:
        nltk.download(corpus, download_dir=str(NLTK_DIR), quiet=True)
    except Exception:
        pass

print(f"NLTK data ready at {NLTK_DIR}")
