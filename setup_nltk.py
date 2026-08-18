"""Download required NLTK corpora for TextBlob."""

import nltk

CORPORA = ["punkt", "punkt_tab", "brown", "wordnet", "averaged_perceptron_tagger"]

for corpus in CORPORA:
    try:
        nltk.download(corpus, quiet=True)
    except Exception:
        pass

print("NLTK data ready.")
