
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

# Preserve negation because it can change a sentence's meaning.
STOP_WORDS = STOP_WORDS - {"not", "no", "nor"}

def preprocess_text(text: str) -> list[str]:
    """Clean and tokenize one query/document into Word2Vec tokens."""
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = wordpunct_tokenize(text)
    return [
        token for token in tokens
        if token.isalpha() and token not in STOP_WORDS
    ]