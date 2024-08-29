import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Preprocesses text for analysis.

    This function performs several text preprocessing steps:
    1. Converts text to lowercase
    2. Tokenizes the text
    3. Removes stopwords and punctuation
    4. Applies stemming

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text as a string of stemmed tokens.
    """
    logger.info("Preprocessing text")
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)
