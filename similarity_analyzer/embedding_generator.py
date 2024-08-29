import tensorflow_hub as hub
import streamlit as st
import numpy as np
import logging
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.api_core import retry
from config import CONFIG
import time

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = {
    "Universal Sentence Encoder": "https://tfhub.dev/google/universal-sentence-encoder/4",
    "Gemini Text Embedding": "models/text-embedding-004"
}

class GeminiModel:
    """A simple class to represent the Gemini model."""
    def __init__(self, model_name):
        self.model_name = model_name

@st.cache_resource
def load_model(model_name: str):
    """
    Loads the specified embedding model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        A TensorFlow Hub model or a GeminiModel instance.
    """
    logger.info(f"Loading model: {model_name}")
    if model_name in EMBEDDING_MODELS:
        if model_name == "Gemini Text Embedding":
            genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
            return GeminiModel(EMBEDDING_MODELS[model_name])
        else:
            model_url = EMBEDDING_MODELS[model_name]
            return hub.load(model_url)
    else:
        logger.error(f"Model '{model_name}' not supported.")
        st.error(f"Model '{model_name}' not supported.")
        return None

@retry.Retry(
    predicate=retry.if_exception_type(
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
    ),
    initial=2.0,
    maximum=60.0,
    multiplier=2,
    timeout=600.0
)
def generate_gemini_embeddings(texts: list) -> np.ndarray:
    """Generates embeddings using the Gemini API with retry logic."""
    try:
        model = EMBEDDING_MODELS["Gemini Text Embedding"]
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=model, 
                content=text,
                request_options={
                    "timeout": 30.0  # Set a reasonable timeout for each request
                }
            )
            embeddings.append(result['embedding'])
        return np.array(embeddings)
    except google_exceptions.GoogleAPICallError as e:
        logger.error(f"Error generating Gemini embeddings: {e}")
        st.error(f"An error occurred while generating embeddings: {e}")
        return None

def generate_embeddings(model, texts: list) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the provided model.

    Args:
        model: The loaded TensorFlow Hub model or GeminiModel instance.
        texts (list): A list of text strings to generate embeddings for.

    Returns:
        numpy.ndarray: An array of embeddings for the input texts.
    """
    logger.info(f"Generating embeddings for {len(texts)} texts")
    if isinstance(model, GeminiModel):
        return generate_gemini_embeddings(texts)
    else:
        return model(texts)