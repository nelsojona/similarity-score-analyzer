import tensorflow_hub as hub
import streamlit as st
import numpy as np
import logging

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = {
    "Universal Sentence Encoder": "https://tfhub.dev/google/universal-sentence-encoder/4",
    # Add more models here
}

@st.cache_resource
def load_model(model_name: str):
    """
    Loads the specified embedding model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        A TensorFlow Hub model: The loaded embedding model.
    """
    logger.info(f"Loading model: {model_name}")
    if model_name in EMBEDDING_MODELS:
        model_url = EMBEDDING_MODELS[model_name]
        return hub.load(model_url)
    else:
        logger.error(f"Model '{model_name}' not supported.")
        st.error(f"Model '{model_name}' not supported.")
        return None

def generate_embeddings(model, texts: list) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the provided model.

    Args:
        model: The loaded TensorFlow Hub model.
        texts (list): A list of text strings to generate embeddings for.

    Returns:
        numpy.ndarray: An array of embeddings for the input texts.
    """
    logger.info(f"Generating embeddings for {len(texts)} texts")
    return model(texts)
