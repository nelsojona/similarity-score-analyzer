from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_similarity(query_embedding: np.ndarray, section_embeddings: np.ndarray) -> list:
    """
    Calculates cosine similarity between query and section embeddings.

    Args:
        query_embedding (numpy.ndarray): The embedding of the query.
        section_embeddings (numpy.ndarray): A 2D array of embeddings for each section.

    Returns:
        list: A list of similarity scores (0-10 scale) for each section.
    """
    logger.info("Calculating similarity scores")
    similarities = cosine_similarity(query_embedding, section_embeddings)
    return [score * 10 for score in similarities[0]]
