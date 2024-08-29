from google.cloud import language_v1
from google.api_core import exceptions
from config import CONFIG
import logging
import time
import asyncio
import functools

logger = logging.getLogger(__name__)

def rate_limited(max_per_second):
    """
    Decorator to rate limit function calls.

    Args:
        max_per_second (int): Maximum number of calls per second.

    Returns:
        function: Decorated function with rate limiting.
    """
    min_interval = 1.0 / max_per_second
    def decorate(func):
        last_time_called = [0.0]
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return rate_limited_function
    return decorate

@rate_limited(10)  # 10 calls per second
@functools.lru_cache(maxsize=128)
def analyze_sentiment(text_content):
    """
    Analyzing Sentiment in a String

    Args:
        text_content: The text content to analyze
    
    Returns:
        The document sentiment, or None if an error occurred
    """
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT, language="en")
        response = client.analyze_sentiment(request={'document': document})
        return response.document_sentiment
    except exceptions.GoogleAPICallError as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return None

@rate_limited(10)  # 10 calls per second
@functools.lru_cache(maxsize=128)
def analyze_entities(text_content):
    """
    Analyzes entities in a string using Google Cloud Natural Language API.

    Args:
      text_content: The text content to analyze.
    
    Returns:
      A list of entities, or an empty list if an error occurred.
    """
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT, language="en")
        response = client.analyze_entities(request={'document': document})
        return list(response.entities)
    except exceptions.GoogleAPICallError as e:
        logger.error(f"Error in entity analysis: {e}")
        return []

async def analyze_sentiment_async(text_content):
    """
    Asynchronous wrapper for sentiment analysis.

    Args:
        text_content (str): The text content to analyze.

    Returns:
        The document sentiment, or None if an error occurred.
    """
    return await asyncio.to_thread(analyze_sentiment, text_content)

async def analyze_entities_async(text_content):
    """
    Asynchronous wrapper for entity analysis.

    Args:
        text_content (str): The text content to analyze.

    Returns:
        A list of entities, or an empty list if an error occurred.
    """
    return await asyncio.to_thread(analyze_entities, text_content)

async def analyze_all_sections(sections):
    """
    Analyzes sentiment and entities for all sections asynchronously.

    Args:
        sections (list): List of text sections to analyze.

    Returns:
        tuple: Two lists containing sentiment and entity analysis results for each section.
    """
    sentiment_tasks = [analyze_sentiment_async(section) for section in sections]
    entity_tasks = [analyze_entities_async(section) for section in sections]
    
    sentiments = await asyncio.gather(*sentiment_tasks)
    entities = await asyncio.gather(*entity_tasks)
    
    return sentiments, entities
