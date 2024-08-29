import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def scrape_webpage(url: str) -> dict:
    """
    Scrapes a webpage using requests and BeautifulSoup and extracts all text content.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        dict: A dictionary containing the page title and a list of text sections,
              or None if an error occurs.

    This function uses the requests library to load a webpage and BeautifulSoup to parse its content.
    It extracts text from common HTML elements such as paragraphs, headings, and list items.
    """
    try:
        logger.info(f"Scraping webpage: {url}")
        
        # Send a GET request to the specified URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)

        # Parse the webpage content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title
        title = soup.title.string if soup.title else 'No Title'

        # Extract text from paragraphs, headings, and list items
        sections = []
        elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'section', 'article', 'li'])
        for element in elements:
            text = element.get_text(strip=True)
            if text:
                sections.append(text)

        # Return the extracted data
        result = {
            'title': title,
            'sections': sections
        }

        if sections:
            return result
        else:
            logger.warning(f"No content found on {url}.")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while fetching the webpage: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while scraping {url}: {type(e).__name__} - {e}")
        return None