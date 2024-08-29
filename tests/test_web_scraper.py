import unittest
from unittest.mock import patch
from similarity_analyzer.web_scraper import scrape_webpage

class TestWebScraper(unittest.TestCase):
    """
    Unit tests for the web_scraper module.
    """

    @patch('similarity_analyzer.web_scraper.requests.get')
    def test_scrape_webpage_success(self, MockGet):
        """
        Test successful webpage scraping.
        """
        MockGet.return_value.status_code = 200
        MockGet.return_value.text = '<html><head><title>Test Page</title></head><body><h1>Header</h1><p>Test content</p></body></html>'

        result = scrape_webpage('http://example.com')
        self.assertEqual(result['title'], 'Test Page')
        self.assertEqual(result['sections'], ['Header', 'Test content'])

    @patch('similarity_analyzer.web_scraper.requests.get')
    def test_scrape_webpage_failure(self, MockGet):
        """
        Test webpage scraping failure.
        """
        MockGet.side_effect = Exception("Failed to scrape webpage")

        result = scrape_webpage('http://example.com')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()