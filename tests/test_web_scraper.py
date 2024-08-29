import unittest
from unittest.mock import patch, AsyncMock
from similarity_analyzer.web_scraper import scrape_webpage

class TestWebScraper(unittest.TestCase):
    """
    Unit tests for the web_scraper module.
    """

    @patch('similarity_analyzer.web_scraper.PlaywrightCrawler')
    async def test_scrape_webpage_success(self, MockCrawler):
        """
        Test successful webpage scraping.
        """
        mock_crawler = AsyncMock()
        mock_crawler.default_dataset.get_items.return_value = [{'title': 'Test Page', 'sections': ['Test content']}]
        MockCrawler.return_value.__aenter__.return_value = mock_crawler

        result = await scrape_webpage('http://example.com')
        self.assertEqual(result['title'], 'Test Page')
        self.assertEqual(result['sections'], ['Test content'])

    @patch('similarity_analyzer.web_scraper.PlaywrightCrawler')
    async def test_scrape_webpage_failure(self, MockCrawler):
        """
        Test webpage scraping failure.
        """
        mock_crawler = AsyncMock()
        mock_crawler.default_dataset.get_items.return_value = []
        MockCrawler.return_value.__aenter__.return_value = mock_crawler

        result = await scrape_webpage('http://example.com')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
