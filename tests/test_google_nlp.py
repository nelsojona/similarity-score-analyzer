import unittest
from unittest.mock import patch
from google.cloud import language_v1
from similarity_analyzer.google_nlp import analyze_sentiment, analyze_entities

class TestGoogleNLP(unittest.TestCase):
    """
    Unit tests for the google_nlp module.
    """

    @patch('google.cloud.language_v1.LanguageServiceClient')
    def test_analyze_sentiment(self, mock_client):
        """
        Test the analyze_sentiment function with a mocked response.
        """
        mock_response = language_v1.AnalyzeSentimentResponse()
        mock_response.document_sentiment.score = 0.8
        mock_response.document_sentiment.magnitude = 0.9
        mock_client.return_value.analyze_sentiment.return_value = mock_response

        result = analyze_sentiment("This is a test.")
        self.assertEqual(result.score, 0.8)
        self.assertEqual(result.magnitude, 0.9)

    @patch('google.cloud.language_v1.LanguageServiceClient')
    def test_analyze_entities(self, mock_client):
        """
        Test the analyze_entities function with a mocked response.
        """
        mock_response = language_v1.AnalyzeEntitiesResponse()
        entity = language_v1.Entity()
        entity.name = "Google"
        entity.type_ = language_v1.Entity.Type.ORGANIZATION
        mock_response.entities.append(entity)
        mock_client.return_value.analyze_entities.return_value = mock_response

        result = analyze_entities("Google is a tech company.")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Google")
        self.assertEqual(result[0].type_, language_v1.Entity.Type.ORGANIZATION)

if __name__ == '__main__':
    unittest.main()
