import unittest
from similarity_analyzer.text_preprocessor import preprocess_text

class TestTextPreprocessor(unittest.TestCase):
    """
    Unit tests for the text_preprocessor module.
    """

    def test_preprocess_text(self):
        """
        Test the preprocess_text function with a sample input.
        """
        input_text = "This is a Test sentence. It has punctuation and Stopwords!"
        expected_output = "test sentenc punctuat stopword"
        self.assertEqual(preprocess_text(input_text), expected_output)

if __name__ == '__main__':
    unittest.main()
