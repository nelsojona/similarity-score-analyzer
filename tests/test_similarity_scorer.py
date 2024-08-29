import unittest
import numpy as np
from similarity_analyzer.similarity_scorer import calculate_similarity

class TestSimilarityScorer(unittest.TestCase):
    """
    Unit tests for the similarity_scorer module.
    """

    def test_calculate_similarity(self):
        """
        Test the calculate_similarity function with sample embeddings.
        """
        query_embedding = np.array([[1.0, 0.0, 0.0]])
        section_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scores = calculate_similarity(query_embedding, section_embeddings)
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 10.0)
        self.assertAlmostEqual(scores[1], 0.0)

if __name__ == '__main__':
    unittest.main()
