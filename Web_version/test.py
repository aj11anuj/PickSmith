# test.py
import unittest
from preprocessing import load_data, compute_tfidf
from rec import recommend_movies

class TestRecommendation(unittest.TestCase):
    def setUp(self):
        self.df = load_data("movies.csv")
        self.tfidf_matrix = compute_tfidf(self.df, 'Description')
    
    def test_recommendation(self):
        result = recommend_movies("The Godfather", self.df, self.tfidf_matrix)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(movie in self.df['Title'].values for movie in result))

if __name__ == "__main__":
    unittest.main()