import importlib
import unittest
from tira.third_party_integrations import ensure_pyterrier_is_loaded

cqot = importlib.import_module('corpus-graph-over-time')
cqot
class TokenizationTest(unittest.TestCase):
    def test_tokenize_query_without_duplicates_01(self):
        ensure_pyterrier_is_loaded()
        query = 'hello world'
        expected = 'world hello'

        actual = cqot.tokenise_query(query, seed=42, top_terms_per_document=50)

        self.assertEqual(expected, actual)

    def test_tokenize_query_without_duplicates_02(self):
        ensure_pyterrier_is_loaded()
        query = 'hello world'
        expected = 'hello world'

        actual = cqot.tokenise_query(query, seed=30, top_terms_per_document=50)

        self.assertEqual(expected, actual)

    def test_tokenize_query_with_duplicates_01(self):
        ensure_pyterrier_is_loaded()
        query = 'hello world hello world world a b c'
        expected = 'world hello'

        actual = cqot.tokenise_query(query, seed=31, top_terms_per_document=2)

        self.assertEqual(expected, actual)

    def test_tokenize_query_with_duplicates_02(self):
        ensure_pyterrier_is_loaded()
        query = 'hello world hello world world a b c d e f g h i j k l g'
        expected = 'world g hello'

        actual = cqot.tokenise_query(query, seed=31, top_terms_per_document=3)

        self.assertEqual(expected, actual)

    def test_tokenize_query_with_duplicates_03(self):
        ensure_pyterrier_is_loaded()
        query = 'hello world hello world world a b c d e f h i j k l g e'
        expected = 'world e hello'

        actual = cqot.tokenise_query(query, seed=31, top_terms_per_document=3)

        self.assertEqual(expected, actual)

    def test_tokenize_query_with_duplicates_04(self):
        ensure_pyterrier_is_loaded()
        query = 'hello world hello world world a l l b hello c d e f h i j k l g e'
        expected = 'l world hello'

        actual = cqot.tokenise_query(query, seed=31, top_terms_per_document=3)

        self.assertEqual(expected, actual)

    def test_tokenize_query_with_duplicates_05(self):
        ensure_pyterrier_is_loaded()
        query = 'hello world hello world world a b c'
        expected = 'hello world hello world world a b c'

        actual = cqot.tokenise_query(query, seed=31, top_terms_per_document=1)

        self.assertEqual(expected, actual)
