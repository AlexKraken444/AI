import unittest
from training.prepare_language import document_key, document_split


class LanguageSplitTests(unittest.TestCase):
    def test_duplicate_whitespace_and_case_stay_in_same_split(self):
        a = document_key("Текст статьи\nдля обучения")
        b = document_key(" текст СТАТЬИ   для обучения ")
        self.assertEqual(a, b)
        self.assertEqual(document_split(a), document_split(b))

    def test_split_is_stable_independent_of_document_order(self):
        keys = [document_key(f"Статья {i}") for i in range(1000)]
        first = {k: document_split(k) for k in keys}
        second = {k: document_split(k) for k in reversed(keys)}
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
