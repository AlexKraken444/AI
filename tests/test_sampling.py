import unittest
import numpy as np
from neural.transformer import select_token


class SamplingTests(unittest.TestCase):
    def test_greedy_and_model_probability(self):
        token, probability = select_token(np.log([.1, .7, .2]), np.random.default_rng(1))
        self.assertEqual(token, 1)
        self.assertAlmostEqual(probability, .7)

    def test_nucleus_includes_threshold_crossing_token(self):
        rng = np.random.default_rng(5)
        drawn = [select_token(np.log([.6, .3, .1]), rng, 1., .8)[0] for _ in range(2000)]
        self.assertEqual(set(drawn), {0, 1})
        self.assertAlmostEqual(drawn.count(0) / len(drawn), 2 / 3, delta=.04)

    def test_seed_reproduces_sequence_and_rejects_invalid_settings(self):
        def draw():
            rng = np.random.default_rng(13)
            return [select_token(np.zeros(10), rng, .8, 1)[0] for _ in range(30)]
        self.assertEqual(draw(), draw())
        for temperature, top_p in [(-1, .9), (float("nan"), .9), (1, 0), (1, 2)]:
            with self.assertRaises(ValueError):
                select_token(np.ones(5), np.random.default_rng(), temperature, top_p)
