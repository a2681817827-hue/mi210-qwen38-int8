import math
import unittest

from fidelity_compare import coarsened_kl, compare


class FidelityComparisonTests(unittest.TestCase):
    def test_nonshared_top_tokens_belong_in_residual_mass(self):
        left = {'token_id:1': math.log(0.6), 'token_id:2': math.log(0.3)}
        right = {'token_id:1': math.log(0.5), 'token_id:3': math.log(0.4)}
        kl, support = coarsened_kl(left, right)
        self.assertAlmostEqual(kl, 0.6 * math.log(0.6 / 0.5) + 0.4 * math.log(0.4 / 0.5))
        self.assertEqual(support, 1)

    def test_diverged_history_is_not_compared(self):
        row = {'index': 0, 'prompt_sha256': 'same', 'tokens': [1, 2, 3],
               'top_logprobs': [{'token_id:1': math.log(0.9)}] * 3}
        reference = {'corpus_sha256': 'same', 'records': [row], 'image_id': 'ref'}
        candidate = {'corpus_sha256': 'same', 'records': [dict(row, tokens=[1, 9, 8])], 'image_id': 'var'}
        result = compare(reference, candidate)
        self.assertEqual(result['comparable_positions'], 2)
        self.assertEqual(result['exact_greedy_token_sequences'], 0)

    def test_missing_logprobs_and_changed_prompts_fail(self):
        row = {'index': 0, 'prompt_sha256': 'same', 'tokens': [1], 'top_logprobs': []}
        capture = {'corpus_sha256': 'same', 'records': [row], 'image_id': 'ref'}
        with self.assertRaises(ValueError):
            compare(capture, capture)
        with self.assertRaises(ValueError):
            compare(capture, dict(capture, corpus_sha256='different'))

    def test_nonfinite_model_probabilities_cannot_pass(self):
        for value in (float('nan'), float('inf'), float('-inf'), 0.5):
            with self.assertRaises(ValueError):
                coarsened_kl({'token_id:1': value}, {'token_id:1': -0.1})


if __name__ == '__main__':
    unittest.main()
