import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from kv_transfer_demo.attention import ATOL, PROMPTS, RTOL, KVCache, TinyAttention


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyAttention()

    def test_two_prompts_generate_two_tokens_with_matching_scores(self):
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt):
                tokens = list(prompt)
                scores, cache = self.model.prefill(tokens)
                for step in range(3):
                    # Compare the complete vectors before advancing either path.
                    expected = self.model.reference(tokens)[-1]
                    self.assertEqual(scores.shape, (16,))
                    self.assertTrue(np.isfinite(scores).all())
                    assert_allclose(scores, expected, rtol=RTOL, atol=ATOL)
                    _, rebuilt = self.model.prefill(tokens)
                    assert_allclose(cache.keys, rebuilt.keys, rtol=RTOL, atol=ATOL)
                    assert_allclose(cache.values, rebuilt.values, rtol=RTOL, atol=ATOL)
                    if step < 2:
                        token = int(np.argmax(expected))
                        self.assertEqual(int(np.argmax(scores)), token)
                        tokens.append(token)
                        scores, cache = self.model.decode(token, cache)
                self.assertEqual(len(cache.keys), len(prompt) + 2)

    def test_reference_is_causal_and_cached_prefixes_match(self):
        tokens = list(PROMPTS[0])
        full = self.model.reference(tokens)
        for length in range(1, len(tokens) + 1):
            prefix = self.model.reference(tokens[:length])
            assert_allclose(full[:length], prefix, rtol=RTOL, atol=ATOL)
        scores, cache = self.model.prefill(tokens[:1])
        assert_allclose(scores, full[0], rtol=RTOL, atol=ATOL)
        for i, token in enumerate(tokens[1:], start=1):
            scores, cache = self.model.decode(token, cache)
            assert_allclose(scores, full[i], rtol=RTOL, atol=ATOL)

    def test_decode_preserves_input_and_owns_its_arrays(self):
        _, cache = self.model.prefill(PROMPTS[0])
        old_keys, old_values = cache.keys.copy(), cache.values.copy()
        _, updated = self.model.decode(4, cache)
        assert_array_equal(cache.keys, old_keys)
        assert_array_equal(cache.values, old_values)
        arrays = (cache.keys, cache.values, updated.keys, updated.values)
        for i, left in enumerate(arrays):
            for right in arrays[i + 1:]:
                self.assertFalse(np.shares_memory(left, right))

    def test_corrupting_either_cached_tensor_changes_scores(self):
        prompt = list(PROMPTS[0])
        _, cache = self.model.prefill(prompt)
        expected = self.model.reference(prompt + [4])[-1]
        for field in ("keys", "values"):
            with self.subTest(field=field):
                corrupted = KVCache(cache.keys.copy(), cache.values.copy())
                getattr(corrupted, field)[0] += 10.0
                actual, _ = self.model.decode(4, corrupted)
                self.assertFalse(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))

    def test_seed_reproduces_weights_and_scores(self):
        assert_array_equal(self.model.reference(PROMPTS[0]),
                           TinyAttention().reference(PROMPTS[0]))
        # Positions have no random component, even when model weights change.
        assert_array_equal(self.model.position, TinyAttention(seed=123).position)
        assert_array_equal(self.model.position[0], [0, 1, 0, 1, 0, 1, 0, 1])

    def test_distinct_embedding_query_key_and_value_dimensions(self):
        class DifferentDimensions(TinyAttention):
            model_dim = 6
            qk_dim = 4
            value_dim = 3

        model = DifferentDimensions()
        tokens = list(PROMPTS[0])
        scores, cache = model.prefill(tokens)
        self.assertEqual(cache.keys.shape, (6, 4))
        self.assertEqual(cache.values.shape, (6, 3))
        assert_allclose(scores, model.reference(tokens)[-1], rtol=RTOL, atol=ATOL)
        for token in (2, 5):
            tokens.append(token)
            scores, cache = model.decode(token, cache)
            assert_allclose(scores, model.reference(tokens)[-1], rtol=RTOL, atol=ATOL)

    def test_invalid_inputs_and_context_limit(self):
        for tokens in ([], [16], [-1], [1.5], [[1]], [True], [0] * 33):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.model.prefill(tokens)
        _, cache = self.model.prefill([1] * 31)
        scores, full = self.model.decode(2, cache)
        assert_allclose(scores, self.model.reference([1] * 31 + [2])[-1],
                        rtol=RTOL, atol=ATOL)
        with self.assertRaises(ValueError):
            self.model.decode(3, full)
        with self.assertRaises(ValueError):
            self.model.decode(3, KVCache(np.zeros((2, 7)), np.zeros((2, 8))))


if __name__ == "__main__":
    unittest.main()
