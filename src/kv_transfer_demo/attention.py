"""One fixed-weight attention head with independent reference and cached paths.

Scores are logits over token IDs, not probabilities or meaningful language.
All arithmetic uses float64. No transfer protocol is implemented here.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
PROMPTS = ((1, 4, 2, 8, 3, 7), (9, 5, 12, 6, 10, 14))
RTOL = 1e-12
ATOL = 1e-12


@dataclass(frozen=True)
class KVCache:
    """Token-major keys (length, qk_dim) and values (length, value_dim).

    Arrays remain mutable so later transfer scenarios can copy or corrupt data.
    Decode returns fresh arrays and never modifies the supplied cache.
    The caller must use a cache from the same model and token history.
    """

    keys: FloatArray
    values: FloatArray


def _softmax(scores: FloatArray) -> FloatArray:
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


class TinyAttention:
    """Causal attention with fixed synthetic weights.

    Computation: embedding + position -> Q/K/V -> logits.
    There is no residual, MLP, normalization layer, or multi-head projection.
    The sinusoidal position table limits contexts to 32 tokens.
    """

    vocab_size = 16
    model_dim = 8  # Token and positional embedding width.
    qk_dim = 8     # Shared query/key width, required by their dot product.
    value_dim = 8  # Value width; independent of query/key width.
    max_context = 32

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.embedding = rng.normal(0, 0.5, (self.vocab_size, self.model_dim))
        # PE(pos, 2i) = sin(pos / 10000**(2i/d)); odd dimensions use cosine.
        positions = np.arange(self.max_context, dtype=np.float64)[:, None]
        frequencies = 10000.0 ** (-np.arange(0, self.model_dim, 2) / self.model_dim)
        angles = positions * frequencies
        self.position = np.empty((self.max_context, self.model_dim), dtype=np.float64)
        self.position[:, 0::2] = np.sin(angles)
        self.position[:, 1::2] = np.cos(angles)
        scale = self.model_dim ** -0.5
        self.wq = rng.normal(0, scale, (self.model_dim, self.qk_dim))
        self.wk = rng.normal(0, scale, (self.model_dim, self.qk_dim))
        self.wv = rng.normal(0, scale, (self.model_dim, self.value_dim))
        self.output = rng.normal(0, self.value_dim ** -0.5,
                                 (self.value_dim, self.vocab_size))
        for weight in (self.embedding, self.position, self.wq, self.wk,
                       self.wv, self.output):
            weight.flags.writeable = False

    def _tokens(self, tokens: ArrayLike) -> NDArray[np.int64]:
        ids = np.asarray(tokens)
        if ids.ndim != 1 or not 1 <= ids.size <= self.max_context:
            raise ValueError(f"Expected 1 to {self.max_context} token IDs")
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError(f"Token IDs must be integers in [0, {self.vocab_size})")
        return ids.astype(np.int64)

    def reference(self, tokens: ArrayLike) -> FloatArray:
        """Compute next-token logits at every position without using a KV cache.

        Returns an array of shape (len(tokens), vocab_size).
        Row i contains next-token logits conditioned on tokens[:i + 1].
        """
        ids = self._tokens(tokens)
        x = self.embedding[ids] + self.position[:len(ids)]
        queries, keys, values = x @ self.wq, x @ self.wk, x @ self.wv
        scores = queries @ keys.T / np.sqrt(self.qk_dim)
        future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
        return (_softmax(scores) @ values) @ self.output

    def _next_scores(self, query: FloatArray, cache: KVCache) -> FloatArray:
        """Compute next-token logits for the final position of a cached context."""
        scores = cache.keys @ query / np.sqrt(self.qk_dim)
        return (_softmax(scores) @ cache.values) @ self.output

    def prefill(self, tokens: ArrayLike) -> tuple[FloatArray, KVCache]:
        """Build prompt K/V and return the next-token logits and cache."""
        ids = self._tokens(tokens)
        x = self.embedding[ids] + self.position[:len(ids)]
        cache = KVCache(keys=x @ self.wk, values=x @ self.wv)
        return self._next_scores(x[-1] @ self.wq, cache), cache

    def _cache_length(self, cache: KVCache) -> int:
        if (cache.keys.ndim != 2 or cache.values.ndim != 2
                or len(cache.keys) != len(cache.values)
                or cache.keys.shape[1] != self.qk_dim
                or cache.values.shape[1] != self.value_dim):
            raise ValueError("Expected keys (length, qk_dim) and values (length, value_dim)")
        length = len(cache.keys)
        if not 1 <= length <= self.max_context:
            raise ValueError("Cache length is outside the model context bounds")
        return length

    def cached_scores(self, last_token: int, cache: KVCache) -> FloatArray:
        """Score an existing context using its K/V, without appending a token.

        last_token must be the token at the final position of this cache.
        This lets a consumer score transferred data without rebuilding prompt K/V.
        """
        ids = self._tokens([last_token])
        length = self._cache_length(cache)
        x = self.embedding[ids[0]] + self.position[length - 1]
        return self._next_scores(x @ self.wq, cache)

    def decode(self, token: int, cache: KVCache) -> tuple[FloatArray, KVCache]:
        """Append one token at position len(cache), returning subsequent logits.

        Only the new token's Q/K/V are computed; previous K/V come from the cache.
        """
        ids = self._tokens([token])
        length = self._cache_length(cache)
        if length == self.max_context:
            raise ValueError("Cache must be nonempty and have room for one token")
        x = self.embedding[ids[0]] + self.position[length]
        updated = KVCache(
            keys=np.concatenate((cache.keys, (x @ self.wk)[None, :]), axis=0),
            values=np.concatenate((cache.values, (x @ self.wv)[None, :]), axis=0),
        )
        return self._next_scores(x @ self.wq, updated), updated
