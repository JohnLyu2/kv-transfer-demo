# KV transfer demo

A CPU-only educational demo of KV-cache transfer correctness. Milestone 1
implements a numerical baseline: full causal recomputation and local cached
decoding agree. Transfers, cancellation, staging pools, and formal checks remain
planned in [PLAN.md](PLAN.md).

## Run the baseline tests

The reference environment is Python 3.13.5 (also recorded in `.python-version`),
with NumPy 2.2.6. The test runner is Python's built-in `unittest`.
From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

No TPU Sync initialization, dependencies, build, model downloads, or accelerator
are required. The optional submodule is only a source reference; see the
[production mapping](docs/tpu-sync-mapping.md).

## Numerical model

`TinyAttention` uses a 16-token vocabulary, embedding dimension 8, one causal attention
head, seeded fixed weights, and a sinusoidal positional embedding table of length 32.
It uses fixed synthetic weights to exercise KV-cache correctness, with no MLP or
residual connection. Generated token IDs are not meaningful text. Computation
uses float64.

Positions are independent of the random seed. For position `p` and dimension
pair `i`, the table uses `sin(p / 10000^(2i/8))` in dimension `2i` and
`cos(p / 10000^(2i/8))` in dimension `2i+1`. Token embeddings and projection
weights remain seeded and synthetic.

The code names three dimensions separately, all defaulting to 8:
`model_dim` for embeddings, `qk_dim` for queries and keys, and `value_dim` for
values. Q and K share a width for their dot product; V may have a different width.
`Wq` and `Wk` have shape `(model_dim, qk_dim)`, `Wv` has shape
`(model_dim, value_dim)`, and the output projection has shape
`(value_dim, vocab_size)`. Attention scores are scaled by `sqrt(qk_dim)`.

- `reference(tokens)` recomputes all causal positions without a cache. Each row
  contains logits predicting the following token.
- `prefill(tokens)` returns next-token logits and token-major K/V arrays.
- `decode(token, cache)` computes only the new token's Q/K/V, appends its K/V to
  fresh arrays, and returns logits for the next token and the extended cache.

For prompts `(1, 4, 2, 8, 3, 7)` and `(9, 5, 12, 6, 10, 14)`, tests compare all
16 scores before each argmax selection and after appending each of two generated
tokens. Both paths follow the same token history, with `rtol=1e-12` and
`atol=1e-12`. Tests also check causal masking, cache independence, context limits,
and that corrupting either K or V produces a detectable score mismatch.

These are numerical regression checks, not verification of a transfer protocol
or of TPU Sync.
