# KV transfer demo

A CPU-only educational demo of KV-cache transfer correctness. The numerical
baseline and staged transfers are implemented: two requests transfer actual K/V
through reusable staging into paged destination storage, then decode with scores
matching full causal recomputation. Cancellation, formal checks, and checker trace
replay remain planned in [PLAN.md](PLAN.md).

## Run the demo and tests

The reference environment is Python 3.13.5 (also recorded in `.python-version`),
with NumPy 2.2.6. The test runner is Python's built-in `unittest`.
From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m kv_transfer_demo.demo
.venv/bin/python -m unittest discover -s tests -v
```

No TPU Sync initialization, dependencies, build, model downloads, or accelerator
are required. The optional submodule is only a source reference; see the
[production mapping](docs/tpu-sync-mapping.md).

Expected demo output (the final floating-point digits can vary by platform):

```text
A: logical blocks -> physical slots [5, 1, 7, 3]; generated [4, 4]
B: logical blocks -> physical slots [0, 6, 2, 4]; generated [4, 4]
PASS: 6 blocks transferred; 60 events; max score error 8.882e-16
Both staging slots are free; destination blocks remain owned until request retirement.
```

Add `--trace` to print each event, transfer ID, copy-piece index, and staging and
destination slot generations. An invalid event raises `ProtocolError` and stops
the schedule; it is never silently skipped.

## Staged transfers and block tables

`protocol.py` implements separate producer snapshots, a two-slot staging pool,
and an eight-slot destination pool. A block holds two tokens, with separate key
and value widths. The demo gives A physical slots `[5, 1, 7]` and B `[0, 6, 2]`,
then transfers logical blocks in order `[2, 0, 1]`. Both requests have copies in
flight together. A single thread advances explicitly selected events; no sleeps,
network calls, or background transfers are involved.

For each block, the scheduler executes:

```text
reserve staging and destination
  -> copy each producer K/V row to staging
  -> complete_staging
  -> copy each staging K/V row to destination
  -> complete_destination
```

Exclusive reservations protect staging and destination until physical completion
is recorded, even after the last row-copy event. Producer snapshots remain
read-only for the request's lifetime. After completion, `release_staging` permits
reuse; `notify` records a completion outcome; `publish` makes the destination
block readable after notification. Staging release can occur before notification:
the demo deliberately reassigns staging slots before delivering old notifications.
Generation checks reject stale physical references, and notifications never
release staging on behalf of a newer transfer.

`poll()` consumes outcomes and changes no copy, readiness, or reservation state.
`gather()` refuses an incomplete context, then follows the request's block table
in logical order and returns independent contiguous arrays for attention. This
models paged KV storage, not the production PagedAttention kernel. Padding in a
partial final block is excluded using the request's token count.

Each generated token's K/V is appended to destination storage: a partial block is
filled first, then a new block is allocated when necessary. Scores are checked
from freshly gathered destination data before generation and after both decode
steps. `release_request()` frees a fully published, retired request's destination
blocks; it cannot cancel an in-flight request. Request and transfer IDs are unique
for one manager lifetime. Access rules assume callers use the checked methods;
the backing arrays are exposed for inspection, not concurrent external mutation.

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
- `cached_scores(last_token, cache)` scores an existing, possibly transferred
  context without rebuilding or appending K/V. The caller supplies the actual
  final token of that context.

For prompts `(1, 4, 2, 8, 3, 7)` and `(9, 5, 12, 6, 10, 14)`, tests compare all
16 scores before each argmax selection and after appending each of two generated
tokens. Both paths follow the same token history, with `rtol=1e-12` and
`atol=1e-12`. Tests also check causal masking, cache independence, context limits,
and that corrupting either K or V produces a detectable score mismatch.

Protocol tests also cover early access, conflicting reservations, pool exhaustion,
stale generations, delayed notifications, partial blocks, and deterministic event
order. These are executable regression checks; no formal proof or verification of
TPU Sync is claimed. A versioned TLA+ trace importer remains future work.
