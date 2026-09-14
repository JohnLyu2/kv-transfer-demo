# Demo plan: Safe staged KV-cache transfers

## Objective

Build a small, CPU-only inference demo that represents TPU Sync's core workload:
one worker produces a KV cache, a transfer manager moves it through reusable host
staging, and another worker uses it to continue decoding.

The demo will compute real attention over real numerical tensors. A small formal
model will check the protocol that coordinates transfers, readiness, cancellation,
and memory reuse. Counterexamples will be replayed against the numerical demo.

The central question is:

> When can a consumer use a transferred KV block, and when can the system reuse
> the physical storage involved in that transfer?

The central distinction is that **request cancellation, transfer completion,
destination readiness, and storage reusability are separate facts**.

This is an educational verification artifact inspired by TPU Sync. It does not
prove the production implementation correct or reproduce TPU performance.

## Relationship to TPU Sync

The optional source reference is `third_party/tpu-sync`, pinned to upstream main
commit `6852486862ad644c2e9febb26d0f5e75be13c981` (reviewed 2026-09-09).
See [the implementation mapping](docs/tpu-sync-mapping.md) for the reviewed
correspondence and its limits. The demo and formal checks must run without
initializing or building this submodule.

The demo models one staged prefill-to-decode transfer route. It abstracts hardware
and networking while preserving the dependencies that make the route difficult.

The CPU demo and formal checks must not import TPU Sync's code or install its
dependencies. See the [production mapping and submodule workflow](docs/tpu-sync-mapping.md)
for planned correspondence, deliberate abstractions, and revision maintenance.
The source links below require initializing the optional submodule.

| TPU Sync concept | Demo representation | Code reference |
| --- | --- | --- |
| Producer and consumer workers | Separate Python objects and arrays | [Example serving topology](third_party/tpu-sync/examples/single_host_disagg/README.md) |
| Host staging and device destination | Separate finite memory pools | [Receive state](third_party/tpu-sync/tpu_sync/core/kv_cache_manager_with_transfer.h) |
| Separate network and H2D completion | Separate scheduled transfer stages | [Receive state](third_party/tpu-sync/tpu_sync/core/kv_cache_manager_with_transfer.h) |
| Pins and release after use | Explicit reservations for outstanding accesses | [Cache-store contract](third_party/tpu-sync/tpu_sync/kv_cache/kv_cache_store.h) |
| Request/plan generations | Transfer identity and slot allocation generation | [Request registry](third_party/tpu-sync/tpu_sync/kv_cache/reshard/request_block_registry.h), [plan generations](third_party/tpu-sync/tpu_sync/core/kv_cache_manager_with_transfer.h) |
| Publication after successful load | Destination readiness transition | [Backend completion callbacks](third_party/tpu-sync/tpu_sync/kv_cache/host_offload_backend.cc) |
| Recorded operation outcomes | Client polling separate from physical completion | [BlockTracker](third_party/tpu-sync/tpu_sync/kv_cache/block_tracker.cc) |

Reservations in the demo are an abstract protocol mechanism. They must not be
equated with PJRT usage holds, raw-buffer references, or cache pins individually.
The real system distributes those responsibilities across several layers.

In this revision, backend callbacks perform load/save completion bookkeeping and
`KVCacheStore` delegates status polling to `BlockTracker`. Remote reads use the
regular load path. The demo should represent that separation without reproducing
the production class hierarchy or adding more features.

## Scope

### Included

- A tiny, fixed-weight autoregressive attention model implemented with NumPy.
- Actual prefill, KV tensors, and incremental decoding.
- Two requests with different prompts.
- Block-based allocation and a small reusable staging pool.
- Two asynchronous copy stages: producer to staging, then staging to destination.
- Explicit destination publication, cancellation, and delayed completion.
- A deterministic event scheduler and reproducible traces.
- A TLA+ model checked with TLC in a bounded configuration.
- Deliberately faulty protocol variants and numerical regression checks.

### Excluded from the first version

- Real TPUs, GPUs, networking, RPC, multiple processes, and threads.
- A trained LLM, meaningful text generation, training, and model downloads.
- Prefix sharing, resharding, weight updates, retries, and crash recovery.
- Hardware memory-ordering proofs and production C++ verification.
- Throughput claims or performance comparisons derived from simulated timing.
- A general-purpose verification framework or custom model checker.

## Executable workload

### Tiny attention model

Start with a vocabulary of 16 token IDs, embedding dimension 8, one attention
head, and one causal attention layer. Use sinusoidal positional embeddings and seeded
weights. Prompts should contain about six tokens, with a block size of two tokens.
Generate two additional tokens per successful request using deterministic argmax.

This attention model uses fixed synthetic weights and an output projection.
Its role is to provide a real computation that depends on correct KV data;
generated token IDs are not intended to represent meaningful text.

Implement two paths:

1. **Reference:** recompute causal attention over the complete token sequence at
   each step, without a cache or transfer manager.
2. **Cached:** compute the prompt's K/V tensors, transfer them, and append the new
   token's K/V entries during incremental decoding.

Compare next-token score vectors within an explicit numerical tolerance. Compare
the same token sequence in both paths before allowing generation to advance;
otherwise one wrong token can obscure the original numerical mismatch. Token
equality alone is too weak, because corrupted scores may retain the same argmax.

### Memory and identity

Use separate NumPy arrays for producer data, staging, and destination data. Ensure
the arrays do not accidentally alias. Begin with two reusable staging slots and
enough destination blocks for both short requests and their generated tokens.
Staging scarcity provides contention without introducing an eviction policy.

Each request has a block table mapping logical token-block indices to physical
destination slots. Exercise noncontiguous, nonmonotonic allocations and transfer
blocks out of logical order. Gather ready blocks through that table in token
order for the numerical attention model; this implements paged storage semantics,
not a production PagedAttention kernel. Track the valid token count to exclude
padding in partial blocks and allocate a new destination block as decoding grows.

Identify logical data by `(cache_key, block_index, piece_index)`, independently
of the request using it. Identify physical storage by `(pool, slot, generation)`.
The demo assumes correct cache-key assignment and one fixed model configuration.

Represent each transferred block as two abstract pieces, for example its two
token rows across K and V. Each copy-piece event performs an actual array copy.
The formal model represents those rows using provenance labels instead of floats.

### Transfer stages and scheduling

```text
Producer K/V arrays -> host staging pool -> decode K/V arrays -> attention
                       stage 1             stage 2
```

A single-threaded event loop selects enabled events. Operations may remain
pending while another request advances. Use explicit events rather than sleeps:

- reserve staging and destination slots;
- copy one producer-to-staging piece;
- complete producer-to-staging transfer;
- start and advance staging-to-destination transfer;
- record physical transfer completion;
- deliver a completion notification;
- publish a ready destination block and record its operation outcome;
- poll recorded outcomes without advancing physical transfers;
- consume ready data;
- cancel a request;
- release reservations and reuse slots.

Physical completion and notification delivery are distinct so a notification may
arrive after cancellation or slot reuse. A slot may only be reused when no old
operation can access it. Generation checks reject stale bookkeeping updates;
they do not prevent an already-issued physical write. Completion can occur
without a client poll. Polling observes accumulated outcomes; it must not be what
makes a copy finish or makes its destination physically valid. A completion record
is not itself a reservation keeping memory alive.

For simplicity, cancellation does not abort a submitted copy. Submitted work
drains to physical completion, its unwanted result is not published, and its
reservations are then released. No new stages are started for a cancelled request.

## Formal specification

### Abstract behavior

A cache block is either unavailable at the destination or ready with its complete
expected contents. A successful consumer read of block K returns K's contents.
An unavailable block causes waiting or a miss, never a partial read.

Publication is per block. Decoding a context requires every block needed by that
context to be ready. Do not claim an atomic update of an entire multi-block cache.

### Concrete state

Track request interest, transfer stage/progress, cache-block identity, slot
allocation generation, source/destination reservations, delivered completion
notifications, recorded client-visible outcomes, and destination readiness separately.

The model's initial finite configuration should use two cache identities, two
requests, two pieces per block, and at most two concurrent transfers. Use one
logical block per request initially; add a two-block context configuration to
check the decode readiness gate. Keep array sizes and model bounds independent:
the executable workload can use larger tensors than the checker.

### Properties

| ID | Property |
| --- | --- |
| S1 | Every successful consumer read returns the expected complete block. |
| S2 | Destination readiness requires successful completion of all required pieces at that destination. |
| S3 | A source remains unchanged while a transfer may read it; destination writes exclude conflicting accesses. |
| S4 | A slot cannot be reassigned while an outstanding operation may access it. |
| S5 | A completion notification cannot publish or release a different transfer or allocation generation. |
| S6 | Cancellation does not grant permission to reuse storage still in use, and prevents subsequent publication for that cancelled operation. |
| S7 | Polling observes recorded outcomes without advancing a physical transfer or bypassing readiness and lifetime rules. |

Assume each modeled piece copy faithfully transfers its source value, memory
regions are disjoint as declared, and consumers follow the readiness interface.
Do not assume whole-block copies are atomic or that cancellation stops hardware.

Initially check safety invariants with TLC. Document configuration bounds and
assumptions alongside results. Optionally check eventual completion or safe
retirement later, under explicit fairness and eventual-transfer-completion
assumptions. A time limit or incomplete state-space search is not a passing check.

The abstract specification guides the invariants. A machine-checked refinement
theorem is a possible extension, not a first-version deliverable.

## Demonstrations

### A. Arrived does not mean ready

Faulty variant: publish the device block when producer-to-staging reception
finishes, before staging-to-device copying completes.

The checker should find a consumer reading an incomplete destination. Replay the
trace against the numerical workload and show its score mismatch against the
reference. Use deterministic inputs that expose the defect.

### B. Cancelled does not mean reusable

Faulty variant: cancellation immediately releases a staging slot even though a
stage-2 transfer is still reading it.

Replay a trace in which request B overwrites request A's staging source. Show
corrupted destination provenance or a failed identity check for the cancelled
transfer. Do not force a cancelled request to decode merely to obtain a numerical
mismatch: the direct witness is the conflicting access or corrupted transfer.
An optional extended scenario can demonstrate corruption reaching a live consumer
if another faulty rule also permits premature destination reuse.

### C. Correct staged protocol

With readiness and lifetime rules enabled, the bad transitions are unavailable.
Show a successful two-request execution, including cancellation and safe reuse,
and demonstrate that unrelated work proceeds while transfers remain pending.

All faulty variants are intentional demo mutations, not allegations about bugs
in TPU Sync. Keep mutation switches isolated from the correct protocol.

## Connection between model and executable

Define a small versioned trace format containing event names, transfer IDs, cache
identities, slots, and generations. Map every relevant TLA+ action to an executable
scheduler event. Replays must reject an event whose precondition is not satisfied;
they must not silently skip or repair it.

The mapping should distinguish model-only bookkeeping from actual array copies.
Model traces use symbolic provenance; the executable records corresponding data
movement and checks actual tensor contents and numerical outputs.

Replay and state comparison provide evidence that the two artifacts agree on
tested executions. They are not a proof of complete implementation refinement.

## Proposed files

```text
kv-transfer-demo/
  README.md
  PLAN.md
  pyproject.toml
  src/kv_transfer_demo/
    attention.py            # Tiny model and recomputation reference
    protocol.py             # Pools, reservations, operations, outcomes
    scheduler.py            # Deterministic event scheduling
    demo.py                 # CLI scenarios and trace replay
  tests/                    # Numerical and protocol regression checks
  formal/
    KVTransfer.tla
    small.cfg
    context.cfg
    assumptions.md
  traces/                   # Small counterexample/replay examples
  tooling/
    check_model.sh          # Pinned TLC runner with explicit failure handling
    export_trace.py         # Converts checker output to replay format
  docs/
    tpu-sync-mapping.md      # Reviewed source mapping and abstractions
  third_party/
    tpu-sync/               # Optional, pinned source-reference submodule
```

Use NumPy plus a lightweight Python test runner for the workload, and Java/TLC
for verification. The baseline pins Python in `.python-version` and NumPy in
`pyproject.toml`; pin TLC when formal implementation begins. The TPU Sync
reference is pinned separately; changing it requires a mapping review. Keep generated
checker state, downloaded tools, and bulky logs out of Git. Start with a terminal
trace/table; a visual trace viewer is optional after the full workflow works.

## Implementation milestones

1. **Numerical baseline (implemented):** full recomputation and local cached
   decoding agree. See [setup and tests](README.md) for the regression tests,
   including both prompts through two generated tokens at explicit tolerances.
2. **Real staged transfers (implemented):** block copies through separate pools
   preserve scores; two requests exercise scattered block tables, staging reuse,
   delayed notifications, and destination growth. The CPU-only demo and protocol
   regression tests run through the [documented commands](README.md).
3. **Protocol specification:** write the action mapping, assumptions, TLA+ model,
   and bounded configurations; verify the correct variant.
4. **Counterexamples:** weaken publication and cancellation cleanup separately;
   obtain checker failures and replay them against the executable.
5. **Reproducible demo:** document setup, commands, expected outputs, bounds,
   limitations, and correspondence to the production repo.

At the end of milestone 2, confirm that the demo remains small and CPU-only. At
milestone 3, validate non-vacuity: the model must reach successful transfers,
reads, cancellation, and slot reuse, rather than passing because useful work is
disabled. Do not add more production features before counterexample replay works.

Current implementation boundary: staging and destination reservations are
exclusive; producer snapshots remain immutable. Events distinguish physical
completion, notification, publication, and outcome polling. Cancellation and safe
retirement are implemented, including tests at every transfer-event boundary and
a runnable cancellation/reuse scenario. `complete_staging` and `start_destination`
are separate events; cancelled producer stages drain without starting the latter.
Cancelled notifications can arrive after retirement and slot reuse, reporting
historical outcomes without accessing storage. Intentional faulty variants,
TLC checking, and model-to-executable trace correspondence remain future work.

## Completion criteria

- One documented command runs a successful numerical workload on a laptop.
- Cached decode scores agree with the full-recomputation reference.
- A documented checker command completely explores each stated finite configuration
  and reports the checked invariants.
- Each intentional mutation produces its expected invariant violation.
- Counterexample traces replay deterministically against the executable, with a
  concrete witness of incomplete data, conflicting access, or incorrect contents.
- The correct protocol demonstrates both useful overlap and eventual safe reuse
  in executable scenarios.
- Documentation distinguishes bounded model results, numerical tests, assumptions,
  and unverified production/hardware behavior.

## Research grounding

The ownership mechanism is an application of established ideas, not a proposed
new logic. [Safe Asynchronous Multicore Memory Operations (ASE 2011)](https://www.doc.ic.ac.uk/~afd/papers/2011/ASE.pdf)
provides the closest precedent for permissions committed to pending copies.
[PagedAttention (SOSP 2023)](https://arxiv.org/pdf/2309.06180) provides the KV-block
management context. [TLA+ and TLC](https://lamport.azurewebsites.net/tla/tools.html)
provide the model-checking infrastructure.

The project's contribution is a compact, executable explanation of staged
KV-cache transfer correctness: real inference results, a precise protocol,
machine-discovered counterexamples, and a clear boundary around what was checked.
