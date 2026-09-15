# Plan: Formal verification of KV-cache transfers

## Objective

Build a formal verification demo of a simplified KV-cache transfer protocol
inspired by TPU Sync. Check correct consumer reads, safe memory reuse, and
progress under concurrent transfers and cancellation. Develop a simulator to
replay selected model executions in code.

The central question is:

> When can a consumer use transferred KV data, and when can its storage be reused?

The central distinction is that **request cancellation, transfer completion,
destination readiness, and storage reusability are separate facts**.

The demo should produce three reusable artifacts:

1. **A transfer specification.** A TLA+ formal model defining state, allowed steps,
   correctness requirements, and assumptions for competing requests, multi-block
   contexts, partial copies, cancellation, and reuse.
2. **A reproducible verification suite.** Safety, liveness, and reachability checks
   with recorded configurations, assumptions, and results.
3. **A replayable scenario suite.** Successful executions and counterexamples from
   deliberately faulty variants, replayed in the simulator and retained as tests.

The later integration plan adapts these artifacts to models of real TPU Sync
paths and tests of their implementations. The [README](README.md) introduces the
motivation and approach; this document details the design and completion criteria.

## Relationship to TPU Sync

The optional source reference is `third_party/tpu-sync`, pinned to upstream main
commit `6852486862ad644c2e9febb26d0f5e75be13c981` (reviewed 2026-09-09).
See [the implementation mapping](docs/tpu-sync-mapping.md) for the reviewed
correspondence and its limits. The demo and formal checks must run without
initializing or building this submodule.

The demo models one staged prefill-to-decode transfer route. It abstracts hardware
and networking while preserving the dependencies that make the route difficult.

The simulator and formal checks must not import TPU Sync's code or install its
dependencies. See the [production mapping and submodule workflow](docs/tpu-sync-mapping.md)
for planned correspondence, deliberate abstractions, and revision maintenance.
The source links below require initializing the optional submodule.

| TPU Sync concept | Demo representation | Code reference |
| --- | --- | --- |
| Producer and consumer workers | Separate Python objects and arrays | [Example serving topology](third_party/tpu-sync/examples/single_host_disagg/README.md) |
| Host staging and device destination | Separate memory pools | [Receive state](third_party/tpu-sync/tpu_sync/core/kv_cache_manager_with_transfer.h) |
| Separate network and H2D completion | Separate scheduled transfer stages | [Receive state](third_party/tpu-sync/tpu_sync/core/kv_cache_manager_with_transfer.h) |
| Pins and release after use | Explicit reservations for outstanding accesses | [Cache-store interface](third_party/tpu-sync/tpu_sync/kv_cache/kv_cache_store.h) |
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

## Scope and limitations

### Included

- Competing requests whose KV caches span multiple blocks, including out-of-order
  arrival and partial final blocks.
- Block-based allocation, reusable staging, and explicit allocation identity.
- Two asynchronous copy stages: producer to staging, then staging to destination.
- Separate publication, cancellation, copy completion, and outcome reporting.
- A symbolic TLA+ model checked with TLC for safety, liveness, and reachability.
- A deterministic simulator, trace export, and comparison of model and simulator state.
- Deliberately faulty variants and replayable regression scenarios.

### Limitations

The demo excludes:

- Real TPUs, GPUs, networking, RPC, multiple processes, and threads.
- A trained LLM, meaningful text generation, training, and model downloads.
- Prefix sharing, resharding, weight updates, transport failures, retries, and
  crash recovery.
- Hardware memory ordering and production C++ verification.
- Throughput claims or performance comparisons derived from simulated timing.
- A general-purpose verification framework or custom model checker.

Model-checking results apply to the model and checked configurations. Simulator
replay and TPU Sync tests provide evidence for particular code executions.
Formally verifying TPU Sync's code would additionally require proving that it
conforms to the checked model; refinement proofs between models do not establish
that connection.

## Simulator design

Develop a simulator that performs the simplified protocol's operations on arrays
in a caller-selected event order. It should expose data contents, readiness,
block mappings, and reservations for comparison with the formal model. A small
numerical attention workload provides supporting validation of transferred K/V;
its outputs are secondary to transfer correctness and ownership checks.

### Memory and identity

Use separate NumPy arrays for producer data, staging, and destination data. Ensure
the arrays do not accidentally alias. Begin with two reusable staging slots and
enough destination blocks for two short requests. Staging scarcity provides
contention without introducing an eviction policy.

Each request has a block table mapping logical token-block indices to physical
destination slots. Exercise noncontiguous, nonmonotonic allocations and transfer
blocks out of logical order. Gather ready blocks through that table in token
order for consumer reads. Track the valid token count to exclude padding in
partial blocks. Model each consumer read as an atomic snapshot of ready data.

Identify logical data by `(request, block, position)` labels and physical storage
by `(pool, slot, generation)`. The demo assumes a fixed association between each
request and its source cache; shared-cache identities are unnecessary for this scope.

Begin with two token rows per full block. Each copy-piece event copies one token's
K/V data. The formal model uses symbolic labels instead of floats and propagates
the label actually present at the source, so overwritten staging produces a
detectable data-identity error.

### Transfer stages and scheduling

```text
Producer K/V arrays -> host staging pool -> destination K/V arrays -> consumer
                       stage 1             stage 2
```

A single-threaded event loop selects enabled events. Operations may remain
pending while another request advances. Use explicit events rather than sleeps:

- reserve staging and destination slots;
- copy one producer-to-staging piece;
- complete producer-to-staging transfer;
- start and advance staging-to-destination transfer;
- record physical transfer completion;
- deliver a completion notification and record its operation outcome;
- publish a ready destination block;
- poll recorded outcomes without advancing physical transfers;
- consume ready data;
- cancel a request;
- release reservations and reuse slots.

The model schedules completion and notification separately to explore asynchronous
reporting. A slot may only be reused when no old operation can access it.
Generation checks reject stale bookkeeping updates;
they do not prevent an already-issued physical write. Completion can occur
without a client poll. Polling observes accumulated outcomes; it must not be what
makes a copy finish or makes its destination physically valid. A completion record
is not itself a reservation keeping memory alive.

This event separation is an abstraction of TPU Sync's execution behavior; the
[source mapping](docs/tpu-sync-mapping.md) records the correspondence.

For simplicity, cancellation does not abort a submitted copy. Submitted work
drains to physical completion, its unwanted result is not published, and its
reservations are then released. No new stages are started for a cancelled request.

## Formal specification

Use TLA+ to describe state and allowed transitions, and TLC to check the resulting
behaviors. Sets, functions, and records express ownership and symbolic data
identity directly; temporal properties state progress requirements and their
assumptions. Refinement supports later connections to a detailed TPU Sync model.

The initial [specification](formal/KVTransfer.tla) and
[configuration](formal/small.cfg) cover one two-row block per request. Extend
this starting point to the full demo scope below. Keep exact check results in
[formal/results.md](formal/results.md), assumptions in
[formal/assumptions.md](formal/assumptions.md), and code correspondence in the
[action mapping](formal/action-mapping.md).

### Abstract behavior

A cache block is either unavailable at the destination or ready with its complete
expected contents. A successful context read returns all required blocks for the
intended request in token order. If any required block is unavailable, the read
waits or reports a miss rather than returning partial data.

Publication is per block. A consumer read requires every block in the context to
be ready; this does not require atomic publication of the entire cache.

### Model state and configurations

Track request interest, transfer stage/progress, cache-block identity, slot
allocation generation, source/destination reservations, delivered completion
notifications, recorded client-visible outcomes, and destination readiness separately.

Start with two requests and two pieces per full block. Include configurations
with multiple logical blocks per request, out-of-order arrival, partial final
blocks, staging contention, and allocation reuse. These are core demo cases.
Keep simulator array sizes independent of checker configurations, but record the
exact configuration used for each replay so the states can be compared.

### Properties

| ID | Property |
| --- | --- |
| S1 | Every successful context read returns all required KV data for the intended request in token order. |
| S2 | Destination readiness requires successful completion of all required pieces at that destination. |
| S3 | A source remains unchanged while a transfer may read it; destination writes exclude conflicting accesses. |
| S4 | A slot cannot be reassigned while an outstanding operation may access it. |
| S5 | Outcome reporting does not change readiness or storage ownership; stale notifications cannot affect a reused allocation. |
| S6 | Cancellation prevents new stages, consumer access, and subsequent publication; submitted work retains its reservations until completion. |
| S7 | Polling observes recorded outcomes without advancing a physical transfer or bypassing readiness and lifetime rules. |

Assume each modeled piece copy faithfully transfers its source value, memory
regions are disjoint as declared, and consumers follow the readiness interface.
Do not assume whole-block copies are atomic or that cancellation stops hardware.

Check three kinds of properties:

- **Safety:** the invariants above hold in every reachable state.
- **Liveness:** a request that remains active eventually becomes readable and a
  cancelled request eventually releases its buffers, under explicit fairness and
  transfer-completion assumptions. Specify which scheduling, publication, and
  cleanup actions must eventually execute, and when resources must become available.
- **Reachability:** successful context reads, cancellation, retirement, and reuse
  are possible. Include overlap between requests and out-of-order block arrival.

TLC checks the selected model configuration without a fixed execution-depth
cutoff. Record configuration limits, assumptions, and results; a timeout or
incomplete search is not a passing check. Reachability witnesses show that useful
behavior is possible, while liveness checks address whether progress is guaranteed
under the stated assumptions.

Refinement proofs connecting this specification to a detailed TPU Sync model are
later integration work, separate from the demo's model checks and simulator replay.

## Demonstrations

### A. Arrived does not mean ready

Faulty variant: publish the device block when producer-to-staging reception
finishes, before staging-to-device copying completes.

The checker should find a consumer reading an incomplete destination. Replay the
trace in the matching faulty simulator variant and compare the returned data with
the expected contents, identifying the premature publication step.

### B. Cancelled does not mean reusable

Faulty variant: cancellation immediately releases a staging slot even though a
stage-2 transfer is still reading it.

Replay the counterexample in the matching faulty simulator variant and show
storage being released while a copy still accesses it. If execution continues
far enough for another request to overwrite that slot, compare the copied data
with the expected source identity. The lifetime violation itself is sufficient
evidence; a downstream numerical error is unnecessary.

### C. Correct staged protocol

With readiness and lifetime rules enabled, verify the safety and liveness
properties and replay successful reachability witnesses. Exercise multi-block
reads, cancellation, and safe reuse, including schedules where unrelated work
proceeds while transfers remain pending.

All faulty variants are intentional demo mutations, not allegations about bugs
in TPU Sync. Keep mutation switches isolated from the correct protocol.

## Replay workflow

1. **Export the trace.** Define a versioned format recording the model variant,
   initial configuration, ordered actions, request and transfer IDs, logical
   blocks, piece indices, slots, and allocation generations. Include observations
   needed for state comparison.
2. **Replay the actions.** Map TLA+ actions to simulator operations using the
   [action mapping](formal/action-mapping.md). Preserve action order and dispatch
   consumer reads and outcome polling explicitly. Counterexamples from intentional
   mutations require corresponding simulator faults. Reject invalid operations
   rather than silently skipping or repairing them.
3. **Compare each step.** Compare data identity, readiness, block mappings,
   reservations, and observed outcomes. Distinguish model-only bookkeeping from
   code operations and report the first mismatch or property violation. Retain
   successful traces and failure scenarios as regression tests; corrected code
   must prevent the original violation, even if it rejects the faulty schedule.

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
  tests/                    # Protocol, replay, and supporting numerical checks
  formal/
    KVTransfer.tla
    small.cfg
    context.cfg
    assumptions.md
    action-mapping.md
    results.md
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
`pyproject.toml`; maintain the runner's pinned TLC version. The TPU Sync
reference is pinned separately; changing it requires a mapping review. Keep generated
checker state, downloaded tools, and bulky logs out of Git. Start with a terminal
trace/table; a visual trace viewer is optional after the full workflow works.

## Implementation milestones

1. **Complete the transfer specification.** Extend the initial model to multi-block
   contexts, out-of-order arrival, and partial final blocks. Document state, steps,
   assumptions, and correspondence to simulator operations.
2. **Complete the verification suite.** Check safety, liveness, and reachability
   across the selected configurations. Record progress assumptions and verify
   that successful reads and reuse are reachable, so safety cannot pass merely
   because useful behavior is disabled.
3. **Complete simulator replay.** Export and replay successful model traces, then
   introduce the publication and cancellation faults separately. Reproduce their
   counterexamples and compare model and simulator state after each step.
4. **Package the verification case study.** Provide commands, expected outcomes,
   regression scenarios, and documentation of assumptions and limitations. Keep
   the specification, verification suite, and scenario suite independently reusable.

The simulator and model can evolve together, but their action mapping must remain
explicit. Record completed checks in [formal/results.md](formal/results.md);
the milestones above describe the full demo goal rather than implementation status.

## Completion criteria

- The specification covers competing requests, multi-block contexts, out-of-order
  arrival, partial copies, cancellation, reporting, and reuse.
- A documented checker command completes safety and liveness checks for each
  designated configuration, recording properties, assumptions, and results.
- Reachability checks produce witnesses for successful complete context reads,
  cancellation, retirement, overlap, and storage reuse.
- Each intentional mutation produces its expected invariant violation.
- Successful traces and counterexamples replay deterministically, with model and
  simulator state agreeing after each step. Counterexamples provide a concrete
  witness of incomplete data, conflicting access, or incorrect contents.
- The corrected simulator prevents each reproduced violation, with retained
  regression scenarios.
- Documentation distinguishes model-checking results, evidence from code tests,
  assumptions, and the separate proof needed for code conformance to the model.

## From Demo to TPU Sync

The demo's formal model, verification suite, and replay scenarios provide a
starting point for checking models of TPU Sync transfer paths and testing their
implementations. This work follows completion of the demo:

1. **Adapt and check the model for TPU Sync.** Extend and revise the demo model
   to represent TPU Sync's remote-loading path. Review its allocations, copies,
   callbacks, and synchronization, documenting how model steps correspond to code
   operations. Revisit abstractions and assumptions, including relevant error
   handling, then check safety, liveness, and reachability. The deliverable is a
   model grounded in that path with reproducible verification results.
2. **Turn model scenarios into implementation tests.** Translate applicable
   scenarios into tests with controlled transfer completion and callbacks. Check
   data correctness, cancellation, and buffer reuse, then assess backend-dependent
   assumptions through integration tests. The deliverable is a TPU Sync regression
   suite derived from behaviors relevant to its implementation.
3. **Use verification to guide changes.** Update the model and rerun checks
   alongside code tests for protocol changes. Include model results and scenarios
   in review, retain regression tests in CI, and extend coverage to additional
   paths. Track properties covered, faults found, and maintenance effort.

Refinement proofs can later establish that the detailed TPU Sync model preserves
the demo specification's correctness requirements. Proving that TPU Sync's code
conforms to that model remains a separate task.

## Research grounding

The ownership mechanism is an application of established ideas, not a proposed
new logic. [Safe Asynchronous Multicore Memory Operations (ASE 2011)](https://www.doc.ic.ac.uk/~afd/papers/2011/ASE.pdf)
provides the closest precedent for permissions committed to pending copies.
[PagedAttention (SOSP 2023)](https://arxiv.org/pdf/2309.06180) provides the KV-block
management context. [TLA+ and TLC](https://lamport.azurewebsites.net/tla/tools.html)
provide the model-checking infrastructure.

The project's contribution is a reproducible verification case study of staged
KV-cache transfers: a formal specification, safety and progress checks, replayable
scenarios, and a documented path toward models and tests of TPU Sync.
