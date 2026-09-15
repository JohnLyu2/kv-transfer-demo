# Formal verification of KV-cache transfers

A formal verification demo of a simplified KV-cache transfer protocol for LLM
serving, focused on correct cache reads and safe memory reuse under concurrent
transfers and cancellation. The demo's scope is model checking and replayable
counterexamples for this simplified protocol.

The future plan is to apply the resulting specification, verification evidence,
and test scenarios to real TPU Sync transfer paths, supporting production
regression tests and protocol changes.

## Background

An autoregressive language model generates tokens one at a time. Attention uses
key and value tensors from earlier tokens; retaining them in a **KV cache** avoids
recomputing them at every step. **Prefill** processes the prompt and builds the
cache. **Decode** uses and extends it as new tokens are generated. Running these
phases on separate workers is called **prefill–decode disaggregation**: the
prefill worker transfers its cache so the decode worker can continue the computation.

[TPU Sync](third_party/tpu-sync/README.md) manages KV caches and data transfers on
Cloud TPUs. It moves data between **accelerator memory** (the TPU's high-bandwidth
memory used by attention), **host memory** (the machine's system RAM, used for
staging or cache storage), and **remote workers** (serving processes on other
machines, reached over the network).

This project uses TPU Sync at Git commit
`6852486862ad644c2e9febb26d0f5e75be13c981` as its fixed implementation reference.
It studies a **simplified abstraction of a staged transfer route**:

```text
Prefill worker           Reusable host staging          Decode worker
Producer K/V arrays  ->  Staging buffers            ->  Destination K/V arrays
                                                       |
                                                       v
                                                   Attention and decoding
```

The [source mapping](docs/tpu-sync-mapping.md) documents how the abstraction
relates to this implementation.

## The verification problem

> When can a consumer use transferred KV data, and when can its storage be reused?

A request's KV cache can span several blocks, copied through staging buffers
shared by competing requests. Correctness depends on coordinating four concerns:

| Concern | Why coordination is needed | Required property |
| --- | --- | --- |
| **Readiness and data identity** | Rows and blocks can arrive out of order, so individual copies may finish before the required cache is ready. | Every successful context read returns all required KV data for the intended request in token order. |
| **Buffer lifetime and reuse** | Requests share storage, and outstanding copies may still access a slot proposed for reuse. | A slot is reused only after all outstanding accesses are confirmed complete; stale references cannot affect its new allocation. |
| **Cancellation** | A request can be cancelled while its submitted copies still access memory. | Cancellation prevents new transfer stages and consumer access; submitted copies drain before storage is released, and their results are not published. |
| **Completion reporting** | Physical completion and client observation occur separately. | Outcome reporting does not change readiness or storage ownership; polling does not advance copies. |

The protocol uses **reservations** to protect buffers, **publication** to make
completed data readable, and **generation numbers** to distinguish successive
allocations of a slot. Verification asks whether these mechanisms preserve the
properties above across all allowed event orderings.

## Demo plan

### Goal and deliverables

Build a reproducible verification case study covering competing requests, caches
spanning multiple blocks, out-of-order arrival, partial copies, cancellation,
and reuse. The demo should produce three reusable artifacts:

1. **A transfer specification.** A simplified formal model, written in TLA+,
   describing how requests share buffers, move KV data, and respond to
   cancellation. It represents these
   operations as changes to abstract state, while implementation code performs
   them on actual data and memory. The model also states requirements such as
   keeping a buffer allocated while a copy still uses it. The model can later be
   adapted to represent TPU Sync's actual transfer behavior and checked against
   these requirements to identify protocol errors.
2. **A reproducible verification suite.** Safety checks, reachability checks for
   successful reads and reuse, and liveness checks for eventual request completion
   or safe retirement under explicit scheduling and transfer-completion
   assumptions, with recorded configurations and results.
3. **A replayable scenario suite.** Successful executions and counterexamples
   from deliberately faulty model variants, replayed in the simulator to show
   which requirements hold or fail. Applicable scenarios can later become
   TPU Sync regression tests.

The [project plan](PLAN.md) details the design and completion criteria;
[formal results](formal/results.md) record exact checks and configuration limits.

### Model KV-cache transfers as a state machine

A state machine describes a system through its possible states and the steps
that change them. This fits KV-cache transfers because their correctness depends
on the order in which copying, cancellation, and buffer reuse occur.

Here, each **state** records request status, copied rows, readable blocks, and
buffer ownership and reservations. Each **transition** represents an operation,
such as copying a row or cancelling a request, with conditions for when it is
allowed and rules for how it updates the state. Examples include:

| Step | When it is allowed | What changes |
| --- | --- | --- |
| Copy a row to the destination | The destination copy has started and holds the required buffers | One destination row receives the staging row's contents |
| Cancel a request | The request is active | Consumer access is withdrawn; submitted copies keep their reservations |
| Record destination-copy completion | Every required row has reached the destination | The copy's staging and destination reservations are released |
| Release a staging buffer | Its submitted copies have completed or been safely discarded | The buffer becomes available for reuse |

An execution is a sequence of these steps. After copying one row for request A,
the next step might advance request B or cancel A. The model represents this
concurrency by allowing any enabled step, rather than prescribing one schedule.
A whole block copy spans several steps, so cancellation and other requests can
occur between its rows.

The model represents each token's K/V data **symbolically**, using a
`(request, block, position)` label instead of numerical tensors. Assuming
individual copies faithfully transfer their source contents, copying propagates
the source label so the checker can detect missing or incorrect data.

### Why TLA+ and TLC

**TLA+ is a language used to write the formal model and its correctness
requirements; TLC is the tool used to check them.** The choice fits this project
for three reasons:

- **Express ownership and data identity directly.** TLA+'s sets, functions, and
  records represent request-to-buffer mappings, reservations, and symbolic K/V
  labels. This keeps the specification focused on transfer behavior without
  reproducing the implementation's threads, callbacks, or tensor arithmetic.
  See the [TLA+ overview](https://lamport.azurewebsites.net/tla/high-level-view.html).
- **Separate safety from progress.** Safety requirements such as “a reserved
  buffer cannot be reassigned” can be written as **invariants**: conditions that
  hold in every reachable state. Temporal properties express **liveness**, such
  as eventual release of a cancelled request's buffers. Explicit fairness and
  transfer-completion assumptions make clear what progress depends on.
- **A path to refinement proofs.** A refinement proof shows that every execution
  of a detailed model corresponds to behavior allowed by a simpler specification.
  TLA+ supports using the demo's simplified model as that specification, then
  proving that a more detailed model of TPU Sync preserves its correctness
  requirements. See the
  [refinement examples](https://lamport.azurewebsites.net/tla/auxiliary/auxiliary.html).

TLC explores reachable states to check safety and liveness, returning a
counterexample if a property is violated. A completed check without violations
establishes these properties for the chosen configuration. Separate reachability
checks confirm that useful behaviors, such as successful reads and reuse, are
possible.

TLA+/TLC provides a practical combination of expressive specifications, executable
checks, and support for relating models at different levels of detail. Other
tools offer overlapping capabilities, with trade-offs in modeling effort,
checking performance, and proof support worth considering as the project develops.

### Replay transfer counterexamples

Develop a simulator of the simplified transfer protocol that executes model
actions in code. Use it to reproduce TLC counterexamples and connect violations
in the formal model to their effects on data and buffer ownership:

1. **Export the trace.** Save the TLC counterexample's initial configuration and
   ordered actions, including the request, block, and buffer allocation involved.
2. **Replay the actions.** Translate the trace into simulator operations using
   the [action mapping](formal/action-mapping.md), preserving the recorded order.
   When the trace comes from a deliberately faulty model variant, reproduce the
   same fault in the simulator.
3. **Verify the reproduction.** Compare data identity, readiness, block mappings,
   and reservations with the model after each step to locate the violation or any
   mismatch. Retain the scenario as a regression test for the corrected behavior.

### Scope and limitations

The demo covers transfer coordination, cancellation, and buffer reuse in a
simplified model. Hardware memory ordering, transport failures, retries, and
crash recovery are outside its scope.

Model-checking results apply to the model and checked configurations. The
integration plan below combines model checking with implementation tests; it
does not formally verify TPU Sync's code. That would additionally require proving
that TPU Sync's code conforms to the checked model. Refinement proofs between models
alone do not establish this connection.

## From Demo to TPU Sync

The demo's formal model, verification suite, and replay scenarios provide a
starting point for checking models of TPU Sync transfer paths and testing their
implementations. The integration plan has three steps:

1. **Adapt and check the model for TPU Sync.** Extend and revise the demo model
   to represent TPU Sync's remote-loading path. Map its steps to code operations
   and check safety, liveness, and reachability under documented assumptions.
2. **Turn model scenarios into implementation tests.** Translate applicable model
   scenarios into tests with controlled event ordering. Check data correctness,
   cancellation, and reuse, and assess backend assumptions through integration tests.
3. **Use verification to guide changes.** Update the model and rerun checks
   alongside code tests for protocol changes. Include results in review, retain
   regression tests in CI, and extend coverage to other paths.

Refinement proofs can later establish that the detailed TPU Sync model preserves
the demo specification's correctness requirements.

## Further reading

- [Project plan](PLAN.md): detailed design, demonstrations, and completion criteria.
- [TPU Sync mapping](docs/tpu-sync-mapping.md): production references and abstractions.
- [Formal assumptions](formal/assumptions.md): execution assumptions and configuration limits.
- [Action mapping](formal/action-mapping.md): correspondence between Python and TLA+.
- [Formal results](formal/results.md): recorded checks and reproduction details.
