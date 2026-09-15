# Formal verification of KV-cache transfers

A research project on formal verification of KV-cache transfer protocols for
LLM serving. It investigates correct cache reads, safe memory reuse, and progress
guarantees under asynchronous transfers, request cancellation, and delayed
completion notifications.

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

Requests share a finite pool of staging buffers, and a block copy progresses in
pieces: some token rows may arrive before others. Reusing a staging buffer while
a pending copy still reads it can cause that copy to transfer another request's
data. Cancelling a request does not stop an already-submitted copy.

A transfer can also finish before the client learns that it is complete.
The protocol must distinguish completion of the copy from reporting its outcome,
and keep buffers protected for as long as a transfer may access them.

These behaviors require separate rules for **readiness, ownership, cancellation,
and reuse**:

- Every successful consumer read returns the complete KV data for the intended
  request and logical block. A request's KV cache can span multiple blocks that
  arrive in any order; decoding requires all of them to be ready and read in token order.
- Reservations prevent storage reuse while copies are in progress. Generation
  numbers identify each use of a slot so software can reject stale references.
- Publication makes completed destination data readable. Cancellation withdraws
  consumer access and prevents subsequent publication; submitted work drains
  before its storage is released.
- Completion records report outcomes; reading them does not advance transfers
  or grant permission to access or release memory.

## Demo goal

Build a reproducible verification case study with two outcomes:

1. **A checked transfer contract.** Specify competing requests whose KV caches
   span multiple blocks, including out-of-order arrival, partial copies,
   cancellation, and reuse. Check safety within explicit finite bounds and show
   that successful reads and safe retirement are possible.
2. **Replayable counterexamples.** Introduce controlled mistakes, such as early
   publication or releasing staging on cancellation. Have TLC find failures and
   replay them over actual K/V tensors, identifying the violated property, its
   consequence, and the rule that prevents it. These are intentional demo faults,
   not claims of bugs in TPU Sync.

Together, these deliverables provide the basis for an ICLR blog post explaining
the correctness problem and its verification. The [project plan](PLAN.md)
details the design and completion criteria; [formal results](formal/results.md)
record exact checks and bounds.

## Verification approach and rationale

### Represent the protocol as states and steps

A **state** is a snapshot of the whole protocol: which requests are active or
cancelled, which rows have been copied, which blocks are ready to read, and who
owns or reserves each buffer. A **transition** is one permitted step that changes
that snapshot. Each step has conditions for when it can happen and rules for
what it changes. For example:

| Step | When it is allowed | What changes |
| --- | --- | --- |
| Copy a row to the destination | The destination copy has started and holds the required buffers | One destination row receives the staging row's contents |
| Cancel a request | The request is active | Consumer access is withdrawn; submitted copies keep their reservations |
| Record copy completion | Every required row has been copied | The copy's reservations are released |
| Release a staging buffer | Its submitted copies have completed or been safely discarded | The buffer becomes available for reuse |

An execution is a sequence of these steps. After copying one row for request A,
the next step might advance request B or cancel A. The model represents this
concurrency by allowing any enabled step, rather than prescribing one schedule.
A whole block copy spans several steps, so cancellation and other requests can
occur between its rows.

Rows carry labels identifying their request, logical block, and position instead
of full numerical tensors. A copy propagates the label actually at its source:
if staging was overwritten with B's data, the destination receives B's label.
This captures data-identity errors while keeping the state small, under the
assumption that individual copies faithfully transfer their source contents.

### Why TLA+ and TLC

**TLA+ specifies system behavior:** initial states, allowed steps, and required
properties. **TLC checks finite instances of that specification.** A property
that must hold in every reachable state is an **invariant**, such as keeping
buffers reserved while copies can access them. If a property fails, TLC returns
a **counterexample**: the sequence of steps leading to the failure.

The reason to choose TLA+ here is the combination of three features:

- **Direct ownership modeling.** Sets, functions, and records describe which
  request owns each slot, which operations may access it, and which blocks are
  ready. These relationships can be specified without reproducing TPU Sync's
  threads or callback structure. This matches TLA+'s emphasis on
  [designs above the code level](https://lamport.azurewebsites.net/tla/high-level-view.html).
- **Explicit progress assumptions.** TLA+ expresses both safety and eventual
  progress, with fairness conditions stating which enabled actions cannot be
  postponed forever. This separates “cancellation never frees a busy buffer”
  from “a cancelled request eventually releases its buffers.” Reachability
  checks show useful execution is possible; liveness reasoning asks whether it
  is guaranteed under stated scheduling and transfer-completion assumptions.
- **A path to refinement proofs.** TLA+ supports relating detailed protocols to
  simpler contracts. Copying a row can leave the abstract block “unavailable”;
  publication changes it to “ready.” Such abstractly unchanged steps are called
  *stuttering*. This provides a foundation for proving that buffer-level steps
  satisfy the abstract read contract. See the
  [refinement examples](https://lamport.azurewebsites.net/tla/auxiliary/auxiliary.html).

TLC makes this specification executable for small configurations containing
competing requests, multiple blocks, and reuse. It explores reachable states
and helps debug proposed invariants before a broader proof effort. Its finite
checks do not establish guarantees for arbitrary system sizes. Other tools also
support these kinds of analysis; TLA+/TLC is a practical fit for combining an
abstract contract, counterexample exploration, and later proof work.

### Relate counterexamples to inference

A deterministic simulator executes selected events over actual K/V tensors in
separate producer, staging, and destination arrays. Logical block tables recover
token order from scattered storage. A small attention model computes next-token
scores from the transferred cache and compares them with full recomputation
from the same token history. Comparing every score can expose errors even when
the selected token is unchanged.

The intended replay workflow follows model-checker counterexamples through
concrete copies to incorrect data and, when consumed, incorrect scores. A
cancelled request need not decode to expose a violation: an outstanding copy
accessing reused storage is already an error.

## Scope

Verification targets the abstract protocol. Bounded checks establish properties
only for the stated configurations; numerical tests and trace replay do not prove
implementation refinement. The model simplifies TPU Sync's execution behavior.
Hardware memory ordering, transport failures,
retries, and crash recovery are outside this abstraction. No production
correctness or performance claim follows from these checks.

## Future directions

Beyond the demo, the research can extend toward stronger guarantees and broader
applicability:

1. **Establish progress guarantees.** Specify when a live request must eventually
   become readable and when a cancelled request must eventually release its
   resources. State the necessary assumptions, such as submitted copies eventually
   finishing and eligible operations not being postponed forever. This addresses
   whether the protocol can get stuck despite preserving safety.

2. **Connect detailed protocols to the abstract contract.** Define how buffers,
   copy stages, and callbacks correspond to the specification's state and steps.
   Investigate a refinement proof: a proof that every allowed execution of the
   detailed protocol satisfies the simpler contract. Begin with a detailed
   protocol model; connecting actual implementation code requires additional
   verification. A further question is whether guarantees can be proved for
   arbitrary numbers of requests and blocks, beyond the checked finite cases.

3. **Evaluate which results generalize.** Apply the contract to selected TPU Sync
   transfer paths and another representative serving implementation. Identify
   shared rules, implementation-specific assumptions, and any changes needed to
   the abstraction. Evaluate the properties covered, faults detected, and cost
   of modeling and checking each case. This tests whether the work offers a
   reusable verification method beyond one example.

These extensions aim toward a research contribution suitable for a venue such
as CAV or ICLR, supported by stronger proofs, reusable methods, or substantive
findings across implementations.

## Quick start

### Bounded formal checks

With Java 11 or newer and Python 3 available:

```sh
sh tooling/check_model.sh --download
sh tooling/check_model.sh
```

The first command downloads the pinned TLC jar if needed; subsequent runs use
the verified local copy. For offline use, pass `--jar /path/to/tla2tools.jar`.
Logs, configurations, witness traces, and `summary.json` go to
`.cache/tlc/latest/`, or the directory selected with `--output PATH`.
`--timeout SECONDS` sets the per-check limit. Incomplete safety searches,
unexpected violations, and checker errors cause the runner to fail.

### Numerical demo and tests

The demo runs on a laptop without accelerators or model downloads. The reference
environment is Python 3.13.5 and NumPy 2.2.6. From the repository root, create an
environment and run the demo and tests:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m kv_transfer_demo.demo
.venv/bin/python -m unittest discover -s tests -v
```

The demo compares decoding scores against full recomputation and reports the
maximum error. To inspect cancellation, safe retirement, and slot reuse:

```sh
.venv/bin/python -m kv_transfer_demo.demo --cancel --trace
```

`--trace` prints the scheduled events and allocation generations. Invalid events
raise an error instead of being silently skipped. The demos require no TPU Sync
build or initialization.

## Further reading

- [Project plan](PLAN.md): detailed design, demonstrations, and completion criteria.
- [TPU Sync mapping](docs/tpu-sync-mapping.md): production references and abstractions.
- [Formal assumptions](formal/assumptions.md): execution assumptions and model bounds.
- [Action mapping](formal/action-mapping.md): correspondence between Python and TLA+.
- [Formal results](formal/results.md): recorded checks and reproduction details.
