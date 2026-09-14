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

## Verification approach and rationale

### Model event ordering and data identity

The correctness questions depend on the order of reads, writes, and buffer reuse.
A TLA+ state machine makes those dependencies explicit through actions for copy
progress, completion, publication, cancellation, and release. TLC explores all
reachable states within a finite configuration, checking safety properties
across the allowed event orderings.

The model tracks logical data identity, physical slots and their allocation
generations, and outstanding operations. Symbolic labels replace numerical
K/V rows because the properties concern data identity and completeness. Each copy
propagates the label actually at its source, so overwritten staging produces
incorrect destination data. This assumes that individual copies faithfully
transfer their source contents.

The verification design uses small configurations with competing requests,
KV caches spanning multiple blocks, partial copies, and buffer reuse to keep
systematic exploration practical. Deliberately weakening a rule, such as
allowing early publication, lets the checker produce an exact sequence showing
why that rule matters.

### Distinguish safety from progress

A protocol that prevents every read could satisfy read safety without serving
any requests. Reachability checks therefore require successful reads and safe
retirement to be possible. Proving that they eventually occur is a separate
liveness question, requiring explicit assumptions about fair scheduling and
physical transfer completion.

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

## Scope and research direction

Verification targets the abstract protocol. Bounded checks establish properties
only for the stated configurations; numerical tests and trace replay do not prove
implementation refinement. The model also separates events that TPU Sync handles
consecutively within a callback, so not every modeled interleaving is established
to occur in that implementation. Hardware memory ordering, transport failures,
retries, and crash recovery are outside this abstraction. No production
correctness or performance claim follows from these checks.

The core plan is to specify and check transfers of KV caches spanning multiple
blocks, including out-of-order arrival, readiness of the entire cache,
cancellation, and safe reuse. Numerical validation and counterexample replay
connect the verified properties to executable behavior. Broader research
directions include liveness verification, refinement proofs connecting
implementations to the contract, and evaluation across representative serving systems.

The intended outcome is a reproducible case study for an ICLR blog post and a
foundation for a broader formal-methods or ML-systems contribution. That research
goal requires generalizable contracts or verification methods and evidence beyond
a single small model. The [project plan](PLAN.md) defines the core artifact's
design and completion criteria; [formal results](formal/results.md) record the
exact checks and their bounds.

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
