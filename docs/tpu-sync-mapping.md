# TPU Sync implementation mapping

## Reference revision

- Upstream: https://github.com/google/tpu-sync
- Submodule: `third_party/tpu-sync`
- Pinned and reviewed commit: `6852486862ad644c2e9febb26d0f5e75be13c981`
- Review date: 2026-09-09
- Previous planning reference: `622f9b5f5f4ca6308c33050923f97dadb643dbb1`

This is a source-review mapping for a planned demo, not verified equivalence.
The local numerical baseline is implemented; staged transfers and the formal
model remain planned. Neither production tests nor hardware
verification were performed for this update. The pin is a reproducible source
reference, not a declaration of deployment stability.

## Reviewed correspondence

| Concern | Production reference | Planned demo abstraction |
| --- | --- | --- |
| Disaggregated prefill/decode | [Serving example](../third_party/tpu-sync/examples/single_host_disagg/README.md) | Separate producer/consumer objects and real NumPy KV tensors |
| Distinct network and H2D stages | [KVCacheManagerWithTransfer::RecvEntry](../third_party/tpu-sync/tpu_sync/core/kv_cache_manager_with_transfer.h) | Independently scheduled copy stages and completion notifications |
| Remote fetch followed by H2D | [HostOffloadBackend](../third_party/tpu-sync/tpu_sync/kv_cache/host_offload_backend.cc) | Fetch into reserved staging, then copy into destination memory |
| Completion status | [BlockTracker](../third_party/tpu-sync/tpu_sync/kv_cache/block_tracker.cc) and [KVCacheStore polling](../third_party/tpu-sync/tpu_sync/kv_cache/kv_cache_store.cc) | Record outcomes on completion; polling reports them without driving data movement |
| Local cache pin contract | [KVCacheStore](../third_party/tpu-sync/tpu_sync/kv_cache/kv_cache_store.h) | Reservations prevent reuse while an operation may access a slot |
| Generations and lifecycle observation | [RequestBlockRegistry](../third_party/tpu-sync/tpu_sync/kv_cache/reshard/request_block_registry.h) | Explicit transfer IDs and slot generations; a stale notification cannot affect a new allocation |
| Raw-buffer hold option | [AcquireCommonRawBuffer](../third_party/tpu-sync/tpu_sync/core/xla_compat.cc) | Backend ownership assumptions documented; no PJRT emulation or proof |

## Changes from the previous reference

Backend callbacks now carry load/save completion bookkeeping that previously
lived in store polling. `KVCacheStore::PollLoadStatus` and `PollSaveStatus`
delegate to `BlockTracker`. The remote read wrapper delegates to `load` and its
status wrapper to load-status polling.

The reviewed remote-load success path waits for fetch completion, submits H2D,
and deallocates host staging in the H2D completion callback before reporting the
outcome. This is a concrete example of the staged lifetime rule. It does not
establish correctness of every failure, timeout, or cancellation path.

`BlockTracker` stores pending entries and accumulated outcome records. Polling
consumes terminal outcome records while reporting current pending entries. It
is not a memory owner or a complete per-transfer generation protocol. The demo
must not conflate its recorded outcomes with slot reservations or assume this
class alone guarantees exactly-once completion.

The request registry adds a status probe with registered, claimed, cancelled,
and unknown results. It purges expired state and has explicit status precedence.
Unknown is not proof that no physical transfer remains active. Full registry TTL
behavior remains outside the first demo.

PJRT acquisition now resides in a compatibility layer. The CommonPjRtBuffer
path still obtains a usage hold and retains it only when skipping is disabled.
The C-API path creates a raw alias. These backend mechanics remain outside the
formal model.

## Deliberate abstractions and assumptions

- Real attention and separate NumPy arrays replace an actual TPU serving stack.
- Deterministic copy-piece events replace network and DMA execution; no hardware
  ordering or throughput claims follow from the demo.
- Producer data starts complete, with a fixed model and correct cache identity.
- Demo cancellation drains already-submitted copies before releasing their
  reservations. This is a proposed demo protocol, not a claim about every
  production cancellation path.
- Cache pins, raw-buffer references, framework holds, and staging reservations
  are different mechanisms. The abstract reservations capture only the stated
  demo access/lifetime contract.
- Resharding, weight synchronization, telemetry, crash recovery, prefix sharing,
  and distributed registry behavior are excluded from the first version.

## Optional checkout and deliberate updates

The CPU workload and formal checks must not require the submodule.
To initialize the recorded reference when source inspection is needed:

```sh
git submodule update --init third_party/tpu-sync
```

To inspect a possible update, first ensure the submodule has no local changes,
then fetch upstream without changing the checkout:

```sh
git -C third_party/tpu-sync status --short
git -C third_party/tpu-sync fetch origin main
git -C third_party/tpu-sync log --oneline HEAD..origin/main
```

Review relevant changes, select the exact commit, check it out detached in the
submodule, update this document and PLAN.md, and stage the submodule pointer in
the parent repo. Commit the pointer and mapping together when ready. Do not
implicitly follow a moving upstream branch in normal demo or CI runs.
