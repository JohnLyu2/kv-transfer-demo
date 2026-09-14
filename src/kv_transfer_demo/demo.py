"""Run the two-request staged-transfer scenario: python -m kv_transfer_demo.demo."""

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .attention import ATOL, PROMPTS, RTOL, KVCache, TinyAttention
from .protocol import TransferManager
from .scheduler import Event, Scheduler


@dataclass
class DemoResult:
    manager: TransferManager
    scheduler: Scheduler
    generated: dict[str, list[int]]
    max_score_error: float


def run_demo() -> DemoResult:
    model = TinyAttention()
    manager = TransferManager(qk_dim=model.qk_dim, value_dim=model.value_dim)
    scheduler = Scheduler(manager)
    tables = {"A": [5, 1, 7], "B": [0, 6, 2]}
    for request_id, prompt in zip(tables, PROMPTS):
        _, producer = model.prefill(prompt)
        manager.add_request(request_id, producer, tables[request_id])
        for block in range(3):
            manager.plan(f"{request_id}{block}", request_id, block)

    previous: list[str] = []
    # Transfer logical blocks out of order; physical block order differs too.
    for block in (2, 0, 1):
        current = [f"A{block}", f"B{block}"]
        scheduler.run(Event("reserve", transfer_id) for transfer_id in current)
        # Old notifications arrive after their staging slots were reassigned.
        for transfer_id in previous:
            scheduler.run([Event("notify", transfer_id), Event("publish", transfer_id)])
        for piece in (0, 1):
            scheduler.run(Event("copy_to_staging", transfer_id, piece)
                          for transfer_id in current)
        scheduler.run(Event("complete_staging", transfer_id) for transfer_id in current)
        for piece in (1, 0):
            scheduler.run(Event("copy_to_destination", transfer_id, piece)
                          for transfer_id in reversed(current))
        for transfer_id in current:
            scheduler.run([Event("complete_destination", transfer_id),
                           Event("release_staging", transfer_id)])
        previous = current
    for transfer_id in previous:
        scheduler.run([Event("notify", transfer_id), Event("publish", transfer_id)])

    outcomes = manager.poll()
    if len(outcomes) != 6 or manager.poll():
        raise AssertionError("Expected six completion outcomes, consumed exactly once")

    generated: dict[str, list[int]] = {}
    max_error = 0.0
    for request_id, prompt in zip(tables, PROMPTS):
        tokens = list(prompt)
        generated[request_id] = []
        cache = manager.gather(request_id)
        assert_array_equal(cache.keys, manager.requests[request_id].source.keys)
        assert_array_equal(cache.values, manager.requests[request_id].source.values)
        for step in range(3):
            # Every score uses a fresh logical-order read from the physical pool.
            cache = manager.gather(request_id)
            scores = model.cached_scores(tokens[-1], cache)
            expected = model.reference(tokens)[-1]
            assert_allclose(scores, expected, rtol=RTOL, atol=ATOL)
            max_error = max(max_error, float(np.max(np.abs(scores - expected))))
            if step < 2:
                token = int(np.argmax(scores))
                if token != int(np.argmax(expected)):
                    raise AssertionError("Argmax differs from reference")
                next_scores, extended = model.decode(token, cache)
                tokens.append(token)
                assert_allclose(next_scores, model.reference(tokens)[-1], rtol=RTOL, atol=ATOL)
                manager.append(request_id, KVCache(extended.keys[-1:], extended.values[-1:]))
                generated[request_id].append(token)
        _, expected_cache = model.prefill(tokens)
        assert_allclose(manager.gather(request_id).keys, expected_cache.keys, rtol=RTOL, atol=ATOL)
        assert_allclose(manager.gather(request_id).values, expected_cache.values, rtol=RTOL, atol=ATOL)
    return DemoResult(manager, scheduler, generated, max_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="Print each executed copy/lifecycle event")
    args = parser.parse_args()
    result = run_demo()
    if args.trace:
        for index, entry in enumerate(result.scheduler.trace):
            staging = entry.staging
            print(f"{index:02d} {entry.event.transfer_id} {entry.event.action:22s} "
                  f"piece={entry.event.piece} staging={staging.slot}:{staging.generation} "
                  f"destination={entry.destination.slot}:{entry.destination.generation}")
    for request_id, tokens in result.generated.items():
        slots = [ref.slot for ref in result.manager.requests[request_id].blocks]
        print(f"{request_id}: logical blocks -> physical slots {slots}; generated {tokens}")
    print(f"PASS: 6 blocks transferred; {len(result.scheduler.trace)} events; "
          f"max score error {result.max_score_error:.3e}")
    print("Both staging slots are free; destination blocks remain owned until request retirement.")


if __name__ == "__main__":
    main()
