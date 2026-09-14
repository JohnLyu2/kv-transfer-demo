"""Run the two-request staged-transfer scenario: python -m kv_transfer_demo.demo."""

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .attention import ATOL, PROMPTS, RTOL, KVCache, TinyAttention
from .protocol import TransferManager
from .scheduler import Event, RequestEvent, Scheduler


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
        scheduler.run(Event("start_destination", transfer_id) for transfer_id in current)
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

    generated, max_error = verify_decoding(model, manager, dict(zip(tables, PROMPTS)))
    return DemoResult(manager, scheduler, generated, max_error)


def verify_decoding(model, manager, prompts):
    """Check transferred data and two decode steps for each live request."""
    generated: dict[str, list[int]] = {}
    max_error = 0.0
    for request_id, prompt in prompts.items():
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
    return generated, max_error


def run_cancellation_demo() -> DemoResult:
    """Cancel A mid-copy, let B progress, then reuse A's slots for C."""
    model = TinyAttention()
    manager = TransferManager(destination_slots=4)
    scheduler = Scheduler(manager)
    prompts = {"A": PROMPTS[0][:2], "B": PROMPTS[1][:2], "C": PROMPTS[0][2:4]}
    for request_id, slot in (("A", 0), ("B", 1)):
        _, source = model.prefill(prompts[request_id])
        manager.add_request(request_id, source, [slot])
        manager.plan(request_id + "0", request_id, 0)

    def events(transfer_id):
        return ([Event("reserve", transfer_id)]
                + [Event("copy_to_staging", transfer_id, piece) for piece in (0, 1)]
                + [Event("complete_staging", transfer_id), Event("start_destination", transfer_id)]
                + [Event("copy_to_destination", transfer_id, piece) for piece in (0, 1)]
                + [Event("complete_destination", transfer_id), Event("release_staging", transfer_id),
                   Event("notify", transfer_id), Event("publish", transfer_id)])

    scheduler.run(events("A0")[:6])  # One row reached the destination.
    scheduler.run([RequestEvent("cancel", "A")])
    scheduler.run(events("B0"))  # A is still draining while B finishes.
    scheduler.run([Event("copy_to_destination", "A0", 1),
                   Event("complete_destination", "A0"), Event("release_staging", "A0"),
                   RequestEvent("retire", "A")])
    _, source = model.prefill(prompts["C"])
    manager.add_request("C", source, [0])  # Reuse A's destination slot.
    manager.plan("C0", "C", 0)
    scheduler.run(events("C0")[:2])  # Reuse A's staging and start writing C.
    scheduler.run([Event("notify", "A0")])  # Metadata only; old slots are untouched.
    scheduler.run(events("C0")[2:])
    outcomes = manager.poll()
    if {o.transfer_id: o.status for o in outcomes} != {
        "A0": "cancelled", "B0": "completed", "C0": "completed"
    } or len(outcomes) != 3 or manager.poll():
        raise AssertionError("Unexpected cancellation outcomes")
    generated, max_error = verify_decoding(model, manager,
                                          {key: prompts[key] for key in ("B", "C")})
    return DemoResult(manager, scheduler, generated, max_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="Print each executed copy/lifecycle event")
    parser.add_argument("--cancel", action="store_true", help="Run cancellation, drain, and reuse scenario")
    args = parser.parse_args()
    result = run_cancellation_demo() if args.cancel else run_demo()
    if args.trace:
        for index, entry in enumerate(result.scheduler.trace):
            staging = entry.staging
            if isinstance(entry.event, RequestEvent):
                print(f"{index:02d} request {entry.request_id} {entry.event.action}")
                continue
            staging_label = f"{staging.slot}:{staging.generation}" if staging else "none"
            print(f"{index:02d} {entry.event.transfer_id} {entry.event.action:22s} "
                  f"piece={entry.event.piece} staging={staging_label} "
                  f"destination={entry.destination.slot}:{entry.destination.generation}")
    for request_id, tokens in result.generated.items():
        slots = [ref.slot for ref in result.manager.requests[request_id].blocks]
        print(f"{request_id}: logical blocks -> physical slots {slots}; generated {tokens}")
    scenario = "A cancelled and retired; B/C verified" if args.cancel else "6 blocks transferred"
    print(f"PASS: {scenario}; {len(result.scheduler.trace)} events; "
          f"max score error {result.max_score_error:.3e}")
    print("Both staging slots are free; destination blocks remain owned until request retirement.")


if __name__ == "__main__":
    main()
