import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from kv_transfer_demo.attention import ATOL, RTOL, KVCache, TinyAttention
from kv_transfer_demo.demo import run_demo
from kv_transfer_demo.protocol import ProtocolError, TransferManager
from kv_transfer_demo.scheduler import Event, Scheduler


def copy_events(transfer_id, pieces):
    return ([Event("reserve", transfer_id)]
            + [Event("copy_to_staging", transfer_id, i) for i in range(pieces)]
            + [Event("complete_staging", transfer_id), Event("start_destination", transfer_id)]
            + [Event("copy_to_destination", transfer_id, i) for i in range(pieces)]
            + [Event("complete_destination", transfer_id),
               Event("release_staging", transfer_id),
               Event("notify", transfer_id), Event("publish", transfer_id)])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyAttention()
        self.manager = TransferManager()
        self.scheduler = Scheduler(self.manager)
        _, self.source = self.model.prefill([1, 4, 2, 8])
        self.manager.add_request("A", self.source, [5, 1])
        self.manager.plan("A0", "A", 0)
        self.manager.plan("A1", "A", 1)

    def event(self, action, transfer_id="A0", piece=None):
        self.scheduler.run([Event(action, transfer_id, piece)])

    def test_two_requests_scattered_blocks_and_staging_reuse(self):
        result = run_demo()
        self.assertEqual(result.manager.staging.free_slots, [0, 1])
        self.assertEqual([r.slot for r in result.manager.requests["A"].blocks], [5, 1, 7, 3])
        self.assertEqual([r.slot for r in result.manager.requests["B"].blocks], [0, 6, 2, 4])
        self.assertTrue(all(len(tokens) == 2 for tokens in result.generated.values()))
        reservations = [r for r in result.scheduler.trace if r.event.action == "reserve"]
        self.assertEqual([r.staging.generation for r in reservations], [1, 1, 2, 2, 3, 3])
        # No completion for A2 is delivered until its staging slot belongs to A0.
        actions = [(r.event.action, r.event.transfer_id) for r in result.scheduler.trace]
        self.assertLess(actions.index(("reserve", "A0")), actions.index(("notify", "A2")))
        self.assertEqual(result.scheduler.trace, run_demo().scheduler.trace)

    def test_readiness_requires_publication_of_every_context_block(self):
        with self.assertRaises(ProtocolError):
            self.manager.gather("A")
        events = copy_events("A0", 2)
        self.scheduler.run(events[:-1])  # Fully copied and notified, but not published.
        with self.assertRaises(ProtocolError):
            self.manager.gather("A")
        self.event("publish")
        with self.assertRaises(ProtocolError):
            self.manager.gather("A")  # Block 1 is still unavailable.
        self.scheduler.run(copy_events("A1", 2))
        gathered = self.manager.gather("A")
        assert_array_equal(gathered.keys, self.source.keys)
        assert_array_equal(gathered.values, self.source.values)

    def test_rejects_early_duplicate_and_unknown_events(self):
        for action in ("publish", "notify", "release_staging", "complete_destination"):
            with self.subTest(action=action), self.assertRaises(ProtocolError):
                self.event(action)
        self.assertEqual(self.scheduler.trace, [])
        self.event("reserve")
        self.event("copy_to_staging", piece=0)
        for action, piece in (("copy_to_staging", 0), ("copy_to_staging", 2),
                              ("complete_staging", None), ("copy_to_destination", 0),
                              ("unknown", None)):
            with self.subTest(action=action, piece=piece), self.assertRaises(ProtocolError):
                self.event(action, piece=piece)
        self.assertEqual(len(self.scheduler.trace), 2)

    def test_inflight_reservations_block_release_and_conflicting_access(self):
        self.event("reserve")
        transfer = self.manager.transfers["A0"]
        for pool, ref in ((self.manager.staging, transfer.staging),
                          (self.manager.destination, transfer.destination)):
            with self.subTest(pool=pool.name):
                with self.assertRaises(ProtocolError):
                    pool.release(ref)
                with self.assertRaises(ProtocolError):
                    pool.read(ref)
                with self.assertRaises(ProtocolError):
                    pool.write_piece(ref, 0, KVCache(np.zeros(8), np.zeros(8)))
        with self.assertRaises(ProtocolError):
            self.manager.release_request("A")
        self.event("copy_to_staging", piece=0)
        self.event("copy_to_staging", piece=1)
        self.event("complete_staging")
        self.event("start_destination")
        self.event("copy_to_destination", piece=0)
        with self.assertRaises(ProtocolError):
            self.event("release_staging")
        with self.assertRaises(ProtocolError):
            self.event("complete_destination")

    def test_staging_exhaustion_preserves_pending_transfer(self):
        self.event("reserve")
        self.event("reserve", "A1")
        self.manager.add_request("B", self.source, [0, 2])
        self.manager.plan("B0", "B", 0)
        with self.assertRaises(ProtocolError):
            self.event("reserve", "B0")
        transfer = self.manager.transfers["B0"]
        self.assertEqual(transfer.phase, "planned")
        self.assertIsNone(transfer.staging)
        self.manager.destination.check(transfer.destination)
        self.scheduler.run(copy_events("A0", 2)[1:])
        self.event("reserve", "B0")

    def test_poll_does_not_advance_or_publish_a_transfer(self):
        self.event("reserve")
        before = self.manager.destination.keys.copy()
        self.assertEqual(self.manager.poll(), ())
        self.assertEqual(self.manager.transfers["A0"].phase, "producer")
        assert_array_equal(self.manager.destination.keys, before)
        self.scheduler.run(copy_events("A0", 2)[1:-1])
        outcomes = self.manager.poll()
        self.assertEqual([outcome.transfer_id for outcome in outcomes], ["A0"])
        self.assertEqual(self.manager.poll(), ())
        self.assertEqual(self.manager.requests["A"].ready, set())
        self.event("publish")  # Polling is not a prerequisite for publication.

    def test_old_references_cannot_access_reallocated_storage(self):
        self.scheduler.run(copy_events("A0", 2))
        old = self.manager.transfers["A0"].staging
        self.event("reserve", "A1")
        new = self.manager.transfers["A1"].staging
        self.assertEqual(old.slot, new.slot)
        self.assertNotEqual(old.generation, new.generation)
        with self.assertRaises(ProtocolError):
            self.manager.staging.read(old)
        with self.assertRaises(ProtocolError):
            self.manager.staging.release(old)
        with self.assertRaises(ProtocolError):
            self.event("notify")  # Duplicate delivery cannot record another outcome.
        self.scheduler.run(copy_events("A1", 2)[1:])
        old_destination = self.manager.requests["A"].blocks[0]
        self.manager.release_request("A")
        with self.assertRaises(ProtocolError):
            self.manager.add_request("A", self.source, [5, 1])
        self.manager.add_request("B", self.source, [5, 1])
        with self.assertRaises(ProtocolError):
            self.manager.destination.read(old_destination)
        with self.assertRaises(ProtocolError):
            self.event("publish")

    def test_arrays_do_not_alias_and_gather_is_a_snapshot(self):
        request = self.manager.requests["A"]
        arrays = (self.source.keys, self.source.values, request.source.keys,
                  request.source.values, self.manager.staging.keys,
                  self.manager.staging.values, self.manager.destination.keys,
                  self.manager.destination.values)
        for i, left in enumerate(arrays):
            for right in arrays[i + 1:]:
                self.assertFalse(np.shares_memory(left, right))
        with self.assertRaises(ValueError):
            request.source.keys[0, 0] = 0
        self.scheduler.run(copy_events("A0", 2) + copy_events("A1", 2))
        snapshot = self.manager.gather("A")
        snapshot.values.fill(0)
        assert_array_equal(self.manager.gather("A").values, self.source.values)

    def test_partial_block_and_distinct_key_value_widths(self):
        class DifferentDimensions(TinyAttention):
            model_dim, qk_dim, value_dim = 6, 4, 3

        model = DifferentDimensions()
        manager = TransferManager(destination_slots=3, qk_dim=4, value_dim=3)
        scheduler = Scheduler(manager)
        tokens = [1, 2, 3]
        _, cache = model.prefill(tokens)
        manager.add_request("A", cache, [2, 0])
        for block, pieces in ((1, 1), (0, 2)):
            manager.plan(str(block), "A", block)
            scheduler.run(copy_events(str(block), pieces))
        assert_array_equal(manager.gather("A").keys, cache.keys)
        # Append fills the last partial block, then allocates a fresh block.
        for token in (4, 5):
            cache = manager.gather("A")
            scores, extended = model.decode(token, cache)
            manager.append("A", KVCache(extended.keys[-1:], extended.values[-1:]))
            tokens.append(token)
            gathered = manager.gather("A")
            self.assertTrue(np.isfinite(gathered.keys).all())
            assert_allclose(model.cached_scores(token, gathered), model.reference(tokens)[-1],
                            rtol=RTOL, atol=ATOL)
            assert_allclose(scores, model.cached_scores(token, gathered), rtol=RTOL, atol=ATOL)
        self.assertEqual([ref.slot for ref in manager.requests["A"].blocks], [2, 0, 1])

    def test_invalid_registration_does_not_allocate_slots(self):
        free = self.manager.destination.free_slots
        for slots in ([0, 0], [0, 5], [0]):
            with self.assertRaises(ProtocolError):
                self.manager.add_request("B", self.source, slots)
            self.assertEqual(self.manager.destination.free_slots, free)


if __name__ == "__main__":
    unittest.main()
