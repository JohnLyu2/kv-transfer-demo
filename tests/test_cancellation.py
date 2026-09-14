import unittest

from numpy.testing import assert_array_equal

from kv_transfer_demo.attention import KVCache, TinyAttention
from kv_transfer_demo.demo import run_cancellation_demo
from kv_transfer_demo.protocol import ProtocolError, TransferManager
from kv_transfer_demo.scheduler import Event, RequestEvent, Scheduler
from test_protocol import copy_events


class CancellationTests(unittest.TestCase):
    def setup_manager(self):
        self.manager = TransferManager(destination_slots=4)
        self.scheduler = Scheduler(self.manager)
        _, self.source = TinyAttention().prefill([1, 2])
        self.manager.add_request("A", self.source, [0])
        self.manager.plan("A0", "A", 0)

    def drain(self, transfer_id):
        transfer = self.manager.transfers[transfer_id]
        if transfer.phase == "producer":
            for piece in range(transfer.pieces):
                if piece not in transfer.producer_pieces:
                    self.manager.step("copy_to_staging", transfer_id, piece)
            self.manager.step("complete_staging", transfer_id)
        elif transfer.phase == "destination":
            for piece in range(transfer.pieces):
                if piece not in transfer.destination_pieces:
                    self.manager.step("copy_to_destination", transfer_id, piece)
            self.manager.step("complete_destination", transfer_id)

    def test_cancel_at_every_event_boundary(self):
        events = copy_events("A0", 2)
        for cut in range(len(events) + 1):
            with self.subTest(after_events=cut):
                self.setup_manager()
                self.scheduler.run(events[:cut])
                transfer = self.manager.transfers["A0"]
                notified_before = transfer.notified
                phase_before = transfer.phase
                self.scheduler.run([RequestEvent("cancel", "A")])
                self.assertTrue(self.manager.requests["A"].cancelled)
                self.assertTrue(transfer.cancelled)
                self.assertEqual(self.manager.requests["A"].ready, set())
                for action in ("reserve", "start_destination", "publish"):
                    with self.assertRaises(ProtocolError):
                        self.manager.step(action, "A0")
                with self.assertRaises(ProtocolError):
                    self.manager.gather("A")
                with self.assertRaises(ProtocolError):
                    self.manager.append("A", KVCache(self.source.keys[:1], self.source.values[:1]))
                if phase_before in ("producer", "destination"):
                    with self.assertRaises(ProtocolError):
                        self.manager.release_request("A")
                    for pool, ref in ((self.manager.staging, transfer.staging),
                                      (self.manager.destination, transfer.destination)):
                        with self.assertRaises(ProtocolError):
                            pool.release(ref)
                    with self.assertRaises(ProtocolError):
                        self.manager.step("notify", "A0")
                self.drain("A0")
                if phase_before == "destination":
                    drained = self.manager.destination.read(transfer.destination)
                    assert_array_equal(drained.keys, self.source.keys)
                    assert_array_equal(drained.values, self.source.values)
                if phase_before in ("planned", "producer", "staged"):
                    self.assertEqual(transfer.destination_pieces, set())
                    self.assertEqual(transfer.phase, "discarded")
                if not transfer.staging_released:
                    self.manager.step("release_staging", "A0")
                self.scheduler.run([RequestEvent("retire", "A")])
                if not notified_before:
                    self.manager.step("notify", "A0")  # May arrive after retirement.
                outcomes = self.manager.poll()
                self.assertEqual(len(outcomes), 1)
                self.assertEqual(outcomes[0].status, "completed" if notified_before else "cancelled")
                self.assertEqual(self.manager.poll(), ())
                self.assertEqual(self.manager.staging.free_slots, [0, 1])
                self.assertEqual(self.manager.destination.free_slots, [0, 1, 2, 3])
                self.assertNotIn("A", self.manager.requests)

    def test_late_notification_after_both_pools_reused_changes_no_memory(self):
        self.setup_manager()
        self.scheduler.run(copy_events("A0", 2)[:6])  # One destination row written.
        old = self.manager.transfers["A0"]
        self.manager.cancel_request("A")
        # Independent B progresses while A still holds its reservations.
        self.manager.add_request("B", self.source, [1])
        self.manager.plan("B0", "B", 0)
        self.scheduler.run(copy_events("B0", 2))
        self.assertEqual(old.phase, "destination")
        assert_array_equal(self.manager.gather("B").keys, self.source.keys)
        self.drain("A0")
        self.manager.step("release_staging", "A0")
        self.manager.release_request("A")
        _, different = TinyAttention().prefill([7, 8])
        self.manager.add_request("C", different, [0])
        self.manager.plan("C0", "C", 0)
        self.scheduler.run(copy_events("C0", 2)[:2])
        new = self.manager.transfers["C0"]
        self.assertEqual(old.staging.slot, new.staging.slot)
        self.assertGreater(new.staging.generation, old.staging.generation)
        self.assertGreater(new.destination.generation, old.destination.generation)
        keys = self.manager.staging.keys.copy(), self.manager.destination.keys.copy()
        self.manager.step("notify", "A0")
        assert_array_equal(self.manager.staging.keys, keys[0])
        assert_array_equal(self.manager.destination.keys, keys[1])
        self.manager.staging.check(new.staging, "C0")
        self.manager.destination.check(new.destination, "C0")
        for action in ("notify", "publish", "release_staging", "complete_destination"):
            with self.assertRaises(ProtocolError):
                self.manager.step(action, "A0")
        self.scheduler.run(copy_events("C0", 2)[2:])
        assert_array_equal(self.manager.gather("C").values, different.values)

    def test_request_wide_cancel_drains_multiple_stages_and_unplanned_blocks(self):
        self.setup_manager()
        manager = TransferManager(destination_slots=4)
        _, source = TinyAttention().prefill([1, 2, 3, 4, 5, 6, 7])
        manager.add_request("A", source, [0, 1, 2, 3])
        for block in range(3):  # Fourth block has no transfer plan.
            manager.plan(f"A{block}", "A", block)
        scheduler = Scheduler(manager)
        scheduler.run(copy_events("A0", 2)[:2])
        scheduler.run(copy_events("A1", 2)[:6])
        manager.cancel_request("A")
        with self.assertRaises(ProtocolError):
            manager.plan("A3", "A", 3)
        self.manager = manager
        self.drain("A0")
        manager.step("release_staging", "A0")
        with self.assertRaises(ProtocolError):
            manager.release_request("A")
        self.drain("A1")
        manager.step("release_staging", "A1")
        manager.release_request("A")
        self.assertEqual(manager.destination.free_slots, [0, 1, 2, 3])

    def test_cancel_before_planning_and_duplicate_operations(self):
        manager = TransferManager()
        _, source = TinyAttention().prefill([1])
        manager.add_request("A", source, [0])
        manager.cancel_request("A")
        with self.assertRaises(ProtocolError):
            manager.cancel_request("A")
        manager.release_request("A")
        with self.assertRaises(ProtocolError):
            manager.release_request("A")
        with self.assertRaises(ProtocolError):
            manager.add_request("A", source, [0])
        self.assertEqual(manager.destination.free_slots, list(range(8)))

    def test_cancellation_demo_preserves_live_requests(self):
        result = run_cancellation_demo()
        self.assertNotIn("A", result.manager.requests)
        self.assertEqual(set(result.generated), {"B", "C"})
        self.assertTrue(all(len(tokens) == 2 for tokens in result.generated.values()))
        self.assertEqual(result.manager.staging.free_slots, [0, 1])
        self.assertFalse(result.manager.transfers["A0"].published)
        self.assertEqual(result.scheduler.trace, run_cancellation_demo().scheduler.trace)


if __name__ == "__main__":
    unittest.main()
