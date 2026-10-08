"""Reject parent-event double counting and account for overlapping streams."""

import json
from pathlib import Path
import tempfile
import unittest

from profile_audit import audit


class ProfileAuditTests(unittest.TestCase):
    def run_audit(self, events):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            return audit(path)

    def test_parent_annotations_and_cpu_events_are_excluded(self):
        events = [{"ph": "X", "cat": cat, "name": "event", "ts": 0,
                   "dur": duration} for cat, duration in (
                       ("gpu_user_annotation", 1000), ("cpu_op", 1000),
                       ("kernel", 20))]
        result = self.run_audit(events)
        self.assertEqual(result["summed_kernel_ms"], 0.02)
        self.assertEqual(result["kernels"][0]["calls"], 1)

    def test_busy_time_merges_overlapping_stream_intervals(self):
        events = [{"ph": "X", "cat": "kernel", "name": "event",
                   "ts": start, "dur": duration, "args": {"device": 0}}
                  for start, duration in ((0, 100), (10, 20), (110, 10))]
        result = self.run_audit(events)
        self.assertEqual(result["summed_kernel_ms"], 0.13)
        self.assertEqual(result["devices"]["0"]["busy_union_ms"], 0.11)
        self.assertEqual(result["devices"]["0"]["first_to_last_kernel_ms"], 0.12)

    def test_empty_and_negative_timings_do_not_produce_a_report(self):
        for events in ([], [{"ph": "X", "cat": "kernel", "name": "bad",
                            "ts": 0, "dur": -1}]):
            with self.assertRaises(ValueError):
                self.run_audit(events)


if __name__ == "__main__":
    unittest.main()
