from pathlib import Path
import tempfile
import unittest

from metrics_delta import acceptance, counters


def metrics(drafts, tokens, accepted):
    return f'''# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{{engine="0"}} {drafts}
vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {tokens}
vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted}
vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="0"}} 900
'''


class MetricsTests(unittest.TestCase):
    def delta(self, a, b):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before", Path(directory) / "after"
            before.write_text(a)
            after.write_text(b)
            return acceptance(before, after)

    def test_workload_uses_deltas_not_lifetime_totals_or_position_counters(self):
        result = self.delta(metrics(100, 1300, 300), metrics(110, 1430, 365))
        self.assertEqual(result["drafts"], 10)
        self.assertEqual(result["accepted_fraction"], 0.5)
        self.assertEqual(result["mean_accepted_per_draft"], 6.5)

    def test_reset_missing_and_impossible_counters_fail(self):
        with self.assertRaises(ValueError):
            self.delta(metrics(100, 1300, 300), metrics(10, 130, 30))
        with self.assertRaises(ValueError):
            counters("# no speculative decoding counters")
        with self.assertRaises(ValueError):
            self.delta(metrics(0, 0, 0), metrics(1, 13, 14))

    def test_idle_workload_has_no_acceptance_estimate(self):
        result = self.delta(metrics(10, 130, 65), metrics(10, 130, 65))
        self.assertIsNone(result["accepted_fraction"])


if __name__ == "__main__":
    unittest.main()
