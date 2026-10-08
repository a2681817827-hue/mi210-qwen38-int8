"""CPU checks that prevent false GPU qualification and wrong TP geometry."""
import json
from pathlib import Path
import tempfile
import unittest

from launch import command
from optimization_campaign import gate_commands, inspect_gate


class CampaignTests(unittest.TestCase):
    def test_default_and_all_splits_have_boundary_and_replay_gates(self):
        jobs = list(gate_commands(Path("/tmp/plan")))
        self.assertEqual(len(jobs), 75)
        self.assertEqual(len({job["output"] for job in jobs}), 75)
        default = [job for job in jobs if job["label"] == "default"]
        self.assertEqual(len(default), 15)
        for job in default:
            self.assertNotIn("--target-splits", job["argv"])
        for job in jobs:
            if job["gate"] == "operator_numerics":
                argv = job["argv"]
                self.assertEqual(argv[argv.index("--segment-capacity") + 1], "64")

    def test_generic_fallback_and_failed_gate_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            job = {"output": str(output), "gate": "operator_numerics"}
            for row in ({"numeric": "PASS", "candidate_launches": 0},
                        {"numeric": "FAIL", "candidate_launches": 5}):
                output.write_text(json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    inspect_gate(job)
            output.write_text(json.dumps({"numeric": "PASS", "candidate_launches": 5}) + "\n")
            self.assertEqual(inspect_gate(job)["numeric"], "PASS")
            job["gate"] = "graph_replay"
            output.write_text(json.dumps({"ragged_numeric": "PASS", "graph_replay": "FAIL",
                                          "candidate_launches": 5}) + "\n")
            with self.assertRaises(ValueError):
                inspect_gate(job)

    def test_replicas_keep_native_attention_and_tp2_rejects_it(self):
        for profile in ("single", "replicas2"):
            env, argv = command(profile, attention_profile="cdna2")
            self.assertEqual(env["VLLM_G128_DECODE_CDNA2"], "1")
            self.assertEqual(env["VLLM_UA_TILE"], "32")
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "1")
        with self.assertRaises(ValueError):
            command("tp2", attention_profile="cdna2")
        env, _ = command("tp2")
        self.assertEqual(env["VLLM_G128_GLUON"], "0")

    def test_tp2_native_is_explicitly_experimental(self):
        env, argv = command("tp2", attention_profile="cdna2-tp2-experimental")
        self.assertEqual(env["VLLM_MI210_TP2_CDNA2"], "1")
        self.assertEqual(env["VLLM_G128_GLUON_MMA"], "bf16")
        self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "2")
        for profile in ("single", "replicas2"):
            with self.assertRaises(ValueError):
                command(profile, attention_profile="cdna2-tp2-experimental")


if __name__ == "__main__":
    unittest.main()
