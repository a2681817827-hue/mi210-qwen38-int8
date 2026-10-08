"""CPU checks for opt-in split selection; GPU numerics remain a separate gate."""
import ast
import importlib.util
import os
import shlex
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = ROOT if (ROOT / "vllm").is_dir() else ROOT / "operators/int8-vllm"
OPS = VLLM_ROOT / "vllm/v1/attention/ops"
SPEC = importlib.util.spec_from_file_location("split_tuning", OPS / "mi210_split_tuning.py")
TUNING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TUNING)


class SplitTests(unittest.TestCase):
    def test_query_tile_values(self):
        for value in ("5", "10", "16"):
            self.assertEqual(TUNING.validate_mi210_query_tile(value), int(value))
        for value in ("0", "8", "32", "bad"):
            with self.assertRaises(ValueError):
                TUNING.validate_mi210_query_tile(value)

    def test_capacity_and_values(self):
        for splits in (8, 16, 32, 64):
            self.assertEqual(TUNING.validate_mi210_splits(str(splits), 64), splits)
        for value, capacity in (("64", 32), ("16", 8), ("0", 64),
                                ("128", 128), ("bad", 64)):
            with self.assertRaises(ValueError):
                TUNING.validate_mi210_splits(value, capacity)

    def selection(self, env=None, arch=True, **changes):
        tree = ast.parse((OPS / "triton_unified_attention.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "unified_attention")
        start = next(i for i, node in enumerate(function.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "mi210_splits"
                             for t in node.targets))
        code = compile(ast.Module(body=function.body[start:start + 2], type_ignores=[]),
                       "split-selection", "exec")
        modules = {"vllm.platforms.rocm": types.SimpleNamespace(on_gfx90a=lambda: arch),
                   "vllm.v1.attention.ops.mi210_split_tuning": TUNING}
        values = dict(os=types.SimpleNamespace(environ=env or {}),
                      actual_num_splits=32, use_3d=True, use_g8=True,
                      kv_quant_mode=1, KVQuantMode=types.SimpleNamespace(INT8_BLOCK_G128=1),
                      head_size=256, num_kv_heads=4, num_queries_per_kv=6,
                      use_causal=True, sliding_window_val=0,
                      current_platform=types.SimpleNamespace(is_rocm=lambda: True),
                      softmax_segm_output=types.SimpleNamespace(shape=(256, 24, 64, 256)),
                      softmax_segm_max=types.SimpleNamespace(shape=(256, 24, 64)),
                      softmax_segm_expsum=types.SimpleNamespace(shape=(256, 24, 64)))
        values.update(changes)
        with patch.dict(sys.modules, modules):
            exec(code, values)
        return values["actual_num_splits"]

    def test_default_and_target_override(self):
        self.assertEqual(self.selection(), 32)
        env = {"VLLM_MI210_FLASH_SPLITS": "16", "VLLM_G128_DECODE_CDNA2": "1"}
        self.assertEqual(self.selection(env), 16)
        for changes in ({"head_size": 128}, {"use_3d": False},
                        {"use_causal": False}, {"sliding_window_val": 2048},
                        {"num_kv_heads": 1}, {"kv_quant_mode": 2}):
            self.assertEqual(self.selection(env, **changes), 32)
        self.assertEqual(self.selection(env, arch=False), 32)

    def test_smallest_buffer_capacity(self):
        env = {"VLLM_MI210_FLASH_SPLITS": "64", "VLLM_G128_DECODE_CDNA2": "1"}
        with self.assertRaises(ValueError):
            self.selection(env, softmax_segm_max=types.SimpleNamespace(shape=(256, 24, 32)))

    def test_tp2_split_requires_explicit_opt_in_and_supported_arch(self):
        env = {"VLLM_MI210_FLASH_SPLITS": "64", "VLLM_G128_DECODE_CDNA2": "1"}
        self.assertEqual(self.selection(env, num_kv_heads=2), 32)
        env["VLLM_MI210_TP2_CDNA2"] = "1"
        self.assertEqual(self.selection(env, num_kv_heads=2), 64)
        self.assertEqual(self.selection(env, arch=False, num_kv_heads=2), 32)
        self.assertEqual(self.selection(env, num_kv_heads=3), 32)
        self.assertEqual(self.selection(env, num_kv_heads=2, sliding_window_val=2048), 32)

    def test_launcher(self):
        env = dict(os.environ, MI210_FLASH_SPLITS="16", MI210_API_KEY="test-only",
                   GPU_NODES="/dev/dri/renderD128")
        command = ["bash", str(ROOT / "scripts/serve_mi210_long_context.sh"),
                   "graph-c8", "--dry-run"]
        result = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
        tokens = shlex.split(result.stdout)
        self.assertIn("VLLM_MI210_FLASH_SPLITS=16", tokens)
        env["MI210_FLASH_SPLITS"] = "12"
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
