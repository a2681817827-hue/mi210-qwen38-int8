"""CPU dispatch checks for the separately gated MI210 output projection."""
import ast
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class HeadDispatchTests(unittest.TestCase):
    def select(self, enabled=False, arch="gfx90a", head=True, bias=None):
        tree = ast.parse((ROOT / "vllm/model_executor/layers/vocab_parallel_embedding.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "UnquantizedEmbeddingMethod")
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                        and n.name == "process_weights_after_loading")
        calls = []
        layer = types.SimpleNamespace(weight=types.SimpleNamespace(is_floating_point=lambda: True),
                                      _gfx908_skip_int8=head, bias=bias)
        envs = types.SimpleNamespace(VLLM_GFX908_INT8_EMBEDDING=False,
                                    VLLM_GFX908_INT8_LM_HEAD=True)
        values = {"torch": types.SimpleNamespace(nn=types.SimpleNamespace(Module=object)),
                  "current_platform": types.SimpleNamespace(is_cpu=lambda: False,
                                                            is_rocm=lambda: True),
                  "os": os, "envs": envs,
                  "_quantize_lm_head_w8a8_": lambda layer: calls.append("head"),
                  "bind_rocm_unquantized_gemm_gfx908": lambda layer: calls.append("float")}
        module = types.SimpleNamespace(on_gfx908=lambda: arch == "gfx908",
                                       on_gfx90a=lambda: arch == "gfx90a")
        code = compile(ast.Module(body=[function], type_ignores=[]), "head-dispatch", "exec")
        with patch.dict(sys.modules, {"vllm.platforms.rocm": module}), patch.dict(
                os.environ, {"VLLM_MI210_INT8_LM_HEAD": "1" if enabled else "0"}):
            exec(code, values)
            values["process_weights_after_loading"](None, layer)
        return calls, getattr(layer, "_mi210_w8a8_head", False)

    def test_mi210_requires_opt_in_and_head_without_bias(self):
        self.assertEqual(self.select(True), (["head"], True))
        for changes in ({"enabled": False}, {"head": False}, {"bias": object()},
                        {"arch": "gfx942"}):
            self.assertEqual(self.select(**changes), ([], False))

    def test_existing_gfx908_head_contract_is_preserved(self):
        self.assertEqual(self.select(arch="gfx908"), (["head"], False))


if __name__ == "__main__":
    unittest.main()
