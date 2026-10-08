"""Check actual patched CK routing without loading GPU libraries."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path('/opt/aiter/aiter/ops/gemm_op_a8w8.py')
        if not source.exists() or '# MI210_GATED_CK_DISPATCH' not in source.read_text():
            raise unittest.SkipTest('Run in the CK candidate image')
        tree = ast.parse(source.read_text())
        cls.function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'gemm_a8w8_CK')
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == '_MI210_CK_SELECTED'
                                  for t in n.targets))
        cls.selections = ast.literal_eval(assignment.value)

    def route(self, enabled=True, arch='gfx90a', cu=104, rows=14, dtype='bf16',
              scale='fp32', bias=None, split=None, contiguous=True, weight_device='cuda:0'):
        def tensor(shape, dtype, device='cuda:0'):
            return SimpleNamespace(shape=shape, dtype=dtype, device=device, is_cuda=True,
                                   is_contiguous=lambda: contiguous,
                                   numel=lambda: shape[0])
        calls = []
        ns = {'os': os, 'Tensor': object,
              'torch': SimpleNamespace(dtype=object, int8='int8', float32='fp32', bfloat16='bf16',
                                       empty=lambda *a, **k: 'OUT'),
              'dtypes': SimpleNamespace(bf16='bf16', fp8='fp8', i8='int8'),
              'get_gfx': lambda: arch, 'get_cu_num': lambda: cu,
              '_MI210_CK_SELECTED': self.selections, '_MI210_CK_SEEN': set(),
              'logger': SimpleNamespace(info=lambda *a: None),
              'AITER_CONFIGS': SimpleNamespace(AITER_CONFIG_GEMM_A8W8_FILE='unused'),
              'get_GEMM_config_with_quant_type': lambda *a: None,
              'gemm_a8w8_ck': lambda *a: calls.append(('default', a[-1])) or 'OUT',
              'gemm_a8w8_tune': lambda *a: calls.append(('tuned', a[-2], a[-1])) or 'OUT'}
        code = compile(ast.Module(body=[self.function], type_ignores=[]), 'actual-ck-wrapper', 'exec')
        with patch.dict(os.environ, {'AITER_MI210_TUNED_CK': '1' if enabled else '0', 'AITER_CK_STRICT': '0'}):
            exec(code, ns)
            ns['gemm_a8w8_CK'](tensor((rows, 17408), 'int8'), tensor((5120, 17408), 'int8', weight_device),
                               tensor((rows, 1), scale), tensor((5120, 1), scale), bias, dtype, split)
        return calls

    def test_selected_shape_uses_the_proven_kernel(self):
        self.assertEqual(self.route(), [('tuned', 34, 0)])

    def test_unmatched_hardware_dtype_layout_and_explicit_split_keep_default(self):
        for overrides in ({'enabled': False}, {'arch': 'gfx908'}, {'cu': 120}, {'rows': 8},
                          {'dtype': 'fp16'}, {'scale': 'fp16'}, {'bias': object()},
                          {'split': 0}, {'contiguous': False}, {'weight_device': 'cuda:1'}):
            with self.subTest(overrides=overrides):
                self.assertEqual(self.route(**overrides), [('default', 0)])


if __name__ == '__main__':
    unittest.main()
