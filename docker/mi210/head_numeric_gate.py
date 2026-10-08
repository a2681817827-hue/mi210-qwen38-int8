"""MI210 head arithmetic gate; random tensors do not establish model fidelity."""
import argparse
import json
import os
from pathlib import Path

import torch

import vllm._aiter_ops  # Registers the same CK op used by the serving model.
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, _apply_w8a8_head,
)
from vllm.model_executor.kernels.linear.mixed_precision.act_quant_rn import pertoken_quant_rn


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ["VLLM_MI210_INT8_LM_HEAD"] = "1"
torch.manual_seed(31)
layer = torch.nn.Module()
layer.weight = torch.nn.Parameter(torch.randn(8192, 5120, device="cuda", dtype=torch.bfloat16) * 0.015,
                                  requires_grad=False)
layer.bias = None
layer._gfx908_skip_int8 = True
UnquantizedEmbeddingMethod().process_weights_after_loading(layer)
assert layer._mi210_w8a8_head
results = []
for rows in (1, 7, 10, 14):
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    actual = _apply_w8a8_head(layer, x).float()
    xq, xs = pertoken_quant_rn(x)
    expected = ((xq.float() @ layer._ck_q.float().T) * xs * layer._ck_s.T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected.float(), atol=0.016, rtol=0.005)
    original = x.float() @ layer.weight.float().T
    relative = (torch.linalg.vector_norm(actual - original) /
                torch.linalg.vector_norm(original)).item()
    assert relative < 0.02, relative
    results.append({"rows": rows, "numeric": "PASS", "relative_l2_to_float_weight": relative,
                    "output_dtype": "bfloat16"})
report = {"operator_numerics": "PASS", "random_tensor_cases": results,
          "model_fidelity": "NOT_ESTABLISHED_BY_THIS_GATE"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
