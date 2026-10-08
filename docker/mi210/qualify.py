# SPDX-License-Identifier: Apache-2.0
"""Record static, runtime-selection and numeric MI210 qualification separately."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def qualify(root: Path, tier: int) -> dict:
    # Docker's source layer clears an inherited single-card restriction with
    # an empty value. HIP interprets that value as hiding every GPU.
    if os.environ.get("HIP_VISIBLE_DEVICES") == "":
        os.environ.pop("HIP_VISIBLE_DEVICES")
    import torch

    checks = {}
    required = {
        "long_context_reduction": ("csrc/rocm/attention.cu", "npar_loops > 16"),
        "free_kernel_guard": ("vllm/platforms/rocm.py", "_FREE_KERNEL_HEAD_MULTIPLE"),
        "singleton_stride_guard": ("vllm/model_executor/layers/utils.py", "contiguous_format"),
        "sharded_tp_guard": ("vllm/model_executor/model_loader/sharded_state_loader.py", "_saved_tp_size"),
        "int4_interleave": ("vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py", "on_gfx9"),
        "ck_int8_gate": ("vllm/_aiter_ops.py", "register_gfx90a_ck_int8_ops_once"),
        "spec_context_partition": ("vllm/v1/attention/ops/prefix_prefill.py", "_verify_partition_enabled"),
        "decode_partition": ("vllm/v1/attention/ops/chunked_prefill_paged_decode.py", "_seq_partition_enabled"),
        "grouped_kv_declared": ("vllm/v1/attention/backends/rocm_aiter_unified_attn.py", '"int8_block_g128"'),
    }
    for name, (path, marker) in required.items():
        checks[name] = marker in (root / path).read_text()
    import vllm
    import vllm._rocm_C
    import vllm._C_stable_libtorch
    import vllm._rust_tool_parser

    checks["source_path"] = Path(vllm.__file__).parent == root / "vllm"
    spec = importlib.util.find_spec("aiter")
    checks["source_aiter"] = spec is not None and spec.origin.startswith("/opt/aiter/")
    if not all(checks.values()):
        raise RuntimeError(f"Static checks failed: {checks}")
    from apply_aiter import patch_checkout

    patch_checkout(Path("/opt/aiter"), check=True)
    # Exercise the actual code generator without importing AITER's runtime.
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GPU_ARCHS": "gfx90a"}
        # Prove the gfx90a default works even without the image's override.
        env.pop("AITER_CK_DTYPES", None)
        subprocess.run([sys.executable, "/opt/aiter/csrc/ck_gemm_a8w8/gen_instances.py",
                        "--working_path", tmp, "--tune_file", "/nonexistent.csv"],
                       check=True, env=env, capture_output=True, text=True)
        combos = (Path(tmp) / "gemm_a8w8_built_combos.h").read_text()
        assert "AITER_BUILT_F8" not in combos
        assert "AITER_BUILT_I8_F32_B16" in combos
        assert "AITER_BUILT_I8_F32_F16" in combos
        checks["gfx90a_int8_codegen"] = True
    gpu_count = torch.cuda.device_count()
    record = {"result": "static_pass", "max_tier": 0, "checks": checks,
              "gpu_count": gpu_count, "hardware_validated": False,
              "torch": torch.__version__, "hip": torch.version.hip,
              "unvalidated": ["kernel_numerics", "serving_quality", "single_gpu_performance",
                              "two_gpu_performance", "spec_acceptance"]}
    if tier > 0:
        if not gpu_count:
            record["result"] = "blocked_no_gpu"
            return record
        props = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        assert all(p.gcnArchName.split(":")[0] == "gfx90a" for p in props)
        from vllm._aiter_ops import rocm_aiter_ops
        assert rocm_aiter_ops.is_int8_linear_enabled()
        assert not rocm_aiter_ops.is_linear_fp8_enabled()
        from vllm.platforms.rocm import use_rocm_custom_paged_attention as gate
        assert gate(torch.bfloat16, 256, 16, 6, 262144, 0, "auto", None, None)
        assert not gate(torch.bfloat16, 256, 544, 6, 262144, 0, "auto", None, None)
        record.update(result="runtime_gate_pass", max_tier=1,
                      arch=[p.gcnArchName for p in props])
    if tier > 1:
        # Same testing order as the repository ladder: micro first, then battery.
        subprocess.run([sys.executable, str(root / "scripts/test_int8_kv_micro.py")], check=True)
        from micro import run
        record["grouped_int8_cases"] = run()
        subprocess.run([sys.executable, str(root / "docker/mi210/verify_ck.py")], check=True)
        record.update(result="numeric_pass", max_tier=2, hardware_validated=True)
        record["unvalidated"].remove("kernel_numerics")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(Path(__file__).resolve().parents[2], args.tier)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["result"] == "blocked_no_gpu":
        sys.exit(2)
