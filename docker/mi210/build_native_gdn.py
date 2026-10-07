"""Build an isolated gfx90a stable-ABI GDN experiment; no GPU is used at build time."""
import hashlib
import json
import os
from pathlib import Path

from torch.utils.cpp_extension import load


root = Path('/opt/int8-vllm')
build = root / 'docker/mi210/native-gdn-build'
build.mkdir(exist_ok=True)
source = root / 'csrc/libtorch_stable/gdn/fused_gdn_decode_kernel_rocm.cu'
body = source.read_text()
changes = {
    '#include "../torch_utils.h"': '#include "mi210_torch_utils.h"',
    'constexpr int kMaxMtpTokens = 8;': 'constexpr int kMaxMtpTokens = 16;',
    'cudaStream_t stream': 'hipStream_t stream',
    # The kernel uses logical 32-thread warps on a wave64 device. A token
    # staging branch can activate only half a wave; HIP's full-mask sync
    # shuffle asserts that every masked lane participates and traps there.
    # Register-only XOR reductions need no barrier. Explicit width 32 keeps
    # the two logical warps separate, including in a partial active wave.
    '__shfl_xor_sync(GDN_WARP_MASK, value, offset)':
        '__shfl_xor(value, offset, 32)',
    '__shfl_xor_sync(GDN_WARP_MASK, x, offset)':
        '__shfl_xor(x, offset, 32)',
    '__shfl_xor_sync(GDN_WARP_MASK, y, offset)':
        '__shfl_xor(y, offset, 32)',
}
# The reference recurrent update writes its raw output to a BF16 tensor
# before RMSNorm. Preserve that rounding boundary inside the fused kernel;
# the recurrent state and the normalization arithmetic remain FP32.
for component in ('dot_hq_01.x', 'dot_hq_01.y', 'dot_hq_23.x', 'dot_hq_23.y'):
    changes['=\n            ' + component + ';'] = (
        '=\n            __bfloat162float(__float2bfloat16(' + component + '));')
for old, new in changes.items():
    if body.count(old) != 1:
        raise ValueError('Native GDN source drift: ' + old)
    body = body.replace(old, new)
token_loop = 'if (warp < num_tokens) {\n    const int t = warp;'
if body.count(token_loop) != 2:
    raise ValueError('Native GDN token staging source drift')
body = body.replace(token_loop, 'for (int t = warp; t < num_tokens; t += kWarps) {')
partitioned = os.environ.get('MI210_NATIVE_GDN_VALUE_PARTITIONS') == '1'
raw_output = os.environ.get('MI210_NATIVE_GDN_RAW_OUTPUT') == '1'
if raw_output and not partitioned:
    raise ValueError('Raw output requires the value-partitioned experiment')
if partitioned:
    from partition_native_gdn import partition_source
    body = partition_source(body, raw_output=raw_output)
kernel = build / 'mi210_gdn.cu'
kernel.write_text(body)
(build / 'mi210_torch_utils.h').write_text('''#pragma once
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>
#include <hip/hip_runtime.h>
inline hipStream_t get_current_cuda_stream(int32_t index) {
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(index, &stream));
  return reinterpret_cast<hipStream_t>(stream);
}
''')
binding = build / 'bindings.cpp'
binding.write_text('''#include "ops.h"
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("fused_gdn_decode_post_conv_mtp(Tensor mixed_qkv, Tensor a, Tensor b, "
          "Tensor A_log, Tensor dt_bias, Tensor state_indices, Tensor cu_seqlens, "
          "Tensor num_accepted_tokens, Tensor! state, Tensor output_gate, "
          "Tensor norm_weight, Tensor! out, float scale, float norm_eps=1e-5) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("fused_gdn_decode_post_conv_mtp", TORCH_BOX(&fused_gdn_decode_post_conv_mtp));
}
''')
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx90a'
os.environ['MAX_JOBS'] = '2'
flags = ['-O3', '-std=c++20', '-DUSE_ROCM', '-DVLLM_ENABLE_FUSED_GDN_DECODE=1',
         '-DTORCH_TARGET_VERSION=0x020B000000000000ULL']
if raw_output:
    # Private schema preserves the standard fused operator's normalized
    # output semantics while allowing the model's compiled norm/quant path.
    flags.append('-Dfused_gdn_decode_post_conv_mtp=mi210_gdn_decode_post_conv_raw_mtp')
    binding.write_text(binding.read_text().replace(
        '"fused_gdn_decode_post_conv_mtp', '"mi210_gdn_decode_post_conv_raw_mtp'))
load(name='mi210_native_gdn', sources=[str(kernel), str(binding)],
     extra_cflags=flags, extra_cuda_cflags=flags,
     extra_include_paths=[str(root / 'csrc/libtorch_stable'), str(build)],
     build_directory=str(build), is_python_module=False, verbose=True)
(build / 'provenance.json').write_text(json.dumps({
    'source': str(source.relative_to(root)),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'generated_sha256': hashlib.sha256(kernel.read_bytes()).hexdigest(),
    'arch': 'gfx90a', 'threads': 256, 'max_verify_tokens': 16,
    'variant': ('value_partitioned_raw' if raw_output else
                'value_partitioned_with_inplace_norm' if partitioned else 'fused_all_values'),
    'token_staging': 'each logical warp iterates over tokens in strides of kWarps',
    'precision': 'BF16 inputs/output, FP32 recurrent state remains supported',
    'numeric_validation': 'NOT_RUN', 'changes': changes,
}, indent=2) + '\n')
