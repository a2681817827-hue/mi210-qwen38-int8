"""Generate an isolated value-partitioned GDN candidate from the gated source."""


def partition_source(body, raw_output=False):
    def replace(old, new):
        nonlocal body
        if body.count(old) != 1:
            raise ValueError('Value-partition source drift: ' + old)
        body = body.replace(old, new)

    replace('  __shared__ StateT shared_state[2][kChunkV][kDimK];\n', '')
    replace('  __shared__ float shared_out[kMaxMtpTokens][kDimV];\n', '')
    start = body.index('  copy_state_chunk<StateT, kChunkV, kDimK>(&shared_state[0][0][0]')
    end = body.index('\n\n', start)
    body = body[:start] + body[end:]
    start = body.index('#pragma unroll 1\n  for (int chunk = 0; chunk < kNumChunks; ++chunk) {')
    end = body.index('    float h[kRowsPerWarp][4];', start)
    body = body[:start] + '  {\n    const int chunk = blockIdx.z;\n' + body[end:]
    replace('load_state4(&shared_state[chunk & 1][rows[row]][k_base])',
            'load_state4(source_state + (chunk * kChunkV + rows[row]) * kDimK + k_base)')
    for row, component in enumerate(('dot_hq_01.x', 'dot_hq_01.y', 'dot_hq_23.x', 'dot_hq_23.y')):
        replace(f'shared_out[t][chunk * kChunkV + rows[{row}]] =\n'
                f'            __bfloat162float(__float2bfloat16({component}));',
                f'out[(static_cast<int64_t>(bos + t) * HV + value_head) * kDimV +\n'
                f'            chunk * kChunkV + rows[{row}]] = __float2bfloat16({component});')
    # The original normalizer spans all 128 values in one block. Independent
    # 32-value state tiles instead write the reference BF16 raw output, then
    # normalize it in place on the same stream after all state tiles finish.
    start = body.index('  __syncthreads();\n\n  for (int t = warp;', body.index('const int destination_slot'))
    end = body.index('\ntemplate <typename StateT, int ValueHeadsPerKeyHead>\nvoid launch_', start)
    body = body[:start] + '}\n' + body[end:]
    replace('linear < num_tokens * kDimV', 'linear < num_tokens * kChunkV')
    replace('const int token = bos + linear / kDimV;', 'const int token = bos + linear / kChunkV;')
    replace('const int value = linear % kDimV;', 'const int value = blockIdx.z * kChunkV + linear % kChunkV;')
    replace('const dim3 grid(num_requests, num_value_heads);',
            'const dim3 grid(num_requests, num_value_heads, kNumChunks);')
    if raw_output:
        return body
    normalizer = r'''
__global__ void gdn_partition_norm_kernel(
    bf16_t* out, const bf16_t* output_gate, const void* norm_weight,
    const int* state_indices, const int* cu_seqlens,
    const int* num_accepted_tokens, int num_requests, int state_width,
    int tokens, int HV, int64_t gate_stride, bool weight_bf16, float eps) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= tokens * HV) return;
  const int token = row / HV;
  const int head = row % HV;
  bool valid = false;
  for (int request = 0; request < num_requests; ++request) {
    if (token >= cu_seqlens[request] && token < cu_seqlens[request + 1]) {
      const int accepted = num_accepted_tokens[request];
      valid = accepted > 0 && accepted <= state_width &&
          state_indices[request * state_width + accepted - 1] > 0;
    }
  }
  float values[4];
  float square = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = valid ? __bfloat162float(out[static_cast<int64_t>(row) * 128 + lane + i * 32]) : 0.0f;
    square += values[i] * values[i];
  }
  const float rstd = rsqrtf(warp_reduce_sum(square) / 128.0f + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int value = lane + i * 32;
    const float weight = weight_bf16
        ? __bfloat162float(static_cast<const bf16_t*>(norm_weight)[value])
        : static_cast<const float*>(norm_weight)[value];
    const float gate = valid ? silu_fast(__bfloat162float(
        output_gate[static_cast<int64_t>(token) * gate_stride + head * 128 + value])) : 0.0f;
    out[static_cast<int64_t>(row) * 128 + value] =
        __float2bfloat16(values[i] * rstd * weight * gate);
  }
}

'''
    replace('template <typename StateT, int ValueHeadsPerKeyHead>\nvoid launch_',
            normalizer + 'template <typename StateT, int ValueHeadsPerKeyHead>\nvoid launch_')
    replace('  const hipError_t error = hipGetLastError();', r'''
  const hipError_t recurrence_error = hipGetLastError();
  STD_TORCH_CHECK(recurrence_error == hipSuccess,
                  "GDN partition recurrence launch failed: ", hipGetErrorString(recurrence_error));
  const int total_tokens = static_cast<int>(mixed_qkv.size(0));
  gdn_partition_norm_kernel<<<(total_tokens * num_value_heads + 3) / 4, 128, 0, stream>>>(
      static_cast<bf16_t*>(out.data_ptr()), output_gate, norm_weight.data_ptr(),
      static_cast<const int*>(state_indices.data_ptr()),
      static_cast<const int*>(cu_seqlens.data_ptr()),
      static_cast<const int*>(num_accepted_tokens.data_ptr()), num_requests,
      static_cast<int>(state_indices.size(1)), total_tokens, num_value_heads,
      strides.gate_row, norm_weight.scalar_type() == ScalarType::BFloat16,
      static_cast<float>(norm_eps));
  const hipError_t error = hipGetLastError();''')
    return body
