#!/usr/bin/env bash
# Reproduce the user's vision baseline; vary only eager vs decode graphs.
set -euo pipefail
arm=${1:-baseline}
spec_tokens=${SPEC_TOKENS:-13}
spec_method=${SPEC_METHOD:-dflash}
case "$spec_method" in
  dflash) speculative_config="{\"method\":\"dflash\",\"model\":\"/models/dflash2-ptqr-r1\",\"num_speculative_tokens\":$spec_tokens,\"kv_cache_dtype\":\"int8_block_g128\"}" ;;
  mtp) speculative_config="{\"method\":\"mtp\",\"model\":\"/models/Qwen3.8-27B-PTQR-R10S60-mtp-view\",\"num_speculative_tokens\":$spec_tokens,\"kv_cache_dtype\":\"int8_block_g128\"}" ;;
  *) echo 'SPEC_METHOD must be dflash or mtp' >&2; exit 2 ;;
esac
[[ $spec_tokens =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || {
  echo 'SPEC_TOKENS must be in [1, 32]' >&2; exit 2;
}
batched_tokens=${BATCHED_TOKENS:-2048}
[[ $batched_tokens =~ ^[1-9][0-9]*$ ]] || {
  echo 'BATCHED_TOKENS must be a positive integer' >&2; exit 2;
}
max_model_len=${MAX_MODEL_LEN:-32768}
[[ $max_model_len =~ ^[1-9][0-9]*$ ]] && ((max_model_len <= 131072)) || {
  echo 'MAX_MODEL_LEN must be in [1, 131072]' >&2; exit 2;
}
concurrency=1
case "$arm" in
  baseline) execution=(--enforce-eager) ;;
  graph) execution=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","pass_config":{"fuse_allreduce_rms":false}}') ;;
  c8) concurrency=8; execution=(--enforce-eager) ;;
  graph-c8) concurrency=8; execution=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","pass_config":{"fuse_allreduce_rms":false}}') ;;
  graph-only) execution=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  graph-only-c8) concurrency=8; execution=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  *) echo 'Usage: bash scripts/serve_mi210_32k_ab.sh {baseline|graph|c8|graph-c8|graph-only|graph-only-c8} [--dry-run]' >&2; exit 2 ;;
esac
cache_args=()
experiment_args=()
for variable in MI210_GDN_TINY_BA MI210_GDN_ZVIEW MI210_NATIVE_GDN MI210_NATIVE_GDN_RAW MI210_INT8_QK; do
  value=${!variable:-}
  if [[ -n $value ]]; then
    [[ $value =~ ^(0|1)$ ]] || { echo "$variable must be 0 or 1" >&2; exit 2; }
    case "$variable" in
      MI210_GDN_TINY_BA) experiment_args+=(-e "VLLM_MI210_GDN_TINY_BA=$value") ;;
      MI210_GDN_ZVIEW) experiment_args+=(-e "VLLM_GDN_ZVIEW=$value") ;;
      MI210_NATIVE_GDN) experiment_args+=(-e "VLLM_MI210_NATIVE_GDN=$value") ;;
      MI210_NATIVE_GDN_RAW) experiment_args+=(-e "VLLM_MI210_NATIVE_GDN_RAW=$value") ;;
      MI210_INT8_QK) experiment_args+=(-e "VLLM_MI210_INT8_QK=$value") ;;
    esac
  fi
done
if [[ -n ${MI210_TUNED_CK:-} ]]; then
  [[ $MI210_TUNED_CK =~ ^(0|1)$ ]] || { echo 'MI210_TUNED_CK must be 0 or 1' >&2; exit 2; }
  experiment_args+=(-e "AITER_MI210_TUNED_CK=$MI210_TUNED_CK")
fi
profile_args=()
if [[ ${PREFIX_CACHING:-1} == 0 ]]; then
  profile_args+=(--no-enable-prefix-caching)
elif [[ ${PREFIX_CACHING:-1} != 1 ]]; then
  echo 'PREFIX_CACHING must be 0 or 1' >&2; exit 2
fi
if [[ ${MI210_PROFILE:-0} == 1 ]]; then
  profile_delay=${MI210_PROFILE_DELAY:-10}
  [[ $profile_delay =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo 'MI210_PROFILE_DELAY must be a nonnegative integer' >&2; exit 2;
  }
  profile_args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/tmp/mi210-profile\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":$profile_delay,\"max_iterations\":20}")
fi
if [[ -n ${MI210_UA_3D_MAXQ:-} ]]; then
  [[ $MI210_UA_3D_MAXQ =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || {
    echo 'MI210_UA_3D_MAXQ must be in [1, 32]' >&2; exit 2;
  }
  experiment_args+=(-e "VLLM_UA_3D_MAXQ=$MI210_UA_3D_MAXQ")
fi
if [[ -n ${MI210_FLASH_SPLITS:-} ]]; then
  [[ $MI210_FLASH_SPLITS =~ ^(8|16|32|64)$ ]] || {
    echo 'MI210_FLASH_SPLITS must be 8, 16, 32 or 64' >&2; exit 2;
  }
  experiment_args+=(-e "VLLM_MI210_FLASH_SPLITS=$MI210_FLASH_SPLITS")
fi
if [[ -n ${MI210_KV_TILE:-} ]]; then
  [[ $MI210_KV_TILE =~ ^(32|64)$ ]] || {
    echo 'MI210_KV_TILE must be 32 or 64' >&2; exit 2;
  }
  experiment_args+=(-e "VLLM_G128_DECODE_CDNA2_TILE=$MI210_KV_TILE")
fi
if [[ -n ${MI210_QUERY_TILE:-} ]]; then
  [[ $MI210_QUERY_TILE =~ ^(5|10|16)$ ]] || {
    echo 'MI210_QUERY_TILE must be 5, 10 or 16' >&2; exit 2;
  }
  experiment_args+=(-e "VLLM_MI210_G128_QUERY_TILE=$MI210_QUERY_TILE")
fi
if [[ -n ${MI210_INT8_LM_HEAD:-} ]]; then
  [[ $MI210_INT8_LM_HEAD =~ ^(0|1)$ ]] || {
    echo 'MI210_INT8_LM_HEAD must be 0 or 1' >&2; exit 2;
  }
  experiment_args+=(-e "VLLM_MI210_INT8_LM_HEAD=$MI210_INT8_LM_HEAD")
fi
if [[ -n ${MI210_JIT_CACHE:-} ]]; then
  cache_args=(-v "$MI210_JIT_CACHE:/mi210-jit" -e AITER_JIT_DIR=/mi210-jit)
fi
if [[ -n ${MI210_TRITON_CACHE:-} ]]; then
  cache_args+=(-v "$MI210_TRITON_CACHE:/root/.triton")
fi
if [[ -n ${MI210_COMPILE_CACHE:-} ]]; then
  cache_args+=(-v "$MI210_COMPILE_CACHE:/root/.cache/vllm")
fi
source "$(dirname "${BASH_SOURCE[0]}")/mi210_gpu_nodes.sh"
mi210_collect_nodes
gpu_index=${MI210_GPU:-0}
[[ $gpu_index =~ ^(0|[1-9][0-9]*)$ ]] && ((gpu_index < ${#mi210_nodes[@]})) || {
  echo 'MI210_GPU must index an available gfx90a render node (starting at 0)' >&2; exit 2;
}
gpu_node=${mi210_nodes[$gpu_index]}
cmd=(docker run -d --name "${MI210_NAME:-int8-vllm-mi210-32k-$arm}"
  --restart unless-stopped --network host --device=/dev/kfd --device="$gpu_node"
  --ipc=host -v "${MODEL_ROOT:-/data/models}:/models:ro"
  -v "${HF_CACHE_ROOT:-/data/cache/huggingface}:/root/.cache/huggingface"
  -e HIP_VISIBLE_DEVICES=0 -e GPU_ARCHS=gfx90a
  -e AITER_USE_SYSTEM_TRITON=1 -e VLLM_ROCM_USE_AITER=1
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 -e VLLM_GFX908_INT8_LM_HEAD=1
  -e VLLM_GFX908_ACT_QUANT=round -e VLLM_DISABLED_KERNELS=TritonW8A16LinearKernel
  "${cache_args[@]}"
  "${experiment_args[@]}"
  --entrypoint vllm "${MI210_IMAGE:-int8-vllm-mi210:32k-vision-baseline}"
  serve /models/Qwen3.8-27B-PTQR-R10S60 --served-model-name qwen-ptqr
  --host "${MI210_HOST:-10.168.1.4}" --port "${PORT:-8080}"
  --api-key "${MI210_API_KEY:-mi210-local-api}"
  --tensor-parallel-size 1 --dtype bfloat16 --kv-cache-dtype int8_block_g128
  --mamba-ssm-cache-dtype float32 --max-model-len "$max_model_len" --max-num-seqs "$concurrency"
  --max-num-batched-tokens "$batched_tokens" --gpu-memory-utilization 0.90
  --limit-mm-per-prompt '{"image":1,"video":0}' "${execution[@]}"
  "${profile_args[@]}"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --speculative-config "$speculative_config")
if [[ ${2:-} == --dry-run ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  [[ -c /dev/kfd ]] || { echo '/dev/kfd is unavailable' >&2; exit 1; }
  [[ -c $gpu_node ]] || { echo "Not a GPU device node: $gpu_node" >&2; exit 1; }
  "${cmd[@]}"
fi
