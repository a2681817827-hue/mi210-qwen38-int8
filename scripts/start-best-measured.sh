#!/usr/bin/env bash
set -euo pipefail
export MI210_IMAGE=${MI210_IMAGE:-int8-vllm-mi210:query5-candidates-20261007}
export MI210_NAME=${MI210_NAME:-mi210-best-measured}
export SPEC_TOKENS=13 MI210_QUERY_TILE=5 MI210_KV_TILE=64 MI210_FLASH_SPLITS=32
export MI210_INT8_LM_HEAD=1 MI210_NATIVE_GDN=1 MI210_INT8_QK=1 MI210_TUNED_CK=1
export MAX_MODEL_LEN=131072 BATCHED_TOKENS=2048
export MI210_HOST=${MI210_HOST:-127.0.0.1}
export GPU_NODES=${GPU_NODES:-/dev/dri/renderD128}
export MI210_COMPILE_CACHE=${MI210_COMPILE_CACHE:-/data/cache/mi210-vllm/mi210-query5-screen-20261007-ck-query5-split32}
: "${MI210_API_KEY:?Set MI210_API_KEY before starting}"
exec bash "$(dirname "$0")/serve_mi210_long_context.sh" graph-c8 "${1:-}"
