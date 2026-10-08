#!/usr/bin/env bash
# Measured 48K C1 median 107.7075 tok/s; 110/120 goals are still unmet.
set -euo pipefail
: "${MI210_API_KEY:?Set MI210_API_KEY for local serving}"
export MI210_IMAGE=${MI210_IMAGE:-int8-vllm-mi210:draft-window-measured-20261008}
export MI210_NAME=${MI210_NAME:-mi210-local-best}
export MI210_HOST=${MI210_HOST:-127.0.0.1}
export PORT=${PORT:-8080}
export MI210_COMPILE_CACHE=${MI210_COMPILE_CACHE:-/data/cache/mi210-vllm/mi210-wave-draft-wired-20261007-draft-window}
export MI210_TRITON_CACHE=${MI210_TRITON_CACHE:-/data/cache/mi210-vllm/triton-cdna2}
export MAX_MODEL_LEN=131072 BATCHED_TOKENS=2048 SPEC_METHOD=dflash SPEC_TOKENS=13
export GPU_NODES=${GPU_NODES:-/dev/dri/renderD128}
export MI210_UA_3D_MAXQ=16
export MI210_QUERY_TILE=5 MI210_KV_TILE=64 MI210_FLASH_SPLITS=32
export MI210_INT8_LM_HEAD=1 MI210_NATIVE_GDN=1 MI210_INT8_QK=1 MI210_TUNED_CK=1
export MI210_DRAFT_GLUON=1 MI210_DRAFT_SPLITS=16 MI210_TARGET_WAVE_LAYOUT=0
export MI210_REGISTER_QUANT_ALIAS=0 MI210_EXTENDED_CK=0
export MI210_QKPV_INTERLEAVE=0 MI210_REDUCTION_POLICY=0
export MI210_GDN_TINY_BA=0 MI210_GDN_ZVIEW=0 MI210_NATIVE_GDN_RAW=0 MI210_PROFILE=0
exec bash "$(dirname "$0")/serve_mi210_long_context.sh" graph-c8 "${1:-}"
