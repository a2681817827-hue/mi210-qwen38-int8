#!/usr/bin/env bash
# Measured MI210 TP1 CDNA2 G128 attention; keep target/draft precision intact.
set -euo pipefail
export MI210_IMAGE=${MI210_IMAGE:-int8-vllm-mi210:long-context-cdna2}
export MI210_NAME=${MI210_NAME:-int8-vllm-mi210-ptqr}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
export MI210_UA_3D_MAXQ=${MI210_UA_3D_MAXQ:-16}
export MI210_COMPILE_CACHE=${MI210_COMPILE_CACHE:-/data/cache/mi210-vllm/long-context-cdna2}
export MI210_TRITON_CACHE=${MI210_TRITON_CACHE:-/data/cache/mi210-vllm/triton-cdna2}
exec bash "$(dirname "$0")/serve_mi210_32k_ab.sh" "${1:-graph-c8}" "${2:-}"
