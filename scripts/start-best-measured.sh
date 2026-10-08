#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/start-mi210-saved-20261008.sh" "${1:-}"
