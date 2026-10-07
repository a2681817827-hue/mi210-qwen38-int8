#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Device-node isolation adapted from davetha/mi210-vllm gpu-nodes.sh.
# HIP visibility alone does not isolate amdsmi's architecture detection.
# Populate mi210_nodes; callers choose TP1, replicas or TP2 explicitly.
mi210_collect_nodes() {
  mi210_nodes=()
  local node id vendor
  if [[ -n ${GPU_NODES:-} ]]; then
    read -r -a mi210_nodes <<< "$GPU_NODES"
    for node in "${mi210_nodes[@]}"; do
      [[ $node == /*/renderD* && ${node##*/} =~ ^renderD[0-9]+$ ]] || {
        echo "GPU_NODES must contain individual render devices: $node" >&2
        return 1
      }
    done
  else
    for node in "${MI210_DRI_ROOT:-/dev/dri}"/renderD*; do
      [[ -e $node ]] || continue
      vendor=$(cat "${MI210_SYSFS_ROOT:-/sys/class/drm}/${node##*/}/device/vendor" 2>/dev/null) || continue
      [[ $vendor == 0x1002 ]] || continue
      id=$(cat "${MI210_SYSFS_ROOT:-/sys/class/drm}/${node##*/}/device/device" 2>/dev/null) || continue
      case "$id" in 0x7408|0x740c|0x740f) mi210_nodes+=("$node") ;; esac
    done
  fi
  if (( ${#mi210_nodes[@]} == 0 )); then
    echo 'No MI210/gfx90a render device. Check host amdgpu initialization or set GPU_NODES.' >&2
    return 1
  fi
}
