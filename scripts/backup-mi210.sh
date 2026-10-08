#!/usr/bin/env bash
# Run on the host after stopping MI210 services. Destination must be new.
set -euo pipefail
umask 077
target=${1:?Usage: bash backup-mi210.sh /mnt/backup/mi210-20261008 [--all-images]}
mode=${2:-}
[[ $target == /* && ! -e $target ]] || { echo 'Use a new absolute backup directory.' >&2; exit 2; }
[[ -z $mode || $mode == --all-images ]] || { echo 'Unknown option.' >&2; exit 2; }
for command in docker zstd rsync tar sha256sum; do command -v "$command" >/dev/null; done
[[ -z $(docker ps --filter name=mi210 --format '{{.Names}}') ]] || {
  echo 'Stop running MI210 containers before backing up.' >&2; exit 2;
}
bundle_dir=$(cd "$(dirname "$0")" && pwd)
bundle_dir=${MI210_HANDOFF_DIR:-$bundle_dir}
if [[ ! -f $bundle_dir/source.tar.gz ]]; then
  bundle_dir=/root/mi210-handoff-20261008
fi
[[ -f $bundle_dir/source.tar.gz && -f $bundle_dir/results.tar.gz && -f $bundle_dir/SHA256SUMS ]] || {
  echo 'The complete handoff bundle is required; set MI210_HANDOFF_DIR to its path.' >&2; exit 2;
}
parent=$(dirname "$target")
mkdir -p "$parent"
available=$(df -PB1 "$parent" | awk 'NR==2 {print $4}')
((available >= 85899345920)) || { echo 'Provide at least 80 GiB free space; current full compressed backup is about 64 GiB.' >&2; exit 2; }
mkdir -p "$target/handoff" "$target/models" "$target/metadata"
cp -a "$bundle_dir/." "$target/handoff/"
images=(int8-vllm-mi210:saved-container-20261008
        int8-vllm-mi210:draft-window-measured-20261008
        int8-vllm-mi210:query5-candidates-20261007
        int8-vllm-mi210:qkpv-candidates-20261008
        int8-vllm-mi210:qkpv-policy-candidates-20261008)
if [[ $mode == --all-images ]]; then
  mapfile -t images < <(docker image ls --filter 'reference=int8-vllm-mi210:*' --format '{{.Repository}}:{{.Tag}}' | sort -u)
fi
docker image inspect "${images[@]}" > "$target/metadata/images.inspect.private.json"
mapfile -t containers < <(docker ps -a --filter name=mi210 --format '{{.Names}}')
if ((${#containers[@]})); then
  docker inspect "${containers[@]}" > "$target/metadata/containers.inspect.private.json"
fi
docker image save "${images[@]}" | zstd -T2 -3 -o "$target/images.tar.zst"
for model in Qwen3.8-27B-PTQR-R10S60 dflash2-ptqr-r1; do
  rsync -aH --numeric-ids "/data/models/$model" "$target/models/"
done
tar -C /data/cache -cf - mi210-vllm huggingface | zstd -T2 -3 -o "$target/caches.tar.zst"
docker export mi210-qkpv-screen-20261008-control-after | zstd -T2 -3 -o "$target/container-rootfs.tar.zst"
(
  cd "$target"
  sha256sum images.tar.zst caches.tar.zst container-rootfs.tar.zst > SHA256SUMS
  find models -type f -print0 | sort -z | xargs -0 sha256sum >> SHA256SUMS
)
echo "Backup complete: $target"
