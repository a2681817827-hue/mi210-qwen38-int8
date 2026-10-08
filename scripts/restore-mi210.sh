#!/usr/bin/env bash
# Run on a reinstalled host with Docker and working MI210 device nodes.
set -euo pipefail
umask 077
backup=${1:?Usage: bash restore-mi210.sh /mnt/backup/mi210-20261008}
[[ $backup == /* ]] || { echo 'Use an absolute backup path.' >&2; exit 2; }
[[ -c /dev/kfd ]] || { echo 'Restore the AMD GPU driver first.' >&2; exit 2; }
docker info >/dev/null
[[ ! -e /root/int8-vllm ]] || { echo '/root/int8-vllm already exists; restore to a clean host.' >&2; exit 2; }
for model in Qwen3.8-27B-PTQR-R10S60 dflash2-ptqr-r1; do
  [[ ! -e /data/models/$model ]] || { echo "Model path already exists: $model" >&2; exit 2; }
done
for cache in mi210-vllm huggingface; do
  [[ ! -e /data/cache/$cache ]] || { echo "Cache path already exists: $cache" >&2; exit 2; }
done
(
  cd "$backup"
  sha256sum -c SHA256SUMS
  cd handoff
  sha256sum -c SHA256SUMS
)
zstd -dc "$backup/images.tar.zst" | docker image load
expected=sha256:23fe3b8ed7cb2dca13cee345a233970b0cc0b165a267bc6d75c3ec4fdc55dbf7
actual=$(docker inspect int8-vllm-mi210:saved-container-20261008 --format '{{.Id}}')
[[ $actual == "$expected" ]] || { echo 'Measured image ID mismatch.' >&2; exit 2; }
mkdir -p /data/models /data/cache
rsync -aH --numeric-ids "$backup/models/" /data/models/
zstd -dc "$backup/caches.tar.zst" | tar -xf - -C /data/cache
tar -xzf "$backup/handoff/source.tar.gz" -C /root
tar -xzf "$backup/handoff/results.tar.gz" -C /root
mkdir -p /root/mi210-handoff-20261008
cp -a "$backup/handoff/." /root/mi210-handoff-20261008/
echo 'Restore complete. Set MI210_API_KEY, then run scripts/start-mi210-saved-20261008.sh.'
