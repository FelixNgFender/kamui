#!/usr/bin/env bash
# Request an interactive GPU session.
# Edit the partition/account/qos/time/gres to match your campus cluster.

set -euo pipefail

PARTITION="${PARTITION:-short}"
ACCOUNT="${ACCOUNT:-bislam}"
TIME="${TIME:-00:15:00}"
GPUS="${GPUS:-4}"
CPUS="${CPUS:-4}"
MEM="${MEM:-64G}"
GPU_TYPE="${GPU_TYPE:-H200|H100|A100-80G|A100}"

echo "requesting interactive session..."
echo "  partition: $PARTITION"
echo "  account:   $ACCOUNT"
echo "  time:      $TIME"
echo "  gpus:      $GPUS"
echo "  cpus:      $CPUS"
echo "  mem:       $MEM"
echo "  gpu type:  $GPU_TYPE"

module load uv gcc python cuda

srun --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --time="$TIME" \
  --gres="gpu:$GPUS" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  -C "$GPU_TYPE" \
  --pty bash
