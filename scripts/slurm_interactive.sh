#!/usr/bin/env bash
# Request an interactive GPU session.
# Edit the partition/account/qos/time/gres to match your campus cluster.

set -euo pipefail

PARTITION="${PARTITION:-academic}"
ACCOUNT="${ACCOUNT:-ece341x}"
TIME="${TIME:-02:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-1}"
MEM="${MEM:-8G}"
# GPU_TYPE="${GPU_TYPE:-H200|H100|A100-80G|L40S|A100|V100}"

echo "requesting interactive session..."
echo "  partition: $PARTITION"
echo "  account:   $ACCOUNT"
echo "  time:      $TIME"
echo "  gpus:      $GPUS"
echo "  cpus:      $CPUS"
echo "  mem:       $MEM"
# echo "  gpu type:  $GPU_TYPE"

srun --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --time="$TIME" \
  --gres="gpu:$GPUS" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --pty bash
  # -C "$GPU_TYPE" \
