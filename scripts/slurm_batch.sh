#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -J "pealm"
#SBATCH -o logs/pealm_%j.out
#SBATCH -e logs/pealm_%j.err
#SBATCH --partition=short
#SBATCH --account=bislam
#SBATCH --qos=normal
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint="H200|H100|A100-80G|A100"

set -euo pipefail

mkdir -p logs

echo "starting batch job on $(hostname) at $(date)"
nvidia-smi

module load gcc/15.1.0 cuda/12.9.0 python/3.13.5 uv/0.7.2

export PYTHONUNBUFFERED=1

uv sync
source .venv/bin/activate
torchrun --standalone --nproc_per_node=4 -m pealm train gpt2 --input.npy-shards data/fineweb_edu10B --micro-batch-size 32 --save-every 5000 2>&1 | tee training.log

echo "done at $(date)"
