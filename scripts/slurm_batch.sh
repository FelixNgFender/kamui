#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -J "picogpt"
#SBATCH -o logs/picogpt_%j.out
#SBATCH -e logs/picogpt_%j.err
#SBATCH -p short
#SBATCH -A bislam
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:8
#SBATCH -C "H200|H100|A100-80G|A100"

set -euo pipefail

mkdir -p logs

echo "starting batch job on $(hostname) at $(date)"
nvidia-smi

module load gcc/15.1.0 cuda/12.9.0 python/3.13.5 uv/0.7.2

export PYTHONUNBUFFERED=1

uv sync
source .venv/bin/activate
uv run picogpt train gpt2 --num-epochs 1 --input.npy-shards data/fineweb_edu10B

echo "done at $(date)"
