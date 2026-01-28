#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -J "picogpt"
#SBATCH -o logs/picogpt_%j.out
#SBATCH -e logs/picogpt_%j.err
#SBATCH -p academic
#SBATCH -t 05:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

echo "starting batch job on $(hostname) at $(date)"
nvidia-smi

module load gcc/15.1.0
module load cuda/12.9.0
module load python/3.13.5
module load uv/0.7.2

export PYTHONUNBUFFERED=1

uv sync
source .venv/bin/activate
picogpt train char-transformer --num-epochs 5

echo "done at $(date)"
