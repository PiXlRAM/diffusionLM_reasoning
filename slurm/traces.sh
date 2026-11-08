#!/bin/bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=dlm_traces
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32gb
#SBATCH --time=0-05:00
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch slurm/traces.sh MODEL [N_TRAIN] [N_TEST]   (defaults 80 and 300)
# Run once per model; then: python -m scripts.select_questions --split train (and --split test)
MODEL=${1:?usage: sbatch slurm/traces.sh llada|qwen [N_TRAIN] [N_TEST]}
N_TRAIN=${2:-80}
N_TEST=${3:-300}

export PROJ=${PROJ:-/insomnia001/depts/edu/COMS_6975_Fall2026/vs3032/dlm}
export HF_HOME=$PROJ/hf_cache

module load anaconda
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/dlm

cd "$PROJ/code"
nvidia-smi --query-gpu=name,memory.total --format=csv
set -e
python -m scripts.generate_traces --model $MODEL --split train --n $N_TRAIN
python -m scripts.generate_traces --model $MODEL --split test --n $N_TEST
