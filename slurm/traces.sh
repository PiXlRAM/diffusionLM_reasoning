#!/bin/bash
#SBATCH --account=edu
#SBATCH --job-name=dlm_traces
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32gb
#SBATCH --time=0-05:00
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch slurm/traces.sh [N_TRAIN] [N_TEST]   (defaults 80 and 300)
N_TRAIN=${1:-80}
N_TEST=${2:-300}

export PROJ=${PROJ:-/insomnia001/depts/edu/COMS_6975_Fall2026/vs3032/dlm}
export HF_HOME=$PROJ/hf_cache

module load anaconda
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/dlm

cd "$PROJ/code"
nvidia-smi --query-gpu=name,memory.total --format=csv
set -e
for model in llada qwen; do
  python -m scripts.generate_traces --model $model --split train --n $N_TRAIN
  python -m scripts.generate_traces --model $model --split test --n $N_TEST
done
python -m scripts.select_questions --split train
python -m scripts.select_questions --split test
