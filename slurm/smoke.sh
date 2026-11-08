#!/bin/bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=dlm_smoke
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32gb
#SBATCH --time=0-00:45
#SBATCH --output=logs/%x_%j.out

export PROJ=${PROJ:-/insomnia001/depts/edu/COMS_6975_Fall2026/vs3032/dlm}
export HF_HOME=$PROJ/hf_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8

module load anaconda
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/dlm

cd "$PROJ/code"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m tests.test_cache
python -m scripts.smoke_test "$@"
