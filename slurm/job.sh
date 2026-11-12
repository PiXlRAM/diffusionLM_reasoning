#!/bin/bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=dlm_job
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32gb
#SBATCH --time=0-04:00
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch slurm/job.sh scripts.MODULE [ARGS...]
# e.g.   sbatch slurm/job.sh scripts.hide_check
#        sbatch slurm/job.sh scripts.pilot_estimator
MODULE=${1:?usage: sbatch slurm/job.sh scripts.MODULE [ARGS...]}
shift

export PROJ=${PROJ:-/insomnia001/depts/edu/COMS_6975_Fall2026/vs3032/dlm}
export HF_HOME=$PROJ/hf_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

module load anaconda
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/dlm

cd "$PROJ/code"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m $MODULE "$@"
