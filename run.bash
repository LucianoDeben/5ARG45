#!/bin/bash
#SBATCH --job-name=5ARG45
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=jhs_tue2022
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=./%x_%A_%a.out

# Reset environment
module purge

# Load necessary modules
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Python/3.9.18

. /etc/bashrc
. ~/.bashrc

source ./venv/bin/activate

# Debugging: print environment details
echo "Running on node:"
hostname
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Loaded Modules:"
module list

srun python src/perturbinator.py