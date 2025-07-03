#!/bin/bash -l
#SBATCH --job-name=iterative_emulation_mpi
#SBATCH --partition=q128
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/iterative_mpi_%j.out
#SBATCH --error=logs/iterative_mpi_%j.err

# Prevent TensorFlow/NumPy from oversubscribing threads
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

echo "========= Job started on $(hostname) at $(date) =========="

# Load modules
ml load anaconda3
ml load gcc
ml load openmpi

# Activate your Conda environment
conda activate ConnectEnvironment

# Launch the MPI-enabled Python script
srun /home/lucajn/.conda/envs/ConnectEnvironment/bin/python -u iterative_mpi.py

echo "========= Job finished at $(date) =========="
