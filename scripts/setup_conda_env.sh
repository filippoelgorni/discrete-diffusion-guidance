#!/bin/bash
#SBATCH --job-name=setup_conda_env
#SBATCH --account=3261535
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Setup conda environment for discrete-diffusion-guidance
# Usage: sbatch setup_conda_env.sh

echo "Setting up conda environment 'discdiff'..."

cd ../ || exit

# Initialize conda
if [ -f "${HOME}/.bashrc" ]; then
    source "${HOME}/.bashrc"
fi

# Create conda environment from requirements.yaml
conda env create -f requirements.yaml

echo "✓ Conda environment 'discdiff' created successfully!"
echo "To activate: conda activate discdiff"
