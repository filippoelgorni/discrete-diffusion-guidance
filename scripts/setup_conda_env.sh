#!/bin/bash
#SBATCH --job-name=setup_conda_env
#SBATCH --account=3261535
#SBATCH --partition=debug_gpunew
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

# Remove existing environment
echo "Removing existing discdiff environment..."
conda env remove -n discdiff -y 2>/dev/null || true

# Aggressively clean conda cache
echo "Cleaning conda cache..."
conda clean --all -y
conda clean --force-pkgs-dirs -y
rm -rf ${HOME}/miniconda3/pkgs/pytorch-* 2>/dev/null || true
rm -rf ${HOME}/.conda/pkgs/pytorch-* 2>/dev/null || true

# Create conda environment from requirements.yaml
conda env create -f requirements.yaml

echo "✓ Conda environment 'discdiff' created successfully!"
echo "To activate: conda activate discdiff"
