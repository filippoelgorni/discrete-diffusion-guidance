#!/bin/bash
#SBATCH --job-name=download_cifar10
#SBATCH --account=3261535
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=3261535+hpc@phd.unibocconi.it

# Download CIFAR-10 dataset
# Usage: sbatch download_cifar10.sh

DOWNLOAD_DIR="${1:-$HOME/datasets/cifar10}"

echo "Downloading CIFAR-10 to: ${DOWNLOAD_DIR}"

cd ../ || exit

# Initialize conda if CONDA_SHELL not set
if [ -z "${CONDA_SHELL}" ]; then
    # Try common conda initialization paths
    if [ -f "${HOME}/.bashrc" ]; then
        export CONDA_SHELL="${HOME}/.bashrc"
    elif [ -f "${HOME}/.bash_profile" ]; then
        export CONDA_SHELL="${HOME}/.bash_profile"
    elif [ -f "/etc/profile.d/conda.sh" ]; then
        export CONDA_SHELL="/etc/profile.d/conda.sh"
    elif [ -n "${CONDA_EXE}" ]; then
        # If conda is available, find conda.sh relative to it
        CONDA_BASE=$(dirname $(dirname ${CONDA_EXE}))
        export CONDA_SHELL="${CONDA_BASE}/etc/profile.d/conda.sh"
    fi
fi

# Create conda environment if it doesn't exist
if command -v conda &> /dev/null; then
    if ! conda env list | grep -q "^discdiff "; then
        echo "Creating discdiff conda environment..."
        conda env create -f requirements.yaml
    fi
fi

source setup_env.sh

python -c "
import torchvision
import os

download_dir = '${DOWNLOAD_DIR}'
os.makedirs(download_dir, exist_ok=True)

print(f'Downloading CIFAR-10 train set to {download_dir}...')
torchvision.datasets.CIFAR10(root=download_dir, train=True, download=True)

print(f'Downloading CIFAR-10 test set to {download_dir}...')
torchvision.datasets.CIFAR10(root=download_dir, train=False, download=True)

print(f'✓ CIFAR-10 downloaded successfully to {download_dir}')
print(f'Set DATASET_PATH={download_dir} in train_cifar10_unet_guidance.sh')
"
