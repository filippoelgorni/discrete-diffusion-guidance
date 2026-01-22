#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=8000
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4

# Download CIFAR-10 dataset
# Usage: sbatch download_cifar10.sh

DOWNLOAD_DIR="${1:-$HOME/datasets/cifar10}"

echo "Downloading CIFAR-10 to: ${DOWNLOAD_DIR}"

cd ../ || exit
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
