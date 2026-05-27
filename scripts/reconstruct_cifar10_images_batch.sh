#!/bin/bash
#SBATCH --job-name=recon_cifar10
#SBATCH --account=3261535
#SBATCH --partition=gpunew
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=3261535+hpc@phd.unibocconi.it

# Usage:
#   cd scripts/
#   sbatch reconstruct_cifar10_images_batch.sh \
#     --checkpoints outputs/cifar10/run1/checkpoints/last.ckpt \
#     --index 42 \
#     --mask-percentage 50 \
#     --output-dir outputs/cifar10/reconstructions
#
#   sbatch reconstruct_cifar10_images_batch.sh \
#     --checkpoints \
#       outputs/cifar10/run1/checkpoints/last.ckpt \
#       outputs/cifar10/run2/checkpoints/last.ckpt \
#     --category 3 \
#     --mask-type random

cd ../ || exit
source setup_env.sh
export HYDRA_FULL_ERROR=1

echo "Launching reconstruction with arguments: $*"

srun python -u reconstruct_cifar10_images.py "$@"
