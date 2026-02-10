#!/bin/bash
#SBATCH --job-name=reconstruct_cifar10
#SBATCH --account=3261535
#SBATCH --partition=long_gpunew
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=3261535+hpc@phd.unibocconi.it

# Script to reconstruct corrupted CIFAR10 images using trained model
# 
# Usage:
#   cd scripts/
#   CHECKPOINT_PATH=/path/to/checkpoint.ckpt \
#   CORRUPTION_LEVEL=0.5 \
#   NUM_STEPS=128 \
#   sbatch reconstruct_cifar10.sh

# Setup environment
cd ../ || exit
REPO_ROOT=$(pwd)
source setup_env.sh
export HYDRA_FULL_ERROR=1

# Configuration
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"???"}
CORRUPTED_DATA_PATH=${CORRUPTED_DATA_PATH:-${HOME}/discrete-diffusion-guidance/data/corrupted_cifar10}
OUTPUT_PATH=${OUTPUT_PATH:-${REPO_ROOT}/outputs/cifar10_reconstruction}
CORRUPTION_LEVEL=${CORRUPTION_LEVEL:-0.5}
NUM_STEPS=${NUM_STEPS:-128}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
SPLIT=${SPLIT:-test}
DISABLE_EMA=${DISABLE_EMA:-false}
SEED=${SEED:-42}

# Check if checkpoint exists
if [ "${CHECKPOINT_PATH}" = "???" ]; then
  echo "ERROR: CHECKPOINT_PATH must be set"
  echo "Example: CHECKPOINT_PATH=/path/to/checkpoint.ckpt sbatch reconstruct_cifar10.sh"
  exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
  exit 1
fi

echo "=============================================="
echo "Reconstructing Corrupted CIFAR10 Images"
echo "=============================================="
echo "Checkpoint:       ${CHECKPOINT_PATH}"
echo "Corrupted data:   ${CORRUPTED_DATA_PATH}"
echo "Output path:      ${OUTPUT_PATH}"
echo "Corruption level: ${CORRUPTION_LEVEL}"
echo "Num steps:        ${NUM_STEPS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Num samples:      ${NUM_SAMPLES}"
echo "Split:            ${SPLIT}"
echo "Disable EMA:      ${DISABLE_EMA}"
echo "Seed:             ${SEED}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_PATH}"

# Build disable_ema flag
if [ "${DISABLE_EMA}" = "true" ]; then
  EMA_FLAG="--disable_ema"
else
  EMA_FLAG=""
fi

# Reconstruct images
srun python -u reconstruct_images.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --corrupted_data_path "${CORRUPTED_DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --corruption_level ${CORRUPTION_LEVEL} \
  --num_steps ${NUM_STEPS} \
  --batch_size ${BATCH_SIZE} \
  --num_samples ${NUM_SAMPLES} \
  --split ${SPLIT} \
  --seed ${SEED} \
  ${EMA_FLAG}

echo "Reconstruction complete!"
echo "Results saved to: ${OUTPUT_PATH}"
