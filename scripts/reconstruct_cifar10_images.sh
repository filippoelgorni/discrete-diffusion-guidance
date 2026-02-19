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

# ============================================================================
# CIFAR-10 Image Reconstruction Script
# ============================================================================
# Reconstructs partially masked CIFAR-10 images using trained model(s).
#
# Usage:
#   cd scripts/
#   sbatch --export=ALL,CHECKPOINTS="path1 path2",INDEX=42 reconstruct_cifar10_images.sh
#
# Examples:
#   # Reconstruct specific image with one checkpoint
#   sbatch --export=ALL,CHECKPOINTS="outputs/cifar10/run1/checkpoints/last.ckpt",INDEX=42 reconstruct_cifar10_images.sh
#
#   # Multiple checkpoints
#   sbatch --export=ALL,CHECKPOINTS="outputs/cifar10/run1/checkpoints/last.ckpt outputs/cifar10/run2/checkpoints/last.ckpt",INDEX=100 reconstruct_cifar10_images.sh
#
#   # Random image from category
#   sbatch --export=ALL,CHECKPOINTS="outputs/cifar10/run1/checkpoints/last.ckpt",CATEGORY=5 reconstruct_cifar10_images.sh
#
#   # Custom masking
#   sbatch --export=ALL,CHECKPOINTS="outputs/cifar10/run1/checkpoints/last.ckpt",MASK_PERCENTAGE=30,MASK_FROM_TOP=true reconstruct_cifar10_images.sh
#
# Required environment variables:
#   CHECKPOINTS    - Space-separated list of checkpoint paths (required)
#
# Optional environment variables:
#   INDEX          - Specific CIFAR-10 image index 0-49999 (default: none, uses category or random)
#   CATEGORY       - CIFAR-10 category 0-9, used if INDEX not set (default: none, picks random)
#   MASK_PERCENTAGE - Percentage to mask, 0-100 (default: 50)
#   MASK_FROM_TOP  - Mask from top instead of bottom (default: false)
#   OUTPUT_DIR     - Output directory (default: auto-generated with timestamp)
#   EPS            - Noise schedule epsilon (default: 1e-5)
#   SEED           - Random seed (default: 42)
#   DATA_DIR       - CIFAR-10 data directory (default: data/cifar10)
# ============================================================================

# Setup environment
cd ../ || exit
source setup_env.sh
export HYDRA_FULL_ERROR=1

# Check required argument
if [ -z "${CHECKPOINTS}" ]; then
  echo "ERROR: CHECKPOINTS is not set"
  echo "Usage: sbatch --export=ALL,CHECKPOINTS=\"path1 path2\" reconstruct_cifar10_images.sh"
  exit 1
fi

# Convert space-separated checkpoints to array
CHECKPOINT_ARRAY=($CHECKPOINTS)

# Set defaults
INDEX=${INDEX:-}
CATEGORY=${CATEGORY:-}
MASK_PERCENTAGE=${MASK_PERCENTAGE:-50.0}
MASK_FROM_TOP=${MASK_FROM_TOP:-false}
OUTPUT_DIR=${OUTPUT_DIR:-}
EPS=${EPS:-1e-5}
SEED=${SEED:-42}
DATA_DIR=${DATA_DIR:-data/cifar10}

echo "=============================================="
echo "CIFAR-10 Image Reconstruction"
echo "=============================================="
echo "Checkpoints:     ${#CHECKPOINT_ARRAY[@]} checkpoint(s)"
for ckpt in "${CHECKPOINT_ARRAY[@]}"; do
  echo "  - ${ckpt}"
done
echo "Index:           ${INDEX:-auto (by category or random)}"
echo "Category:        ${CATEGORY:-auto (random)}"
echo "Mask percentage: ${MASK_PERCENTAGE}%"
echo "Mask from top:   ${MASK_FROM_TOP}"
echo "Output dir:      ${OUTPUT_DIR:-auto (timestamped)}"
echo "Epsilon:         ${EPS}"
echo "Seed:            ${SEED}"
echo "Data dir:        ${DATA_DIR}"
echo "=============================================="

# Build command arguments
CMD_ARGS="--checkpoints ${CHECKPOINTS}"
CMD_ARGS="${CMD_ARGS} --mask-percentage ${MASK_PERCENTAGE}"
CMD_ARGS="${CMD_ARGS} --eps ${EPS}"
CMD_ARGS="${CMD_ARGS} --seed ${SEED}"
CMD_ARGS="${CMD_ARGS} --data-dir ${DATA_DIR}"

# Add optional arguments
if [ -n "${INDEX}" ]; then
  CMD_ARGS="${CMD_ARGS} --index ${INDEX}"
fi
if [ -n "${CATEGORY}" ]; then
  CMD_ARGS="${CMD_ARGS} --category ${CATEGORY}"
fi
if [ -n "${OUTPUT_DIR}" ]; then
  CMD_ARGS="${CMD_ARGS} --output-dir ${OUTPUT_DIR}"
fi
if [ "${MASK_FROM_TOP}" = "true" ]; then
  CMD_ARGS="${CMD_ARGS} --no-mask-from-bottom"
fi

# Run reconstruction
echo ""
echo "Running reconstruction..."
srun python -u reconstruct_cifar10_images.py ${CMD_ARGS}

echo ""
echo "=============================================="
echo "Reconstruction complete!"
echo "Check the output directory for results:"
echo "  - 00_original.png"
echo "  - 01_masked.png"
echo "  - 02_reconstructed_*.png (one per checkpoint)"
echo "=============================================="
