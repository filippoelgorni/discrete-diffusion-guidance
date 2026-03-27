#!/bin/bash
#SBATCH --job-name=gen_cifar10
#SBATCH --account=3261535
#SBATCH --partition=gpunew
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=3261535+hpc@phd.unibocconi.it

# ============================================================================
# CIFAR-10 Sample Generation Script
# ============================================================================
# Generates samples from a trained model and saves them as PNG images.
# This is step 1 of 3 for computing memorization metrics.
#
# Usage:
#   cd scripts/
#   sbatch --export=ALL,RUN_NAME=<run_folder_name> generate_cifar10_samples.sh
#
# Example:
#   sbatch --export=ALL,RUN_NAME=mdlm_v1 generate_cifar10_samples.sh
#   sbatch --export=ALL,RUN_NAME=mdlm_v1,CHECKPOINT=epoch=0-step=200000.ckpt,EVAL_STEP=200k generate_cifar10_samples.sh
#
# Required environment variables:
#   RUN_NAME       - Name of the run folder in outputs/cifar10/ (required)
#
# Optional environment variables:
#   CHECKPOINT     - Checkpoint file to load (default: last.ckpt)
#   EVAL_STEP      - Step identifier for output folder (default: final)
#   OUTPUT_DIR     - Custom output directory (default: outputs/cifar10/<run_name>/generated_samples_<eval_step>)
#   NUM_SAMPLES    - Number of samples to generate (default: 10000)
#   BATCH_SIZE     - Batch size for generation (default: uses config)
#   SAMPLING_STEPS - Number of diffusion sampling steps (default: uses config)
#   SEED           - Random seed (default: 42)
#   USE_CFG        - Enable classifier-free guidance (default: true)
#   CFG_CONDITION  - Class condition for CFG, 0-9 (default: 0)
#   CFG_GAMMA      - Guidance strength, 0=unconditional, 1=conditional (default: 1.0)
#
# After generation, download the images and run:
#   1. extract_cifar10_reference.py to extract CIFAR-10 reference images
#   2. compute_cifar10_metrics.py to compute memorization metrics locally
# ============================================================================

# Setup environment
cd ../ || exit
source setup_env.sh
export HYDRA_FULL_ERROR=1

# Check required argument
if [ -z "${RUN_NAME}" ]; then
  echo "ERROR: RUN_NAME is not set"
  echo "Usage: sbatch --export=ALL,RUN_NAME=<run_folder_name> generate_cifar10_samples.sh"
  exit 1
fi

# Set defaults
CHECKPOINT=${CHECKPOINT:-last.ckpt}
EVAL_STEP=${EVAL_STEP:-final}
OUTPUT_DIR=${OUTPUT_DIR:-}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
BATCH_SIZE=${BATCH_SIZE:-}
SAMPLING_STEPS=${SAMPLING_STEPS:-}
SEED=${SEED:-42}
USE_CFG=${USE_CFG:-true}
CFG_CONDITION=${CFG_CONDITION:-0}
CFG_GAMMA=${CFG_GAMMA:-1.0}

echo "=============================================="
echo "CIFAR-10 Sample Generation"
echo "=============================================="
echo "Run name:        ${RUN_NAME}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Eval step:       ${EVAL_STEP}"
echo "Output dir:      ${OUTPUT_DIR:-auto}"
echo "Num samples:     ${NUM_SAMPLES}"
echo "Batch size:      ${BATCH_SIZE:-from config}"
echo "Sampling steps:  ${SAMPLING_STEPS:-from config}"
echo "Seed:            ${SEED}"
echo "Use CFG:         ${USE_CFG}"
echo "CFG condition:   ${CFG_CONDITION}"
echo "CFG gamma:       ${CFG_GAMMA}"
echo "=============================================="

# Build CFG arguments
CFG_ARGS=""
if [ "${USE_CFG}" = "true" ]; then
  CFG_ARGS="--use-cfg --cfg-condition ${CFG_CONDITION} --cfg-gamma ${CFG_GAMMA}"
fi

# Build optional arguments
OPTIONAL_ARGS=""
if [ -n "${OUTPUT_DIR}" ]; then
  OPTIONAL_ARGS="${OPTIONAL_ARGS} --output-dir ${OUTPUT_DIR}"
fi
if [ -n "${BATCH_SIZE}" ]; then
  OPTIONAL_ARGS="${OPTIONAL_ARGS} --batch-size ${BATCH_SIZE}"
fi
if [ -n "${SAMPLING_STEPS}" ]; then
  OPTIONAL_ARGS="${OPTIONAL_ARGS} --sampling-steps ${SAMPLING_STEPS}"
fi

# Run generation
srun python -u generate_cifar10_samples.py \
  --run-name "${RUN_NAME}" \
  --outputs-dir "outputs/cifar10" \
  --checkpoint "${CHECKPOINT}" \
  --eval-step "${EVAL_STEP}" \
  --num-samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  ${CFG_ARGS} \
  ${OPTIONAL_ARGS}

echo ""
echo "=============================================="
echo "Generation complete!"
echo "Next steps:"
echo "  1. Download generated images from the output directory"
echo "  2. Run: python extract_cifar10_reference.py --cifar10-path <path> --output-dir <path> --split train"
echo "  3. Run: python compute_cifar10_metrics.py --generated-dir <path> --reference-dir <path> --output-dir <path>"
echo "=============================================="
