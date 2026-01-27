#!/bin/bash
#SBATCH --job-name=eval_cifar10
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
#SBATCH --mail-user=3261535@phd.unibocconi.it

# ============================================================================
# CIFAR-10 Evaluation Script
# ============================================================================
# Computes FID and memorization metrics for a trained model.
#
# Usage:
#   cd scripts/
#   sbatch --export=ALL,RUN_NAME=<run_folder_name> eval_cifar10_fid_memorization.sh
#
# Example:
#   sbatch --export=ALL,RUN_NAME=mdlm_v1 eval_cifar10_fid_memorization.sh
#   sbatch --export=ALL,RUN_NAME=mdlm_v1,CHECKPOINT=epoch=0-step=200000.ckpt,EVAL_STEP=200k eval_cifar10_fid_memorization.sh
#
# Optional environment variables:
#   RUN_NAME       - Name of the run folder in outputs/cifar10/ (required)
#   CHECKPOINT     - Checkpoint file to load (default: last.ckpt)
#   EVAL_STEP      - Step identifier for output folder (default: final)
#   NUM_SAMPLES    - Number of samples to generate (default: 10000)
#   BATCH_SIZE     - Batch size for generation (default: 64)
#   SAMPLING_STEPS - Number of diffusion sampling steps (default: 128)
#   MEM_THRESHOLD  - Memorization threshold k (default: 1/3)
#   SEED           - Random seed (default: 42)
# ============================================================================

# Dataset path
CIFAR10_PATH=${CIFAR10_PATH:-${HOME}/discrete-diffusion-guidance/data/cifar10}

# Setup environment
cd ../ || exit
source setup_env.sh
export HYDRA_FULL_ERROR=1

# Check required argument
if [ -z "${RUN_NAME}" ]; then
  echo "ERROR: RUN_NAME is not set"
  echo "Usage: sbatch --export=ALL,RUN_NAME=<run_folder_name> eval_cifar10_fid_memorization.sh"
  exit 1
fi

# Set defaults
CHECKPOINT=${CHECKPOINT:-last.ckpt}
EVAL_STEP=${EVAL_STEP:-final}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
BATCH_SIZE=${BATCH_SIZE:-64}
SAMPLING_STEPS=${SAMPLING_STEPS:-128}
MEM_THRESHOLD=${MEM_THRESHOLD:-0.3333}
SEED=${SEED:-42}

echo "=============================================="
echo "CIFAR-10 Evaluation"
echo "=============================================="
echo "Run name:        ${RUN_NAME}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Eval step:       ${EVAL_STEP}"
echo "Num samples:     ${NUM_SAMPLES}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Sampling steps:  ${SAMPLING_STEPS}"
echo "Mem threshold:   ${MEM_THRESHOLD}"
echo "Seed:            ${SEED}"
echo "CIFAR-10 path:   ${CIFAR10_PATH}"
echo "=============================================="

# Run evaluation
srun python -u guidance_eval/cifar10_eval.py \
  --run-name "${RUN_NAME}" \
  --outputs-dir "outputs/cifar10" \
  --checkpoint "${CHECKPOINT}" \
  --eval-step "${EVAL_STEP}" \
  --cifar10-path "${CIFAR10_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --mem-threshold "${MEM_THRESHOLD}" \
  --seed "${SEED}"

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: outputs/cifar10/${RUN_NAME}/eval_step${EVAL_STEP}/"
echo "=============================================="
