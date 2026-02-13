#!/bin/bash
#SBATCH --job-name=train_cifar10
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

# NOTE: Need to set the (local) dataset path for downloaded cifar-10 data
# For subset training, point to the preprocessed subset directory instead
DATASET_PATH=${DATASET_PATH:-${HOME}/discrete-diffusion-guidance/data/cifar10}

<<comment
#  Usage:
cd scripts/
MODEL=<mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_cifar10_${MODEL} \
  train_cifar10_unet_guidance.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
REPO_ROOT=$(pwd)

# Convert DATASET_PATH to absolute path relative to repo root if not already absolute
if [[ "${DATASET_PATH}" != /* ]]; then
  DATASET_PATH="${REPO_ROOT}/${DATASET_PATH}"
fi

source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi

T=0
if [ "${MODEL}" = "mdlm" ]; then
  PARAMETERIZATION=subs
  DIFFUSION="absorbing_state"
  ZERO_RECON_LOSS=False
  time_conditioning=False
  sampling_use_cache=True
elif [ "${MODEL}" = "udlm" ]; then
  PARAMETERIZATION=d3pm
  DIFFUSION="uniform"
  ZERO_RECON_LOSS=True
  time_conditioning=True
  sampling_use_cache=False
else
  echo "MODEL must be one of mdlm, udlm"
  exit 1
fi

# Optional: Set BATCH_SIZE for training (default: 250)
BATCH_SIZE=${BATCH_SIZE:-250}

# Optional: Set MAX_STEPS to train for more/fewer steps (default: 300000)
MAX_STEPS=${MAX_STEPS:-300000}

CHECKPOINT_EVERY_N_STEPS=${CHECKPOINT_EVERY_N_STEPS:-10000}

# Optional: Set VAL_CHECK_INTERVAL for validation frequency (default: 10000)
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-10000}

echo "=============================================="
echo "Training Configuration"
echo "=============================================="
echo "MODEL:           ${MODEL}"
echo "Parameterization: ${PARAMETERIZATION}"
echo "Diffusion:       ${DIFFUSION}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Max steps:       ${MAX_STEPS}"
echo "Checkpoint every n steps: ${CHECKPOINT_EVERY_N_STEPS}"
echo "=============================================="

# To enable preemption re-loading, set `hydra.run.dir`
srun python -u -m main \
  is_vision=True \
  diffusion=${DIFFUSION} \
  parameterization=${PARAMETERIZATION} \
  T=${T} \
  time_conditioning=${time_conditioning} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=cifar10 \
  data.train=${DATASET_PATH} \
  data.valid=${DATASET_PATH} \
  loader.global_batch_size=${BATCH_SIZE} \
  loader.eval_global_batch_size=64 \
  backbone=unet \
  model=unet \
  optim.lr=2e-4 \
  lr_scheduler=constant_warmup \
  lr_scheduler.num_warmup_steps=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
  +trainer.check_val_every_n_epoch=null \
  training.guidance.cond_dropout=0.1 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="cifar10_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/cifar10/${RUN_NAME}"

