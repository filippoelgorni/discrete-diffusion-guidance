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

# Optional: Set SUBSET_FRACTION to use only a portion of the dataset
# Examples: 0.1 (10%), 0.01 (1%), 0.001 (0.1%), or 1.0 (100% - full dataset)
SUBSET_FRACTION=${SUBSET_FRACTION:-1.0}

# Optional: Set BATCH_SIZE for training (default: 250)
# Note: To use very small SUBSET_FRACTION values, you may need smaller batch sizes.
# For SUBSET_FRACTION=0.001, need at least ~50 samples per batch to get ≥1 batch.
BATCH_SIZE=${BATCH_SIZE:-250}

# Optional: Set LIMIT_VAL_BATCHES separately from SUBSET_FRACTION (default: same as SUBSET_FRACTION)
# Use this if you get "limit_val_batches < 1" errors with small SUBSET_FRACTION
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-${SUBSET_FRACTION}}

# Optional: Set MAX_STEPS to train for more/fewer steps (default: 300000)
MAX_STEPS=${MAX_STEPS:-300000}

echo "=============================================="
echo "Training Configuration"
echo "=============================================="
echo "MODEL:           ${MODEL}"
echo "Parameterization: ${PARAMETERIZATION}"
echo "Diffusion:       ${DIFFUSION}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Subset fraction: ${SUBSET_FRACTION}"
echo "Val limit:       ${LIMIT_VAL_BATCHES}"
echo "Max steps:       ${MAX_STEPS}"
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
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=10_000 \
  +trainer.check_val_every_n_epoch=null \
  trainer.limit_train_batches=${SUBSET_FRACTION} \
  trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
  training.guidance.cond_dropout=0.1 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="cifar10_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/cifar10/${RUN_NAME}"

