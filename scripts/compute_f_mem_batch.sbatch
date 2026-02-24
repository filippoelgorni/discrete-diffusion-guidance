#!/bin/bash
#SBATCH --job-name=f_mem_batch
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

# Usage: sbatch compute_f_mem_batch.sbatch models_config.txt output.csv

CONFIG=${1:-f_mem_models.txt}
OUTPUT=${2:-f_mem_results.csv}

echo "Config file: $CONFIG"
echo "Output file: $OUTPUT"

python compute_f_mem_batch.py "$CONFIG" --output "$OUTPUT" --mem-threshold 0.333333

echo "Done. Results in $OUTPUT"
