#!/bin/bash
#SBATCH --job-name=f_mem_batch
#SBATCH --account=3261535
#SBATCH --partition=gpunew
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=3261535+hpc@phd.unibocconi.it

# Usage: sbatch compute_f_mem_batch.sh models_config.txt output.csv [--num-samples 1000] [--batch-size 64]

CONFIG=${1:-f_mem_models_100000.txt}
OUTPUT=${2:-f_mem_results.csv}
shift 2  # Remove first two args, pass remaining to Python

echo "Config file: $CONFIG"
echo "Output file: $OUTPUT"

python ../compute_f_mem_batch.py "$CONFIG" --output "$OUTPUT" "$@"

echo "Done. Results in $OUTPUT"
