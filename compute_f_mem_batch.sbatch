#!/bin/bash
#SBATCH --job-name=f_mem_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/f_mem_batch_%j.log

# Usage: sbatch compute_f_mem_batch.sbatch models_config.txt output.csv

CONFIG=${1:-f_mem_models.txt}
OUTPUT=${2:-f_mem_results.csv}

echo "Config file: $CONFIG"
echo "Output file: $OUTPUT"

python compute_f_mem_batch.py "$CONFIG" --output "$OUTPUT" --mem-threshold 0.333333

echo "Done. Results in $OUTPUT"
