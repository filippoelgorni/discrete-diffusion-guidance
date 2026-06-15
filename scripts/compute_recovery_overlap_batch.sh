#!/bin/bash
#SBATCH --job-name=recovery_overlap
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

# Usage:
#   sbatch compute_recovery_overlap_batch.sh models_config.txt output.csv [extra python args]
#
# Example:
#   sbatch compute_recovery_overlap_batch.sh f_mem_models_100000.txt recovery_overlap_results.csv \
#       --n-samples 1000 --times 0.0:0.1:1.0 --cfg-gamma 1.0
#
# Add --channelwise-random-mask to mask individual RGB entries instead of whole
# spatial pixels shared across RGB channels.

CONFIG=${1:-f_mem_models_100000.txt}
OUTPUT=${2:-recovery_overlap_results.csv}

# Remove first two args, pass remaining args to Python.
# This also works when CONFIG/OUTPUT are supplied explicitly.
if [ "$#" -ge 2 ]; then
    shift 2
else
    shift "$#"
fi

mkdir -p out err

echo "Config file: $CONFIG"
echo "Output file: $OUTPUT"
echo "Extra args: $@"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"

conda activate discdiff
python ../compute_recovery_overlap_batch.py "$CONFIG" --output "$OUTPUT" "$@"

STATUS=$?
echo "Finished at: $(date)"
echo "Exit status: $STATUS"
echo "Results in $OUTPUT"
exit $STATUS
