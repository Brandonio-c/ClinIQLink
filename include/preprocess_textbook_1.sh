#!/bin/bash
#SBATCH --partition=norm                # Use the 'norm' partition for standard CPU jobs
#SBATCH --nodes=1                       # Request a single node
#SBATCH --ntasks=1                      # One task per node
#SBATCH --cpus-per-task=32               # Allocate 4 CPU cores for the task
#SBATCH --mem=32G                       # Allocate 16 GB of memory
#SBATCH --time=01:00:00                 # Set a time limit of 1 hour
#SBATCH --job-name=preprocess_textbook_1  # Name the job
#SBATCH --output=preprocess_textbook_1.out # Redirect output to a file

# Set temporary directories
export TMPDIR=/data/coleloughbc/tmp
export HOME=/data/coleloughbc

# Initialize Conda and re-source .bashrc to apply changes
conda init
source ~/.bashrc

# Activate the DOLA environment
conda activate ClinIQLink
# Navigate to the ClinIQLink directory
cd /data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink


# Define directories
BASEDIR="/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink"
SCRIPT_DIR="$BASEDIR/include"
# DATA_DIR="$BASEDIR/data"
DATA_DIR="/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink/data/physiology"
OUTPUT_DIR="$BASEDIR/preprocessed_dataset/physiology"

# Ensure the output directory exists
# mkdir -p "$OUTPUT_DIR"

# Run the preprocessing script
echo "Starting preprocessing..."
/data/coleloughbc/miniconda3/envs/ClinIQLink/bin/python "$BASEDIR/include/preprocess_textbook_paragraph.py" "$DATA_DIR" "$OUTPUT_DIR"
# python "$BASEDIR/include/preprocess_textbook.py" "$DATA_DIR" "$OUTPUT_DIR"
echo "Preprocessing completed."
