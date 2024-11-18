#!/bin/bash
#SBATCH --partition=gpu                             # Request the GPU partition
#SBATCH --gres=gpu:a100:1                           # Request 1 A100 GPU
#SBATCH --nodes=1                                   # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                         # Number of tasks per node
#SBATCH --cpus-per-task=32                          # 32 CPUs per node
#SBATCH --mem=64G                                   # 64 GB memory per node
#SBATCH --time=00:30:00                             # 1 hr for testing
#SBATCH --job-name=generate_qa_dataset              # Set job name
#SBATCH --output=generate_qa_dataset.out            # Output file name

## Unloading all currently loaded CUDA modules
module unload cuDNN
module unload CUDA

# Loading the latest CUDA Toolkit and corresponding cuDNN
echo "Loading latest CUDA Toolkit..."
module load gcc/11.3.0
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

#export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-12.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# Initialize Conda and re-source .bashrc to apply changes
conda init
source /data/coleloughbc/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

# Activate the DOLA environment
conda activate ClinIQLink
#conda activate llama3-1
# Navigate to the ClinIQLink directory
cd /data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink

# Define paths and parameters
BASEDIR="/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink"
SCRIPT_DIR="$BASEDIR/src"
DATA_DIR="$BASEDIR/data"
OUTPUT_DIR="$BASEDIR/QA_dataset"
PREPROCESSED_DATA_CSV="/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink/preprocessed_dataset/preprocessed_data.csv"
MODEL_PATH="/data/coleloughbc/GraphMind/LLMs/llama-models/Meta-Llama-3.1-8B-hf-converted"
MAX_ENTRIES=100  # Set to desired number or leave blank for all
MAX_LENGTH=2048
CHECKPOINT=10

# Run the Python script with specified arguments
echo "Starting QA dataset generation..."
/data/coleloughbc/miniconda3/envs/ClinIQLink/bin/python "$SCRIPT_DIR/generate_QA_dataset.py" \
    "$PREPROCESSED_DATA_CSV" \
    "$MODEL_PATH" \
    "$OUTPUT_DIR" \
    --max_length ${MAX_LENGTH:-} \
    --checkpoint ${CHECKPOINT:-} \
    --max_entries ${MAX_ENTRIES:-} \
    --debugging

echo "QA dataset generation completed."
