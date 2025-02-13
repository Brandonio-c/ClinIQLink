#!/bin/bash
#SBATCH --partition=gpu                                                         # Request the GPU partition
#SBATCH --gres=gpu:a100:2                                                       # Request 2 A100 GPUs per node
#SBATCH --nodes=1                                                               # Number of nodes to allocate
#SBATCH --ntasks-per-node=2                                                     # Number of tasks per node
#SBATCH --cpus-per-task=32                                                      # Number of CPUs per task
#SBATCH --mem=164G                                                              # Request Ram memory per node
#SBATCH --time=48:00:00                                                         # 48 hours 
#SBATCH --job-name=generate_qa_dataset_llama3-3_70B                                          # Set job name
#SBATCH --output=generate_qa_dataset_llama3-3_70B.out                                        # Output file name
#SBATCH --error=generate_qa_dataset_llama3-3_70B.err                                         # set error file name 

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
PREPROCESSED_DATA_CSV="/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink/preprocessed_dataset/978-0323393041.csv"
MODEL_PATH="/data/coleloughbc/LLAMA-3-2/HF_Converted_llama-3-3_70B_instruct_HF"
MAX_ENTRIES=10000  # Set to desired number or leave blank for all
MAX_SEQUENCE_LENGTH=2048
MAX_NEW_TOKENS=1024  # (or adjust dynamically so prompt + generated tokens ≤ 2048)
MODEL_MAX_LENGTH=131072 
CHECKPOINT=25
START_PARAGRAPH=1  # Change this value to resume from a specific paragraph


# Run the Python script with specified arguments
echo "Starting QA dataset generation..."

/data/coleloughbc/miniconda3/envs/ClinIQLink/bin/python  "$SCRIPT_DIR/generate_QA_dataset.py" \
    "$PREPROCESSED_DATA_CSV" \
    "$MODEL_PATH" \
    "$OUTPUT_DIR" \
    --max_new_tokens ${MAX_NEW_TOKENS:-} \
    --checkpoint ${CHECKPOINT:-} \
    --max_entries ${MAX_ENTRIES:-} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH:-} \
    --model_max_length ${MODEL_MAX_LENGTH:-} \
    --start_paragraph ${START_PARAGRAPH:-} \
    --debugging

echo "QA dataset generation completed."

# change nproc for number of GPU's in use! 
