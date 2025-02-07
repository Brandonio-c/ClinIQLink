#!/bin/bash
#SBATCH --partition=gpu                                                         # Request the GPU partition
#SBATCH --gres=gpu:a100:1                                                       # Request 2 A100 GPUs per node
#SBATCH --nodes=1                                                               # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                                                     # Number of tasks per node
#SBATCH --cpus-per-task=32                                                      # Number of CPUs per task
#SBATCH --mem=86G                                                              # Request Ram memory per node
#SBATCH --time=04:00:00                                                         # 2 hours 0 minutes for testing
#SBATCH --job-name=generate_qa_dataset_llama3-3_70B                                          # Set job name
#SBATCH --output=generate_qa_dataset_llama3-3_70B.out                                        # Output file name
#SBATCH --error=generate_qa_dataset_llama3-3_70B.err                                         # set error file name 

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=0  # Allow direct GPU communication
export NCCL_DEBUG=INFO  # Debug NCCL issues if needed
export OMP_NUM_THREADS=1  # use 1 thread for testing! 
export CUDA_VISIBLE_DEVICES=0 #,1  # Explicitly allocate GPUs for multi-GPU setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Set distributed training environment variables
export RANK=${SLURM_PROCID} 
export WORLD_SIZE=${SLURM_NTASKS}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
while ss -tuln | awk '{print $4}' | grep -q ":$MASTER_PORT$"; do
    export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
done
echo "Using MASTER_PORT: $MASTER_PORT"

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
OUTPUT_DIR="$BASEDIR/QA_dataset_multi"
PREPROCESSED_DATA_CSV="/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink/preprocessed_dataset/preprocessed_data.csv"
MODEL_PATH="/data/coleloughbc/LLAMA-3-2/HF_Converted_llama-3-3_70B_instruct_HF"
MAX_ENTRIES=10000  # Set to desired number or leave blank for all
MAX_NEW_TOKENS=1024
MAX_SEQUENCE_LENGTH=1024
CHECKPOINT=50

# Run the Python script with specified arguments
echo "Starting QA dataset generation..."
/data/coleloughbc/miniconda3/envs/ClinIQLink/bin/torchrun --nproc_per_node=1  "$SCRIPT_DIR/generate_QA_dataset_multi_processing.py" \
    "$PREPROCESSED_DATA_CSV" \
    "$MODEL_PATH" \
    "$OUTPUT_DIR" \
    --max_new_tokens ${MAX_NEW_TOKENS:-} \
    --checkpoint ${CHECKPOINT:-} \
    --max_entries ${MAX_ENTRIES:-} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH:-} \
    --debugging

echo "QA dataset generation completed."

# change nproc for number of GPU's in use! 
