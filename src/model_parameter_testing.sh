#!/bin/bash
#SBATCH --partition=gpu                        # Request the GPU partition
#SBATCH --gres=gpu:a100:1                      # Request 1 A100 GPU
#SBATCH --nodes=1                              # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                    # Number of tasks per node
#SBATCH --cpus-per-task=16                     # Number of CPUs per task
#SBATCH --mem=84G                              # Request RAM per node
#SBATCH --time=00:30:00                        # 4-hour job runtime limit
#SBATCH --job-name=debug_llama3                # Job name
#SBATCH --output=debug_llama3.out              # Output file
#SBATCH --error=debug_llama3.err               # Error file

# Load necessary modules
module unload cuDNN
module unload CUDA

echo "Loading CUDA Toolkit..."
module load gcc/11.3.0
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

# Initialize Conda
conda init
source /data/coleloughbc/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

# Activate the correct Conda environment
conda activate ClinIQLink

# Navigate to the correct directory
cd /data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink/src

# Run the Python script
echo "Starting model debugging..."
/data/coleloughbc/miniconda3/envs/ClinIQLink/bin/python model_parameter_testing.py

echo "Debugging complete!"
