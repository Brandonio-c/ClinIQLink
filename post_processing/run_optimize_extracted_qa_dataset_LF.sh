#!/bin/bash
#SBATCH --mem-bind=local
#SBATCH --partition=gpu                   # GPU partition
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --job-name=optimize_cliniqlink_qa_longformer
#SBATCH --output=optimize_cliniqlink_qa_longformer.out

# Activate conda env that already has PyTorch 2.3 / Transformers ≥ 4.41
conda init
source /data/coleloughbc/miniconda3/etc/profile.d/conda.sh
conda activate llama-stack

module purge
module load gcc/11.3.0
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Define working directories
WORK_DIR=/data/coleloughbc/ClinIQLink-Challenge/data/fix
SCRIPT_PATH=$WORK_DIR/optimize_extracted_qa_dataset.py
SOURCE_TEXTS=$WORK_DIR/source_texts
INPUT_DIR=$WORK_DIR/extracted_db_qa_pairs_human_expert_annotated_verified
OUTPUT_DIR=$INPUT_DIR/updated_longformer
WORDLIST_UNIX=$WORK_DIR/nCloze/dict-unix.txt
WORDLIST_INFO=$WORK_DIR/nCloze/dict-info.txt
PROFANITY_JSON=$WORK_DIR/nCloze/profanity.json
NLTK_CACHE=$WORK_DIR/nltk_cache
#MODEL_PATH=/data/coleloughbc/LLAMA-3-2/HF_Converted_llama-3-3_70B_instruct_HF

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set cache location for nltk
export NLTK_DATA=$NLTK_CACHE

# Activate conda environment if needed
# source ~/.bashrc
# conda activate your-env-name

# Run script
python "$SCRIPT_PATH" \
  --source-dir "$SOURCE_TEXTS" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --unix-wordlist "$WORDLIST_UNIX" \
  --info-wordlist "$WORDLIST_INFO" \
  --profanity-file "$PROFANITY_JSON" \
  --max-subwords 200 \
  --distractor-pool 128 \
  --top-k 50 \
  --select-n 3 \
  --alpha 0.4 \
  --beta 0.2 \
  --clean \
  --distractors \
  --method default  \
  --log-file "$OUTPUT_DIR/optimize_cliniqlink_qa_LF.log" \
  --debug
#  --mask-model-path "$MODEL_PATH" \
#  --embed-model-path "$MODEL_PATH"
