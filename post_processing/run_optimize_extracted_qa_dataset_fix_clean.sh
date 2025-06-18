#!/bin/bash
#SBATCH --mem-bind=local
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --job-name=clean_short_inverse
#SBATCH --output=clean_short_inverse.out

# Activate conda env that already has PyTorch 2.3 / Transformers ≥ 4.41
conda init
source /data/coleloughbc/miniconda3/etc/profile.d/conda.sh
conda activate llama-stack

# Define paths
WORK_DIR=/data/coleloughbc/ClinIQLink-Challenge/data/fix
SCRIPT_PATH=$WORK_DIR/optimize_extracted_qa_dataset_llama_3.py
SOURCE_TEXTS=$WORK_DIR/source_texts
INPUT_DIR=$WORK_DIR/extracted_db_qa_pairs_human_expert_annotated_verified
OUTPUT_DIR=$INPUT_DIR/updated_clean_short_inv
WORDLIST_UNIX=$WORK_DIR/nCloze/dict-unix.txt
WORDLIST_INFO=$WORK_DIR/nCloze/dict-info.txt
PROFANITY_JSON=$WORK_DIR/nCloze/profanity.json
NLTK_CACHE=$WORK_DIR/nltk_cache

# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"

# Set NLTK cache location
export NLTK_DATA=$NLTK_CACHE

# Run script to clean only short_inverse.json
python "$SCRIPT_PATH" \
  --source-dir "$SOURCE_TEXTS" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --unix-wordlist "$WORDLIST_UNIX" \
  --info-wordlist "$WORDLIST_INFO" \
  --profanity-file "$PROFANITY_JSON" \
  --clean
