#!/bin/bash
#SBATCH --mem-bind=local
#SBATCH --partition=gpu                   # GPU partition
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --job-name=optimize_cliniqlink_qa_llama_3_small
#SBATCH --output=optimize_cliniqlink_qa_llama_3_small.out

# Activate conda env that already has PyTorch 2.3 / Transformers ≥ 4.41
conda init
source /data/coleloughbc/miniconda3/etc/profile.d/conda.sh
conda activate llama-stack

module purge
module load gcc/11.3.0
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
# export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Define working directories
WORK_DIR=/data/coleloughbc/ClinIQLink-Challenge/data/fix
SCRIPT_PATH=$WORK_DIR/optimize_extracted_qa_dataset_llama_3.py
SOURCE_TEXTS=$WORK_DIR/source_texts
INPUT_DIR=$WORK_DIR/extracted_db_qa_pairs_human_expert_annotated_verified
OUTPUT_DIR=$INPUT_DIR/updated_llama_3_small
WORDLIST_UNIX=$WORK_DIR/nCloze/dict-unix.txt
WORDLIST_INFO=$WORK_DIR/nCloze/dict-info.txt
PROFANITY_JSON=$WORK_DIR/nCloze/profanity.json
NLTK_CACHE=$WORK_DIR/nltk_cache
MODEL_PATH=/data/coleloughbc/LLAMA-3-2/HF_Converted_llama-3-1_8B_instruct_HF
INPUT_FILE=/data/coleloughbc/ClinIQLink-Challenge/data/fix/extracted_db_qa_pairs_human_expert_annotated_verified/MC.json

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set cache location for nltk
export NLTK_DATA=$NLTK_CACHE

# Activate conda environment if needed
# source ~/.bashrc
# conda activate your-env-name

# Run script
# Run script
python "$SCRIPT_PATH" \
  --source-dir "$SOURCE_TEXTS" \
  --input-dir "$INPUT_DIR" \
  --input_file "$INPUT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --unix-wordlist "$WORDLIST_UNIX" \
  --info-wordlist "$WORDLIST_INFO" \
  --profanity-file "$PROFANITY_JSON" \
  --extend_subwords \
  --max-subwords 3 \
  --distractor-pool 64 \
  --top-k 100 \
  --select-n 3 \
  --alpha 0.4 \
  --beta 0.3 \
  --gamma 0.4 \
  --its 200 \
  --distractors-from-text \
  --distractors \
  --method default \
  --mask-model-path "$MODEL_PATH" \
  --embed-model-path "$MODEL_PATH" \
  --log-file "$OUTPUT_DIR/optimize_cliniqlink_qa_llama_3_small.log" \
  --debug # \
  # --extend_subwords \
  # --debug

  # ======== REQUIRED PATHS ========
# --source-dir: Directory of *_processed.json source texts used for reference context.
# --input-dir: Directory containing the QA datasets (e.g., short_inverse.json, MC.json).
# --output-dir: Where processed QA files will be saved.
# --unix-wordlist: Path to a list of common Unix words for filtering.
# --info-wordlist: Path to a domain-specific wordlist (e.g., medical terms).
# --profanity-file: Path to a JSON file listing words to be filtered as offensive.

# ======== MCQ DISTRACTOR GENERATION PARAMETERS ========
# --top-k: Number of top candidates to retrieve from the masked LM (higher = more diverse).
# --select-n: Number of distractors to keep per question (typically 3–4).
# --alpha: Weight on the incorrectness score (e.g., dissimilarity to the correct answer).
# --beta: Weight on the distinctiveness score (e.g., distractors differ from each other).

# ======== OPERATIONAL MODES ========
# --clean: Only run short_inverse.json cleaning (e.g., filtering or formatting).
# --distractors: Only run distractor regeneration for MC.json.

# ======== METHOD SELECTION ========
# --method: Distractor generation method. 'default' uses masked LM + scoring. 'cdgp' uses a generation-based approach.

# ======== OPTIONAL INPUT TWEAKS ========
# --input_file: If provided, use a specific file instead of the default MC.json.
# --distractor-pool: Number of candidates passed to scoring (after filtering).
# --min-dist: Minimum edit distance between distractors and correct answer.
# --min-sent-words: Minimum length for a sentence in source text to be used for distractors.
# --max-subwords: Max number of subword tokens allowed in any distractor (controls simplicity).
# --extend_subwords: If enabled, tries to extend subwords into full tokens (slow but improves fluency).
# --its: Number of iterations for simulated annealing in distractor selection (higher = better, slower).

# ======== SOURCE TEXT CONTROL ========
# --distractors-from-text: Force all distractors to be selected only from the source context text.

# ======== MODEL CONFIGURATION ========
# --mask-model-path: HuggingFace model name or path for masked language modeling (e.g., fill-mask).
# --embed-model-path: HuggingFace model name or path for computing sentence embeddings.

# ======== LOGGING ========
# --log-file: Where to save logs (debug/info/warning messages).
# --debug: If set, enables verbose logging (useful for debugging, slows down runs).
