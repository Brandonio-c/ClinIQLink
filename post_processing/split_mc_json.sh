#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=breakup_json
#SBATCH --output=breakup_json.out

# Bash script to run the MC.json splitting task

INPUT_FILE=/data/coleloughbc/ClinIQLink-Challenge/data/fix/extracted_db_qa_pairs_human_expert_annotated_verified/MC.json
OUTPUT_DIR=/data/coleloughbc/ClinIQLink-Challenge/data/fix/extracted_db_qa_pairs_human_expert_annotated_verified/MC_files

# Run the Python script
python3 split_mc_json.py --input "$INPUT_FILE" --output_dir "$OUTPUT_DIR"
