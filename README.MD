# ClinIQLink: QA Dataset Generation

### Author: Brandon Colelough  
### Last Updated: November 14, 2024  
### License: MIT  

---

## Table of Contents
- [1. Project Overview](#project-overview)
- [2. Folder Structure](#folder-structure)
- [3. Installation](#installation)
- [4. Usage](#usage)
- [5. Configuration](#configuration)
- [6. Methods](#methods)
- [7. Output Files](#output-files)
- [8. Troubleshooting](#troubleshooting)
- [9. License](#license)

---

## 1. Project Overview

ClinIQLink is a project for generating question-answer (QA) datasets from medical textbooks, using neuro-symbolic AI techniques. This project processes JSON files of preprocessed textbook paragraphs, applies NLP methods to generate relevant QA pairs, and outputs the results in JSON and CSV formats. The generated dataset is intended for training AI systems, especially in domains like medical education, where complex, domain-specific knowledge is essential.

The QA generation process uses Named Entity Recognition (NER), semantic entropy, and language model pipelines to determine the most appropriate question type for each paragraph. The generated questions are factual, true/false, or list types, providing a diverse dataset suitable for training AI on comprehension and reasoning tasks.

---


## 2. Folder Structure

``` bash
ClinIQLink/
├── include/
│   ├── bash_scripts_process_textbook/
│   ├── preprocess_textbook_paragraph.py
│   ├── preprocess_textbook_sentence.py
│   └── preprocess_textbook.sh
├── LICENSE
├── post_install.py
├── post_processing/
│   ├── nCloze
│   ├── optimize_extracted_qa_dataset.py
│   ├── optimize_extracted_qa_dataset_llama_3.py
│   ├── requirements_optimize.txt
│   ├── run_optimize_extracted_qa_dataset_fix_clean.sh
│   ├── run_optimize_extracted_qa_dataset_LF.sh
│   ├── run_optimize_extracted_qa_dataset_llama_3.sh
│   ├── run_optimize_extracted_qa_dataset_llama_3_small.sh
│   ├── split_mc_json.py
│   └── split_mc_json.sh
├── pre_processing/
│   ├── preprocess_textbook_paragraph.py
│   ├── preprocess_textbook_sentence.py
│   └── preprocess_textbook.sh
├── prompt_templates/
│   ├── QA_LIST.prompt
│   ├── QA_MC.prompt
│   ├── QA_MULTI-HOP-Inverse.prompt
│   ├── QA_MULTI-HOP.prompt
│   ├── QA_SHORT-Inverse.prompt
│   ├── QA_SHORT.prompt
│   └── QA_TF.prompt
├── README.MD
├── requirements.txt
└── src/
    ├── generate_QA_dataset_multi_processing.py
    ├── generate_QA_dataset_multi_processing.sh
    ├── generate_QA_dataset.sh
    ├── generate_QA_dataset.py
    ├── model_parameter_testing.py
    └── model_parameter_testing.sh
```

---

## 3. Installation

### A. Clone the Repository

Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/Brandonio-c/ClinIQLink.git
cd ClinIQLink
```
### B. Set Up the Python Environment
Use Conda to create and activate a virtual environment for the project.

```bash
conda create -n ClinIQLink python=3.10 -y
conda activate ClinIQLink
```

### C. Install Requirements
Install the required Python packages and download the SpaCy model.

```bash
pip install -r requirements.txt
python post_install.py  # Ensures SpaCy 'en_core_web_sm' model is downloaded
```

## 4. Usage
To generate a QA dataset, run the main script generate_QA_dataset.py with the required arguments:

```bash
python src/generate_QA_dataset.py <preprocessed_csv> <llama3_model_path> <output_dir> --max_entries <optional_max_entries>
python src/generate_QA_dataset.py preprocess_dataset/preprocessed_data.csv /path/to/llama3_model preprocess_dataset/QA_Dataset --max_entries 10
```

## 5. Arguments
preprocessed_csv: Path to the CSV file containing links to the preprocessed JSON files.
llama3_model_path: Path to the LLaMA3 model directory.
output_dir: Directory where the generated QA files will be saved.
--max_entries: (Optional) Maximum number of paragraphs to process per JSON file.
Configuration
Prompt Templates
The prompt templates for generating questions are located in the prompt_templates directory. Each template is customized for generating different question types:

QA_SHORT.prompt: For generating factual questions.
QA_MC.prompt: For generating true/false questions.
QA_LIST.prompt: For generating list-type questions.
Modify these templates if needed to customize QA generation.

## 6. Methods
Main Class: QADatasetGenerator
Key Methods:
generate_factual_question: Generates factual QA pairs.
generate_true_false_question: Generates true/false QA pairs.
generate_list_question: Generates list QA pairs.
is_content_informative: Determines if a paragraph is informative enough for QA generation, based on content density, named entity presence, and coherence.
determine_question_type_v2: Uses semantic entropy and entity recognition to decide on the question type for each paragraph.
save_qa_pairs: Saves generated QA pairs in JSON and CSV formats.
For more details, see inline documentation in generate_QA_dataset.py.

## 7. Output Files
Generated QA pairs are saved in the specified output directory as follows:

factual_qa.json: Contains factual questions generated from paragraphs.
true_false_qa.json: Contains true/false questions.
list_qa.json: Contains list-type questions.
QA_dataset.csv: Consolidated CSV file of all QA pairs, including fields like question, answer, type, ISBN, paragraph ID, and page number.


## 8.Troubleshooting
Common Errors and Fixes
FileNotFoundError: Ensure that all JSON files referenced in preprocessed_csv are accessible and that relative paths are correctly resolved.

OSError (SpaCy Model): Run python post_install.py if SpaCy model errors occur, or manually download the model using:

```bash
python -m spacy download en_core_web_sm
```


## 9. License
This project is licensed under the MIT License. See the LICENSE file for details.


