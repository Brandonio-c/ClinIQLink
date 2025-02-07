"""
QADatasetGenerator

Author: Brandon Colelough
Date Last Edited: 2025-01-28
License: MIT

Description:
    The QADatasetGenerator class is designed to generate question-answer pairs from textbook data.
    This class processes data from preprocessed JSON files, identifies appropriate question types 
    (factual, true/false, or list) based on content analysis, and generates question-answer pairs.
    The generated QA pairs are saved in JSON and CSV formats, organized by question type.

    The class uses NLP techniques such as Named Entity Recognition (NER) and sentence embeddings 
    to analyze the content and determine the most suitable question type for each paragraph. 
    It also allows customization of the number of entries processed through the `max_entries` parameter.

"""
import os
import json
import sys
import random
import csv
from pathlib import Path
import transformers
import torch
from transformers import pipeline
import argparse
import re
import spacy
import numpy as np
from enum import Enum
import time
from datetime import datetime
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import dispatch_model, infer_auto_device_map
from torch.nn.utils.rnn import pad_sequence
from transformers.utils.logging import set_verbosity_debug
set_verbosity_debug()
#adding to fix: "Disabling tokenizer parallelism, we're using DataLoader multithreading already" suggests the Hugging Face tokenizer might be blocking parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QAType(Enum):
    TRUE_FALSE = "tf"
    MULTIPLE_CHOICE = "mc"
    SHORT = "short"
    LIST = "list"
    MULTI_HOP = "multi_hop"  # New question type

class QADatasetGenerator:
    def __init__(self, preprocessed_csv, llama3_model_path, output_dir, max_new_tokens, max_sequence_length, checkpoint, max_entries=None, debugging=False):
        """
        Initializes the QADatasetGenerator with necessary paths and models.

        Parameters:
            preprocessed_csv (str): Path to the CSV file containing JSON file links for each book.
            llama3_model_path (str): Path to the LLaMA3 model for generating questions.
            output_dir (str): Directory to save the generated QA files.
            max_entries (int, optional): Maximum number of paragraphs to process per book file.
        
        This constructor sets up:
            - The NLP models (NER and sentence embedding)
            - The output directory for storing generated QA pairs
            - The QA pipeline model for text generation
            - The prompt templates for generating different question types
        """

        # Automatically detect GPU and set the device
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        self.start_time = time.time()  # Start the overall timer
        self.init_start_time = datetime.now()
        print(f"[DEBUG] Initialization started at: {self.init_start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        self.preprocessed_csv = Path(preprocessed_csv)
        self.llama3_model_path = llama3_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.max_sequence_length = max_sequence_length
        # Load NER model (SpaCy) and sentence embedding model (Hugging Face)
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
        # Add missing pipeline components only if they are not already present
        if "attribute_ruler" not in self.nlp.pipe_names:
            self.nlp.add_pipe("attribute_ruler")

        if "lemmatizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("lemmatizer", after="attribute_ruler")  # Ensure proper order

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")  # Ensures sentence segmentation

        self.embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        self.max_entries = max_entries
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llama3_model_path, padding_side="left", truncation=True, model_max_length=self.max_sequence_length)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Initialize Distributed Training
        # dist.init_process_group(backend="nccl")  # Set up multi-GPU communication

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    llama3_model_path, 
                    torch_dtype=torch.bfloat16,  # Use bfloat16 as specified
                    device_map="auto" #,  # Automatically map model to available devices (e.g., GPU/CPU)
                    # device_map="balanced"
                    # device_map="balanced_low_0" # ensures the model is evenly split across both A100 GPUs, preventing one GPU from being overloaded.
                    # offload_folder="/tmp",  # Optional: Offload to CPU to save GPU memory
                    # offload_state_dict=True, # Helps in managing large models across multiple GPUs
                    ).eval()  # Put model in inference mode to reduce overhead by disabling gradient computation.
        
        # Ensure padding token is set
        self.model.config.pad_token_id = self.tokenizer.eos_token_id   

        # Move the model to the correct device
        # self.model.to(torch.cuda.current_device())
        # self.tokenizer.to(torch.cuda.current_device())

        
        # if torch.cuda.is_available():
        #     self.model.to(f"cuda:{torch.cuda.current_device()}")

        # torch.cuda.empty_cache()  # Free up memory

        # Wrap the model with DistributedDataParallel (DDP)
        #device_map = infer_auto_device_map(self.model, max_memory={0: "80GB", 1: "80GB"}, dtype=torch.bfloat16)
        # device_map = infer_auto_device_map(self.model)
        # self.model = dispatch_model(self.model, device_map=device_map)
 
        self.qa_pipeline = pipeline(
                                    "text-generation",
                                    model=self.model.module if hasattr(self.model, "module") else self.model,  # Ensure compatibility with DDP
                                    tokenizer=self.tokenizer,
                                    truncation=True,  # Enable truncation
                                    #device=self.device,
                                    batch_size=4,  # Adjust batch size based on available VRAM - A100 GPUs have enough memory to handle batch size 16-32
                                    max_new_tokens=self.max_new_tokens,  # Set the maximum token length to 2048
                                    pad_token_id=self.tokenizer.eos_token_id,  # Explicitly set padding token
                                    return_tensors="pt",
                                    # device=torch.cuda.current_device()
                                )
        
        # Define generation arguments
        # Define generation arguments
        self.generation_args = {
            #"max_new_tokens": 400,  # Limit the number of new tokens per response
            "return_full_text": False,  # Exclude the input text from the response
            "temperature": 0.7,  # Balance between determinism and creativity
            "do_sample": True,   # Enable sampling for varied responses
            "top_k": 50,         # Consider the top 50 tokens for diversity
            "top_p": 0.9,        # Nucleus sampling for balanced output quality
            "repetition_penalty": 1.2,  # Penalize repetition for natural responses
            "eos_token_id": self.tokenizer.eos_token_id,  # Use the model's EOS token
        }

        
        # Load prompt templates
        self.prompt_templates_dir = Path("/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink/prompt_templates")
        self.QA_SHORT_template = self.load_prompt_template(self.prompt_templates_dir / "QA_SHORT.prompt")
        self.QA_MC_template = self.load_prompt_template(self.prompt_templates_dir / "QA_MC.prompt")
        self.QA_LIST_template = self.load_prompt_template(self.prompt_templates_dir / "QA_LIST.prompt")
        self.QA_TF_template = self.load_prompt_template(self.prompt_templates_dir / "QA_TF.prompt")
        self.QA_MULTI_HOP_template = self.load_prompt_template(self.prompt_templates_dir / "QA_MULTI-HOP.prompt")

        
        # Initialize containers for each type of QA pair
        self.short_QA_pairs = []
        self.TF_QA_pairs = []
        self.list_QA_pairs = []
        self.MC_QA_pairs = []
        self.multi_hop_QA_pairs = []
        self.max_entries = max_entries
        self.save_every = checkpoint
        self.debugging=debugging

        self.qa_counts = {  # Initialize counters for each QA type
            QAType.SHORT: 0,
            QAType.TRUE_FALSE: 0,
            QAType.LIST: 0,
            QAType.MULTIPLE_CHOICE: 0,
            QAType.MULTI_HOP: 0,
        }

        self.balance_threshold = 10 # the above two variables are used to balance the number of QA quesiton types are made / distributed. 

        if self.debugging:
            print("Debugging enabled!", flush=True)

        # Print initialization duration
        self.init_end_time = datetime.now()
        print(f"[DEBUG] Initialization completed at: {self.init_end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"[DEBUG] Initialization duration: {(time.time() - self.start_time):.2f} seconds\n", flush=True)

    def load_prompt_template(self, file_path):
        """
        Loads a QA prompt template from a specified file.

        Parameters:
            file_path (Path): The path to the template file.
        
        Returns:
            str: The content of the template file as a string.
        """
        with open(file_path, "r") as file:
            return file.read()

    def load_preprocessed_data(self):
        """
        Loads preprocessed data from the CSV file specified in the initialization.

        Returns:
            list: A list of dictionaries, each containing metadata for a JSON file entry.
        """
        with open(self.preprocessed_csv, "r") as csvfile:
            data = list(csv.DictReader(csvfile))
            return data

    def is_content_informative(self, paragraph_text, i, total_paragraphs, title, isbn):
        """
        Determines if a paragraph is informative enough to generate a QA pair.

        Parameters:
            paragraph_text (str): The paragraph text to evaluate.
        
        Returns:
            bool: True if the paragraph is informative; False otherwise.
        """
        try:
            # Check basic length and complexity
            doc = self.nlp(paragraph_text)
            word_count = len(paragraph_text.split())
            sentence_count = len(list(doc.sents))

            if word_count < 15 or sentence_count < 2:
                if self.debugging:
                    print(f"[DEBUG] Processing paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Word count and / or sentence count too low. Word count: {word_count}, Sentence count: {sentence_count}", flush=True)
                return False  # Too short or simple
            
            # Check for irrelevant content
            irrelevant_keywords = [
            "copyright", "ISBN", "publisher", "permission", "disclaimer", "all rights reserved",
            "reproduction", "permission required", "for educational use only",
            "appendix", "bibliography", "terms and conditions", "privacy policy", "trademark", "advertisement",
            "footnote",  "terms of use", "legal notice", "author biography", "editorial note", "foreword",
            "publication date", "previous editions", "contributor", "acknowledgement",
            "funding disclosure", "license", "disclosure statement", "author note",
            "supporting information", "recommended readings", "learning objectives", "key terms"
            ]
            if any(keyword in paragraph_text.lower() for keyword in irrelevant_keywords):
                if self.debugging:
                    found_keywords = [keyword for keyword in irrelevant_keywords if keyword in paragraph_text.lower()]
                    print(f"[DEBUG] Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Found irrelevant keyword(s): {', '.join(found_keywords)}", flush=True)
                return False  # Contains irrelevant metadata
            
            """ # Detect tables, figures, and metadata
            if re.search(r"(\d+\.\d+|Fig\.|Table|•)", paragraph_text):
                if self.debugging:
                    match = re.search(r"(\d+\.\d+|Fig\.|Table|•)", paragraph_text)
                    print(f"Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Found pattern match: '{match.group(0)}'", flush=True)
                return False """

            # Named Entity Density
            entities = [ent.label_ for ent in doc.ents]
            entity_density = len(entities) / word_count
            if entity_density < 0.01:  # Less than 1% of words are entities - Tunable parameter TODO - Play around with this to determine what is a good threshhold 
                if self.debugging:
                    print(
                            f"[DEBUG] Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - "
                            f"Entity Density too low. Word count: {word_count}, Entities found: {len(entities)}, "
                            f"Entity Density: {entity_density:.3f}, Entities: {entities}"
                        )
                return False  # Not enough entities to be informative

            # Topic Coherence using Semantic Embeddings
            sentences = [sent.text for sent in doc.sents]
            
            # Generate embeddings
            sentence_embeddings = self.embedding_model(sentences, truncation=True, batch_size=4)

            # Ensure embeddings are correctly formatted as PyTorch tensors
            sentence_embeddings = [torch.tensor(embed).squeeze() for embed in sentence_embeddings]

            # Pad sequences to the length of the longest sequence
            padded_embeddings = pad_sequence(sentence_embeddings, batch_first=True, padding_value=0.0)

            # Move to the appropriate device
            sentence_embeddings_tensor = padded_embeddings.to(self.device)

            # Compute variance for coherence check
            embedding_variance = torch.var(sentence_embeddings_tensor).item()

            if embedding_variance > 0.5:
                if self.debugging:
                    print(
                        f"[DEBUG] Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - "
                        f"High embedding variance. Variance: {embedding_variance:.3f}, "
                        f"Sentences analyzed: {[sent.text for sent in doc.sents]}"
                    )
                return False  # High variance suggests incoherent topic

            return True  # Paragraph passed all checks
        
        except Exception as e:
            import traceback
            print(f"[ERROR] Exception in is_content_informative: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            return True  # Default return value if error occurs
    
    def extract_text_between_markers(self, response_text, occurrence=1):
        """
        Extracts text between the <startofQAtext> and <endofQAtext> markers.

        Parameters:
            response_text (str): The text containing the markers.
            occurrence (int): The occurrence of the start and end markers to use (1-based index).

        Returns:
            str: Extracted text if markers are found, otherwise an error message.
        """
        # Define markers (these are set in the prompt templates and are always:)
        start_marker = "<startofQAtext>"
        end_marker = "<endofQAtext>"

        if self.debugging:
            print(f"\n[DEBUG] Extracting text between markers (Occurrence: {occurrence})...", flush=True)
            print(f"[DEBUG] Raw response text (first 500 chars): {response_text[:500]}", flush=True)


        # Find all occurrences of the markers
        start_indices = [i for i in range(len(response_text)) if response_text.startswith(start_marker, i)]
        end_indices = [i for i in range(len(response_text)) if response_text.startswith(end_marker, i)]

        if self.debugging:
            print(f"[DEBUG] Start marker indices: {start_indices}", flush=True)
            print(f"[DEBUG] End marker indices: {end_indices}", flush=True)

        # Ensure the specified occurrence exists
        if len(start_indices) >= occurrence and len(end_indices) >= occurrence:
            start_index = start_indices[occurrence - 1] + len(start_marker)
            end_index = end_indices[occurrence - 1]
            extracted_text = response_text[start_index:end_index].strip()

            if self.debugging:
                print(f"[DEBUG] Extracted text (first 300 chars): {extracted_text[:300]}", flush=True)

            return extracted_text
        else:
            if self.debugging:
                print(f"[WARNING] Start or end marker occurrence {occurrence} not found in the generated output!", flush=True)
            return None
       
    def generate_short_answer_QA(self, paragraph_text, source_info):
        """
        Generates a short-answer QA pair from a paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to generate a question from.
            source_info (dict): Metadata about the paragraph's source.

        Returns:
            dict or None: A dictionary with question-answer data or None if not generated.
        """
        # Prepare the prompt for the short-answer QA
        prompt = self.QA_SHORT_template.replace("{paragraph_text}", paragraph_text)
        if self.debugging:
            print("[DEBUG] - Prompt used to generate short answer QA sets", flush=True)
            print(prompt)
        try:
            tokenized_input = self.tokenizer(prompt, return_tensors="pt")
            input_length = tokenized_input["input_ids"].shape[1]  # Get sequence length

            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Truncating!", flush=True)
                prompt = self.tokenizer.decode(
                            self.tokenizer(prompt, return_tensors="pt", max_length=self.max_sequence_length, truncation=True)["input_ids"][0], 
                            skip_special_tokens=True
                        )

            # Generate response using the QA pipeline
            response = self.qa_pipeline(
                                            prompt, 
                                            max_new_tokens=self.max_new_tokens,  # Ensures the response adheres to max_new_tokens
                                            truncation=True,  # Ensures long inputs are truncated properly
                                            num_return_sequences=1,  # Generates only one response
                                            **self.generation_args  # Passes additional generation arguments
                                        )
            if self.debugging:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print("[DEBUG]", flush=True)
                print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if "not found" in extracted_text:
                print(f"[WARNING] Markers not found in response: {qa_text}", flush=True)
            

            # Split the generated text into question and answer
            if "\nAnswer:" in extracted_text:
                qa_parts = extracted_text.split("\nAnswer:", 1)
                question = qa_parts[0].strip()  # Text before "Answer:" is the question
                answer = qa_parts[1].strip()  # Text after "Answer:" is the answer

                # Return the structured QA pair
                return {
                    "question": question,
                    "answer": answer,
                    "type": "short_answer",
                    "source": source_info
                }
            else:
                print(f"[ERROR] Invalid QA format found in short answer QA: {qa_text}", flush=True)
                return None

        except Exception as e:
            print(f"[ERROR] Error generating short-answer QA: {e}", flush=True)
            return None

    def generate_TF_QA(self, paragraph_text, source_info):
        """
        Generates a true/false QA pair from a paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to generate a question from.
            source_info (dict): Metadata about the paragraph's source.

        Returns:
            dict or None: A dictionary with question-answer data or None if not generated.
        """

        # Prepare the prompt for the true/false QA
        prompt = self.QA_TF_template.replace("{paragraph_text}", paragraph_text)
        if self.debugging:
            print("[DEBUG] - Prompt used to generate TF QA sets", flush=True)
            print(prompt, flush=True)

        try:
            tokenized_input = self.tokenizer(prompt, return_tensors="pt")
            input_length = tokenized_input["input_ids"].shape[1]  # Get sequence length

            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Truncating!", flush=True)
                prompt = self.tokenizer.decode(
                            self.tokenizer(prompt, return_tensors="pt", max_length=self.max_sequence_length, truncation=True)["input_ids"][0], 
                            skip_special_tokens=True
                        )

            # Generate response using the QA pipeline
            response = self.qa_pipeline(
                                            prompt, 
                                            max_new_tokens=self.max_new_tokens,  # Ensures the response adheres to max_new_tokens
                                            truncation=True,  # Ensures long inputs are truncated properly
                                            num_return_sequences=1,  # Generates only one response
                                            **self.generation_args  # Passes additional generation arguments
                                        )
            if self.debugging:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print("[DEBUG]", flush=True)
                print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if "not found" in extracted_text:
                print(f"[WARNING] Markers not found in response: {qa_text}", flush=True)

            # Split the generated text into question and answer
            if "\nAnswer:" in extracted_text:
                qa_parts = extracted_text.split("\nAnswer:", 1)
                question = qa_parts[0].strip()  # Text before "Answer:" is the question
                generated_answer = qa_parts[1].strip()  # Text after "Answer:" is the answer

                # Validate the generated answer matches the chosen "True" or "False"
                if generated_answer in ["True", "False"]:
                    return {
                        "question": question,
                        "answer": generated_answer,
                        "type": "true_false",
                        "source": source_info
                    }
                else:
                    print(f"[ERROR] Generated answer mismatch: {generated_answer}", flush=True)
                    return None
            else:
                print(f"[ERROR] Invalid QA format found in TF paragraph: {qa_text}", flush=True)
                return None

        except Exception as e:
            print(f"[ERROR] Error generating true/false QA: {e}", flush=True)
            return None

    def generate_list_QA(self, paragraph_text, source_info):
        """
        Generates a list-type QA pair from a paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to generate a question from.
            source_info (dict): Metadata about the paragraph's source.

        Returns:
            dict or None: A dictionary with question-answer data or None if not generated.
        """
        # Prepare the prompt using the QA_LIST_template
        prompt = self.QA_LIST_template.replace("{paragraph_text}", paragraph_text)
        if self.debugging:
            print("[DEBUG] - Prompt used to generate list QA sets", flush=True)
            print(prompt, flush=True)

        try:
            tokenized_input = self.tokenizer(prompt, return_tensors="pt")
            input_length = tokenized_input["input_ids"].shape[1]  # Get sequence length

            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Truncating!", flush=True)
                prompt = self.tokenizer.decode(
                            self.tokenizer(prompt, return_tensors="pt", max_length=self.max_sequence_length, truncation=True)["input_ids"][0], 
                            skip_special_tokens=True
                        )

            # Generate response using the QA pipeline
            response = self.qa_pipeline(
                                            prompt, 
                                            max_new_tokens=self.max_new_tokens,  # Ensures the response adheres to max_new_tokens
                                            truncation=True,  # Ensures long inputs are truncated properly
                                            num_return_sequences=1,  # Generates only one response
                                            **self.generation_args  # Passes additional generation arguments
                                        )
            if self.debugging:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print("[DEBUG]", flush=True)
                print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if "not found" in extracted_text:
                print(f"[WARNING] Markers not found in response: {qa_text}", flush=True)

            # Validate and split the response into question and answer
            if "\nAnswer:" in extracted_text:
                qa_parts = extracted_text.split("\nAnswer:", 1)
                question = qa_parts[0].strip()  # Text before "Answer:" is the question
                answer = qa_parts[1].strip()  # Text after "Answer:" is the answer

                # Ensure the answer is a list format (e.g., comma-separated or bullet points)
                if re.search(r"(•|,|;|\n)", answer):  # Match common list delimiters or line breaks
                    return {
                        "question": question,
                        "answer": answer,
                        "type": "list",
                        "source": source_info
                    }
                else:
                    print(f"[ERROR] Invalid list format in generated answer: {answer}", flush=True)
                    return None
            else:
                print(f"[ERROR] Invalid QA format found in LIST question: {qa_text}", flush=True)
                return None

        except Exception as e:
            print(f"[ERROR] Error generating list QA: {e}", flush=True)
            return None
  
    def generate_MC_QA(self, paragraph_text, source_info):
        """
        Generates a multiple-choice QA pair from a paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to generate a question from.
            source_info (dict): Metadata about the paragraph's source.

        Returns:
            dict or None: A dictionary with question-answer data or None if not generated.
        """

        # Prepare the MC-specific prompt
        prompt = self.QA_MC_template.replace("{paragraph_text}", paragraph_text)
        if self.debugging:
            print("[DEBUG] - Prompt used to generate MC QA sets", flush=True)
            print(prompt, flush=True)
        try:
            tokenized_input = self.tokenizer(prompt, return_tensors="pt")
            input_length = tokenized_input["input_ids"].shape[1]  # Get sequence length

            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Truncating!", flush=True)
                prompt = self.tokenizer.decode(
                            self.tokenizer(prompt, return_tensors="pt", max_length=self.max_sequence_length, truncation=True)["input_ids"][0], 
                            skip_special_tokens=True
                        )

            # Run the QA pipeline
            response = self.qa_pipeline(
                                                prompt, 
                                                max_new_tokens=self.max_new_tokens,  # Ensures the response adheres to max_new_tokens
                                                truncation=True,  # Ensures long inputs are truncated properly
                                                num_return_sequences=1,  # Generates only one response
                                                **self.generation_args  # Passes additional generation arguments
                                            )
            if self.debugging:
                    print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                    print("[DEBUG]", flush=True)
                    print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if "not found" in extracted_text:
                print(f"[WARNING] Markers not found in response: {qa_text}", flush=True)

            # Split the generated response into the question, correct answer, and distractors
            qa_parts = re.split(r"\n(A|B|C|D):", extracted_text)  # Split at multiple-choice options (A, B, C, D)
            if len(qa_parts) < 5:  # Ensure the response contains at least one question and four options
                print(f"[ERROR] Invalid MC format: {qa_text}", flush=True)
                return None

            question = qa_parts[0].strip()  # The text before the first option is the question
            options = {  # The rest are options
                "A": qa_parts[2].strip(),
                "B": qa_parts[4].strip(),
                "C": qa_parts[6].strip(),
                "D": qa_parts[8].strip()
            }

            # Identify the correct answer (assumed to be marked in the template or pipeline response)
            correct_option = None
            for key, option in options.items():
                if "[Correct]" in option:  # Assume correct answer is marked with [Correct]
                    correct_option = key
                    options[key] = option.replace("[Correct]", "").strip()  # Remove marker from correct option

            if not correct_option:
                print(f"[ERROR] No correct option identified in: {qa_text}", flush=True)
                return None

            # Return the structured QA pair
            return {
                "question": question,
                "options": options,
                "correct_answer": correct_option,
                "type": "multiple_choice",
                "source": source_info
            }
        except Exception as e:
            print(f"[ERROR] Error generating MC QA: {e}", flush=True)
            return None

    def generate_multi_hop_QA(self, paragraph_text, source_info):
        """
        Generates a multi-hop QA pair from a paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to generate a question from.
            source_info (dict): Metadata about the paragraph's source.

        Returns:
            dict or None: A dictionary with question-answer data or None if not generated.
        """
        # Prepare the prompt for the multi-hop QA
        prompt = self.QA_MULTI_HOP_template.replace("{paragraph_text}", paragraph_text)
        if self.debugging:
            print("[DEBUG] - Prompt used to generate multi-hop QA sets", flush=True)
            print(prompt, flush=True)

        try:
            tokenized_input = self.tokenizer(prompt, return_tensors="pt")
            input_length = tokenized_input["input_ids"].shape[1]  # Get sequence length

            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Truncating!", flush=True)
                prompt = self.tokenizer.decode(
                            self.tokenizer(prompt, return_tensors="pt", max_length=self.max_sequence_length, truncation=True)["input_ids"][0], 
                            skip_special_tokens=True
                        )

            # Generate response using the QA pipeline
            response = self.qa_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                truncation=True,
                num_return_sequences=1,
                **self.generation_args,
            )
            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if "not found" in extracted_text:
                print(f"[WARNING] Markers not found in response: {qa_text}", flush=True)

            # Parse the extracted text into question, answer, and reasoning
            if "\nAnswer:" in extracted_text and "\nReasoning:" in extracted_text:
                qa_parts = extracted_text.split("\nAnswer:", 1)
                question = qa_parts[0].strip()
                answer_and_reasoning = qa_parts[1].split("\nReasoning:", 1)
                answer = answer_and_reasoning[0].strip()
                reasoning = answer_and_reasoning[1].strip()

                # Return the structured QA pair
                return {
                    "question": question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "type": "multi_hop",
                    "source": source_info,
                }
            else:
                print(f"[ERROR] Invalid QA format found in multi-hop QA: {extracted_text}", flush=True)
                return None

        except Exception as e:
            print(f"[ERROR] Error generating multi-hop QA: {e}", flush=True)
            return None
        
    def generate_qa_for_type(self, paragraph_text, question_type, source_info):
        """
        Generates a QA pair using the appropriate function based on the question type.

        Parameters:
            paragraph_text (str): The paragraph to generate a question from.
            question_type (QAType): The type of question to generate.
            source_info (dict): Metadata about the paragraph.

        Returns:
            dict or None: Generated QA pair
        """
        if self.debugging:
            print(f"[DEBUG] Generating QA for type: {question_type.value} | Paragraph ID: {source_info.get('paragraph_id', 'Unknown')}", flush=True)

        # # Force model execution on the correct device
        # torch.cuda.set_device(self.device)  

        qa_pair = None

        if question_type == QAType.SHORT:
            if self.debugging:
                print(f"[DEBUG] Generating short answer QA...", flush=True)
            qa_pair = self.generate_short_answer_QA(paragraph_text, source_info)

        elif question_type == QAType.TRUE_FALSE:
            if self.debugging:
                print(f"[DEBUG] Generating True/False QA...", flush=True)
            qa_pair = self.generate_TF_QA(paragraph_text, source_info)

        elif question_type == QAType.LIST:
            if self.debugging:
                print(f"[DEBUG] Generating List QA...", flush=True)
            qa_pair = self.generate_list_QA(paragraph_text, source_info)

        elif question_type == QAType.MULTIPLE_CHOICE:
            if self.debugging:
                print(f"[DEBUG] Generating Multiple Choice QA...", flush=True)
            qa_pair = self.generate_MC_QA(paragraph_text, source_info)

        elif question_type == QAType.MULTI_HOP:
            if self.debugging:
                print(f"[DEBUG] Generating Multi-Hop QA...", flush=True)
            qa_pair = self.generate_multi_hop_QA(paragraph_text, source_info)

        if self.debugging:
            print(f"[DEBUG] QA Generation Result: {qa_pair if qa_pair else 'No QA generated'}", flush=True)

        return qa_pair


    def generate_qa_batch(self, paragraphs, question_types, sources):
        """
        Generates QA pairs in batches while calling the specific QA function for each question type.

        Parameters:
            paragraphs (list): List of paragraph texts.
            question_types (list): List of corresponding question types.
            sources (list): List of metadata dictionaries for each paragraph.

        Returns:
            None (QA pairs are stored directly in their respective lists)
        """
        if self.debugging:
            print(f"[DEBUG] Starting QA batch generation for {len(paragraphs)} paragraphs...", flush=True)

        results = []
        for para, q_type, source in zip(paragraphs, question_types, sources):
            results.append(self.generate_qa_for_type(para, q_type, source)) 

        # Debug collected QA pairs
        if self.debugging:
            print(f"[DEBUG] Completed QA batch generation. Collected {len(results)} QA pairs.", flush=True)

        # Store QA pairs properly
        for qa_pair in results:
            if qa_pair:
                if qa_pair["type"] == "short_answer":
                    self.short_QA_pairs.append(qa_pair)
                elif qa_pair["type"] == "true_false":
                    self.TF_QA_pairs.append(qa_pair)
                elif qa_pair["type"] == "list":
                    self.list_QA_pairs.append(qa_pair)
                elif qa_pair["type"] == "multiple_choice":
                    self.MC_QA_pairs.append(qa_pair)
                elif qa_pair["type"] == "multi_hop":
                    self.multi_hop_QA_pairs.append(qa_pair)

        if self.debugging:
            print(f"[DEBUG] QA pairs stored - Short: {len(self.short_QA_pairs)}, TF: {len(self.TF_QA_pairs)}, "
                f"List: {len(self.list_QA_pairs)}, MC: {len(self.MC_QA_pairs)}, Multi-Hop: {len(self.multi_hop_QA_pairs)}", flush=True)

    def process_paragraph(self, paragraph):
        """
        Processes a single paragraph to determine if it's informative.
        
        Parameters:
            paragraph (dict): Dictionary containing paragraph information.
        
        Returns:
            tuple: (is_informative, paragraph_text, source_info)
        """
        paragraph_text = paragraph.get("Text", "")
        source_info = {
            "isbn": paragraph.get("ISBN", "Unknown ISBN"),
            "paragraph_id": paragraph.get("Paragraph ID", "Unknown ID"),
            "page": paragraph.get("Page", "Unknown Page")
        }

        # Check if the paragraph is informative
        is_informative = self.is_content_informative(paragraph_text, i=0, total_paragraphs=1, title="Unknown", isbn="Unknown")

        return is_informative, paragraph_text, source_info
    
    def generate_qa_pairs_v2(self, data, batch_size=4, num_workers=8):
        """
        Generates QA pairs for each paragraph in each book entry in batches while maintaining prompt-specific functions.

        Parameters:
            data (list): List of dictionaries with metadata and file paths for each book.
            batch_size (int): Number of paragraphs to process in parallel.
            num_workers (int): Number of CPU workers for parallel preprocessing.
        """
        total_entries = len(data)
        process_start_time = time.time()

        if self.debugging:
            print(f"[DEBUG] Starting QA generation for {total_entries} books...", flush=True)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for index, entry in enumerate(data, start=1):
                json_file_path = (self.preprocessed_csv.parent / entry["JSON File"]).resolve()

                try:
                    if self.debugging:
                        print(f"\n[DEBUG] Processing book {index}/{total_entries} - {json_file_path}", flush=True)

                    with open(json_file_path, "r", encoding="utf-8") as json_file:
                        book_data = json.load(json_file)

                    isbn = book_data.get("ISBN", "Unknown ISBN")
                    title = book_data.get("Title", "Unknown Title")
                    paragraphs = book_data.get("Paragraphs", [])

                    total_paragraphs = min(len(paragraphs), self.max_entries or len(paragraphs))

                    if self.debugging:
                        print(f"[DEBUG] Found {total_paragraphs} paragraphs in book: {title} (ISBN: {isbn})", flush=True)

                    # Process paragraphs in parallel
                    batch_paragraphs = []
                    batch_sources = []

                    # Step 1: **Parallel Preprocessing**
                    results = list(pool.map(self.process_paragraph, paragraphs[:total_paragraphs]))

                    for i, (is_informative, paragraph_text, source_info) in enumerate(results):
                        if not is_informative:
                            continue

                        batch_paragraphs.append(paragraph_text)
                        batch_sources.append(source_info)

                        # Process a batch when full
                        if len(batch_paragraphs) >= batch_size or i == total_paragraphs - 1:
                            if self.debugging:
                                print(f"[DEBUG] Processing batch of {len(batch_paragraphs)} paragraphs...", flush=True)

                            # Step 2: **Parallel Question Type Classification**
                            question_types = list(pool.map(self.determine_question_type_v3, batch_paragraphs))

                            # Step 3: **Parallel QA Generation Using Prompt-Specific Functions**
                            self.generate_qa_batch(batch_paragraphs, question_types, batch_sources)

                            # Reset batch lists
                            batch_paragraphs, batch_sources = [], []

                            if i % self.save_every == 0:
                                self.save_qa_pairs(isbn)
                                if self.debugging:
                                    print(f"[DEBUG] Saved QA pairs after {i} paragraphs.", flush=True)

                    # Final save after processing the book
                    self.save_qa_pairs(isbn)

                    if self.debugging:
                        print(f"[DEBUG] Processed book {index}/{total_entries}: {title} (ISBN: {isbn})", flush=True)

                except Exception as e:
                    print(f"[ERROR] Error with file {json_file_path}: {e}", flush=True)

        process_end_time = time.time()
        if self.debugging:
            print(f"[DEBUG] Total processing time: {process_end_time - process_start_time:.2f} seconds", flush=True)

        # Free up GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[DEBUG] CUDA memory cleared successfully.", flush=True)

        # Properly destroy the NCCL process group
        if dist.is_initialized():
            dist.destroy_process_group()
            print("[DEBUG] Process group destroyed successfully.", flush=True)
    
    def generate_qa_pairs(self, data):
        """
        Generates QA pairs for each paragraph in each book entry in the data and saves the dataset iteratively.
        Provides progress updates after processing each entry.

        Parameters:
            data (list): List of dictionaries with metadata and file paths for each book.
        """
        total_entries = len(data)
        process_start_time = time.time()  # Start timer for processing
        checkpoint_start_time = process_start_time  # Initialize checkpoint timer

        if self.debugging:
            print(f"[DEBUG] Starting QA generation for {total_entries} books...", flush=True)

        for index, entry in enumerate(data, start=1):  # Enumerate to track the current entry index
            # Join the relative path with the directory of `preprocessed_csv` to form an absolute path
            json_file_path = (self.preprocessed_csv.parent / entry["JSON File"]).resolve()

            try:
                if self.debugging:
                    print(f"\n[DEBUG] Processing book {index}/{total_entries} - {json_file_path}", flush=True)
                # Attempt to open and read the JSON file
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    book_data = json.load(json_file)
                    
                isbn = book_data.get("ISBN", "Unknown ISBN")
                title = book_data.get("Title", "Unknown Title")

                # Process each paragraph in the JSON file, up to max_entries
                total_paragraphs = len(book_data["Paragraphs"])
                if total_paragraphs > self.max_entries:
                    total_paragraphs = self.max_entries

                if self.debugging:
                    print(f"[DEBUG] Found {total_paragraphs} paragraphs in book: {title} (ISBN: {isbn})", flush=True)

                for i, paragraph in enumerate(book_data["Paragraphs"]):
                    if self.max_entries is not None and i >= self.max_entries:
                        break  # Stop processing further paragraphs if max_entries limit is reached
                    
                    # Print progress update
                    if self.debugging:
                        print(f"\n[DEBUG] Processing paragraph {i+1}/{total_paragraphs} in '{title}' (ISBN: {isbn})", flush=True)

                    paragraph_text = paragraph["Text"]
                    source_info = {
                        "isbn": isbn,
                        "paragraph_id": paragraph.get("Paragraph ID", "Unknown ID"),
                        "page": paragraph.get("Page", "Unknown Page")
                    }
                    if not self.is_content_informative(paragraph_text, i, total_paragraphs, title, isbn):
                        if self.debugging:
                            print(f"[DEBUG] Paragraph {i+1} skipped (not informative).", flush=True)
                            print("Skipped paragraph text is:", flush=True)
                            print(paragraph_text, flush=True)
                        continue # skip if not informative

                    # Print progress update
                    # print(f"Determining question type for paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn})", flush=True)
                    # Determine question type based on content
                    if self.debugging:
                        print(f"[DEBUG] Determining question type for paragraph {i+1}...", flush=True)
                    question_type = self.determine_question_type_v2(paragraph_text)
                    if self.debugging:
                        print(f"[DEBUG] Selected question type: {question_type.value}", flush=True)

                    # Print progress update
                    # print(f"Generating QA pair for paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn})", flush=True)
                    # Generate QA pairs based on determined question type
                    qa_pair = None  # Initialize the QA pair as None before the block

                    if question_type == QAType.SHORT:
                        if self.debugging:
                            print(f"[DEBUG] Generating short answer QA at paragraph {i+1}...", flush=True)
                        qa_pair = self.generate_short_answer_QA(paragraph_text, source_info)
                        if qa_pair:
                            if self.debugging:
                                print(f"[DEBUG] Success for generating short answer QA at paragraph {i+1}...", flush=True)
                            self.short_QA_pairs.append(qa_pair)

                    elif question_type == QAType.TRUE_FALSE:
                        if self.debugging:
                            print(f"[DEBUG] Generating True/False QA at paragraph {i+1}...", flush=True)
                        qa_pair = self.generate_TF_QA(paragraph_text, source_info)
                        if qa_pair:
                            if self.debugging:
                                print(f"[DEBUG] Success for generating True/False QA at paragraph {i+1}...", flush=True)
                            self.TF_QA_pairs.append(qa_pair)

                    elif question_type == QAType.LIST:
                        if self.debugging:
                            print(f"[DEBUG] Generating List QA at paragraph {i+1}...", flush=True)
                        qa_pair = self.generate_list_QA(paragraph_text, source_info)
                        if qa_pair:
                            if self.debugging:
                                print(f"[DEBUG] Success for generating List QA at paragraph {i+1}...", flush=True)
                            self.list_QA_pairs.append(qa_pair)

                    elif question_type == QAType.MULTIPLE_CHOICE:
                        if self.debugging:
                            print(f"[DEBUG] Generating Multiple Choice QA at paragraph {i+1}...", flush=True)
                        qa_pair = self.generate_MC_QA(paragraph_text, source_info)
                        if qa_pair:
                            if self.debugging:
                                print(f"[DEBUG] Success for generating Multiple Choice QA at paragraph {i+1}...", flush=True)
                            self.MC_QA_pairs.append(qa_pair)

                    elif question_type == QAType.MULTI_HOP:
                        if self.debugging:
                            print(f"[DEBUG] Generating Multi-Hop QA at paragraph {i+1}...", flush=True)
                        qa_pair = self.generate_multi_hop_QA(paragraph_text, source_info)
                        if qa_pair:
                            if self.debugging:
                                print(f"[DEBUG] Success for generating Multi-Hop QA at paragraph {i+1}...", flush=True)
                            self.multi_hop_QA_pairs.append(qa_pair)


                    # Debugging step to check if a QA pair was generated
                    if self.debugging:
                        if qa_pair:
                            if question_type in QAType:  # Validate the question type against the enum
                                print(f"[DEBUG] Successfully generated a {question_type.value.upper()} QA pair for Paragraph ID: {source_info['paragraph_id']}", flush=True)
                            else:
                                print(f"[DEBUG] Generated an invalid QA type: {question_type} for Paragraph ID: {source_info['paragraph_id']}", flush=True)
                        else:
                            print(f"[DEBUG] Failed to generate a QA pair for Paragraph ID: {source_info['paragraph_id']}", flush=True)


                    

                    # Save QA pairs every %save_every% iterations 
                    if i % self.save_every == 0:
                        if any([self.short_QA_pairs, self.TF_QA_pairs, self.list_QA_pairs, self.MC_QA_pairs]):
                            self.save_qa_pairs(isbn)
                            if self.debugging:
                                print(f"[DEBUG] Saved QA pairs after processing {i} paragraphs.", flush=True)
                        else:
                            if self.debugging:
                                print(f"[DEBUG] No valid QA pairs generated for {isbn} up to paragraph {i}.", flush=True)
                        
                        checkpoint_end_time = time.time()
                        if self.debugging:
                            print(f"[DEBUG] Checkpoint reached after {i} iterations. Time elapsed: {checkpoint_end_time - checkpoint_start_time:.2f} seconds", flush=True)
                        checkpoint_start_time = checkpoint_end_time  # Reset checkpoint timer



                    # Print progress update
                    if self.debugging:
                        print(f"[DEBUG] Processed entry paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn})", flush=True)
                
                # Save QA pairs iteratively after each entry is processed as a catch all 
                self.save_qa_pairs(isbn)

                # Print progress update
                if self.debugging:
                    print(f"[DEBUG] Processed entry {index}/{total_entries}: {title} (ISBN: {isbn})", flush=True)

                # Print total processing time
                process_end_time = time.time()
                if self.debugging:
                    print(f"[DEBUG] Total processing time: {process_end_time - process_start_time:.2f} seconds\n", flush=True)

            except FileNotFoundError:
                print(f"[ERROR] Error: File '{json_file_path}' not found. Skipping this entry.", flush=True)
            except json.JSONDecodeError:
                print(f"[ERROR] Error: Failed to decode JSON in file '{json_file_path}'. Skipping this entry.", flush=True)
            except Exception as e:
                print(f"[ERROR] An unexpected error occurred with file '{json_file_path}': {e}", flush=True)

    def determine_question_type_v1(self, paragraph_text):
        """
        Determine the type of question based on the content of the paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to analyze.

        Returns:
            QAType: The determined question type (QAType.LIST, QAType.SHORT, QAType.MULTIPLE_CHOICE, QAType.TRUE_FALSE).
        """
        # Check for enumeration/list-like content (e.g., bullet points, lists, or semicolon-separated phrases)
        if re.search(r"(•|,|;|and|or)", paragraph_text) and len(paragraph_text.split()) > 15:
            return QAType.LIST
        
        # Check for multiple-choice cues (e.g., paragraphs mentioning options or structured lists)
        if re.search(r"(option|choice|select|choose|correct answer|following)", paragraph_text, re.IGNORECASE):
            return QAType.MULTIPLE_CHOICE
        
        # Check for factual cues: dates, technical terms, or names
        if re.search(r"\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|[A-Z][a-z]+ [A-Z][a-z]+)\b", paragraph_text):
            return QAType.SHORT
        
        # Default to true/false if none of the above conditions are met
        return QAType.TRUE_FALSE

    def determine_question_type_v2(self, paragraph_text):
        """
        Determines the question type based on combined semantic entropy and entity recognition.

        Parameters:
            paragraph_text (str): The paragraph text to analyze.

        Returns:
            QAType: The determined question type (QAType.SHORT, QAType.TRUE_FALSE, QAType.LIST, or QAType.MULTIPLE_CHOICE).
        """
        # Semantic Analysis
        sentences = [sent.text for sent in self.nlp(paragraph_text).sents]
        sentence_embeddings = self.embedding_model(sentences)

        # Calculate variance as a proxy for semantic entropy
        sentence_embeddings = np.array([np.mean(embed[0], axis=0) for embed in sentence_embeddings])
        semantic_entropy = np.var(sentence_embeddings)

        # Use NER to detect entities
        doc = self.nlp(paragraph_text)
        entities = [ent.label_ for ent in doc.ents]
        unique_entities = set(entities)

        # Detect relationships between entities (e.g., PERSON and DISEASE)
        entity_pairs = [(ent1.label_, ent2.label_) for ent1 in doc.ents for ent2 in doc.ents if ent1 != ent2]

        # Decision Logic Using Both Semantic Analysis and NER


        # Initialize scores for all question types
        scores = {
            QAType.SHORT: 0,
            QAType.TRUE_FALSE: 0,
            QAType.LIST: 0,
            QAType.MULTIPLE_CHOICE: 0,
            QAType.MULTI_HOP: 0,
        }

        # Scoring for 'list' type
        if semantic_entropy > 0.5:
            scores[QAType.LIST] += 1
        if len(unique_entities) > 2:
            scores[QAType.LIST] += 1
        if re.search(r"(•|,|;|and|or)", paragraph_text):
            scores[QAType.LIST] += 1

        # Scoring for 'short' type
        if unique_entities.intersection({'DATE', 'PERSON', 'ORG', 'GPE', 'NORP', 'EVENT'}):
            scores[QAType.SHORT] += 2
        if 0.3 <= semantic_entropy <= 0.5:
            scores[QAType.SHORT] += 1

        # Scoring for 'true/false' type
        if semantic_entropy < 0.3:
            scores[QAType.TRUE_FALSE] += 1
        if len(sentences) <= 2:
            scores[QAType.TRUE_FALSE] += 1
        if not unique_entities:
            scores[QAType.TRUE_FALSE] += 1

        # Scoring for 'multiple-choice' type
        if re.search(r"(option|choice|select|following|correct answer|choose|questionnaire|quiz)", paragraph_text, re.IGNORECASE):
            scores[QAType.MULTIPLE_CHOICE] += 2
        if len(sentences) > 1 and len(unique_entities) > 1:
            scores[QAType.MULTIPLE_CHOICE] += 1
        if re.search(r"(A:|B:|C:|D:)", paragraph_text):  # Explicit multiple-choice format
            scores[QAType.MULTIPLE_CHOICE] += 2

        # Scoring for 'multi-hop' type
        multi_hop_keywords = re.search(
            r"(reasoning|steps|cause|effect|conclusion|linked|sequence|dependencies|relationship|process|interconnected|logic|rationale|pathway|progression|correlation|connections|inference|derivation|justification|explanation|analysis|causal|chain|flow|interaction|dependencies|association|linkages|framework|structure|integration|contextual|stepwise|systematic|hierarchical|multi-step|dynamic)", 
            paragraph_text, 
            re.IGNORECASE
        )
        if multi_hop_keywords:
            scores[QAType.MULTI_HOP] += 2
        if len(unique_entities) > 3 and len(sentences) > 3:
            scores[QAType.MULTI_HOP] += 1
        if len(set(entity_pairs)) > 1:  # Checks for relationships between entities
            scores[QAType.MULTI_HOP] += 1

        # Determine question type with the highest score
        question_type = max(scores, key=scores.get)

        return question_type
    
    def determine_question_type_v3(self, paragraph_text):
        """
        Determines the most suitable question type based on content analysis.

        Parameters:
            paragraph_text (str): The paragraph text to analyze.

        Returns:
            QAType: The determined question type (QAType.SHORT, QAType.TRUE_FALSE, QAType.LIST, QAType.MULTIPLE_CHOICE, QAType.MULTI_HOP).
        """

        # Extract sentences
        sentences = [sent.text for sent in self.nlp(paragraph_text).sents]
        num_sentences = len(sentences)

        # Compute semantic entropy (variance in embeddings)
        sentence_embeddings = self.embedding_model(sentences)
        sentence_embeddings = np.array([np.mean(embed[0], axis=0) for embed in sentence_embeddings])
        semantic_entropy = np.var(sentence_embeddings)

        # Named Entity Recognition (NER)
        doc = self.nlp(paragraph_text)
        entities = [ent.label_ for ent in doc.ents]
        unique_entities = set(entities)

        # Part-of-Speech (POS) Tagging
        pos_tags = [token.pos_ for token in doc]

        # Detecting entity relationships (for multi-hop)
        entity_pairs = [(ent1.label_, ent2.label_) for ent1 in doc.ents for ent2 in doc.ents if ent1 != ent2]

        # Word Frequency & TF-IDF Analysis
        word_counts = {word.lower(): paragraph_text.lower().split().count(word.lower()) for word in paragraph_text.split()}
        
        # Balanced keyword lists for each QA type
        list_keywords = [
            "include", "consist", "list", "examples", "such as", "contains", "types of", "categories",
            "components", "enumerate", "group", "varieties", "kinds of", "different types", "comprises"
        ]
        
        mc_keywords = [
            "option", "choice", "select", "following", "correct answer", "quiz", "test", "exam",
            "multiple-choice", "choose", "best answer", "most likely", "which of these", "distractors", "responses"
        ]
        
        tf_keywords = [
            "true", "false", "fact", "statement", "correct", "incorrect", "yes or no", "valid", "invalid",
            "agree", "disagree", "always", "never", "right or wrong", "affirmative"
        ]
        
        short_keywords = [
            "who", "what", "where", "when", "how many", "how much", "define", "describe", "identify",
            "name", "explain", "mention", "summarize", "tell me about", "state"
        ]
        
        reasoning_keywords = [
            "reasoning", "cause", "effect", "conclusion", "linked", "sequence", "dependencies",
            "relationship", "process", "interconnected", "logic", "rationale", "progression", "inference"
        ]

        # Initialize scores for all question types
        scores = {
            QAType.SHORT: 0,
            QAType.TRUE_FALSE: 0,
            QAType.LIST: 0,
            QAType.MULTIPLE_CHOICE: 0,
            QAType.MULTI_HOP: 0,
        }

        
        # LIST-Type Questions
        if any(keyword in paragraph_text.lower() for keyword in list_keywords):
            scores[QAType.LIST] += 1
        if "," in paragraph_text or ";" in paragraph_text:
            scores[QAType.LIST] += 2
        if len(unique_entities) > 2:
            scores[QAType.LIST] += 1
        if pos_tags.count("NOUN") > 5:  # Lists tend to have more nouns
            scores[QAType.LIST] += 1

        # MULTIPLE CHOICE Questions
        if any(keyword in paragraph_text.lower() for keyword in mc_keywords):
            scores[QAType.MULTIPLE_CHOICE] += 1
        if re.search(r"(A:|B:|C:|D:)", paragraph_text):  # Explicit multiple-choice format
            scores[QAType.MULTIPLE_CHOICE] += 3
        if num_sentences > 1 and len(unique_entities) > 1:
            scores[QAType.MULTIPLE_CHOICE] += 1
        if "which of the following" in paragraph_text.lower():
            scores[QAType.MULTIPLE_CHOICE] += 2

        # TRUE/FALSE Questions
        if any(keyword in paragraph_text.lower() for keyword in tf_keywords):
            scores[QAType.TRUE_FALSE] += 1
        if num_sentences <= 2:  # Shorter paragraphs favor T/F
            scores[QAType.TRUE_FALSE] += 2
        if len(unique_entities) == 0:
            scores[QAType.TRUE_FALSE] += 1
        if semantic_entropy < 0.3:  # Low semantic variance suggests simple binary facts
            scores[QAType.TRUE_FALSE] += 1

        #  SHORT ANSWER Questions
        if any(keyword in paragraph_text.lower() for keyword in short_keywords):
            scores[QAType.SHORT] += 1
        if 0.3 <= semantic_entropy <= 0.5:
            scores[QAType.SHORT] += 1
        if len(unique_entities) >= 1 and num_sentences > 1:
            scores[QAType.SHORT] += 1
        if paragraph_text.endswith("?"):  # If paragraph contains a question-like structure
            scores[QAType.SHORT] += 2

        #  MULTI-HOP Questions
        if any(keyword in paragraph_text.lower() for keyword in reasoning_keywords):
            scores[QAType.MULTI_HOP] += 1
        if len(unique_entities) > 3 and num_sentences > 3:
            scores[QAType.MULTI_HOP] += 2
        if len(set(entity_pairs)) > 1:  # Checks for relationships between entities
            scores[QAType.MULTI_HOP] += 2
        if semantic_entropy > 0.5:
            scores[QAType.MULTI_HOP] += 1

        # Select highest-scoring QA type
        question_type = max(scores, key=scores.get)

        # --------------- TODO - Fix - This is some hamburger code I jumbled together to make the QA type generation process more deterministic but it isn't great and should look to fix this. 

        # Check QA distribution
        min_count = min(self.qa_counts.values())  # Find least-used QA type count
        max_count = max(self.qa_counts.values())  # Find most-used QA type count
        imbalance = max_count - min_count  # Difference between most & least frequent QA types

        # Only balance if imbalance exceeds threshold
        if imbalance >= self.balance_threshold:
            underrepresented_types = [qa for qa, count in self.qa_counts.items() if count == min_count]
            if question_type not in underrepresented_types:
                question_type_type = random.choice(underrepresented_types)  # Prioritize balance

        # Update QA count
        self.qa_counts[question_type] += 1

        # if self.debugging:
        #     print("\n[DEBUG] QA Type Scores:")
        #     for qa_type, score in scores.items():
        #         print(f"  {qa_type.value}: {score}")
        #     print(flush=True)

        #     print(f"[DEBUG] Selected QA Type: {question_type.value}\n", flush=True)
        #     print(f"Unique Entities: {unique_entities}\n", flush=True)
        #     print(f"Semantic Entropy: {semantic_entropy}\n", flush=True)

        return question_type

    def save_json_async(self, file_path, data):
        """Saves JSON data asynchronously to avoid blocking the main thread."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def save_qa_pairs(self, isbn):
        """
        Saves generated QA pairs to JSON files asynchronously and writes a consolidated CSV file.
        """
        if self.debugging:
            print(f"[DEBUG] Saving QA pairs for ISBN: {isbn}", flush=True)

        json_files = {
            f"short_QA-{isbn}.json": self.short_QA_pairs,
            f"TF_QA-{isbn}.json": self.TF_QA_pairs,
            f"list_QA-{isbn}.json": self.list_QA_pairs,
            f"MC_QA-{isbn}.json": self.MC_QA_pairs,
            f"multi_hop_QA-{isbn}.json": self.multi_hop_QA_pairs
        }

        # Debugging QA pair counts
        for file_name, data in json_files.items():
            print(f"[DEBUG] {file_name}: {len(data)} QA pairs", flush=True)

        # Start threads for each JSON file
        threads = []
        for file_name, data in json_files.items():
            file_path = self.output_dir / file_name
            thread = threading.Thread(target=self.save_json_async, args=(file_path, data))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete before proceeding
        for thread in threads:
            thread.join()

        # Write metadata for generated JSON files to the CSV
        csv_path = self.output_dir / "QA_dataset.csv"
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["file_name", "type", "isbn"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for file_name in json_files.keys():
                question_type = file_name.split("_")[0].lower()
                writer.writerow({"file_name": file_name, "type": question_type, "isbn": isbn})

        if self.debugging:
            print(f"[DEBUG] QA dataset CSV saved at {csv_path}", flush=True)



if __name__ == "__main__":
    script_start_time = time.time()
    script_actual_start_time = datetime.now()
    print(f"[DEBUG] Script started at: {script_actual_start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    parser = argparse.ArgumentParser(description="Generate QA dataset from preprocessed text")
    parser.add_argument("preprocessed_csv", help="Path to the preprocessed data CSV file")
    parser.add_argument("llama3_model_path", help="Path to the LLaMA3 model")
    parser.add_argument("output_dir", help="Directory to save the generated QA files")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum length for input and output tokens") #default is 2048 as that is max allowed for LLama 3-1 7B
    parser.add_argument("--max_sequence_length", type=int, default=1024, help="Maximum length for input and output tokens") #default is 2048 as that is max allowed for LLama 3-1 7B
    parser.add_argument("--checkpoint", type=int, default=None, help="number of iterations to save for each book")
    parser.add_argument("--max_entries", type=int, default=None, help="Maximum number of paragraph entries to process")
    parser.add_argument("--debugging", action="store_true", help="Enable debugging mode to print detailed information about skipped paragraphs and processing steps")


    args = parser.parse_args()

    generator = QADatasetGenerator(args.preprocessed_csv, args.llama3_model_path, args.output_dir, args.max_new_tokens, args.max_sequence_length, args.checkpoint, args.max_entries, args.debugging)
    print("[DEBUG] Loading Pre-Processed Data", flush=True)
    data = generator.load_preprocessed_data()
    print("[DEBUG] Pre-Processed Data Loaded!", flush=True)
    print("[DEBUG] Generating QA Pairs...", flush=True)
    generator.generate_qa_pairs_v2(data)

    script_end_time = time.time()
    script_actual_end_time = datetime.now()
    print(f"[DEBUG] Script completed at: {script_actual_end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[DEBUG] Total script execution time: {script_end_time - script_start_time:.2f} seconds\n", flush=True)
