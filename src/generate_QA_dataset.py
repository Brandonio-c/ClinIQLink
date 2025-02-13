"""
QADatasetGenerator

Author: Brandon Colelough
Date Last Edited: 2025-02-03
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
from torch.nn.utils.rnn import pad_sequence

# from transformers.utils.logging import set_verbosity_debug
# set_verbosity_debug()

class QAType(Enum):
    TRUE_FALSE = "tf"
    MULTIPLE_CHOICE = "mc"
    SHORT = "short"
    LIST = "list"
    MULTI_HOP = "multi_hop"
    MULTI_HOP_INVERSE = "multi_hop_inverse"  # New QA type for multi-hop inverse
    SHORT_INVERSE = "short_inverse"  # New QA type for short inverse


class QADatasetGenerator:
    def __init__(self, preprocessed_csv, llama3_model_path, output_dir, max_new_tokens, max_sequence_length, model_max_length, checkpoint, max_entries=None, start_paragraph=0, debugging=False, debugging_verbose=False):
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
        self.start_paragraph = start_paragraph 

        # Automatically detect GPU and set the device
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        self.start_time = time.time()  # Start the overall timer
        self.init_start_time = datetime.now()
        print(f"[DEBUG] Initialization started at: {self.init_start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        self.name = Path(preprocessed_csv).stem
        self.preprocessed_csv = Path(preprocessed_csv)
        self.llama3_model_path = llama3_model_path
        self.output_dir = Path(output_dir) / self.name
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.max_new_tokens = max_new_tokens
        self.max_sequence_length = max_sequence_length
        self.max_entries = max_entries
        self.model_max_length = model_max_length

        #settings for NER used in determining if sentnece is informative 

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

        #settings for NER used in determining if sentnece is informative 

        # LLama 3.3. 70B params 
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(llama3_model_path, padding_side="left", truncation=True, model_max_length=self.max_sequence_length)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                                                            llama3_model_path,
                                                            padding_side="left",
                                                            model_max_length=self.model_max_length,  
                                                            trust_remote_code=True,   # Ensures custom tokenizers work
                                                        )

        # Ensure padding token is correctly set
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    llama3_model_path, 
                    torch_dtype=torch.bfloat16,  # Use bfloat16 as specified
                    device_map="auto" #,  # Automatically map model to available devices (e.g., GPU/CPU)
                    # device_map="balanced"
                    # device_map="balanced_low_0" # ensures the model is evenly split across both A100 GPUs, preventing one GPU from being overloaded.
                    # offload_folder="/tmp",  # Optional: Offload to CPU to save GPU memory
                    # offload_state_dict=True, # Helps in managing large models across multiple GPUs
                    ).eval()  # Put model in inference mode to reduce overhead by disabling gradient computation.

        if self.model.config.max_position_embeddings < 131072:
            print(f"[WARNING] Model's max_position_embeddings ({self.model.config.max_position_embeddings}) is less than expected (131072). Overriding.")
            self.model.config.max_position_embeddings = 131072
        # Ensure padding token is set
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
 
        self.qa_pipeline = pipeline(
                                    "text-generation",
                                    model=self.model,
                                    tokenizer=self.tokenizer,
                                    max_length=self.max_sequence_length,  # set to 2048 (practical prompt length, original setup used 2 A100 GPU's)
                                    #device=self.device,  
                                    truncation=True,
                                    #batch_size=1,  # Reduce batch size for memory efficiency
                                    max_new_tokens=self.max_new_tokens,     # set to 1024 tokens generated (again for practical prompt length, original setup used 2 A100 GPU's)
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    return_tensors="pt",
                                    torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory overhead  
                                )
        
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
        
        if self.debugging:
            print("\n=== MODEL PARAMETER DEBUGGING START ===")

            # Tokenizer Debugging
            print("Loading tokenizer parameters...")

            print("\n[Tokenizer Configuration]")
            print(f"Tokenizer class: {type(self.tokenizer)}")
            print(f"Model max length: {self.tokenizer.model_max_length}")
            print(f"Padding side: {self.tokenizer.padding_side}")
            print(f"Pad token: {self.tokenizer.pad_token}")
            print(f"Pad token ID: {self.tokenizer.pad_token_id}")
            print(f"EOS token ID: {self.tokenizer.eos_token_id}")
            print(f"BOS token ID: {self.tokenizer.bos_token_id}")
            print(f"Vocab size: {self.tokenizer.vocab_size}")

            # Tokenizing a long input to confirm truncation behavior
            test_text = "This is a test sentence. " * 500  # Create a long sequence
            tokens = self.tokenizer(test_text, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)


            print(f"Tokenized sequence length: {tokens['input_ids'].shape[1]}")
            if tokens["input_ids"].shape[1] > self.tokenizer.model_max_length:
                print("ERROR: Tokenized sequence exceeds model_max_length!")

            # Model Debugging
            print("\nLoading model parameters...")

            print("\n[Model Configuration]")
            print(f"Model class: {type(self.model)}")
            print(f"Max position embeddings: {self.model.config.max_position_embeddings}")
            print(f"Hidden size: {self.model.config.hidden_size}")
            print(f"Number of attention heads: {self.model.config.num_attention_heads}")
            print(f"Number of hidden layers: {self.model.config.num_hidden_layers}")
            print(f"Intermediate size: {self.model.config.intermediate_size}")
            print(f"Rope scaling: {self.model.config.rope_scaling}")
            print(f"Rope theta: {self.model.config.rope_theta}")
            print(f"Vocabulary size: {self.model.config.vocab_size}")
            print(f"Model dtype: {self.model.config.torch_dtype}")

            # Verify model max position embeddings
            if self.model.config.max_position_embeddings < 8192:
                print("WARNING: Model max_position_embeddings is unexpectedly low!")

            # Check if model supports higher token limits
            print("\nChecking if model parameters allow long sequences...")
            if self.model.config.max_position_embeddings < 131072:
                print(f"ERROR: Model max position ({self.model.config.max_position_embeddings}) is lower than expected (131072).")
            


        
        # Load prompt templates
        self.prompt_templates_dir = Path("/data/coleloughbc/NIH_ACL_shared-task_LLM-lie-detector/ClinIQLink/prompt_templates")
        self.QA_SHORT_template = self.load_prompt_template(self.prompt_templates_dir / "QA_SHORT.prompt")
        self.QA_MC_template = self.load_prompt_template(self.prompt_templates_dir / "QA_MC.prompt")
        self.QA_LIST_template = self.load_prompt_template(self.prompt_templates_dir / "QA_LIST.prompt")
        self.QA_TF_template = self.load_prompt_template(self.prompt_templates_dir / "QA_TF.prompt")
        self.QA_MULTI_HOP_template = self.load_prompt_template(self.prompt_templates_dir / "QA_MULTI-HOP.prompt")
        self.QA_MULTI_HOP_INVERSE_template = self.load_prompt_template(self.prompt_templates_dir / "QA_MULTI-HOP-Inverse.prompt")
        self.QA_SHORT_INVERSE_template = self.load_prompt_template(self.prompt_templates_dir / "QA_SHORT-Inverse.prompt")


        
        # Initialize containers for each type of QA pair
        self.short_QA_pairs = []
        self.TF_QA_pairs = []
        self.list_QA_pairs = []
        self.MC_QA_pairs = []
        self.multi_hop_QA_pairs = []
        self.multi_hop_inverse_QA_pairs = []  # Stores multi-hop inverse QA pairs
        self.short_inverse_QA_pairs = []  # Stores short-answer inverse QA pairs
        self.max_entries = max_entries
        self.save_every = checkpoint
        self.debugging=debugging
        self.debugging_verbose = debugging_verbose

        self.qa_counts = {  # Initialize counters for each QA type
            QAType.SHORT: 0,
            QAType.TRUE_FALSE: 0,
            QAType.LIST: 0,
            QAType.MULTIPLE_CHOICE: 0,
            QAType.MULTI_HOP: 0,
            QAType.SHORT_INVERSE: 0,
            QAType.MULTI_HOP_INVERSE: 0,
        }

        self.balance_threshold = 10 # the above two variables are used to balance the number of QA quesiton types are made / distributed. 

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
                if self.debugging_verbose:
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
                if self.debugging_verbose:
                    found_keywords = [keyword for keyword in irrelevant_keywords if keyword in paragraph_text.lower()]
                    print(f"[DEBUG] Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Found irrelevant keyword(s): {', '.join(found_keywords)}", flush=True)
                return False  # Contains irrelevant metadata
            
            """ # Detect tables, figures, and metadata
            if re.search(r"(\d+\.\d+|Fig\.|Table|•)", paragraph_text):
                if self.debugging_verbose:
                    match = re.search(r"(\d+\.\d+|Fig\.|Table|•)", paragraph_text)
                    print(f"Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Found pattern match: '{match.group(0)}'", flush=True)
                return False """

            # Named Entity Density
            entities = [ent.label_ for ent in doc.ents]
            entity_density = len(entities) / word_count
            if entity_density < 0.01:  # Less than 1% of words are entities - Tunable parameter TODO - Play around with this to determine what is a good threshhold 
                if self.debugging_verbose:
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
                if self.debugging_verbose:
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

        if self.debugging_verbose:
            print(f"\n[DEBUG] Extracting text between markers (Occurrence: {occurrence})...", flush=True)
            print(f"[DEBUG] Raw response text (first 500 chars): {response_text[:500]}", flush=True)


        # Find all occurrences of the markers
        start_indices = [i for i in range(len(response_text)) if response_text.startswith(start_marker, i)]
        end_indices = [i for i in range(len(response_text)) if response_text.startswith(end_marker, i)]

        if self.debugging_verbose:
            print(f"[DEBUG] Start marker indices: {start_indices}", flush=True)
            print(f"[DEBUG] End marker indices: {end_indices}", flush=True)

        # Ensure the specified occurrence exists
        if len(start_indices) >= occurrence and len(end_indices) >= occurrence:
            start_index = start_indices[occurrence - 1] + len(start_marker)
            end_index = end_indices[occurrence - 1]
            extracted_text = response_text[start_index:end_index].strip()

            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text (first 300 chars): {extracted_text[:300]}", flush=True)

            return extracted_text
        else:
            if self.debugging_verbose:
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
        if self.debugging_verbose:
            print("[DEBUG] - Prompt used to generate short answer QA sets", flush=True)
            print(prompt, flush=True)
        try:
            #tokenize the prompt (with truncation enabled, but without forcing a max_length) to measure its length.
            tokenized_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            input_length = tokenized_input["input_ids"].shape[1]

            # If the prompt is too long, reduce it to reserve tokens for generation.
            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Reducing prompt to allow room for output.", flush=True)
                # Reserve self.max_new_tokens tokens for output.
                allowed_prompt_length = self.max_sequence_length - self.max_new_tokens
                # Re-tokenize with the allowed prompt length.
                tokenized_input = self.tokenizer(prompt, return_tensors="pt", max_length=allowed_prompt_length, truncation=True)
                input_length = tokenized_input["input_ids"].shape[1]
                # Decode back to text so that the prompt is now shortened.
                prompt = self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            # Now compute how many tokens are available for generation.
            available_tokens = self.max_sequence_length - input_length
            dynamic_max_new_tokens = min(self.max_new_tokens, available_tokens)
            if available_tokens < self.max_new_tokens:
                print(f"[WARNING] Only {available_tokens} tokens available for generation (prompt length: {input_length}).", flush=True)

            # Generate the response using the QA pipeline with the dynamically computed new-token limit.
            response = self.qa_pipeline(
                                        prompt,
                                        max_new_tokens=dynamic_max_new_tokens,
                                        truncation=True,
                                        num_return_sequences=1,
                                        **self.generation_args
                                    )
           
            if self.debugging_verbose:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print("[DEBUG]", flush=True)
                print(response[0]["generated_text"], flush=True)

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if not extracted_text:
                print(f"[ERROR] No valid QA content extracted from: {qa_text}", flush=True)
                return None

            # Debugging: print extracted text before processing
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text:\n{extracted_text}\n", flush=True)

            # Define a regex pattern to extract question and answer reliably
            qa_pattern = re.compile(
                r"\*\*Question:\*\*\s*(?P<question>.*?)\s*\*\*Answer:\*\*\s*(?P<answer>.*)",
                re.DOTALL
            )

            # Match against extracted text
            match = qa_pattern.search(extracted_text)

            if match:
                question = match.group("question").strip()
                answer = match.group("answer").strip()

                # Ensure the answer is properly formatted (remove unwanted trailing text)
                answer = re.split(r"(\n|\s{2,})", answer)[0].strip()  # Keep only the first sentence/phrase

                if self.debugging_verbose:
                    print(f"[DEBUG] Extracted Question: {question}", flush=True)
                    print(f"[DEBUG] Extracted Answer: {answer}", flush=True)

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
        
    def generate_short_inverse_QA(self, paragraph_text, source_info):
        """
        Generates a short-answer inverse QA pair (with a false answer) from a paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to generate a question from.
            source_info (dict): Metadata about the paragraph's source.

        Returns:
            dict or None: A dictionary with question-answer data or None if not generated.
        """
        # Prepare the prompt for the short-answer inverse QA
        prompt = self.QA_SHORT_INVERSE_template.replace("{paragraph_text}", paragraph_text)
        if self.debugging_verbose:
            print("[DEBUG] - Prompt used to generate short inverse QA sets", flush=True)
            print(prompt, flush=True)

        try:
            # Tokenize the prompt to measure its length
            tokenized_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            input_length = tokenized_input["input_ids"].shape[1]

            # If the prompt is too long, reduce it to reserve tokens for generation
            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Adjusting.", flush=True)
                allowed_prompt_length = self.max_sequence_length - self.max_new_tokens
                tokenized_input = self.tokenizer(prompt, return_tensors="pt", max_length=allowed_prompt_length, truncation=True)
                input_length = tokenized_input["input_ids"].shape[1]
                prompt = self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            # Calculate available tokens for generation
            available_tokens = self.max_sequence_length - input_length
            dynamic_max_new_tokens = min(self.max_new_tokens, available_tokens)
            if available_tokens < self.max_new_tokens:
                print(f"[WARNING] Only {available_tokens} tokens available for generation (prompt length: {input_length}).", flush=True)

            # Generate the response using the QA pipeline
            response = self.qa_pipeline(
                prompt,
                max_new_tokens=dynamic_max_new_tokens,
                truncation=True,
                num_return_sequences=1,
                **self.generation_args
            )

            if self.debugging_verbose:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print(response[0]["generated_text"], flush=True)

            qa_text = response[0]["generated_text"]

            # Extract text between markers
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if not extracted_text:
                print(f"[ERROR] No valid QA content extracted from: {qa_text}", flush=True)
                return None

            # Debugging: Print extracted text before processing
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text:\n{extracted_text}\n", flush=True)

            # Define regex pattern to extract Question, Incorrect Answer, Explanation, and Correct Answer
            qa_pattern = re.compile(
                r"\*\*Question:\*\*\s*(?P<question>.*?)\s*"
                r"\*\*Answer:\*\*\s*(?P<answer>.*?)\s*"
                r"\*\*Incorrect Answer Explanation:\*\*\s*"
                r"- \*\*False Answer Given:\*\* (?P<false_answer>.*?)\s*"
                r"- \*\*Why It Is Incorrect:\*\* (?P<explanation>.*?)\s*"
                r"- \*\*Correct Answer:\*\* (?P<correct_answer>.*)",
                re.DOTALL
            )

            # Match against extracted text
            match = qa_pattern.search(extracted_text)

            if match:
                question = match.group("question").strip()
                answer = match.group("answer").strip()
                false_answer = match.group("false_answer").strip()
                incorrect_explanation = match.group("explanation").strip()
                correct_answer = match.group("correct_answer").strip()

                # Ensure all fields are present
                missing_fields = []
                if not question:
                    missing_fields.append("Question")
                if not answer:
                    missing_fields.append("Answer")
                if not false_answer:
                    missing_fields.append("False Answer Given")
                if not incorrect_explanation:
                    missing_fields.append("Explanation for Incorrect Answer")
                if not correct_answer:
                    missing_fields.append("Correct Answer")

                if missing_fields:
                    print(f"[ERROR] Missing fields in extracted short inverse QA: {', '.join(missing_fields)}", flush=True)
                    print(f"[DEBUG] Extracted QA content:\n{extracted_text}", flush=True)
                    return None

                return {
                    "question": question,
                    "answer": answer,
                    "false_answer": false_answer,
                    "incorrect_explanation": incorrect_explanation,
                    "correct_answer": correct_answer,
                    "type": "short_inverse",
                    "source": source_info
                }
            else:
                print(f"[ERROR] Invalid QA format in short inverse QA.", flush=True)
                
                # Additional debugging for incorrect formatting
                if "**Question:**" not in extracted_text:
                    print("[DEBUG] 'Question:' keyword not found in extracted text.", flush=True)
                if "**Answer:**" not in extracted_text:
                    print("[DEBUG] 'Answer:' keyword not found in extracted text.", flush=True)
                if "**Incorrect Answer Explanation:**" not in extracted_text:
                    print("[DEBUG] 'Incorrect Answer Explanation:' keyword not found in extracted text.", flush=True)
                if "**False Answer Given:**" not in extracted_text:
                    print("[DEBUG] 'False Answer Given:' keyword not found in extracted text.", flush=True)
                if "**Why It Is Incorrect:**" not in extracted_text:
                    print("[DEBUG] 'Why It Is Incorrect:' keyword not found in extracted text.", flush=True)
                if "**Correct Answer:**" not in extracted_text:
                    print("[DEBUG] 'Correct Answer:' keyword not found in extracted text.", flush=True)

                print(f"[DEBUG] Full extracted text:\n{extracted_text}", flush=True)
                return None

        except Exception as e:
            print(f"[ERROR] Error generating short inverse QA: {e}", flush=True)
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
        if self.debugging_verbose:
            print("[DEBUG] - Prompt used to generate TF QA sets", flush=True)
            print(prompt, flush=True)

        try:
            #tokenize the prompt (with truncation enabled, but without forcing a max_length) to measure its length.
            tokenized_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            input_length = tokenized_input["input_ids"].shape[1]

            # If the prompt is too long, reduce it to reserve tokens for generation.
            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Reducing prompt to allow room for output.", flush=True)
                # Reserve self.max_new_tokens tokens for output.
                allowed_prompt_length = self.max_sequence_length - self.max_new_tokens
                # Re-tokenize with the allowed prompt length.
                tokenized_input = self.tokenizer(prompt, return_tensors="pt", max_length=allowed_prompt_length, truncation=True)
                input_length = tokenized_input["input_ids"].shape[1]
                # Decode back to text so that the prompt is now shortened.
                prompt = self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            # Now compute how many tokens are available for generation.
            available_tokens = self.max_sequence_length - input_length
            dynamic_max_new_tokens = min(self.max_new_tokens, available_tokens)
            if available_tokens < self.max_new_tokens:
                print(f"[WARNING] Only {available_tokens} tokens available for generation (prompt length: {input_length}).", flush=True)

            # Generate the response using the QA pipeline with the dynamically computed new-token limit.
            response = self.qa_pipeline(
                                        prompt,
                                        max_new_tokens=dynamic_max_new_tokens,
                                        truncation=True,
                                        num_return_sequences=1,
                                        **self.generation_args
                                    )
           

            if self.debugging_verbose:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print("[DEBUG]", flush=True)
                print(response[0]["generated_text"], flush=True)

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)

            if not extracted_text:
                print(f"[ERROR] No valid QA content extracted from: {qa_text}", flush=True)
                return None

            # Debugging: Print extracted text before processing
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text:\n{extracted_text}\n", flush=True)

            # Define regex pattern to reliably extract the statement and answer
            qa_pattern = re.compile(
                r"\*\*Statement:\*\*\s*(?P<statement>.*?)\s*"
                r"\*\*Answer:\*\*\s*(?P<answer>\bTrue\b|\bFalse\b)",
                re.DOTALL | re.IGNORECASE
            )

            # Match against extracted text
            match = qa_pattern.search(extracted_text)

            if match:
                statement = match.group("statement").strip()
                answer = match.group("answer").capitalize()  # Normalize case (True/False)

                # Debugging output
                if self.debugging_verbose:
                    print(f"[DEBUG] Extracted Statement: {statement}", flush=True)
                    print(f"[DEBUG] Extracted Answer: {answer}", flush=True)

                return {
                    "question": statement,  # Treating "Statement" as the question
                    "answer": answer,
                    "type": "true_false",
                    "source": source_info
                }
            else:
                print(f"[ERROR] Invalid QA format found in TF paragraph: {qa_text}", flush=True)

                # Additional debugging for incorrect formatting
                if "**Statement:**" not in extracted_text:
                    print("[DEBUG] Missing '**Statement:**' field.", flush=True)
                if "**Answer:**" not in extracted_text:
                    print("[DEBUG] Missing '**Answer:**' field.", flush=True)
                
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
        if self.debugging_verbose:
            print("[DEBUG] - Prompt used to generate list QA sets", flush=True)
            print(prompt, flush=True)

        try:
            #tokenize the prompt (with truncation enabled, but without forcing a max_length) to measure its length.
            tokenized_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            input_length = tokenized_input["input_ids"].shape[1]

            # If the prompt is too long, reduce it to reserve tokens for generation.
            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Reducing prompt to allow room for output.", flush=True)
                # Reserve self.max_new_tokens tokens for output.
                allowed_prompt_length = self.max_sequence_length - self.max_new_tokens
                # Re-tokenize with the allowed prompt length.
                tokenized_input = self.tokenizer(prompt, return_tensors="pt", max_length=allowed_prompt_length, truncation=True)
                input_length = tokenized_input["input_ids"].shape[1]
                # Decode back to text so that the prompt is now shortened.
                prompt = self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            # Now compute how many tokens are available for generation.
            available_tokens = self.max_sequence_length - input_length
            dynamic_max_new_tokens = min(self.max_new_tokens, available_tokens)
            if available_tokens < self.max_new_tokens:
                print(f"[WARNING] Only {available_tokens} tokens available for generation (prompt length: {input_length}).", flush=True)

            # Generate the response using the QA pipeline with the dynamically computed new-token limit.
            response = self.qa_pipeline(
                                        prompt,
                                        max_new_tokens=dynamic_max_new_tokens,
                                        truncation=True,
                                        num_return_sequences=1,
                                        **self.generation_args
                                    )

            if self.debugging_verbose:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print("[DEBUG]", flush=True)
                print(response[0]["generated_text"], flush=True)

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)

            if not extracted_text:
                print(f"[ERROR] No valid QA content extracted from: {qa_text}", flush=True)
                return None

            # Debugging: Print extracted text before processing
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text:\n{extracted_text}\n", flush=True)

            # Define regex pattern to reliably extract question, options list, and answer list
            qa_pattern = re.compile(
                r"\*\*Question:\*\*\s*(?P<question>.*?)\s*"
                r"\*\*Options:\*\*\s*(?P<options>(?:- .+\n?)+)?"
                r"\*\*Answer:\*\*\s*(?P<answer>(?:- .+\n?)+)?",
                re.DOTALL
            )

            # Match against extracted text
            match = qa_pattern.search(extracted_text)

            if match:
                question = match.group("question").strip()

                # Extract and normalize options list
                options_text = match.group("options") or ""
                options = [opt.strip("- ").strip() for opt in options_text.split("\n") if opt.strip()]

                # Extract and normalize answer list
                answer_text = match.group("answer") or ""
                answer_list = [ans.strip("- ").strip() for ans in answer_text.split("\n") if ans.strip()]

                # Debugging output
                if self.debugging_verbose:
                    print(f"[DEBUG] Extracted Question: {question}", flush=True)
                    print(f"[DEBUG] Extracted Options: {options}", flush=True)
                    print(f"[DEBUG] Extracted Answer List: {answer_list}", flush=True)

                # Validate extracted lists
                if not options:
                    print(f"[ERROR] Missing options in extracted QA: {extracted_text}", flush=True)
                    return None
                if not answer_list:
                    print(f"[ERROR] Missing answer list in extracted QA: {extracted_text}", flush=True)
                    return None

                return {
                    "question": question,
                    "options": options,
                    "answer": answer_list,
                    "type": "list",
                    "source": source_info
                }
            else:
                print(f"[ERROR] Invalid QA format found in LIST question: {qa_text}", flush=True)
                
                # Additional debugging for incorrect formatting
                if "**Question:**" not in extracted_text:
                    print("[DEBUG] Missing '**Question:**' field.", flush=True)
                if "**Options:**" not in extracted_text:
                    print("[DEBUG] Missing '**Options:**' field.", flush=True)
                if "**Answer:**" not in extracted_text:
                    print("[DEBUG] Missing '**Answer:**' field.", flush=True)
                
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
        if self.debugging_verbose:
            print("[DEBUG] - Prompt used to generate MC QA sets", flush=True)
            print(prompt, flush=True)
        try:
            #tokenize the prompt (with truncation enabled, but without forcing a max_length) to measure its length.
            tokenized_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            input_length = tokenized_input["input_ids"].shape[1]

            # If the prompt is too long, reduce it to reserve tokens for generation.
            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Reducing prompt to allow room for output.", flush=True)
                # Reserve self.max_new_tokens tokens for output.
                allowed_prompt_length = self.max_sequence_length - self.max_new_tokens
                # Re-tokenize with the allowed prompt length.
                tokenized_input = self.tokenizer(prompt, return_tensors="pt", max_length=allowed_prompt_length, truncation=True)
                input_length = tokenized_input["input_ids"].shape[1]
                # Decode back to text so that the prompt is now shortened.
                prompt = self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            # Now compute how many tokens are available for generation.
            available_tokens = self.max_sequence_length - input_length
            dynamic_max_new_tokens = min(self.max_new_tokens, available_tokens)
            if available_tokens < self.max_new_tokens:
                print(f"[WARNING] Only {available_tokens} tokens available for generation (prompt length: {input_length}).", flush=True)

            # Generate the response using the QA pipeline with the dynamically computed new-token limit.
            response = self.qa_pipeline(
                                        prompt,
                                        max_new_tokens=dynamic_max_new_tokens,
                                        truncation=True,
                                        num_return_sequences=1,
                                        **self.generation_args
                                    )
           
            if self.debugging_verbose:
                    print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                    print("[DEBUG]", flush=True)
                    print(response[0]["generated_text"], flush=True)

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)

            if not extracted_text:
                print(f"[ERROR] No valid QA content extracted from: {qa_text}", flush=True)
                return None

            # Debugging: Print extracted text before processing
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text before processing:\n{extracted_text}\n", flush=True)

            # Extract the question
            question_match = re.search(r"\*\*Question:\*\*\s*(.*?)\s*(?=\*\*Options:\*\*)", extracted_text, re.DOTALL)
            if not question_match:
                # Try finding "Question" in a more relaxed way
                question_match = re.search(r"Question:\s*(.*?)\s*(?=\*\*Options:\*\*|Options:)", extracted_text, re.DOTALL)

            if not question_match:
                print("[ERROR] Failed to locate the '**Question:**' field in extracted text.", flush=True)
                print(f"[DEBUG] Full Extracted Text:\n{extracted_text}", flush=True)
                return None

            question = question_match.group(1).strip()
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted Question: {question}", flush=True)

            # Extract the options block
            options_match = re.search(r"\*\*Options:\*\*\s*(.*?)\s*(?=\*\*Answer:\*\*)", extracted_text, re.DOTALL)
            if not options_match:
                # Try a more relaxed pattern if strict formatting is missing
                options_match = re.search(r"Options:\s*(.*?)\s*(?=\*\*Answer:\*\*|Answer:)", extracted_text, re.DOTALL)

            if not options_match:
                print("[ERROR] Failed to locate the '**Options:**' section in extracted text.", flush=True)
                print(f"[DEBUG] Full Extracted Text:\n{extracted_text}", flush=True)
                return None

            options_block = options_match.group(1).strip()
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted Options Block:\n{options_block}\n", flush=True)

            # Extract the answer
            answer_match = re.search(r"\*\*Answer:\*\*\s*([ABCD])", extracted_text)
            if not answer_match:
                # Try a more relaxed pattern if strict formatting is missing
                answer_match = re.search(r"Answer:\s*([ABCD])", extracted_text)

            if not answer_match:
                print("[ERROR] Failed to locate the '**Answer:**' field in extracted text.", flush=True)
                print(f"[DEBUG] Full Extracted Text:\n{extracted_text}", flush=True)
                return None

            correct_answer = answer_match.group(1).strip()
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted Correct Answer: {correct_answer}", flush=True)

            # Parse individual options using multiple extraction methods
            options = {}

            # Primary regex method (structured extraction)
            option_patterns = [
                (r"A[:.]?\s*(.*?)\s*(?=\nB[:.]?)", "A"),
                (r"B[:.]?\s*(.*?)\s*(?=\nC[:.]?)", "B"),
                (r"C[:.]?\s*(.*?)\s*(?=\nD[:.]?)", "C"),
                (r"D[:.]?\s*(.*?)$", "D")
            ]

            for pattern, label in option_patterns:
                match = re.search(pattern, options_block, re.DOTALL)
                if match:
                    options[label] = match.group(1).strip()

            # Fallback 1: If primary regex fails, attempt line-by-line parsing
            if len(options) < 4:
                if self.debugging_verbose:
                    print(f"[WARNING] Standard regex extraction failed, attempting line-based extraction...", flush=True)

                lines = options_block.split("\n")
                temp_options = {}
                for line in lines:
                    line = line.strip()
                    if line.startswith("A:") or line.startswith("A."):
                        temp_options["A"] = line[2:].strip()
                    elif line.startswith("B:") or line.startswith("B."):
                        temp_options["B"] = line[2:].strip()
                    elif line.startswith("C:") or line.startswith("C."):
                        temp_options["C"] = line[2:].strip()
                    elif line.startswith("D:") or line.startswith("D."):
                        temp_options["D"] = line[2:].strip()

                if len(temp_options) == 4:
                    options = temp_options

            # Fallback 2: If still missing options, try extracting based on newlines
            if len(options) < 4:
                if self.debugging_verbose:
                    print(f"[WARNING] Line-based extraction failed, attempting newline splitting...", flush=True)

                option_lines = [line.strip() for line in options_block.split("\n") if line.strip()]
                if len(option_lines) >= 4:
                    options = {
                        "A": option_lines[0][2:].strip(),
                        "B": option_lines[1][2:].strip(),
                        "C": option_lines[2][2:].strip(),
                        "D": option_lines[3][2:].strip(),
                    }

            # Final Validation
            if len(options) != 4:
                print(f"[ERROR] Incorrect number of options extracted. Expected 4, found {len(options)}.", flush=True)
                print(f"[DEBUG] Extracted Options: {options}", flush=True)
                return None

            if correct_answer not in options:
                print(f"[ERROR] Correct answer '{correct_answer}' is not among the extracted options.", flush=True)
                print(f"[DEBUG] Extracted Options: {options}", flush=True)
                return None

            # Debugging output for final parsed MC question
            if self.debugging_verbose:
                print(f"[DEBUG] Final Parsed Multiple-Choice QA Pair:", flush=True)
                print(f"  Question: {question}", flush=True)
                for key, value in options.items():
                    print(f"  {key}: {value}", flush=True)
                print(f"  Correct Answer: {correct_answer}\n", flush=True)

            # Return the extracted MC question
            return {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
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
        if self.debugging_verbose:
            print("[DEBUG] - Prompt used to generate multi-hop QA sets", flush=True)
            print(prompt, flush=True)

        try:
            #tokenize the prompt (with truncation enabled, but without forcing a max_length) to measure its length.
            tokenized_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            input_length = tokenized_input["input_ids"].shape[1]

            # If the prompt is too long, reduce it to reserve tokens for generation.
            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Reducing prompt to allow room for output.", flush=True)
                # Reserve self.max_new_tokens tokens for output.
                allowed_prompt_length = self.max_sequence_length - self.max_new_tokens
                # Re-tokenize with the allowed prompt length.
                tokenized_input = self.tokenizer(prompt, return_tensors="pt", max_length=allowed_prompt_length, truncation=True)
                input_length = tokenized_input["input_ids"].shape[1]
                # Decode back to text so that the prompt is now shortened.
                prompt = self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            # Now compute how many tokens are available for generation.
            available_tokens = self.max_sequence_length - input_length
            dynamic_max_new_tokens = min(self.max_new_tokens, available_tokens)
            if available_tokens < self.max_new_tokens:
                print(f"[WARNING] Only {available_tokens} tokens available for generation (prompt length: {input_length}).", flush=True)

            # Generate the response using the QA pipeline with the dynamically computed new-token limit.
            response = self.qa_pipeline(
                                        prompt,
                                        max_new_tokens=dynamic_max_new_tokens,
                                        truncation=True,
                                        num_return_sequences=1,
                                        **self.generation_args
                                    )
           
            if self.debugging_verbose:
                    print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                    print("[DEBUG]", flush=True)
                    print(response[0]["generated_text"], flush=True)

            qa_text = response[0]["generated_text"]

            # Ensure the correct field is passed as `response_text`
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if "not found" in extracted_text:
                print(f"[WARNING] Markers not found in response: {qa_text}", flush=True)

            # Debugging: Print extracted text before processing
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text:\n{extracted_text}\n", flush=True)

            # Define regex pattern to match Question, Answer, and Reasoning
            qa_pattern = re.compile(
                r"\*\*Question:\*\*\s*(?P<question>.*?)\s*"
                r"\*\*Answer:\*\*\s*(?P<answer>.*?)\s*"
                r"\*\*Reasoning:\*\*\s*(?P<reasoning>.*)",
                re.DOTALL
            )

            # Match against extracted text
            match = qa_pattern.search(extracted_text)

            if match:
                question = match.group("question").strip().replace("**", "")  # Remove ** formatting
                answer = match.group("answer").strip().replace("**", "")
                reasoning = match.group("reasoning").strip().replace("**", "")

                # Check for missing fields explicitly
                missing_fields = []
                if not question:
                    missing_fields.append("Question")
                if not answer:
                    missing_fields.append("Answer")
                if not reasoning:
                    missing_fields.append("Reasoning")

                if missing_fields:
                    print(f"[ERROR] Missing fields in extracted QA: {', '.join(missing_fields)}", flush=True)
                    print(f"[DEBUG] Extracted QA content:\n{extracted_text}", flush=True)
                    return None

                return {
                    "question": question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "type": "multi_hop",
                    "source": source_info,
                }
            else:
                # Provide more detailed error information
                print(f"[ERROR] Invalid QA format detected.", flush=True)
                
                # Check if Question is missing
                if "Question:" not in extracted_text:
                    print("[DEBUG] 'Question:' keyword not found in extracted text.", flush=True)

                # Check if Answer is missing
                if "Answer:" not in extracted_text:
                    print("[DEBUG] 'Answer:' keyword not found in extracted text.", flush=True)

                # Check if Reasoning is missing
                reasoning_keywords = ["Reasoning:", "Step 1:", "Explanation:", "Steps:"]
                if not any(keyword in extracted_text for keyword in reasoning_keywords):
                    print(f"[DEBUG] No valid reasoning section found. Expected one of: {reasoning_keywords}", flush=True)

                # Print the extracted text for debugging
                print(f"[DEBUG] Full extracted text:\n{extracted_text}", flush=True)
                
                return None

        except Exception as e:
            print(f"[ERROR] Error generating multi-hop QA: {e}", flush=True)
            return None
        
    def generate_multi_hop_inverse_QA(self, paragraph_text, source_info):
        """
        Generates a multi-hop inverse QA pair (with a false answer) from a paragraph.

        Parameters:
            paragraph_text (str): The paragraph text to generate a question from.
            source_info (dict): Metadata about the paragraph's source.

        Returns:
            dict or None: A dictionary with question-answer data or None if not generated.
        """
        # Prepare the prompt for multi-hop inverse QA
        prompt = self.QA_MULTI_HOP_INVERSE_template.replace("{paragraph_text}", paragraph_text)
        if self.debugging_verbose:
            print("[DEBUG] - Prompt used to generate multi-hop inverse QA sets", flush=True)
            print(prompt, flush=True)

        try:
            # Tokenize the prompt to measure its length
            tokenized_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            input_length = tokenized_input["input_ids"].shape[1]

            # If the prompt is too long, reduce it to reserve tokens for generation
            if input_length > self.max_sequence_length:
                print(f"[WARNING] Input length {input_length} exceeds max allowed ({self.max_sequence_length}). Adjusting.", flush=True)
                allowed_prompt_length = self.max_sequence_length - self.max_new_tokens
                tokenized_input = self.tokenizer(prompt, return_tensors="pt", max_length=allowed_prompt_length, truncation=True)
                input_length = tokenized_input["input_ids"].shape[1]
                prompt = self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            # Calculate available tokens for generation
            available_tokens = self.max_sequence_length - input_length
            dynamic_max_new_tokens = min(self.max_new_tokens, available_tokens)
            if available_tokens < self.max_new_tokens:
                print(f"[WARNING] Only {available_tokens} tokens available for generation (prompt length: {input_length}).", flush=True)

            # Generate the response using the QA pipeline
            response = self.qa_pipeline(
                prompt,
                max_new_tokens=dynamic_max_new_tokens,
                truncation=True,
                num_return_sequences=1,
                **self.generation_args
            )

            if self.debugging_verbose:
                print(f"[DEBUG] Pipeline raw output: {response}", flush=True)
                print(response[0]["generated_text"], flush=True)

            qa_text = response[0]["generated_text"]

            # Extract text between markers
            extracted_text = self.extract_text_between_markers(qa_text, occurrence=1)
            if not extracted_text:
                print(f"[ERROR] No valid QA content extracted from: {qa_text}", flush=True)
                return None

            # Debugging: Print extracted text before processing
            if self.debugging_verbose:
                print(f"[DEBUG] Extracted text:\n{extracted_text}\n", flush=True)

            # Define regex pattern to extract Question, False Answer, Reasoning, and Incorrect Step
            qa_pattern = re.compile(
                r"\*\*Question:\*\*\s*(?P<question>.*?)\s*"
                r"\*\*Answer:\*\*\s*(?P<answer>.*?)\s*"
                r"\*\*Reasoning:\*\*\s*(?P<reasoning>.*?)\s*"
                r"\*\*Incorrect Reasoning Step:\*\*\s*(?P<incorrect_step>.*)",
                re.DOTALL
            )

            # Match against extracted text
            match = qa_pattern.search(extracted_text)

            if match:
                question = match.group("question").strip()
                answer = match.group("answer").strip()
                reasoning = match.group("reasoning").strip()
                incorrect_step = match.group("incorrect_step").strip()

                # Ensure all fields are present
                missing_fields = []
                if not question:
                    missing_fields.append("Question")
                if not answer:
                    missing_fields.append("Answer")
                if not reasoning:
                    missing_fields.append("Reasoning")
                if not incorrect_step:
                    missing_fields.append("Incorrect Reasoning Step")

                if missing_fields:
                    print(f"[ERROR] Missing fields in extracted multi-hop inverse QA: {', '.join(missing_fields)}", flush=True)
                    print(f"[DEBUG] Extracted QA content:\n{extracted_text}", flush=True)
                    return None

                return {
                    "question": question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "incorrect_reasoning_step": incorrect_step,
                    "type": "multi_hop_inverse",
                    "source": source_info
                }
            else:
                print(f"[ERROR] Invalid QA format in multi-hop inverse QA.", flush=True)
                
                # Additional debugging for incorrect formatting
                if "**Question:**" not in extracted_text:
                    print("[DEBUG] 'Question:' keyword not found in extracted text.", flush=True)
                if "**Answer:**" not in extracted_text:
                    print("[DEBUG] 'Answer:' keyword not found in extracted text.", flush=True)
                if "**Reasoning:**" not in extracted_text:
                    print("[DEBUG] 'Reasoning:' keyword not found in extracted text.", flush=True)
                if "**Incorrect Reasoning Step:**" not in extracted_text:
                    print("[DEBUG] 'Incorrect Reasoning Step:' keyword not found in extracted text.", flush=True)

                print(f"[DEBUG] Full extracted text:\n{extracted_text}", flush=True)
                return None

        except Exception as e:
            print(f"[ERROR] Error generating multi-hop inverse QA: {e}", flush=True)
            return None

        
    def process_paragraph(self, paragraph, isbn):
        """
        Processes a single paragraph to determine if it's informative.
        
        Parameters:
            paragraph (dict): Dictionary containing paragraph information.
        
        Returns:
            tuple: (is_informative, paragraph_text, source_info)
        """
        paragraph_text = paragraph.get("Text", "")
        source_info = {
            "isbn": isbn,
            "paragraph_id": paragraph.get("Paragraph ID", "Unknown ID"),
            "page": paragraph.get("Page", "Unknown Page")
        }

        # Check if the paragraph is informative
        is_informative = self.is_content_informative(paragraph_text, i=0, total_paragraphs=1, title="Unknown", isbn="Unknown")

        return is_informative, paragraph_text, source_info
    
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

                for i, paragraph in enumerate(paragraphs[:total_paragraphs]):
                    if i < self.start_paragraph:  # Skip paragraphs before start_paragraph
                        continue    

                    if self.max_entries is not None and i >= self.max_entries:
                        break  # Stop processing further paragraphs if max_entries limit is reached

                    if self.debugging:
                        print(f"\n[DEBUG] Processing paragraph {i+1}/{total_paragraphs} in '{title}' (ISBN: {isbn})", flush=True)

                    is_informative, paragraph_text, source_info = self.process_paragraph(paragraph, isbn)
                    if not is_informative:
                        if self.debugging:
                            print(f"[DEBUG] Paragraph {i+1} skipped (not informative).", flush=True)
                        continue

                    if self.debugging_verbose:
                        print(f"[DEBUG] Determining question type for paragraph {i+1}...", flush=True)
                    question_type = self.determine_question_type(paragraph_text)
                    if self.debugging_verbose:
                        print(f"[DEBUG] Selected question type: {question_type.value}", flush=True)

                    qa_pair = None
                    if question_type == QAType.SHORT:
                        qa_pair = self.generate_short_answer_QA(paragraph_text, source_info)
                    elif question_type == QAType.TRUE_FALSE:
                        qa_pair = self.generate_TF_QA(paragraph_text, source_info)
                    elif question_type == QAType.LIST:
                        qa_pair = self.generate_list_QA(paragraph_text, source_info)
                    elif question_type == QAType.MULTIPLE_CHOICE:
                        qa_pair = self.generate_MC_QA(paragraph_text, source_info)
                    elif question_type == QAType.MULTI_HOP:
                        qa_pair = self.generate_multi_hop_QA(paragraph_text, source_info)
                    elif question_type == QAType.MULTI_HOP_INVERSE:
                        qa_pair = self.generate_multi_hop_inverse_QA(paragraph_text, source_info)
                    elif question_type == QAType.SHORT_INVERSE:
                        qa_pair = self.generate_short_inverse_QA(paragraph_text, source_info)

                    if qa_pair:
                        if self.debugging_verbose:
                            print(f"[DEBUG] Successfully generated a {question_type.value.upper()} QA pair", flush=True)
                        if question_type == QAType.SHORT:
                            self.short_QA_pairs.append(qa_pair)
                        elif question_type == QAType.TRUE_FALSE:
                            self.TF_QA_pairs.append(qa_pair)
                        elif question_type == QAType.LIST:
                            self.list_QA_pairs.append(qa_pair)
                        elif question_type == QAType.MULTIPLE_CHOICE:
                            self.MC_QA_pairs.append(qa_pair)
                        elif question_type == QAType.MULTI_HOP:
                            self.multi_hop_QA_pairs.append(qa_pair)
                        elif question_type == QAType.MULTI_HOP_INVERSE:
                            self.multi_hop_inverse_QA_pairs.append(qa_pair)  # New inverse multi-hop QA pair
                        elif question_type == QAType.SHORT_INVERSE:
                            self.short_inverse_QA_pairs.append(qa_pair)  # New inverse short-answer QA pair
                        
                    else:
                        if self.debugging:
                            print(f"[DEBUG] Failed to generate a QA pair for Paragraph ID: {source_info['paragraph_id']}", flush=True)

                    if i % self.save_every == 0:
                        self.save_qa_pairs(isbn)
                        if self.debugging:
                            print(f"[DEBUG] Saved QA pairs after processing {i} paragraphs.", flush=True)

                self.save_qa_pairs(isbn)  # Final save for the book

                if self.debugging:
                    print(f"[DEBUG] Processed entry {index}/{total_entries}: {title} (ISBN: {isbn})", flush=True)

            except FileNotFoundError:
                print(f"[ERROR] Error: File '{json_file_path}' not found. Skipping this entry.", flush=True)
            except json.JSONDecodeError:
                print(f"[ERROR] Error: Failed to decode JSON in file '{json_file_path}'. Skipping this entry.", flush=True)
            except Exception as e:
                print(f"[ERROR] An unexpected error occurred with file '{json_file_path}': {e}", flush=True)

        process_end_time = time.time()
        if self.debugging:
            print(f"[DEBUG] Total processing time: {process_end_time - process_start_time:.2f} seconds", flush=True)
    
    def determine_question_type(self, paragraph_text):
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
            QAType.SHORT_INVERSE: 0,  # New inverse short-answer QA type
            QAType.MULTI_HOP_INVERSE: 0,  # New inverse multi-hop QA type
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
            scores[QAType.MULTIPLE_CHOICE] += 2
        if re.search(r"(A:|B:|C:|D:)", paragraph_text):  # Explicit multiple-choice format
            scores[QAType.MULTIPLE_CHOICE] += 4
        if num_sentences > 1 and len(unique_entities) > 1:
            scores[QAType.MULTIPLE_CHOICE] += 2
        if "which of the following" in paragraph_text.lower():
            scores[QAType.MULTIPLE_CHOICE] += 3

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

        # SHORT_INVERSE Questions (False Short Answers)
        if scores[QAType.SHORT] > 0:  # If a short-answer question is likely, add inverse variation
            scores[QAType.SHORT_INVERSE] = scores[QAType.SHORT]  # Equal probability as SHORT

        # MULTI_HOP_INVERSE Questions (False Multi-Hop Answers)
        if scores[QAType.MULTI_HOP] > 0:  # If a multi-hop question is likely, add inverse variation
            scores[QAType.MULTI_HOP_INVERSE] = scores[QAType.MULTI_HOP]  # Equal probability as MULTI_HOP


        # Select highest-scoring QA type
        # question_type = max(scores, key=scores.get)

        # --------------- TODO - Fix - This is some hamburger code I jumbled together to make the QA type generation process more deterministic but it isn't great and should look to fix this. 

        # # Get the count of each QA type
        # min_count = min(self.qa_counts.values())  # Find the lowest count
        # max_count = max(self.qa_counts.values())  # Find the highest count

        # # Identify the least-used types
        # underrepresented_types = [qa for qa, count in self.qa_counts.items() if count == min_count]

        # # Apply Discounting Factor to Underrepresented Types
        # discounting_factors = {
        #     qa_type: 1 + ((max_count - count) / max(self.balance_threshold, max_count))  
        #     for qa_type, count in self.qa_counts.items()
        # }

        # # Adjust Scores Using Discounting Factors
        # adjusted_scores = {
        #     qa_type: scores[qa_type] * discounting_factors[qa_type]
        #     for qa_type in scores
        # }

        # #  Select the Question Type Based on Adjusted Scores
        # balanced_question_type = max(adjusted_scores, key=adjusted_scores.get)

        # #  Ensure Severely Underrepresented Types Get Selected
        # if max_count - min_count >= self.balance_threshold:
        #     severely_underrepresented_types = [
        #         qa for qa, count in self.qa_counts.items() if max_count - count >= self.balance_threshold
        #     ]
        #     if severely_underrepresented_types:
        #         balanced_question_type = random.choice(severely_underrepresented_types)

        # # Update QA Type Counts
        # self.qa_counts[balanced_question_type] += 1

        # Group question types into 5 main categories (treat inverse types as their equivalent)
        # Group question types into 5 main categories (treat inverse types as their equivalent)
        grouped_counts = {
            "short": self.qa_counts[QAType.SHORT] + self.qa_counts[QAType.SHORT_INVERSE],
            "true_false": self.qa_counts[QAType.TRUE_FALSE],
            "list": self.qa_counts[QAType.LIST],
            "multiple_choice": self.qa_counts[QAType.MULTIPLE_CHOICE],
            "multi_hop": self.qa_counts[QAType.MULTI_HOP] + self.qa_counts[QAType.MULTI_HOP_INVERSE],
        }

        # Find the lowest and highest counts among grouped categories
        min_count = min(grouped_counts.values())
        max_count = max(grouped_counts.values())

        # Identify underrepresented and severely underrepresented types
        underrepresented_categories = [category for category, count in grouped_counts.items() if count == min_count]
        severely_underrepresented_categories = [
            category for category, count in grouped_counts.items() if max_count - count >= self.balance_threshold
        ]

        # Apply Discounting Factor to Encourage Balance
        discounting_factors = {
            qa_type: 1 + ((max_count - grouped_counts[category]) / max(self.balance_threshold, max_count))
            for qa_type, category in {
                QAType.SHORT: "short",
                QAType.SHORT_INVERSE: "short",
                QAType.TRUE_FALSE: "true_false",
                QAType.LIST: "list",
                QAType.MULTIPLE_CHOICE: "multiple_choice",
                QAType.MULTI_HOP: "multi_hop",
                QAType.MULTI_HOP_INVERSE: "multi_hop",
            }.items()
        }

        # Adjust Scores Using Discounting Factors
        adjusted_scores = {
            qa_type: scores[qa_type] * discounting_factors[qa_type]
            for qa_type in scores
        }

        # Select the Question Type Based on Adjusted Scores
        balanced_question_type = max(adjusted_scores, key=adjusted_scores.get)

        # If a category is severely underrepresented, force selection from it
        if severely_underrepresented_categories:
            eligible_types = [qa for qa, cat in {
                QAType.SHORT: "short",
                QAType.SHORT_INVERSE: "short",
                QAType.TRUE_FALSE: "true_false",
                QAType.LIST: "list",
                QAType.MULTIPLE_CHOICE: "multiple_choice",
                QAType.MULTI_HOP: "multi_hop",
                QAType.MULTI_HOP_INVERSE: "multi_hop",
            }.items() if cat in severely_underrepresented_categories]

            balanced_question_type = random.choice(eligible_types)

        # Update QA Type Counts
        self.qa_counts[balanced_question_type] += 1


        if self.debugging_verbose:
            print(f"\n[DEBUG] QA Type Selection Scores (Before Balancing): {scores}", flush=True)
            print(f"[DEBUG] Discounting Factors: {discounting_factors}", flush=True)
            print(f"[DEBUG] QA Type Selection Scores (After Balancing): {adjusted_scores}", flush=True)
            print(f"[DEBUG] Underrepresented Types: {underrepresented_categories}", flush=True)
            print(f"[DEBUG] Severely Underrepresented Types: {severely_underrepresented_categories}", flush=True)
            print(f"[DEBUG] Selected QA Type: {balanced_question_type.value}\n", flush=True)

        return balanced_question_type


    def save_qa_pairs(self, isbn):
        """
        Saves generated QA pairs to JSON files without overwriting existing data.
        """
        if self.debugging:
            print(f"[DEBUG] Saving QA pairs for ISBN: {isbn}", flush=True)

        json_files = {
            f"short_QA-{isbn}.json": self.short_QA_pairs,
            f"TF_QA-{isbn}.json": self.TF_QA_pairs,
            f"list_QA-{isbn}.json": self.list_QA_pairs,
            f"MC_QA-{isbn}.json": self.MC_QA_pairs,
            f"multi_hop_QA-{isbn}.json": self.multi_hop_QA_pairs,
            f"multi_hop_inverse_QA-{isbn}.json": self.multi_hop_inverse_QA_pairs,  # New inverse multi-hop
            f"short_inverse_QA-{isbn}.json": self.short_inverse_QA_pairs  # New inverse short-answer
        }

        for file_name, new_data in json_files.items():
            file_path = self.output_dir / file_name

            # Load existing data if the file exists
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, list):  # Ensure the file contains a list
                        existing_data = []
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
            else:
                existing_data = []

            # Append new data
            updated_data = existing_data + new_data

            # Save updated data
            with open(file_path, "w") as f:
                json.dump(updated_data, f, indent=4)

            if self.debugging:
                print(f"[DEBUG] {file_name}: {len(updated_data)} QA pairs saved (previous: {len(existing_data)}, new: {len(new_data)})", flush=True)

        # Write metadata for generated JSON files to the CSV
        csv_path = self.output_dir / "QA_dataset.csv"
        with open(csv_path, "a", newline="") as csvfile:  # Use "a" to append instead of "w"
            fieldnames = ["file_name", "type", "isbn"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Only write the header if the file is empty
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                writer.writeheader()

            for file_name in json_files.keys():
                question_type = file_name.split("_")[0].lower()
                writer.writerow({"file_name": file_name, "type": question_type, "isbn": isbn})

        if self.debugging:
            print(f"[DEBUG] QA dataset CSV updated at {csv_path}", flush=True)



if __name__ == "__main__":
    script_start_time = time.time()
    script_actual_start_time = datetime.now()
    print(f"[DEBUG] Script started at: {script_actual_start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    parser = argparse.ArgumentParser(description="Generate QA dataset from preprocessed text")
    parser.add_argument("preprocessed_csv", help="Path to the preprocessed data CSV file")
    parser.add_argument("llama3_model_path", help="Path to the LLaMA3 model")
    parser.add_argument("output_dir", help="Directory to save the generated QA files")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum length for input and output tokens") #default is 2048 as that is max allowed for LLama 3-1 7B
    parser.add_argument("--max_sequence_length", type=int, default=2048, help="Maximum length for input and output tokens") #default is 2048 as that is max allowed for LLama 3-1 7B
    parser.add_argument("--model_max_length", type=int, default=131072, help="Maximum length for input and output tokens") #default is 2048 as that is max allowed for LLama 3-1 7B
    parser.add_argument("--checkpoint", type=int, default=None, help="number of iterations to save for each book")
    parser.add_argument("--max_entries", type=int, default=None, help="Maximum number of paragraph entries to process")
    parser.add_argument("--debugging", action="store_true", help="Enable debugging mode to print detailed information about skipped paragraphs and processing steps")
    parser.add_argument("--debugging_verbose", action="store_true", help="Enable verobse debugging mode to print detailed information about skipped paragraphs and processing steps")
    parser.add_argument("--start_paragraph", type=int, default=0, help="Paragraph index to start processing from")


    args = parser.parse_args()

    generator = QADatasetGenerator(args.preprocessed_csv, args.llama3_model_path, args.output_dir, 
                                   args.max_new_tokens, args.max_sequence_length, args.model_max_length, args.checkpoint, 
                                   args.max_entries, args.start_paragraph, args.debugging, 
                                   args.debugging_verbose)
    
    print("[DEBUG] Loading Pre-Processed Data", flush=True)
    data = generator.load_preprocessed_data()
    print("[DEBUG] Pre-Processed Data Loaded!", flush=True)
    print("[DEBUG] Generating QA Pairs...", flush=True)
    generator.generate_qa_pairs(data)

    script_end_time = time.time()
    script_actual_end_time = datetime.now()
    print(f"[DEBUG] Script completed at: {script_actual_end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[DEBUG] Total script execution time: {script_end_time - script_start_time:.2f} seconds\n", flush=True)
