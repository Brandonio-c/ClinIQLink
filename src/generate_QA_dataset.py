"""
QADatasetGenerator

Author: Brandon Colelough
Date Last Edited: 2024-11-14
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

class QAType(Enum):
    TRUE_FALSE = "tf"
    MULTIPLE_CHOICE = "mc"
    SHORT = "short"
    LIST = "list"

class QADatasetGenerator:
    def __init__(self, preprocessed_csv, llama3_model_path, output_dir, max_length, checkpoint, max_entries=None, debugging=False):
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
        self.preprocessed_csv = Path(preprocessed_csv)
        self.llama3_model_path = llama3_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Automatically detect GPU and set the device
        self.device = 0 if torch.cuda.is_available() else -1
        self.max_length = max_length
        # Load NER model (SpaCy) and sentence embedding model (Hugging Face)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        self.max_entries = max_entries
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llama3_model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    llama3_model_path, 
                    torch_dtype=torch.bfloat16,  # Use bfloat16 as specified
                    device_map="auto"  # Automatically map model to available devices (e.g., GPU/CPU)
                    )
        # Set pad_token_id to suppress warnings
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.qa_pipeline = pipeline(
                                        "text-generation",
                                        model=self.model,
                                        tokenizer=self.tokenizer,
                                        truncation=True,  # Enable truncation
                                        max_length=self.max_length,  # Set the maximum token length to 2048
                                        pad_token_id=0,
                                        return_tensors="pt",
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
        self.prompt_templates_dir = Path("/data/coleloughbc/llm_lie_detector_shared_task/ClinIQLink/prompt_templates")
        self.QA_SHORT_template = self.load_prompt_template(self.prompt_templates_dir / "QA_SHORT.prompt")
        self.QA_MC_template = self.load_prompt_template(self.prompt_templates_dir / "QA_MC.prompt")
        self.QA_LIST_template = self.load_prompt_template(self.prompt_templates_dir / "QA_LIST.prompt")
        self.QA_TF_template = self.load_prompt_template(self.prompt_templates_dir / "QA_TF.prompt")
        
        # Initialize containers for each type of QA pair
        self.short_QA_pairs = []
        self.TF_QA_pairs = []
        self.list_QA_pairs = []
        self.MC_QA_pairs = []
        self.max_entries = max_entries
        self.save_every = checkpoint
        self.debugging=debugging

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
        # Check basic length and complexity
        word_count = len(paragraph_text.split())
        sentence_count = len(list(self.nlp(paragraph_text).sents))
        if word_count < 15 or sentence_count < 2:
            if self.debugging:
                print(f"[DEBUG] Processing paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Word count and / or sentence count too low. Word count: {word_count}, Sentence count: {sentence_count}")
            return False  # Too short or simple
        
        # Check for irrelevant content
        irrelevant_keywords = ["copyright", "ISBN", "publisher", "permission", "disclaimer"]
        if any(keyword in paragraph_text.lower() for keyword in irrelevant_keywords):
            if self.debugging:
                found_keywords = [keyword for keyword in irrelevant_keywords if keyword in paragraph_text.lower()]
                print(f"[DEBUG] Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Found irrelevant keyword(s): {', '.join(found_keywords)}")
            return False  # Contains irrelevant metadata
        
        """ # Detect tables, figures, and metadata
        if re.search(r"(\d+\.\d+|Fig\.|Table|•)", paragraph_text):
            if self.debugging:
                match = re.search(r"(\d+\.\d+|Fig\.|Table|•)", paragraph_text)
                print(f"Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - Found pattern match: '{match.group(0)}'")
            return False """

        # Named Entity Density
        doc = self.nlp(paragraph_text)
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
        sentence_embeddings = self.embedding_model([sent.text for sent in doc.sents])
        sentence_embeddings = np.array([np.mean(embed[0], axis=0) for embed in sentence_embeddings])
        embedding_variance = np.var(sentence_embeddings)
        if embedding_variance > 0.5:
            if self.debugging:
                print(
                    f"[DEBUG] Paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn}) NOT INFORMATIVE - "
                    f"High embedding variance. Variance: {embedding_variance:.3f}, "
                    f"Sentences analyzed: {[sent.text for sent in doc.sents]}"
                )
            return False  # High variance suggests incoherent topic

        return True  # Paragraph passed all checks
    
    
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
            print("[DEBUG] - Prompt used to generate short answer QA sets")
            print(prompt)
        try:
            # Generate response using the QA pipeline
            response = self.qa_pipeline(
                                            prompt, 
                                            max_length=self.max_length,  # Ensures the response adheres to max_length
                                            truncation=True,  # Ensures long inputs are truncated properly
                                            num_return_sequences=1,  # Generates only one response
                                            **self.generation_args  # Passes additional generation arguments
                                        )
            if self.debugging:
                #print(f"[DEBUG] Pipeline raw output: {response}")
                print("[DEBUG]")
                print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Split the generated text into question and answer
            if "\nAnswer:" in qa_text:
                qa_parts = qa_text.split("\nAnswer:", 1)
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
                print(f"Invalid QA format found in short answer QA: {qa_text}")
                return None

        except Exception as e:
            print(f"Error generating short-answer QA: {e}")
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
            print("[DEBUG] - Prompt used to generate TF QA sets")
            print(prompt)

        try:
            # Generate response using the QA pipeline
            response = self.qa_pipeline(
                                            prompt, 
                                            max_length=self.max_length,  # Ensures the response adheres to max_length
                                            truncation=True,  # Ensures long inputs are truncated properly
                                            num_return_sequences=1,  # Generates only one response
                                            **self.generation_args  # Passes additional generation arguments
                                        )
            if self.debugging:
                #print(f"[DEBUG] Pipeline raw output: {response}")
                print("[DEBUG]")
                print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Split the generated text into question and answer
            if "\nAnswer:" in qa_text:
                qa_parts = qa_text.split("\nAnswer:", 1)
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
                    print(f"Generated answer mismatch: {generated_answer}")
                    return None
            else:
                print(f"Invalid QA format found in TF paragraph: {qa_text}")
                return None

        except Exception as e:
            print(f"Error generating true/false QA: {e}")
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
            print("[DEBUG] - Prompt used to generate list QA sets")
            print(prompt)

        try:
            # Generate response using the QA pipeline
            response = self.qa_pipeline(
                                            prompt, 
                                            max_length=self.max_length,  # Ensures the response adheres to max_length
                                            truncation=True,  # Ensures long inputs are truncated properly
                                            num_return_sequences=1,  # Generates only one response
                                            **self.generation_args  # Passes additional generation arguments
                                        )
            if self.debugging:
                #print(f"[DEBUG] Pipeline raw output: {response}")
                print("[DEBUG]")
                print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Validate and split the response into question and answer
            if "\nAnswer:" in qa_text:
                qa_parts = qa_text.split("\nAnswer:", 1)
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
                    print(f"Invalid list format in generated answer: {answer}")
                    return None
            else:
                print(f"Invalid QA format found in LIST question: {qa_text}")
                return None

        except Exception as e:
            print(f"Error generating list QA: {e}")
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
            print("[DEBUG] - Prompt used to generate MC QA sets")
            print(prompt)
        try:
            # Run the QA pipeline
            response = self.qa_pipeline(
                                                prompt, 
                                                max_length=self.max_length,  # Ensures the response adheres to max_length
                                                truncation=True,  # Ensures long inputs are truncated properly
                                                num_return_sequences=1,  # Generates only one response
                                                **self.generation_args  # Passes additional generation arguments
                                            )
            if self.debugging:
                    #print(f"[DEBUG] Pipeline raw output: {response}")
                    print("[DEBUG]")
                    print(response[0]["generated_text"])

            qa_text = response[0]["generated_text"]

            # Split the generated response into the question, correct answer, and distractors
            qa_parts = re.split(r"\n(A|B|C|D):", qa_text)  # Split at multiple-choice options (A, B, C, D)
            if len(qa_parts) < 5:  # Ensure the response contains at least one question and four options
                print(f"Invalid MC format: {qa_text}")
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
                print(f"No correct option identified in: {qa_text}")
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
            print(f"Error generating MC QA: {e}")
            return None


    def generate_qa_pairs(self, data):
        """
        Generates QA pairs for each paragraph in each book entry in the data and saves the dataset iteratively.
        Provides progress updates after processing each entry.

        Parameters:
            data (list): List of dictionaries with metadata and file paths for each book.
        """
        total_entries = len(data)
        for index, entry in enumerate(data, start=1):  # Enumerate to track the current entry index
            # Join the relative path with the directory of `preprocessed_csv` to form an absolute path
            json_file_path = (self.preprocessed_csv.parent / entry["JSON File"]).resolve()

            try:
                # Attempt to open and read the JSON file
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    book_data = json.load(json_file)
                    
                isbn = book_data.get("ISBN", "Unknown ISBN")
                title = book_data.get("Title", "Unknown Title")

                # Process each paragraph in the JSON file, up to max_entries
                total_paragraphs = len(book_data["Paragraphs"])
                if total_paragraphs > self.max_entries:
                    total_paragraphs = self.max_entries

                for i, paragraph in enumerate(book_data["Paragraphs"]):
                    if self.max_entries is not None and i >= self.max_entries:
                        break  # Stop processing further paragraphs if max_entries limit is reached
                    
                    # Print progress update
                    print(f"Processing paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn})")

                    paragraph_text = paragraph["Text"]
                    source_info = {
                        "isbn": isbn,
                        "paragraph_id": paragraph.get("Paragraph ID", "Unknown ID"),
                        "page": paragraph.get("Page", "Unknown Page")
                    }
                    if not self.is_content_informative(paragraph_text, i, total_paragraphs, title, isbn):
                        if self.debugging:
                            print(paragraph_text)
                        continue # skip if not informative

                    # Print progress update
                    # print(f"Determining question type for paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn})")
                    # Determine question type based on content
                    question_type = self.determine_question_type_v2(paragraph_text)

                    # Print progress update
                    # print(f"Generating QA pair for paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn})")
                    # Generate QA pairs based on determined question type
                    qa_pair = None  # Initialize the QA pair as None before the block

                    if question_type == QAType.SHORT:
                        qa_pair = self.generate_short_answer_QA(paragraph_text, source_info)
                        if qa_pair:
                            self.short_QA_pairs.append(qa_pair)

                    elif question_type == QAType.TRUE_FALSE:
                        qa_pair = self.generate_TF_QA(paragraph_text, source_info)
                        if qa_pair:
                            self.TF_QA_pairs.append(qa_pair)

                    elif question_type == QAType.LIST:
                        qa_pair = self.generate_list_QA(paragraph_text, source_info)
                        if qa_pair:
                            self.list_QA_pairs.append(qa_pair)

                    elif question_type == QAType.MULTIPLE_CHOICE:
                        qa_pair = self.generate_MC_QA(paragraph_text, source_info)
                        if qa_pair:
                            self.MC_QA_pairs.append(qa_pair)

                    # Debugging step to check if a QA pair was generated
                    if qa_pair:
                        if question_type in QAType:  # Validate the question type against the enum
                            print(f"[DEBUG] Successfully generated a {question_type.value.upper()} QA pair for Paragraph ID: {source_info['paragraph_id']}")
                        else:
                            print(f"[DEBUG] Generated an invalid QA type: {question_type} for Paragraph ID: {source_info['paragraph_id']}")
                    else:
                        print(f"[DEBUG] Failed to generate a QA pair for Paragraph ID: {source_info['paragraph_id']}")


                    

                    # Save QA pairs every 200 iterations 
                    if i % self.save_every == 0:
                        if any([self.short_QA_pairs, self.TF_QA_pairs, self.list_QA_pairs, self.MC_QA_pairs]):
                            self.save_qa_pairs(isbn)
                            print(f"Saved QA pairs after processing {i} paragraphs.")
                        else:
                            print(f"No valid QA pairs generated for {isbn} up to paragraph {i}.")


                    # Print progress update
                    print(f"Processed entry paragraph {i}/{total_paragraphs}: {title} (ISBN: {isbn})")
                
                # Save QA pairs iteratively after each entry is processed as a catch all 
                self.save_qa_pairs(isbn)

                # Print progress update
                print(f"Processed entry {index}/{total_entries}: {title} (ISBN: {isbn})")

            except FileNotFoundError:
                print(f"Error: File '{json_file_path}' not found. Skipping this entry.")
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON in file '{json_file_path}'. Skipping this entry.")
            except Exception as e:
                print(f"An unexpected error occurred with file '{json_file_path}': {e}")



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

        # Decision Logic Using Both Semantic Analysis and NER

        # Check for multiple-choice cues
        if re.search(r"(option|choice|select|following|correct answer|choose|questionnaire|quiz)", paragraph_text, re.IGNORECASE):
            return QAType.MULTIPLE_CHOICE

        # Initialize scores for each question type
        scores = {
            QAType.SHORT: 0,
            QAType.TRUE_FALSE: 0,
            QAType.LIST: 0,
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

        # Determine question type with the highest score
        question_type = max(scores, key=scores.get)

        return question_type



    def save_qa_pairs(self, isbn):
        """
        Saves generated QA pairs to JSON files and a consolidated CSV file in the output directory.
        """
        # Save to individual JSON files
        with open(self.output_dir / f"short_QA-{isbn}.json", "w") as f:
            json.dump(self.short_QA_pairs, f, indent=4)
        with open(self.output_dir / f"TF_QA-{isbn}.json", "w") as f:
            json.dump(self.TF_QA_pairs, f, indent=4)
        with open(self.output_dir / f"list_QA-{isbn}.json", "w") as f:
            json.dump(self.list_QA_pairs, f, indent=4)
        with open(self.output_dir / f"MC_QA-{isbn}.json", "w") as f:
            json.dump(self.MC_QA_pairs, f, indent=4)
        
        # Consolidate all QA pairs for the CSV
        all_qa_pairs = self.short_QA_pairs + self.TF_QA_pairs + self.list_QA_pairs
        csv_path = self.output_dir / "QA_dataset.csv"

        # Gather the list of generated JSON files
        json_files = [
            f"short_QA-{isbn}.json",
            f"TF_QA-{isbn}.json",
            f"list_QA-{isbn}.json",
            f"MC_QA-{isbn}.json"
        ]

        # Write the list of JSON files to the CSV
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["file_name", "type", "isbn"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for file_name in json_files:
                # Infer the question type from the file name
                question_type = file_name.split("_")[0].lower()
                writer.writerow({
                    "file_name": file_name,
                    "type": question_type,
                    "isbn": isbn
                })
        
        # Write consolidated CSV file
        # with open(csv_path, "w", newline="") as csvfile:
        #     fieldnames = ["question", "answer", "type", "isbn", "paragraph_id", "page"]
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     for qa_pair in all_qa_pairs:
        #         writer.writerow({
        #             "question": qa_pair["question"],
        #             "answer": qa_pair["answer"],
        #             "type": qa_pair["type"],
        #             "isbn": qa_pair["source"]["isbn"],
        #             "paragraph_id": qa_pair["source"]["paragraph_id"],
        #             "page": qa_pair["source"]["page"]
        #         })
        
        print(f"Saved QA pairs to {self.output_dir}")
        print(f"QA dataset CSV saved at {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA dataset from preprocessed text")
    parser.add_argument("preprocessed_csv", help="Path to the preprocessed data CSV file")
    parser.add_argument("llama3_model_path", help="Path to the LLaMA3 model")
    parser.add_argument("output_dir", help="Directory to save the generated QA files")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for input and output tokens") #default is 2048 as that is max allowed for LLama 3-1 7B
    parser.add_argument("--checkpoint", type=int, default=None, help="number of iterations to save for each book")
    parser.add_argument("--max_entries", type=int, default=None, help="Maximum number of paragraph entries to process")
    parser.add_argument("--debugging", action="store_true", help="Enable debugging mode to print detailed information about skipped paragraphs and processing steps")


    args = parser.parse_args()

    generator = QADatasetGenerator(args.preprocessed_csv, args.llama3_model_path, args.output_dir, args.max_length, args.checkpoint, args.max_entries, args.debugging)
    data = generator.load_preprocessed_data()
    generator.generate_qa_pairs(data)
