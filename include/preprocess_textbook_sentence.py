"""
PreprocessTextbookSentence Module
Author: Brandon Colelough
Date: November 14, 2024
License: MIT

Description:
This module contains the `PreprocessTextbookSentence` class, which processes a collection of textbook PDFs to create structured JSON files, each containing sentence-level text extracted from each page. The `PreprocessTextbookSentence` class reads a CSV catalog of textbook metadata, processes each specified PDF, extracts text into sentences, and saves the output as JSON files. Additionally, it generates a catalog CSV file that lists all processed JSON files, providing a summary of the processed textbooks for easy reference.

Usage:
    This script can be run from the command line with the following arguments:
    
        python preprocess_textbook_sentence.py <data_dir> <output_dir>

    - `<data_dir>`: Path to the directory containing the input CSV and PDF files.
    - `<output_dir>`: Path to the directory where JSON output files and catalog CSV will be saved.

Example:
    To run the preprocessing on a data directory and save output to a specific directory:
    
        python preprocess_textbook_sentence.py /path/to/data_dir /path/to/output_dir
"""
import os
import csv
import re
import json
import sys
from pathlib import Path
from PyPDF2 import PdfReader

class PreprocessTextbook:
    """
    A class to preprocess textbook PDFs into structured JSON files, splitting text by sentences and storing metadata.
    Also generates a catalog CSV file listing all processed JSON files for easy reference.

    Attributes:
        data_dir (Path): Path to the directory containing the input CSV and PDF files.
        output_dir (Path): Path to the directory where JSON output files and catalog CSV are saved.
        catalog (list): A list storing metadata for each processed textbook, to be written to a CSV file.
    """
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = []  # Catalog for CSV output describing JSON files

    def process(self):
        csv_path = self.data_dir / 'data.csv'  # Use the passed data_dir for CSV path
        try:
            with open(csv_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                
                for row in reader:
                    self.process_row(row)
        
        except FileNotFoundError as e:
            print(f"Error: CSV file not found at {csv_path}. {e}")
        except Exception as e:
            print(f"Unexpected error reading CSV file: {e}")

        # Write catalog to CSV file
        self.write_catalog()

    def process_row(self, row):
        try:
            # Extract details from the CSV
            title = row.get("Title", "Unknown Title")
            isbn_10 = row.get("ISBN-10", "Unknown ISBN-10")
            isbn_13 = row.get("ISBN-13", "Unknown ISBN-13")
            file_link = row.get("File Link", "")
            
            # Validate that required fields are present
            if not file_link:
                print(f"Warning: No file link found for {title}")
                return
            
            textbook_path = self.data_dir / file_link
            if not textbook_path.is_file():
                print(f"Warning: File not found for {title} at {textbook_path}")
                return
            
            # Prepare the JSON output file
            json_file_path = self.output_dir / f"{isbn_13}_processed.json"
            json_data = {
                "ISBN": isbn_13,
                "Title": title,
                "Sentences": []
            }
            
            # Open and read the PDF with a general exception handler
            with open(textbook_path, "rb") as pdf_file:
                try:
                    reader = PdfReader(pdf_file)
                except Exception as e:
                    print(f"Error reading PDF for {title} ({file_link}): {e}")
                    return

                for page_num, page in enumerate(reader.pages):
                    self.process_page(page, page_num, json_data, title, isbn_13)
            
            # Write JSON data to file
            with open(json_file_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
            
            # Add entry to catalog
            self.catalog.append({
                "Title": title,
                "ISBN-10": isbn_10,
                "ISBN-13": isbn_13,
                "File Link": file_link,
                "JSON File": os.path.relpath(json_file_path, self.output_dir)
            })
            
            print(f"Processed {title} and saved to {json_file_path}")

        except KeyError as e:
            print(f"Error with CSV entry for {title}: missing key {e}")
        except Exception as e:
            print(f"Unexpected error processing {title}: {e}")

    def process_page(self, page, page_num, json_data, title, isbn_13):
        try:
            text = page.extract_text()
            if not text:
                print(f"Warning: Page {page_num+1} of {title} contains no text.")
                return
            
            # Split text into sentences based on basic punctuation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence_num, sentence in enumerate(sentences):
                if sentence.strip():  # Ensure it's a non-empty sentence
                    # Construct unique identifiers
                    sentence_id = f"{isbn_13}-p{page_num+1}-s{sentence_num+1}"
                    section_id = f"{isbn_13}-p{page_num+1}"
                    
                    # Append sentence data to JSON structure
                    json_data["Sentences"].append({
                        "Sentence ID": sentence_id,
                        "Section": section_id,
                        "Page": page_num + 1,
                        "Text": sentence.strip()
                    })
        except Exception as e:
            print(f"Error processing page {page_num+1} of {title}: {e}")

    def write_catalog(self):
        catalog_csv_path = self.output_dir / "preprocessed_data.csv"
        try:
            with open(catalog_csv_path, "w", newline="") as catalog_file:
                fieldnames = ["Title", "ISBN-10", "ISBN-13", "File Link", "JSON File"]
                writer = csv.DictWriter(catalog_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.catalog)
            
            print(f"Catalog saved to {catalog_csv_path}")
        except Exception as e:
            print(f"Error writing catalog CSV: {e}")

if __name__ == "__main__":
    # Accept data_dir and output_dir from command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python preprocess_textbook.py <data_dir> <output_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Instantiate and run the PreprocessTextbook class
    preprocessor = PreprocessTextbook(data_dir, output_dir)
    preprocessor.process()