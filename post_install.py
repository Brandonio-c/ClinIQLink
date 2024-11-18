# post_install.py
import subprocess

# Install the 'en_core_web_sm' SpaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
