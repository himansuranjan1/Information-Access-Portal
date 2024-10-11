import os
from pathlib import Path
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of files involved in the system
list_of_files = [
    "src/__init__.py",  # Corrected the typo _init_ to __init__
    "src/helper.py",
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    "test.py"
]

def create_file(filepath):
    """Create a file and any parent directories if they do not exist."""
    filepath = Path(filepath)
    
    # Create parent directories if they do not exist
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {filepath.parent}")

    # Create the file if it doesn't exist
    if not filepath.exists():
        filepath.touch()
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")

def check_and_create_files(file_list):
    """Check if the given list of files exists, create them if missing."""
    for file in file_list:
        create_file(file)

if __name__ == "__main__":
    logging.info("Checking for the existence of files...")
    check_and_create_files(list_of_files)
    logging.info("File check and creation process completed.")
