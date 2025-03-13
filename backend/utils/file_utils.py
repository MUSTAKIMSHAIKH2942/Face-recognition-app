import os
import json
import csv

def ensure_folder_exists(folder_path):
    """
    Ensure that a folder exists. If it doesn't, create it.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_json(file_path):
    """
    Load data from a JSON file.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_json(file_path, data):
    """
    Save data to a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def append_to_csv(file_path, row):
    """
    Append a row to a CSV file.
    """
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)