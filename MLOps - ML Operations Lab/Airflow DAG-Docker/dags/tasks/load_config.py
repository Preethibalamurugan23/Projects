# tasks/load_config.py
import json

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r', encoding='utf-8') as file:
        return json.load(file)
