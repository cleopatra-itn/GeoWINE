import os
import json
from pathlib import Path

def get_file_name(path):
    head, file_name = os.path.split(path)
    return head, file_name

def get_root_path():
    return Path(os.path.dirname(__file__))

def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def open_json(fileName):
    try:
        with open(fileName,encoding='utf8') as json_data:
            d = json.load(json_data)
    except Exception as s:
        d=s
        print(d)
    return d
