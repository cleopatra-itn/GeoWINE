import os
import json
from glob import glob
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__))

entitiy_files = glob(f'{ROOT_PATH}/cached_entities/cached_entities_*.json')

all_entities = []
for entity_file in entitiy_files:
    all_entities.extend(json.load(open(entity_file)))

entites_path = ROOT_PATH / f'cached_entities_with_descriptions.json'
with open(entites_path, 'w') as json_file:
    json.dump(all_entities, json_file, ensure_ascii=False, indent=4)