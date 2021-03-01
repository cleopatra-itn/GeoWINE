import os
import sys
import time
import json
import h5py
import wikipediaapi
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__))

initial_path = ROOT_PATH / f'entities.json'
entities = []
with open(initial_path) as json_file:
    entities = json.load(json_file)

def get_wikipedia_summary(url):
    wikipedia = wikipediaapi.Wikipedia('en')
    id = url.rsplit('/', 1)[-1]
    lang_page = wikipedia.page(id, unquote=True)
    if lang_page.exists():
        return lang_page.summary
    else:
        return ''

start = time.perf_counter()
for i, entity in enumerate(entities):
    entity['en_description'] = get_wikipedia_summary(entity['wikipedia_page']) if entity['wikipedia_page'] else ''
    toc = time.perf_counter()
    print(f'====> Finished id {entity["id"]} -- {((i+1)/len(entities))*100:.2f}% -- {toc - start:0.2f}s')

entites_path = ROOT_PATH / f'entities_cached.json'
with open(entites_path, 'w') as json_file:
    json.dump(entities, json_file, ensure_ascii=False, indent=4)