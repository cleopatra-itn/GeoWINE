import os
import time
import json
import argparse
import wikipediaapi
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__))

# Add arguments to parser
parser = argparse.ArgumentParser(description='Download descriptions')
parser.add_argument('--chunk', default=1, type=int, help='number of chunk')
args = parser.parse_args()

assert args.chunk in range(1, 11)

# load data
data_path = ROOT_PATH / f'entities/entities_{args.chunk}.json'
entities = {}
with open(data_path) as json_file:
    entities = json.load(json_file)

def get_wikipedia_summary(url):
    try:
        wikipedia = wikipediaapi.Wikipedia('en')
        id = url.rsplit('/', 1)[-1]
        lang_page = wikipedia.page(id, unquote=True)
        if lang_page.exists():
            return lang_page.summary
        else:
            print(f'----> Failed to get description for {id}')
            return ''
    except Exception as e:
        print(e)
        return ''

start = time.perf_counter()
for i, entity in enumerate(entities):
    entity['en_description'] = get_wikipedia_summary(entity['wikipedia_page']) if entity['wikipedia_page'] else ''
    toc = time.perf_counter()
    print(f'====> Finished id {entity["id"]} -- {((i+1)/len(entities))*100:.2f}% -- {toc - start:0.2f}s')

entites_path = ROOT_PATH / f'cached_entities/cached_entities_{args.chunk}.json'
with open(entites_path, 'w') as json_file:
    json.dump(entities, json_file, ensure_ascii=False, indent=4)