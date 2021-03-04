import os
import sys
import time
import json
import h5py
from pathlib import Path
from multiprocessing import Pool

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from embedding import Embedding

ROOT_PATH = Path(os.path.dirname(__file__))

initial_path = ROOT_PATH / f'entities_cached.json'
entities = []
with open(initial_path) as json_file:
    entities = json.load(json_file)

data = ([i, ent['image_url'], ent['id']] for i, ent in enumerate(entities))

embedding_model = Embedding()

# def get_embeddings(args):
#     i, image, id = args
#     embed = embedding_model.embed_url_image(image, id=id)
#     print(f'====> Finished id {id} -- {((i+1)/len(entities))*100:.2f}%')

# with Pool(1) as p:
#     p.map(get_embeddings, data)

start = time.perf_counter()
for i, entity in enumerate(entities):
    embed = embedding_model.embed_url_image(entity['image_url'], id=entity['id'])
    toc = time.perf_counter()
    print(f'====> Finished id {entity["id"]} -- {((i+1)/len(entities))*100:.2f}% -- {toc - start:0.2f}s')

embedding_model.cached_embeddings.close()