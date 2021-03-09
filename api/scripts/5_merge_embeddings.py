import os
import time
import h5py
from glob import glob
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__))

embedding_files = glob(f'{ROOT_PATH}/embeddings/cached_embeddings_*.h5')

cached_embeddings = h5py.File(f'{ROOT_PATH}/cached_embeddings.h5', 'a')

start = time.perf_counter()
for i, embed_file in enumerate(embedding_files):
    embeddings = h5py.File(embed_file, 'r')
    for j, embed_key in enumerate(embeddings.keys()):
        if embed_key not in cached_embeddings:
            cached_embeddings.create_dataset(name=embed_key, data=embeddings[embed_key][()], compression="gzip", compression_opts=9)
        toc = time.perf_counter()
        print(f'==> Finished id {embed_key} -- {embed_file.split("/")[-1]} -- {((j+1)/len(embeddings.keys()))*100:.2f}% -- {toc - start:0.2f}s')
    toc = time.perf_counter()
    print(f'====> Finished file {embed_file.split("/")[-1]} -- {((i+1)/len(embedding_files))*100:.2f}% -- {toc - start:0.2f}s')

cached_embeddings.close()