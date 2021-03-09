import os
import time
import json
import h5py
import argparse
from glob import glob
from PIL import Image
from pathlib import Path
from resizeimage import resizeimage

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from embedding import Embedding

ROOT_PATH = Path(os.path.dirname(__file__))

# Add arguments to parser
parser = argparse.ArgumentParser(description='Create entity embeddings')
parser.add_argument('--chunk', default=1, type=int, help='number of chunk')
args = parser.parse_args()

embedding_model = Embedding(embedding_file=f'embeddings/cached_embeddings_{args.chunk}.h5')

images = glob(f'{ROOT_PATH}/images/part_{args.chunk}/*')

start = time.perf_counter()
for i, image in enumerate(images):
    id = image.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    print(f'Embedding image for id {id}')

    try:
        img = Image.open(image, 'r')
        img = resizeimage.resize_contain(img, [256, 256])
        img = img.convert('RGB')
        embed = embedding_model.embed_pil_image(img, id=id)
    except:
        print(f'----> Failed embedding image for id {id}')
        continue

    toc = time.perf_counter()
    print(f'====> Finished id {id} -- {((i+1)/len(images))*100:.2f}% -- {toc - start:0.2f}s')

embedding_model.cached_embeddings.close()