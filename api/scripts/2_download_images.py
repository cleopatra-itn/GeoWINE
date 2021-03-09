import os
import time
import json
import argparse
import cairosvg
import numpy as np
from PIL import Image
from glob import glob
from urllib import parse
from pathlib import Path
import urllib.request as req
from resizeimage import resizeimage

ROOT_PATH = Path(os.path.dirname(__file__))

# Add arguments to parser
parser = argparse.ArgumentParser(description='Download images')
parser.add_argument('--chunk', default=1, type=int, help='number of chunk')
args = parser.parse_args()

assert args.chunk in range(1, 11)

# load data
data_path = ROOT_PATH / f'entities/entities_{args.chunk}.json'
data = []
with open(data_path) as json_file:
    data = json.load(json_file)

# create directory
temp_path = ROOT_PATH / 'temp/'
thumbnails_path = ROOT_PATH / f'images/part_{args.chunk}'

existing_imgs = [im.rsplit('/', 1)[-1].split('.', 1)[0] for im in glob(f'{thumbnails_path}/*')]

data_with_images = {}
count_images = 0
tic = time.perf_counter()
for i, d in enumerate(data):
    id = d['id']
    print(f'Getting image for id {id}')

    if id in existing_imgs:
        print(f'====>Image alread exists')
        count_images += 1
        continue

    image = d['image_url']
    name = image.rsplit('/', 1)[-1]
    img_format = name.rsplit('.', 1)[-1].lower()

    if img_format not in ['svg', 'jpeg', 'jpg', 'png']:
        print(f'---->Image format not supported: {img_format}. Skipping id {id}')
        continue

    img_name = f'{id}.{img_format}'
    img_path = f'{temp_path}/{img_name}'

    try:
        req.urlretrieve(image, img_path)
    except:
        print(f'---->Failed downloading image {image}')
        continue

    if img_format == 'svg':
        try:
            img_name = img_name.replace('.svg', '.png')
            cairosvg.svg2png(url=img_path, write_to=f'{temp_path}/{img_name}')
            try:
                os.remove(img_path) # delete image
            except OSError:
                print (f'---->Failed to delete {img_path}')
                continue
            img_path = f'{temp_path}/{img_name}' # set path the new image
        except:
            print(f'---->Failed converting {img_path} to PNG.')
            try:
                os.remove(img_path) # delete image
            except OSError:
                print (f'---->Failed to delete {img_path}')
            continue

    try:
        img = Image.open(img_path, 'r')
        img = resizeimage.resize_contain(img, [256, 256])
        img = img.convert('RGB')
        if np.array(img).shape == (256, 256, 3):
            img.save(f'{thumbnails_path}/{img_name}')
            count_images += 1
            try:
                os.remove(img_path) # delete image
            except OSError:
                print (f'---->Failed to delete {img_path}')
            print(f'==>Finished image {img_name}')
    except Exception as e:
        print(f'---->Failed to read and resize image {img_path}')
        try:
            os.remove(img_path) # delete image
        except OSError:
            print (f'---->Failed to delete {img_path}')

    toc = time.perf_counter()
    print(f'====>Finished id {id} -- {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')

print(f'------------------------------------------------------')
print(f'Total items to download images: {len(data)}')
print(f'We downloaded and resized {count_images} images')
print(f'Finished chunk {args.chunk}')
print(f'------------------------------------------------------')