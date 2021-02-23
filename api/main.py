from argparse import ArgumentParser
from math import ceil
import torch
from tqdm.auto import tqdm
from geo_train_base import MultiPartitioningClassifier
from geo_dataset import FiveCropImageDataset
from query_radius import save_radius_entities
from utils import *
from utils_image import GeoEstimator
from utils_image import SceneClassificator
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import h5py
import time
import numpy as np
from torch.autograd import Variable
import os
from scipy import spatial
from eventregistry import EventRegistry, QueryArticlesIter
import imagehash
import shutil

def parse_args():
    args = ArgumentParser()
    args.add_argument("--checkpoint", type=Path, default=Path("models/base_M/epoch=014-val_loss=18.4833.ckpt"), help="Checkpoint to already trained model (*.ckpt)" )
    args.add_argument("--gpu",  action="store_true", default=False)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--hparams",type=Path,default=Path("models/base_M/hparams.yaml"), help="Path to hparams file (*.yaml) generated during training")

    args.add_argument("--path_output", type=str, default='data')
    args.add_argument("--path_models", type=Path, default=Path('models'))
    args.add_argument("--path_entity_images", type=Path, default=Path("data/media/entity_images"))
    args.add_argument("--dir_input_image", type=Path, default=Path('data/media/user_input_images/'))

    return args.parse_args()


class GeolocationEstimation():
    def __init__(self):
        self.args = parse_args()
        print("Load model from ", self.args.checkpoint)
        model = MultiPartitioningClassifier.load_from_checkpoint(
            checkpoint_path=str(self.args.checkpoint),
            hparams_file=str(self.args.hparams),
            map_location=None,
        )
        self.geo_model = model
        dataloader = torch.utils.data.DataLoader(
            FiveCropImageDataset(meta_csv=None, image_dir=self.args.dir_input_image),
            batch_size=ceil(self.args.batch_size / 5),
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        print("Number of images: ", len(dataloader.dataset))
        self.dataloader = dataloader

    def geo_esitmation(self):
        image_dir = self.args.dir_input_image
        args = self.args
        print("Load model from ", args.checkpoint)
        model = MultiPartitioningClassifier.load_from_checkpoint(
            checkpoint_path=str(args.checkpoint),
            hparams_file=str(args.hparams),
            map_location=None,
        )

        print("Init dataloader")
        dataloader = torch.utils.data.DataLoader(
            FiveCropImageDataset(meta_csv=None, image_dir=image_dir),
            batch_size=ceil(args.batch_size / 5),
            shuffle=False,
            num_workers=args.num_workers,
        )
        # model = self.geo_model
        # dataloader = self.dataloader

        model.eval()
        print("Number of images: ", len(dataloader.dataset))
        dataloader = self.dataloader
        if len(dataloader.dataset) == 0:
            raise RuntimeError(f"No images found in {image_dir}")
        rows = []
        for X in tqdm(dataloader):
            if args.gpu:
               X[0] = X[0].cuda()
            img_paths, pred_classes, pred_latitudes, pred_longitudes = model.inference(X)
            print(pred_latitudes)
            for p_key in pred_classes.keys():
                for img_path, pred_class, pred_lat, pred_lng in zip(img_paths,pred_classes[p_key].cpu().numpy(), pred_latitudes[p_key].cpu().numpy(),pred_longitudes[p_key].cpu().numpy(),):
                    rows.append( { "img_id": Path(img_path).stem, "p_key": p_key, "pred_class": pred_class, "pred_lat": pred_lat, "pred_lng": pred_lng,})

        output_lat = rows[0]['pred_lat']
        output_lng = rows[0]['pred_lng']

        return output_lat, output_lng


class Embeddings():
    def __init__(self, path_models):
        hierarchy = f"{path_models}/scene/scene_hierarchy_places365.csv"
        labels = f"{path_models}/scene/categories_places365.txt"
        model_path_scene = f"{path_models}/scene/resnet50_places365.pth.tar"
        model_path_loc = f"{path_models}/location/base_M/"

        # set device
        self.DEVICE = torch.device('cpu')
        if args.gpu:
            torch.cuda.set_device(0)
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.location_obj = GeoEstimator(model_path_loc, use_cpu=True)
        self.scence_obj = SceneClassificator(model_path=model_path_scene, labels_file=labels, hierarchy_file=hierarchy)

        # Import ResNet-152
        print('Loading ResNet-152...')
        resnet152 = models.resnet152(pretrained=True)
        modules = list(resnet152.children())[:-1]
        for p in resnet152.parameters():
            p.requires_grad = False
        resnet152 = nn.Sequential(*modules).to(self.DEVICE)
        self.resnet = resnet152


    def get_embeddings(self, path_entity_images, path_uploaded_image, path_output, radius_entities, hash_input_image):
            scaler = transforms.Resize((224, 224))
            to_tensor = transforms.ToTensor()

            radius_images_hash = [r['image_hash'] for r in radius_entities['retrieved_entities']]

            images_path = [path_uploaded_image]
            images_path.extend( [f"{path_entity_images}/{im_hash}.jpeg" for im_hash in radius_images_hash])  # reads entity image hashes from json

            # write train data
            HDF5_DIR = f'{path_output}/embeddings.h5'
            tic = time.perf_counter()
            errors = 0

            if os.path.exists(HDF5_DIR):
                h5f = h5py.File(HDF5_DIR, 'r+')
            else:
                h5f = h5py.File(HDF5_DIR, 'a')

            for i, img in enumerate(images_path):
                    if i==0: # it is input image (since input image is not saved with hash in the input_dir)
                        id = hash_input_image
                    else:
                        _, img_name = get_file_name(img)
                        id = img_name.split('.')[0]

                    print(f'Getting results for id {id}')
                    if f'{id}_location' in h5f:
                        print(f'====>ID {id} already exists in the dataset!')
                        continue

                    try:
                        im_pil = Image.open(img).convert('RGB')
                        result = Variable(to_tensor(scaler(im_pil)).to(self.DEVICE).unsqueeze(0))
                    except Exception as e:
                        print(e)
                        continue

                    try:
                        resnet_features = self.resnet(result)  # resnet
                        object_features = resnet_features.data.view(result.size(0), 2048).detach().cpu().numpy()[0]
                        location_features = self.location_obj.get_img_embedding(img)
                        scene_features = self.scence_obj.get_img_embedding(img)
                        l_s = np.concatenate((location_features[0], scene_features[0]))
                        all_features = np.concatenate((l_s, object_features))
                    except Exception as e:
                        errors += 1
                        print(e)
                        continue

                    h5f.create_dataset(name=f'{id}_location', data=location_features[0], compression="gzip", compression_opts=9)
                    h5f.create_dataset(name=f'{id}_scene', data=scene_features[0], compression="gzip", compression_opts=9)
                    h5f.create_dataset(name=f'{id}_object', data=object_features, compression="gzip", compression_opts=9)
                    h5f.create_dataset(name=f'{id}_all', data=all_features, compression="gzip", compression_opts=9)
                    toc = time.perf_counter()
                    print( f'====> Finished image for id {id} -- {((i + 1) / len(images_path)) * 100:.2f}% -- {toc - tic:0.2f}s')
            h5f.close()
            print(f'Done with {errors} errors!')


    def get_similarity(self, path_embeddings, input_image_hash, radius_entities):

        embeddings = h5py.File(f'{path_embeddings}/embeddings.h5', 'r')
        sim = {}

        for r in radius_entities['retrieved_entities']:
                entity_hash = r['image_hash']
                sim[entity_hash] = {}

                try:
                    sim[entity_hash]['location'] = 1 - spatial.distance.cosine(embeddings[f'{input_image_hash}_location'], embeddings[f'{entity_hash}_location'])
                    sim[entity_hash]['scene'] = 1 - spatial.distance.cosine(embeddings[f'{input_image_hash}_scene'], embeddings[f'{entity_hash}_scene'])
                    sim[entity_hash]['object'] = 1 - spatial.distance.cosine(embeddings[f'{input_image_hash}_object'], embeddings[f'{entity_hash}_object'])
                    sim[entity_hash]['all'] = 1 - spatial.distance.cosine(embeddings[f'{input_image_hash}_all'], embeddings[f'{entity_hash}_all'])
                except:
                    sim[entity_hash]['location'] = 0
                    sim[entity_hash]['scene'] = 0
                    sim[entity_hash]['object'] = 0
                    sim[entity_hash]['all'] = 0

                r['similarity_location'] = sim[entity_hash]['location']
                r['similarity_scene'] = sim[entity_hash]['scene']
                r['similarity_object'] = sim[entity_hash]['object']
                r['similarity_all'] = sim[entity_hash]['all']

        sorted_object = sorted(sim, key=lambda x: (sim[x]['object']), reverse=True)
        sorted_location = sorted(sim, key=lambda x: (sim[x]['location']), reverse=True)
        sorted_scene = sorted(sim, key=lambda x: (sim[x]['scene']), reverse=True)
        sorted_all = sorted(sim, key=lambda x: (sim[x]['all']), reverse=True)
        sorted_hashes = {'scene': sorted_scene, 'location': sorted_location, 'object': sorted_object, 'all': sorted_all}
        radius_entities['sorted_similarity'] = sorted_hashes

        return radius_entities


class NewsArticlesApi():
    def __init__(self, api_key=''):
        self.news_api = EventRegistry(apiKey=api_key, repeatFailedRequestCount=1)
        self.lang_code = {
            'en': 'eng',
            'de': 'deu',
            'fr': 'fra',
            'it': 'ita',
            'es': 'spa',
            'pl': 'pol',
            'ro': 'ron',
            'nl': 'nld',
            'hu': 'hun',
            'pt': 'por'
        }
        self._no_tokens = False

    def get_news_articles(self, keyword, lang, sort_by='date', max_items=10):

        keyword_query = QueryArticlesIter(keywords=keyword,
                                        keywordsLoc='body',
                                        locationUri=self.news_api.getLocationUri(keyword),
                                        lang=self.lang_code[lang],
                                        dataType='news')

        # if no tokens available returnn no results
        if self._no_tokens:
            return []

        # in case of any exception return no news articles
        try:
            keyword_articles = []
            for article in keyword_query.execQuery(self.news_api, sortBy=sort_by, maxItems=max_items):
                keyword_articles.append({
                    'title': article['title'],
                    'date': article['date'],
                    'source': article['source']['uri'],
                    'url': article['url'],
                    'body': article['body']
                })
            return keyword_articles
        except:
            self._no_tokens = True
            return []

    def reset(self):
        self._no_tokens = False


    def save_news(self, entity_id, path_output, input_image_hash):
        radius_entities0 = open_json(f'{path_output}/radius_entities_{input_image_hash}.json')
        radius_entities = radius_entities0['retrieved_entities']
        for r in radius_entities:
                if r['id'] == entity_id:
                    clicked_entity = r
                    break
        news = self.get_news_articles(clicked_entity['label'], 'en')
        clicked_entity['news'] = news
        save_file(f'{path_output}/radius_entities_{input_image_hash}.json', radius_entities0)


class GetResultsInputImage():
    def __init__(self, path_input_image):  # save each uploaded image with name including "new_"

        img_dir, img_name = get_file_name(path_input_image)

        input_image_hash = str(imagehash.average_hash(Image.open(path_input_image)))
        shutil.copy(path_input_image, f'{img_dir}_archive/{input_image_hash}.jpeg')  # move the input image with its hash code to 'archive' folder
        os.rename(path_input_image, f'{img_dir}/new_image.jpeg')  # only one image is saved (per upload)
        self.input_image_hash = input_image_hash

    def get_results(self,
                        input_entities,
                        true_coords ,
                    pred_coords,  # for test
                        path_input_image,
                        radius_km,
                        path_entity_images,
                        path_output,
                        embeddings_obj):
            input_img_dir, input_img_name = get_file_name(path_input_image)
            # geo_obj = GeolocationEstimation()
            # lat, lng = geo_obj.geo_esitmation()   #to-do get embeddings from here
            # pred_coords = (round(lat, 4), round(lng, 4))
            input_image_hash = self.input_image_hash
            if input_image_hash == None:
                return 0

            path_input_image = f'{input_img_dir}_archive/{input_image_hash}.jpeg'

            # if os.path.exists(f'{path_output}/radius_entities_{input_image_hash}.json'):
            #     radius_entities = open_json(f'{path_output}/radius_entities_{input_image_hash}.json')
            # else:
            radius_entities = save_radius_entities(pred_coords, true_coords, radius_km, input_image_hash, 'input image url', path_output, path_entity_images, path_input_image, input_entities)

            embeddings_obj.get_embeddings(path_entity_images=path_entity_images, path_uploaded_image=path_input_image, path_output=path_output, radius_entities=radius_entities, hash_input_image=input_image_hash)
            radius_entities_with_similarity = embeddings_obj.get_similarity(path_embeddings=path_output, input_image_hash=input_image_hash, radius_entities= radius_entities)
            save_file(f'{path_output}/radius_entities_{input_image_hash}.json', radius_entities_with_similarity)


# UI call
args = parse_args()
path_output = args.path_output
path_models = args.path_models
path_entity_images = args.path_entity_images



# user inputs
input_radiuses = {'street level' :1, 'city level': 25, 'region level': 200, 'country level' :750}

input_radius = input_radiuses['region level']
input_entities={'Q570116': 'tourist_attraction',
                'Q839954': 'archaeological_site',
                'Q2065736': 'cultural_property'}


dic_data = {'hiroshima.jpeg':{'true':(34.391472, 132.453056), 'pred': (34.3869, 132.4513)},
            'new_image.jpeg':{'true':(48.852448, 2.3488266), 'pred': (48.8531,	2.3494)},
            'Ptolemaic-Temple-of-Horus-Edfu-Egypt.jpeg':{'true':( 24.977778, 32.873333), 'pred': (23.5966, 32.5727 )},
            'Tabriz-Bazaar-Carpet-Section.jpeg':{'true':(38.080772, 46.292286), 'pred': (31.4477,	52.40519)},
            'times_square.jpeg': {'true': ( 40.75773, -73.985708), 'pred': ( 40.7573, -73.9863)},
            'vatican.jpeg': {'true': (41.906389, 12.454444 ), 'pred': (41.9053, 12.4541)},
            'osaka.jpeg':{'true':(34.834, 135.606), 'pred':(38.6042, 141.2442)}
            }
input_image_name = 'new_image.jpeg'
true_coords = dic_data[input_image_name]['true']
pred_coords = dic_data[input_image_name]['pred']
path_input_image = f"{path_output}/media/user_input_images/{input_image_name}"  # upload new image to path_input_image (only one) - all uploaed images will be saved by their hash code in 'archive' folder


get_results_input_image = GetResultsInputImage(path_input_image)
embeddings_obj = Embeddings(path_models=path_models)

get_results_input_image.get_results(
            input_entities = input_entities,
            true_coords = true_coords,
    pred_coords = pred_coords,
            path_input_image = path_input_image,
            radius_km = input_radius,
            path_entity_images = path_entity_images,
            path_output= path_output,
            embeddings_obj=embeddings_obj)

# if user selects an entity on the map:

news_api = NewsArticlesApi(api_key='041e85db-cf3a-481c-8559-45f4b45ee47a')
news_api.save_news(entity_id='Q323767', # selected entity id on the map
                   path_output=path_output,
                   input_image_hash='fffffff379000000')

