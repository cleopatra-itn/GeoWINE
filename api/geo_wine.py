from embedding import Embedding
from news import NewsArticlesApi
from utils import ImageCropper, Cosine, ImageReader
from geolocation import GeolocationEstimator
from entity_retriever import EntityRetriever
from events import OekgEventsApi

class GeoWINE():
    '''
    GeoWINE: Geolocation Wiki-Image-News-Events
    '''
    def __init__(self):
        self.img_cropper = ImageCropper()
        self.geolocation_model = GeolocationEstimator()
        self.embedding_model = Embedding()
        self.entity_retriever = EntityRetriever()
        self.news_api = NewsArticlesApi()
        self.events_api = OekgEventsApi()

    def _read_image_from_path(self, path):
        return ImageReader.read_from_path(path)

    def _read_image_from_url(self, url):
        return ImageReader.read_from_url(url)

    def _crop_image(self, image):
        return self.img_cropper.crop(image)

    def _predict_coordinates(self, image):
        return self.geolocation_model.predict(image)

    def _get_embeddings_from_pil(self, image, id=None):
        return self.embedding_model.embed_pil_image(image, id)

    def _get_embeddings_from_url(self, url, id=None):
        return self.embedding_model.embed_url_image(url, id)

    def _get_similarity(self, u, v):
        return Cosine.similarity(u, v)

    def _get_entities(self, coords, radius, entity_type):
        return self.entity_retriever.retrieve(coords, radius, entity_type)

    def _retrieve_entities(self, image, radius, entity_type, true_coords=None):
        image_cropped = self._crop_image(image)
        pred_coords = self._predict_coordinates(image_cropped)
        entities = self._get_entities(pred_coords, radius, entity_type)

        input_img_embed = self._get_embeddings_from_pil(image)

        for entity in entities:
            entity_img_embed = self._get_embeddings_from_url(entity['image_url'], entity['id'])

            entity['similarity'] = {
                'object': self._get_similarity(input_img_embed['object'], entity_img_embed['object']),
                'location': self._get_similarity(input_img_embed['location'], entity_img_embed['location']),
                'scene': self._get_similarity(input_img_embed['scene'], entity_img_embed['scene']),
                'all': self._get_similarity(input_img_embed['all'], entity_img_embed['all'])
            }

        true_coord_dict = {}
        if true_coords is not None:
            true_coord_dict['true_coords'] = {
                'lat': true_coords[0],
                'lng': true_coords[1]
            }

        return {
            **{
                'pred_coords': pred_coords,
                'query': {
                    'radius': radius,
                    'entity_type': entity_type
                    },
                'retrieved_entities': entities
            },
            **true_coord_dict
        }

    def _get_news(self, keyword):
        return self.news_api.get_news_articles(keyword)

    def _get_events(self, id):
        return self.events_api.retrieve(id)

    def retrieve_entities_with_image_url(self, url, radius=25, entity_type='Q33506', true_coords=None):
        image = self._read_image_from_url(url)
        return self._retrieve_entities(image=image, radius=radius, entity_type=entity_type, true_coords=true_coords)

    def retrieve_entities_with_image_path(self, path, radius=25, entity_type='Q33506', true_coords=None):
        image = self._read_image_from_path(path)
        return self._retrieve_entities(image=image, radius=radius, entity_type=entity_type, true_coords=true_coords)

    def retrieve_news_events(self, entity):
        return {
            'news': self._get_news(entity['label']),
            'events': self._get_events((entity['id']))
        }


# input_radiuses = {'street level' :1, 'city level': 25, 'region level': 200, 'country level' :750}

# input_radius = input_radiuses['region level']
# input_entities={'Q570116': 'tourist_attraction',
#                 'Q839954': 'archaeological_site',
#                 'Q2065736': 'cultural_property'}


# dic_data = {'hiroshima.jpeg':{'true':(34.391472, 132.453056), 'pred': (34.3869, 132.4513)},
#             'new_image.jpeg':{'true':(48.852448, 2.3488266), 'pred': (48.8531,	2.3494)},
#             'Ptolemaic-Temple-of-Horus-Edfu-Egypt.jpeg':{'true':( 24.977778, 32.873333), 'pred': (23.5966, 32.5727 )},
#             'Tabriz-Bazaar-Carpet-Section.jpeg':{'true':(38.080772, 46.292286), 'pred': (31.4477,	52.40519)},
#             'times_square.jpeg': {'true': ( 40.75773, -73.985708), 'pred': ( 40.7573, -73.9863)},
#             'vatican.jpeg': {'true': (41.906389, 12.454444 ), 'pred': (41.9053, 12.4541)},
#             'osaka.jpeg':{'true':(34.834, 135.606), 'pred':(38.6042, 141.2442)}
#             }

# input_image_name = 'hiroshima.jpeg'
# true_coords = dic_data[input_image_name]['true']
# pred_coords = dic_data[input_image_name]['pred']

# path_input_image = f"/data/s6enkacu/Projects/geolocation-demo/api/data/media/user_input_images/notreparis.jpg"

# geo_wine = GeoWINE()
# ent_res = geo_wine.retrieve_entities_with_image_path(path_input_image, radius=1, true_coords=[48.852737, 2.350699])

# ent_res2 = geo_wine.retrieve_entities_with_image_path(path_input_image, radius=1, true_coords=[48.852737, 2.350699])

# news_events_res = geo_wine.retrieve_news_events(ent_res['retrieved_entities'][0])