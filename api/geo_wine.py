from events import OekgEventsApi
from news import NewsArticlesApi
from entity_retriever import EntityRetriever
from geolocation import GeolocationEstimator
from utils import ImageCropper, Cosine, ImageReader

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from embedding import Embedding

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
        print('Loaded GeoWINE successfully.')

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

    def _get_embeddings_from_cache(self, id):
        return self.embedding_model.get_cached_embeddings(id)

    def _get_similarity(self, u, v):
        return Cosine.similarity(u, v)

    def _get_entities(self, coords, radius, entity_type):
        return self.entity_retriever.retrieve(coords, radius, entity_type)

    def _retrieve_entities(self, image, radius, entity_type, true_coords=None):
        image_cropped = self._crop_image(image)
        print('Image cropped done!')
        pred_coords = self._predict_coordinates(image_cropped)
        print('Predict coordinates done!')
        print(f'Predicted coordinates: {pred_coords}')
        print(f'Retrieving entities...')
        entities = self._get_entities(pred_coords, radius, entity_type)
        print(f'Retrieving entities done!')

        print(f'Creating embeddings...')

        input_img_embed = self._get_embeddings_from_pil(image)

        retrieved_entities = []
        for entity in entities:
            entity_img_embed = self._get_embeddings_from_cache(entity['id'])

            if entity_img_embed is None:
                continue

            entity['similarity'] = {
                'object': self._get_similarity(input_img_embed['object'], entity_img_embed['object']),
                'location': self._get_similarity(input_img_embed['location'], entity_img_embed['location']),
                'scene': self._get_similarity(input_img_embed['scene'], entity_img_embed['scene']),
                'all': self._get_similarity(input_img_embed['all'], entity_img_embed['all'])
            }

            retrieved_entities.append(entity)

        print(f'Creating embeddings done.')

        true_coord_dict = {}
        if true_coords is not None:
            true_coord_dict['true_coords'] = {
                'lat': true_coords[0],
                'lng': true_coords[1]
            }

        print(f'Finished! Retrived {len(retrieved_entities)} entities.')

        return {
            **{
                'pred_coords': pred_coords,
                'query': {
                    'radius': radius,
                    'entity_type': entity_type
                    },
                'retrieved_entities': retrieved_entities
            },
            **true_coord_dict
        }

    def _get_news(self, keyword):
        return self.news_api.get_news_articles(keyword)

    def _get_events(self, id):
        return self.events_api.retrieve(id)

    def retrieve_entities_with_image_url(self, url, radius=25, entity_type=['Q33506'], true_coords=None):
        image = self._read_image_from_url(url)
        return self._retrieve_entities(image=image, radius=radius, entity_type=entity_type, true_coords=true_coords)

    def retrieve_entities_with_image_path(self, path, radius=25, entity_type=['Q33506'], true_coords=None):
        image = self._read_image_from_path(path)
        return self._retrieve_entities(image=image, radius=radius, entity_type=entity_type, true_coords=true_coords)

    def retrieve_news_events(self, entity):
        return {
            'news': self._get_news(entity['label']),
            'events': self._get_events((entity['id']))
        }
