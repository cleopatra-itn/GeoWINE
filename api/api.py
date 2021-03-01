import os
import json
from pathlib import Path
from geo_wine import GeoWINE
from flask import Flask, request

ROOT_PATH = Path(os.path.dirname(__file__))

geowine = GeoWINE()

app = Flask(__name__)

SAMPLE_IMAGES = {
    'notreparis.jpg': {
        'image': f'{ROOT_PATH}/sample_images/notreparis.jpg',
        'true_coords': [48.852966, 2.349902]
    }
}

@app.route('/api/select_image_entities', methods=['POST'])
def selected_image_entities():
    data = request.get_json()

    id = data['id']
    img_path = SAMPLE_IMAGES[id]['image']
    true_coords = SAMPLE_IMAGES[id]['true_coords']

    radius = data['radius']
    entity_type = data['type']

    return geowine.retrieve_entities_with_image_path(path=img_path, radius=radius, entity_type=entity_type, true_coords=true_coords)

@app.route('/api/select_image_news_events', methods=['POST'])
def selected_image_news_events():
    data = request.get_json()
    # id = data['id']
    # label = data['label']
    return {
        'id': data['id'],
        **geowine.retrieve_news_events(data)
    }

@app.route('/upload', methods=['POST'])
def query_upload_image():
    return {}