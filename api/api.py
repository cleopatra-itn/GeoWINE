import os
import json
from pathlib import Path
from geo_wine import GeoWINE
from utils import ImageReader
from flask import Flask, request

ROOT_PATH = Path(os.path.dirname(__file__))

SAMPLE_DATA = json.load(open(f'{ROOT_PATH}/sample_data.json'))
ENTITY_TYPES = json.load(open(f'{ROOT_PATH}/entity_types.json'))

geowine = GeoWINE()

app = Flask(__name__, static_folder='../build', static_url_path='/')

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

@app.route('/api/select_image_entities', methods=['POST'])
def selected_image_entities():
    data = request.get_json()

    id = data['id']
    img_path = f'{ROOT_PATH.parent}/{SAMPLE_DATA[id]["image"]}'
    true_coords = SAMPLE_DATA[id]['true_coords']

    radius = data['radius']
    entity_types = [ent_type['value'] for ent_type in data['type']]
    query_types = [typ for ent_type in entity_types for typ in list(ENTITY_TYPES[ent_type].keys())]

    return geowine.retrieve_entities_with_image_path(path=img_path, radius=radius, entity_type=query_types, true_coords=true_coords)

@app.route('/api/select_image_news_events', methods=['POST'])
def selected_image_news_events():
    data = request.get_json()
    return {
        'id': data['id'],
        **geowine.retrieve_news_events(data)
    }

@app.route('/api/upload_image_entities', methods=['POST'])
def query_upload_image():
    image_file = request.files['file']

    radius = json.loads(request.form['radius'])
    entity_types = json.loads(request.form['type'])
    entity_types = [ent_type['value'] for ent_type in entity_types]
    query_types = [typ for ent_type in entity_types for typ in list(ENTITY_TYPES[ent_type].keys())]

    pil_image = ImageReader.read_and_resize(image_file.stream)

    return geowine.retrieve_entities_with_pil_path(image=pil_image, radius=radius, entity_type=query_types)