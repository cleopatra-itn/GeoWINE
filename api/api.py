import os
import json
from pathlib import Path
from geo_wine import GeoWINE
from flask import Flask, request

ROOT_PATH = Path(os.path.dirname(__file__))

SAMPLE_DATA = json.load(open(f'{ROOT_PATH}/sample_data.json'))

geowine = GeoWINE()

app = Flask(__name__, static_folder='../build', static_url_path='/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/select_image_entities', methods=['POST'])
def selected_image_entities():
    data = request.get_json()

    id = data['id']
    img_path = f'{ROOT_PATH}/{SAMPLE_DATA[id]["image"]}'
    true_coords = SAMPLE_DATA[id]['true_coords']

    radius = data['radius']
    entity_type = data['type']

    return geowine.retrieve_entities_with_image_path(path=img_path, radius=radius, entity_type=entity_type, true_coords=true_coords)

@app.route('/api/select_image_news_events', methods=['POST'])
def selected_image_news_events():
    data = request.get_json()
    return {
        'id': data['id'],
        **geowine.retrieve_news_events(data)
    }

@app.route('/upload', methods=['POST'])
def query_upload_image():
    return {}