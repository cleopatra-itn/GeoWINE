import json
from flask import Flask, request

app = Flask(__name__)

SAMPLE_RESULTS_ENTITIES = {
    'Q2981': json.load(open('sample_data/entities/Q2981.json')),
    'Q82878': json.load(open('sample_data/entities/Q82878.json'))
}

SAMPLE_RESULTS_NEWS = json.load(open('sample_data/news/news.json'))
SAMPLE_RESULTS_EVENTS = json.load(open('sample_data/events/events.json'))

@app.route('/select_image_entities', methods=['POST'])
def selected_image_entities():
    data = request.get_json()
    id = data['id']
    return SAMPLE_RESULTS_ENTITIES[id]

@app.route('/select_image_news_events', methods=['POST'])
def selected_image_news_events():
    data = request.get_json()
    id = data['id']
    return {
        'id': id,
        'news': SAMPLE_RESULTS_NEWS[id],
        'events': SAMPLE_RESULTS_EVENTS[id]
    }

@app.route('/upload', methods=['POST'])
def query_upload_image():
    return {}