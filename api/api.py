import json
from flask import Flask, request

app = Flask(__name__)

sample_results = [
    {
        'id': 'Q172251',
        'label': 'Yabrud',
        'link': 'https://www.wikidata.org/wiki/Q172251',
        'coordinates': [33.967222, 36.657222],
        'news_articles': [],
        'events': []
    }
]

@app.route('/select', methods=['POST'])
def query_selected_image():
    data = request.get_json()
    id = data['id']
    return 'Response data send succefully!' # sample_results

@app.route('/upload', methods=['POST'])
def query_upload_image():
    return sample_results