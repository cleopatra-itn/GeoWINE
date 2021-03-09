import os
import sys
import time
import json
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__))

TYPES = {
    "Q34038": "Waterfall",
    "Q23413": "Castle",
    "Q751876": "Chateau",
    "Q16560": "Palace",
    "Q174782": "Square",
    "Q16970": "Church Building",
    "Q44613": "Monastery",
    "Q317557": "Parish Church",
    "Q2977": "Cathedral",
    "Q108325": "Chapel",
    "Q1088552": "Catholic Church Building",
    "Q56242215": "Catholic Cathedral",
    "Q2031836": "Eastern Orthodox Church",
    "Q34627": "Synagogue",
    "Q5393308": "Buddhist Temple",
    "Q845945": "Shinto Shrine",
    "Q32815": "Mosque",
    "Q11303": "Skyscraper",
    "Q12518": "Tower",
    "Q570116": "Tourist Attraction",
    "Q41176": "Building",
    "Q1497375": "Architectural Ensemble",
    "Q811979": "Architectural Structure",
    "Q4989906": "Monument",
    "Q5003624": "Memorial",
    "Q575759": "War Memorial",
    "Q162875": "Mausoleum",
    "Q1081138": "Historic site",
    "Q839954": "Archaeological Site",
    "Q17715832": "Castle Ruin",
    "Q12280": "Bridge",
    "Q33506": "Museum",
    "Q207694": "Art Museum",
    "Q17431399": "National Museum",
    "Q2772772": "Military Museum"
}

QUERY_TEMPLATE = """
    SELECT DISTINCT ?item ?itemLabel ?image ?coords ?commons ?enwikipedia
    WHERE
    {
    ?item wdt:P31 wd:WDID .
    ?item wdt:P18 ?image .
    ?item wdt:P625 ?coords .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }
    OPTIONAL { ?item wdt:P373 ?commons . ?enwikipedia schema:isPartOf <https://en.wikipedia.org/>; schema:about ?item . }
    }
    """

def get_results(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    # user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    user_agent = 'Myy User Agent 1.0'
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def parse_wikidata_coords(coords):
    lng, lat = coords.split('Point(')[1].split(' ')
    lat = lat.replace(')', '')

    return [lat, lng]

entities = []
seen_ids = set()
start = time.perf_counter()
for i, ent_class in enumerate(list(TYPES.keys())):
    try:
        results = get_results(QUERY_TEMPLATE.replace('WDID', ent_class))

        if len(results["results"]["bindings"]) == 0:
            print(f'No results for class: {ent_class}')
            continue

        for result in results["results"]["bindings"]:
            id = result['item']['value'].split('/')[-1]
            if id in seen_ids or 'Point' not in result['coords']['value']:
                continue
            entities.append({
                'id': id,
                'label': result['itemLabel']['value'],
                'entity_uri': result['item']['value'],
                'coordinates': parse_wikidata_coords(result['coords']['value']),
                'image_url': result['image']['value'],
                'wikipedia_page': result['enwikipedia']['value'] if 'enwikipedia' in result else '', # 'https://en.wikipedia.org/wiki/Main_Page',
                'wikimedia_commons': f"https://commons.wikimedia.org/wiki/Category:{result['commons']['value'].replace(' ', '_')}" if 'commons' in result else '', # 'https://commons.wikimedia.org/wiki/Main_Page'
                'type_id': ent_class,
                'type': TYPES[ent_class],
            })
            seen_ids.add(id)
    except Exception as e:
        print(f'====> Could not get results for item {ent_class}')
        print(str(e))
        continue
    toc = time.perf_counter()
    print(f'====> Finished id {ent_class} -- {((i+1)/len(list(TYPES.keys())))*100:.2f}% -- {toc - start:0.2f}s')
end = time.perf_counter()

def chunk_list(seq, num=10):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

chunked_entities = chunk_list(entities)

entites_path = ROOT_PATH / 'entities'

for i, chunk_data in enumerate(chunked_entities):
    chunk_path = f'{entites_path}/entities_{i+1}.json'
    with open(chunk_path, 'w') as json_file:
        json.dump(chunk_data, json_file, ensure_ascii=False, indent=4)

with open(f'{ROOT_PATH}/cached_entities.json', 'w') as json_file:
    json.dump(entities, json_file, ensure_ascii=False, indent=4)

print(f'------------------------------------------------------')
print(f'Total time to extarct MLM entities: {end - start:0.4f}')
print(f'Numner of unique entities: {len(entities)}')
print(f'------------------------------------------------------')