import os
import sys
import time
import json
import wikipediaapi
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__))

CLASSES = [
    'Q570116',
    'Q33506',
    'Q839954',
    'Q618123',
    'Q2319498',
    'Q43229',
    'Q327333',
    'Q1802963',
    'Q162875',
    'Q2221906',
    'Q2065736',
    'Q41176'
]

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

def get_wikipedia_summary(url):
    wikipedia = wikipediaapi.Wikipedia('en')
    id = url.rsplit('/', 1)[-1]
    lang_page = wikipedia.page(id, unquote=True)
    if lang_page.exists():
        return lang_page.summary
    else:
        return ''

entities = []
seen_ids = set()
start = time.perf_counter()
for i, ent_class in enumerate(CLASSES):
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
                # 'en_description': get_wikipedia_summary(result['enwikipedia']['value']),
                'wikipedia_page': result['enwikipedia']['value'] if 'enwikipedia' in result else '', # 'https://en.wikipedia.org/wiki/Main_Page',
                'wikimedia_commons': f"https://commons.wikimedia.org/wiki/Category:{result['commons']['value'].replace(' ', '_')}" if 'commons' in result else '' # 'https://commons.wikimedia.org/wiki/Main_Page'
            })
            seen_ids.add(id)
    except Exception as e:
        print(f'====> Could not get results for item {ent_class}')
        print(str(e))
        continue
    toc = time.perf_counter()
    print(f'====> Finished id {ent_class} -- {((i+1)/len(CLASSES))*100:.2f}% -- {toc - start:0.2f}s')
end = time.perf_counter()

entites_path = ROOT_PATH / f'entities.json'
with open(entites_path, 'w') as json_file:
    json.dump(entities, json_file, ensure_ascii=False, indent=4)

print(f'------------------------------------------------------')
print(f'Total time to extarct MLM entities: {end - start:0.4f}')
print(f'Numner of unique entities: {len(entities)}')
print(f'------------------------------------------------------')