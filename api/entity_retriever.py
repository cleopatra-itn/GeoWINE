import os
import json
import wikipediaapi
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__))

BASE_QUERY = f"""
            SELECT ?place ?location
            WHERE{{

            VALUES ?types {{ TYPES }}
            ?place wdt:P31 ?types .

            SERVICE wikibase:around
            {{
                ?place wdt:P625 ?location .
                bd:serviceParam wikibase:center "Point(LNG LAT)"^^geo:wktLiteral .
                bd:serviceParam wikibase:radius RADIUS .
            }}

            SERVICE wikibase:label
            {{
                bd:serviceParam wikibase:language "en" .
            }}
            }}

            """

COMPLETE_QUERY = f"""
            SELECT ?place ?placeLabel ?location ?image ?ensummarylink ?commons
            WHERE{{

            ?place wdt:P31 wd:WID .
            ?place wdt:P18 ?image .
            ?place wdt:P373 ?commons .

            SERVICE wikibase:around
            {{
                ?place wdt:P625 ?location .
                bd:serviceParam wikibase:center "Point(LNG LAT)"^^geo:wktLiteral .
                bd:serviceParam wikibase:radius RADIUS .
            }}

            ?ensummarylink schema:isPartOf <https://en.wikipedia.org/>; # Get item english wikipedia
                schema:about ?place .
            SERVICE wikibase:label
            {{
                bd:serviceParam wikibase:language "en" .
            }}
            }}
            """

class EntityRetriever:
    def __init__(self):
        self.wikidata_endpoint = 'https://query.wikidata.org/sparql'
        self.user_agent = 'Myy User Agent 1.0'
        self.sparql = SPARQLWrapper(self.wikidata_endpoint, agent=self.user_agent)
        self.wikipedia = wikipediaapi.Wikipedia('en')
        self.cached_entities = {ent['id']: ent for ent in json.load(open(f'{ROOT_PATH}/cached_entities.json'))}

    def retrieve(self, coords, radius, entity_type):
        geo_query_res = self._run_geospatial_wikidata_query(coords, radius, entity_type)
        return self._process_results_cached(geo_query_res)

    def _get_query_results(self, query):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)

        return self.sparql.query().convert()

    def _run_geospatial_wikidata_query(self, coords, radius, entity_type):
        query = BASE_QUERY.replace('TYPES', ' '.join([f'wd:{ent_typ}' for ent_typ in entity_type]))\
                        .replace('LNG', str(coords['lng']))\
                        .replace('LAT', str(coords['lat']))\
                        .replace('RADIUS', str(radius))

        try:
            results = self._get_query_results(query)
            return results['results']['bindings']
        except Exception as e:
            print(e)
            return None

    def _get_wikipedia_summary(self, url):
        id = url.rsplit('/', 1)[-1]

        lang_page = self.wikipedia.page(id, unquote=True)

        if lang_page.exists():
            return lang_page.summary
        else:
            return ''

    def _parse_wikidata_coords(self, coords):
        lng, lat = coords.split('Point(')[1].split(' ')
        lat = lat.replace(')', '')

        return [lat, lng]

    def _process_results(self, results):
        entities = []
        seen_ids = set()

        for r in results:
            id = r['place']['value'].split('/')[-1]

            if id in seen_ids:
                continue

            entities.append({
                'id': id,
                'label': r['placeLabel']['value'],
                'entity_uri': r['place']['value'],
                'coordinates': self._parse_wikidata_coords(r['location']['value']),
                'image_url': r['image']['value'],
                'en_description': self._get_wikipedia_summary(r['ensummarylink']['value']),
                'wikipedia_page': r['ensummarylink']['value'],
                'wikimedia_commons': f"https://commons.wikimedia.org/wiki/Category:{r['commons']['value'].replace(' ', '_')}"
            })

            seen_ids.add(id)

        return entities

    def _process_results_cached(self, results):
        entities = []
        seen_ids = set()

        for r in results:
            id = r['place']['value'].split('/')[-1]

            if id in seen_ids or id not in self.cached_entities:
                continue

            entities.append(self.cached_entities[id])
            seen_ids.add(id)

        return entities
