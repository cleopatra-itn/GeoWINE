from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.request as req
import wikipediaapi
from utils import *
from PIL import Image
import imagehash
import os


def run_query(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    # user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    user_agent = 'Myy User Agent 1.0'
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    p2 = sparql.query()
    p3 = p2.convert()
    return p3



def get_summary(url):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    id = url.rsplit('/', 1)[-1]
    lang_page = wiki_wiki.page(id, unquote=True)
    if lang_page.exists():
        return lang_page.summary, url
    else:
        return 0


def query_radius_pure(entity_type, coord, radius):
    query = f"""SELECT ?place ?placeLabel ?location ?image ?ensummarylink
        WHERE{{

        ?place wdt:P31 wd:{entity_type} .
        ?place wdt:P18 ?image.

        SERVICE wikibase:around 
        {{
        ?place wdt:P625 ?location.
        bd:serviceParam wikibase:center "{coord}"^^geo:wktLiteral.
        bd:serviceParam wikibase:radius {radius}.
        }}
        ?ensummarylink schema:isPartOf <https://en.wikipedia.org/>; # Get item english wikipedia
              schema:about ?place.
        SERVICE wikibase:label{{bd:serviceParam wikibase:language "en".
        }}
        }} """
    try:
        results0 = run_query(query)
        results = results0['results']['bindings']
    except Exception as e:
        print(e)
        results = None
    return results

def query_radius_entities(coord, radius, entity_types, entity_labels, path_entities_images):
    if not os.path.exists(path_entities_images):
        os.mkdir(path_entities_images)

    valid_entities = []
    track_ids = []
    count = 0

    for entity_type, entity_label in zip(entity_types, entity_labels):
        results = query_radius_pure(entity_type, coord, radius)
        if results ==None:
            continue


        for r in results:

                id = r['place']['value'].split('/')[-1]
                if id in track_ids:
                    continue
                img_path = f'{path_entities_images}/{id}.jpeg'

                try:
                    req.urlretrieve(r['image']['value'], img_path)
                except:
                    print(f'---->Failed downloading image {id}')
                    continue

                image_hash = str(imagehash.average_hash(Image.open(f'{img_path}')))
                if not os.path.exists(f'{path_entities_images}/{image_hash}.jpeg'):
                    os.rename(img_path, f'{path_entities_images}/{image_hash}.jpeg')
                summary_content, summary_url = get_summary(r['ensummarylink']['value'])

                valid_entity = {}
                valid_entity['id'] = id
                valid_entity['label'] = r['placeLabel']['value']
                valid_entity['en_description'] = summary_content
                valid_entity['entity_uri'] = r['place']['value']
                lat, lng = r['location']['value'].split('Point(')[1].split(' ')
                lng = lng.replace(')', '')
                valid_entity['coordinates'] = [lat, lng]
                valid_entity['types'] = {entity_type: entity_label}
                valid_entity['image_url'] = r['image']['value']
                valid_entity['image_path'] = f'{path_entities_images}/{image_hash}.jpeg'
                valid_entity['image_hash'] = image_hash
                valid_entity['wikipedia_page'] = summary_url

                valid_entities.append(valid_entity)
                count += 1
                print(count)
                track_ids.append(id)

    return valid_entities



def save_radius_entities(pred_coords, ture_coords, radius, user_image_id, user_input_image_url, output_path, path_entity_images, path_input_image, input_entities):
    lat, lng = pred_coords
    entity_types = list(input_entities.keys())
    entity_labels = list(input_entities.values())
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    radius_entities = query_radius_entities(f'Point({lng} {lat})', radius, entity_types, entity_labels, path_entity_images)  # query all the entities within a radius k of the input coord

    dic = {
           'image_path': f'{path_input_image}',
           'image_url': user_input_image_url,
           'pred_coords': (lat, lng),
           'true_coords': ture_coords,
           'query': {'entity_types': entity_types, 'radius':radius},
           'retrieved_entities': radius_entities
           }
    save_file(f'{output_path}/radius_entities_{user_image_id}.json', dic)
    return dic








