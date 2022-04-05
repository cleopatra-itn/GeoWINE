from SPARQLWrapper import SPARQLWrapper, JSON

class OekgEventsApi:
    def __init__(self):
        self.oekg_endpoint = 'http://oekg.l3s.uni-hannover.de/sparql'
        self.user_agent = 'Myy User Agent 1.0'
        self.sparql = SPARQLWrapper(self.oekg_endpoint, agent=self.user_agent)

    def retrieve(self, id):
        geo_query_res = self._run_oekg_query(id)
        return self._process_results(geo_query_res)

    def _run_oekg_query(self, id):
        query = f"""
                SELECT ?event ?label ?date ?description ?wikidataId ?predicate ?relation
                WHERE
                {{

                ?relation rdf:type oekg-s:Relation .
                ?relation ?subjectOrObject ?event .
                ?relation ?objectOrSubject ?location .

                ?relation sem:roleType ?predicate .
                ?event rdf:type sem:Event .
                ?event rdfs:label ?label .
                ?event sem:hasBeginTimeStamp ?date .
                ?event dcterms:description ?description .
                ?location owl:sameAs wd:{id} .
                ?event owl:sameAs ?wikidataId .
                ?event skos:prefLabel ?eventLabel .
                FILTER(REGEX(STR(?wikidataId),"^http://www.wikidata.org/entity/")) .
                FILTER (langMatches(lang(?description),"en")) .
                FILTER (langMatches(lang(?label),"en")) .
                }}
            """

        try:
            results = self._get_query_results(query)
            return results['results']['bindings'] if results else []
        except Exception as e:
            print(e)
            return []

    def _get_query_results(self, query):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)

        return self.sparql.query().convert()

    def _process_results(self, results):
        events = []
        seen_ids = set()

        for r in results:
            id = r['event']['value'].split('/')[-1]

            if id in seen_ids:
                continue

            events.append({
                'id': id,
                'label': r['label']['value'],
                'oekg_uri': r['event']['value'],
                'date': r['date']['value'],
                'en_description': r['description']['value']
            })

            seen_ids.add(id)

        return events
