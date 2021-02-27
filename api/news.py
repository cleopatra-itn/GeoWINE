import os
from pathlib import Path
from eventregistry import EventRegistry, QueryArticlesIter

ROOT_PATH = Path(os.path.dirname(__file__))

class NewsArticlesApi():
    def __init__(self, api_key=open(f'{ROOT_PATH}/ER_API_KEY').readline().rstrip()):
        self.news_api = EventRegistry(apiKey=api_key, repeatFailedRequestCount=1)
        self.lang_code = {
            'en': 'eng',
            'de': 'deu',
            'fr': 'fra',
            'it': 'ita',
            'es': 'spa',
            'pl': 'pol',
            'ro': 'ron',
            'nl': 'nld',
            'hu': 'hun',
            'pt': 'por'
        }
        self._no_tokens = False

    def get_news_articles(self, keyword, lang='en', sort_by='date', max_items=10):

        keyword_query = QueryArticlesIter(keywords=keyword,
                                        keywordsLoc='body',
                                        locationUri=self.news_api.getLocationUri(keyword),
                                        lang=self.lang_code[lang],
                                        dataType='news')

        # if no tokens available returnn no results
        if self._no_tokens:
            return []

        # in case of any exception return no news articles
        try:
            keyword_articles = []
            for article in keyword_query.execQuery(self.news_api, sortBy=sort_by, maxItems=max_items):
                keyword_articles.append({
                    'title': article['title'],
                    'date': article['date'],
                    'source': article['source']['uri'],
                    'url': article['url'],
                    'body': article['body']
                })
            return keyword_articles
        except:
            self._no_tokens = True
            return []

    def reset(self):
        self._no_tokens = False