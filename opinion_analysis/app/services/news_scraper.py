# app/services/news_scraper.py
import os
from typing import List, Dict
from app.services.ner_extraction import ner_extractor
import newspaper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool


def extract_ner(extractor, text):
    doc = extractor(text)
    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )
    return entities


@tool("search_news", return_direct=True)
def search_news(query: str) -> List[Dict]:
    """
    Search for news articles related to the query if the query is related to opinion analysis, public sentiment,

    Args:
        query: The search query

    Returns:
        List of news articles with title, url, publish_date, and content
    """
    region = os.getenv("REGION", "tw-tzh")
    wrapper = DuckDuckGoSearchAPIWrapper(region=region, time="d", max_results=15)
    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper, source="news", output_format="list"
    )

    output_list = search.invoke(query)

    # Then, scrape each article
    articles = []
    for item in output_list:
        try:
            article = newspaper.article(item["link"])
            if not article.text:
                continue
            articles.append(
                {
                    "title": item["title"],
                    "url": item["link"],
                    "publish_date": (
                        article.publish_date.strftime("%Y-%m-%d %H:%M:%S")
                        if article.publish_date
                        else "Unknown"
                    ),
                    "content": article.text,
                    "ner": extract_ner(ner_extractor, article.text),
                }
            )
        except UnicodeEncodeError:
            continue
        except newspaper.exceptions.ArticleException:
            continue

    return articles[-10:]
