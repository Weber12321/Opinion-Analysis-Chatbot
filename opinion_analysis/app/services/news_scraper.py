# app/services/news_scraper.py
from typing import List, Dict
import newspaper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults



class NewsScraperService:
    def __init__(self, region: str = "tw-tzh", max_results: int = 10):
        """Initialize the news scraper service"""
        wrapper = DuckDuckGoSearchAPIWrapper(region=region, time="d", max_results=max_results)
        self.search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news", output_format="list")

    def search_news(self, query: str) -> List[Dict]:
        """
        Search for news articles related to the query
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 5)
            
        Returns:
            List of news articles with title, url, publish_date, and content
        """
        # First, get search results using a search engine
        output_list = self.search.invoke(query)
        
        # Then, scrape each article
        articles = []
        for item in output_list:
            article = newspaper.article(item['link'])
            articles.append({
                "title": item['title'],
                "url": item['link'],
                "publish_date": article.publish_date.strftime("%Y-%m-%d %H:%M:%S") if article.publish_date else "Unknown",
                "content": article.text if article.text else "No content available"
            })

        return articles
