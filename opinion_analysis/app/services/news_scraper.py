# app/services/news_scraper.py
import httpx
import asyncio
from typing import List, Dict
from datetime import datetime
from newspaper import Article
from bs4 import BeautifulSoup

class NewsScraperService:
    def __init__(self):
        """Initialize the news scraper service"""
        self.client = httpx.Client(timeout=30.0)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    def search_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for news articles related to the query
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 5)
            
        Returns:
            List of news articles with title, url, publish_date, and content
        """
        # First, get search results using a search engine
        search_urls = self._get_search_results(query, max_results)
        
        # Then, scrape each article
        articles = []
        for url in search_urls:
            try:
                article = self._scrape_article(url)
                if article and article["content"].strip():
                    articles.append(article)
                    if len(articles) >= max_results:
                        break
            except Exception as e:
                print(f"Error scraping article {url}: {e}")
                continue
                
        return articles
    
    def _get_search_results(self, query: str, max_results: int = 10) -> List[str]:
        """
        Get search results for a news query
        
        This is a simplified implementation that would normally use a proper search API.
        In a production environment, you would use Google Search API, Bing News API, or similar.
        """
        # For demo purposes - would normally use a proper search API
        # This is a placeholder that would be replaced with actual search API calls
        try:
            # Use DuckDuckGo as it doesn't require an API key
            search_url = f"https://duckduckgo.com/html/?q={query}+news"
            response = self.client.get(search_url, headers=self.headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                # Extract links from search results
                for link in soup.select('.result__a'):
                    url = link.get('href')
                    if url and ('news' in url or 'article' in url) and not url.endswith(('.pdf', '.doc', '.xls')):
                        results.append(url)
                        if len(results) >= max_results:
                            break
                            
                return results
            return []
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _scrape_article(self, url: str) -> Dict:
        """
        Scrape a news article from the given URL
        
        Args:
            url: The URL of the article to scrape
            
        Returns:
            Dictionary with title, url, publish_date, and content
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Format the publish date
            publish_date = "Unknown"
            if article.publish_date:
                publish_date = article.publish_date.strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "title": article.title,
                "url": url,
                "publish_date": publish_date,
                "content": article.text
            }
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None
