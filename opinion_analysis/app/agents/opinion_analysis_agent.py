
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from app.services.llm_service import llm_service
from app.services.news_scraper import news_scraper
from app.services.ner_service import ner_service

from app.utils.logger import setup_logger

# Create a logger for this module
logger = setup_logger("opinion_analysis_agent")


class OpinionAnalysisAgent:
    class State(TypedDict):
        query: str
        response: str
        is_opinion_related: bool = False
        news_articles: List[Dict] = []
        analysis_results: Dict = {}
        
    def __init__(self):
        """Initialize the Opinion Analysis Agent with necessary services"""
        self.llm_service = llm_service
        self.news_scraper = news_scraper
        self.ner_service = ner_service
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the opinion analysis agent"""
        
        # Define the state schema
        # Create the workflow
        workflow = StateGraph(self.State)
        
        # Add nodes
        workflow.add_node("check_relevance", self.check_if_opinion_related)
        workflow.add_node("fetch_news", self.fetch_news_articles)
        workflow.add_node("analyze_news", self.analyze_articles)
        workflow.add_node("generate_response", self.generate_response)
        
        # Define edges with routing function
        workflow.add_conditional_edges("check_relevance", self.route_from_check_relevance)
        workflow.add_edge("fetch_news", "analyze_news")
        workflow.add_edge("analyze_news", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("check_relevance")
        
        return workflow.compile()

    # Define the workflow nodes
    def check_if_opinion_related(self, state):
        """Check if the query is related to opinion analysis"""
        query = state["query"]
        is_related = self.llm_service.is_opinion_related(query)
        state["is_opinion_related"] = is_related
        return state
    
    def fetch_news_articles(self, state):
        """Fetch news articles based on the query"""
        query = state["query"]
        articles = self.news_scraper.search_news(query)
        state["news_articles"] = articles
        return state
    
    def analyze_articles(self, state):
        """Analyze the fetched news articles"""
        articles = state["news_articles"]
        analysis_results = {}
        
        for article in articles:
            # Extract named entities
            ner_results = self.ner_service.extract_entities(article["content"])
            
            # Perform sentiment analysis
            sentiment = self.llm_service.analyze_sentiment(article["content"])
            
            # Create summary using LLM
            summary = self.llm_service.generate_summary(article["content"])
            
            analysis_results[article["title"]] = {
                "title": article["title"],
                "publish_date": article["publish_date"],
                "content": article["content"],
                "summary": summary,
                "sentiment": sentiment,
                "named_entities": ner_results
            }
        state["analysis_results"] = analysis_results
        return state
    
    def route_from_check_relevance(self, state):
        if state["is_opinion_related"]:
            return "fetch_news"
        else:
            return "generate_response" 
        
    def generate_response(self, state):
        """Generate a response based on the analysis results"""
        if not state["is_opinion_related"]:
            response = self.llm_service.generate_rejection_message()
        else:
            response = self.llm_service.generate_analysis_response(
                state["query"], 
                state["analysis_results"]
            )
        state["response"] = response
        return state


def create_opinion_analysis_agent() -> OpinionAnalysisAgent:
    """Factory function to create an instance of OpinionAnalysisAgent"""
    return OpinionAnalysisAgent()
