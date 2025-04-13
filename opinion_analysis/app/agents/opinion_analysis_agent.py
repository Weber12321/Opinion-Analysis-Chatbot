
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from app.services.llm_service import LLMService
from app.services.news_scraper import NewsScraperService
from app.services.ner_service import NERService
from app.services.sentiment_analysis import SentimentAnalysisService
from app.utils.logger import setup_logger

# Create a logger for this module
logger = setup_logger("opinion_analysis_agent")

class OpinionAnalysisAgent:
    def __init__(self):
        """Initialize the Opinion Analysis Agent with necessary services"""
        self.llm_service = LLMService()
        self.news_scraper = NewsScraperService()
        self.ner_service = NERService()
        self.sentiment_service = SentimentAnalysisService()
        self.conversations = {}
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the opinion analysis agent"""
        
        # Define the state schema
        class State:
            messages: List[Any]
            query: str
            is_opinion_related: bool = False
            news_articles: List[Dict] = []
            analysis_results: Dict = {}
            
        # Define the workflow nodes
        def check_if_opinion_related(state):
            """Check if the query is related to opinion analysis"""
            query = state["query"]
            is_related = self.llm_service.is_opinion_related(query)
            return {"is_opinion_related": is_related}
        
        def fetch_news_articles(state):
            """Fetch news articles based on the query"""
            query = state["query"]
            articles = self.news_scraper.search_news(query)
            return {"news_articles": articles}
        
        def analyze_articles(state):
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
            
            return {"analysis_results": analysis_results}
        
        def route_from_check_relevance(state):
            if state["is_opinion_related"]:
                return "fetch_news"
            else:
                return "generate_response" 
            
        def generate_response(state):
            """Generate a response based on the analysis results"""
            if not state["is_opinion_related"]:
                response = self.llm_service.generate_rejection_message()
            else:
                response = self.llm_service.generate_analysis_response(
                    state["query"], 
                    state["analysis_results"]
                )
            
            return {"messages": state["messages"] + [AIMessage(content=response)]}
        
        # Create the workflow
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("check_relevance", check_if_opinion_related)
        workflow.add_node("fetch_news", fetch_news_articles)
        workflow.add_node("analyze_news", analyze_articles)
        workflow.add_node("generate_response", generate_response)
        
        # Define edges with routing function
        workflow.add_conditional_edges("check_relevance", route_from_check_relevance)
        workflow.add_edge("fetch_news", "analyze_news")
        workflow.add_edge("analyze_news", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("check_relevance")
        
        return workflow.compile()

graph = OpinionAnalysisAgent().workflow
