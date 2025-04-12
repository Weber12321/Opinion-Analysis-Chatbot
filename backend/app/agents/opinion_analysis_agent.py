# app/agents/opinion_analysis_agent.py
import uuid
import asyncio
from typing import Tuple, Dict, List, Any, Optional
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
                sentiment = self.sentiment_service.analyze_sentiment(article["content"])
                
                # Create summary using LLM
                summary = self.llm_service.generate_summary(article["content"])
                
                analysis_results[article["url"]] = {
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

    async def process_message(self, message: str, conversation_id: Optional[str] = None) -> Tuple[str, str]:
        """Process a message and return the response and conversation ID"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []
            logger.info(f"Created new conversation with ID: {conversation_id}")
        
        logger.info(f"Processing message in conversation {conversation_id}: {message[:50]}...")
        
        # Add user message to conversation history
        self.conversations[conversation_id].append(HumanMessage(content=message))
        
        # Initialize state for workflow
        initial_state = {
            "messages": self.conversations[conversation_id],
            "query": message,
            "is_opinion_related": False,
            "news_articles": [],
            "analysis_results": {}
        }
        logger.info("------")

        # Return the result generator that streams the workflow execution
        return self._stream_workflow_execution(initial_state, conversation_id)
        
    def _stream_workflow_execution(self, initial_state, conversation_id):
        """Stream the execution of the workflow to provide real-time updates"""

        self.workflow.astream()

        # First, yield a starting message
        # yield "Starting to process your request...", conversation_id
        
        # Check if the query is opinion related
        # is_related = await asyncio.to_thread(
        #     self.llm_service.is_opinion_related, 
        #     initial_state["query"]
        # )
        
        # if not is_related:
        #     logger.info(f"Query not opinion-related in conversation {conversation_id}")
        #     yield "This query doesn't appear to be related to opinion analysis...", conversation_id
            
        #     # Generate rejection message
        #     rejection = await asyncio.to_thread(
        #         self.llm_service.generate_rejection_message
        #     )
            
        #     # Update conversation history
        #     self.conversations[conversation_id].append(AIMessage(content=rejection))
            
        #     # Return final response
        #     yield rejection, conversation_id
        #     return
            
        # # If we get here, the query is opinion-related
        # yield "This is an opinion-related query. Searching for relevant news articles...", conversation_id
        
        # # Fetch news articles
        # articles = await asyncio.to_thread(
        #     self.news_scraper.search_news,
        #     initial_state["query"]
        # )
        
        # if not articles:
        #     logger.warning(f"No articles found for query in conversation {conversation_id}")
        #     yield "I couldn't find any relevant news articles. Let me provide a general response.", conversation_id
            
        #     # Generate a general response without specific articles
        #     general_response = await asyncio.to_thread(
        #         self.llm_service.generate_analysis_response,
        #         initial_state["query"],
        #         {}
        #     )
            
        #     # Update conversation history
        #     self.conversations[conversation_id].append(AIMessage(content=general_response))
            
        #     # Return final response
        #     yield general_response, conversation_id
        #     return
            
        # # If we have articles, process them
        # yield f"Found {len(articles)} relevant news articles. Analyzing content...", conversation_id
        
        # # Process articles one by one and stream updates
        # analysis_results = {}
        # for i, article in enumerate(articles):
        #     yield f"Analyzing article {i+1}/{len(articles)}: {article['title']}", conversation_id
            
        #     # Extract named entities
        #     ner_results = await asyncio.to_thread(
        #         self.ner_service.extract_entities,
        #         article["content"]
        #     )
            
        #     # Perform sentiment analysis
        #     sentiment = await asyncio.to_thread(
        #         self.sentiment_service.analyze_sentiment,
        #         article["content"]
        #     )
            
        #     # Create summary using LLM
        #     summary = await asyncio.to_thread(
        #         self.llm_service.generate_summary,
        #         article["content"]
        #     )
            
        #     # Store analysis results
        #     analysis_results[article["url"]] = {
        #         "title": article["title"],
        #         "publish_date": article["publish_date"],
        #         "content": article["content"],
        #         "summary": summary,
        #         "sentiment": sentiment,
        #         "named_entities": ner_results
        #     }
        
        # yield "All articles analyzed. Generating comprehensive response...", conversation_id
        
        # # Generate final response based on all analyses
        # final_response = await asyncio.to_thread(
        #     self.llm_service.generate_analysis_response,
        #     initial_state["query"],
        #     analysis_results
        # )
        
        # # Update conversation history
        # self.conversations[conversation_id].append(AIMessage(content=final_response))
        
        # logger.info(f"Workflow completed for conversation {conversation_id}")
        
        # # Return the final response
        # yield final_response, conversation_id
