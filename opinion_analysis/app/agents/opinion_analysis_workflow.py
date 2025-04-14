from typing import Dict, List, TypedDict, Annotated, Sequence, operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from app.services.llm_service import OpinionLLMService


class OpinionAnalysisWorkflow:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        is_search_related: bool
        search_results: str | List[Dict]
        analysis_results: Dict = {}

    def __init__(self):
        """Initialize the Opinion Analysis Agent with necessary services"""
        self.llm_service = OpinionLLMService()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the opinion analysis agent"""

        # Define the state schema
        # Create the workflow
        workflow = StateGraph(self.State)

        # Add nodes
        workflow.add_node("search_news", self.search_news)
        workflow.add_node(
            "generate_summary_and_sentiment", self.generate_summary_and_sentiment
        )
        workflow.add_node("generate_response", self.generate_response)

        # Define edges with routing function
        workflow.add_conditional_edges("search_news", self.route_from_check_relevance)
        workflow.add_edge("generate_summary_and_sentiment", "generate_response")
        workflow.add_edge("generate_response", END)

        # Set the entry point
        workflow.set_entry_point("search_news")

        return workflow.compile()

    # Define the workflow nodes
    def search_news(self, state):
        """Check if the query is related to opinion analysis"""
        current_messages = state["messages"]
        response = self.llm_service.opinian_agent.invoke(
            {"query": current_messages[-1].content}
        )
        if isinstance(response["output"], str):
            state["is_search_related"] = False

        else:
            state["is_search_related"] = True
        state["search_results"] = response["output"]
        return state

    def generate_summary_and_sentiment(self, state):
        """Generate a summary and sentiment analysis of the news articles"""
        articles = state["search_results"]
        analysis_results = {}

        for article in articles:
            # Generate summary
            summary = self.llm_service.summary_chain.invoke(
                {"text": article["content"]}
            )

            # Perform sentiment analysis
            sentiment = self.llm_service.sentiment_chain.invoke(
                {"context": article["content"]}
            ).sentiment

            analysis_results[article["title"]] = {
                "title": article["title"],
                "summary": summary,
                "sentiment": sentiment,
                "url": article["url"],
                "publish_date": article["publish_date"],
                "content": article["content"],
                "ner": article["ner"],
            }
        state["analysis_results"] = analysis_results
        return state

    def route_from_check_relevance(self, state):
        if state["is_search_related"]:
            return "generate_summary_and_sentiment"
        else:
            return "generate_response"

    def generate_response(self, state):
        """Generate a response based on the analysis results"""
        if not state["is_search_related"]:
            response = state["search_results"]
        else:
            response = self.llm_service.generate_analysis_response(
                state["messages"][-1].content, state["analysis_results"]
            )
        state["messages"] = [AIMessage(content=response)]
        return state
