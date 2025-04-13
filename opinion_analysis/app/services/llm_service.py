# app/services/ner_service.py
import os

from typing import Dict

import os
from typing import Dict
from opinion_analysis.app.services.news_scraper import search_news
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



class SentimentGrader(BaseModel):
    """Grade the sentiment with the given context"""
    sentiment: str = Field(description="sentiment of the comment (positive, neutral or negative")

    
class LLMService:
    def __init__(self):
        """Initialize the LLM service with either Gemini"""
        # ==== config LLM service ====
        api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCle6jmFfSjUcUr-D15ieqd-ZOFeKAdOWc")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        self.opinian_agent = self._create_news_search_agent()
        self.summary_chain = self._create_summary_chain()
        self.sentiment_chain = self._create_sentiment_chain()
        self.analysis_chain = self._create_analysis_chain()

    
    def _create_summary_chain(self):
        summary_prompt = """
        Summarize the following text in 3-5 sentences, focusing on key points:
        
        {text}
        """
        summary_prompt_template =  PromptTemplate.from_template(summary_prompt)
        return summary_prompt_template | self.llm | StrOutputParser()

    def _create_sentiment_chain(self):
        structured_llm_sentiment_grader = self.llm.with_structured_output(SentimentGrader)      
        sentiment_prompt = """
        Analyze the sentiment of the following context.

        context: {context}

        Return positive, neutral or negative.
        Answer in one word.
        Answer:
        """
        sentiment_prompt_template = PromptTemplate.from_template(sentiment_prompt)
        return sentiment_prompt_template | structured_llm_sentiment_grader

    def _create_news_search_agent(self) -> AgentExecutor:
        tools = [search_news] # List of actual tool objects

        # Create a prompt suitable for an agent with tool calling instructions
        agent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "If the following query is related to news, opinion analysis, public sentiment, market research, social media analysis, or news analysis. then search it with tool Otherwise answer with: Please enter a query which is related to opinion analysis, public sentiment, market research, social media analysis, or news analysis.",
                ),
                ("human", "{query}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        # Create the agent runnable
        agent = create_openai_tools_agent(self.llm, tools, agent_prompt)
        
        # Create the executor which handles the tool call loop
        return AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True shows the steps
        
    def _create_analysis_chain(self):
        analysis_prompt = """
        Based on the following query and news analysis results, provide a comprehensive answer.
        Focus on trends, common sentiments, important entities, and insights from the news.
        
        Query: {query}
        
        Analysis Results:
        {formatted_results}
        
        Format your response with clear sections:
        1. Overview of findings
        2. Key sentiment trends
        3. Important entities mentioned
        4. Detailed insights
        5. Conclusion
        """
        analysis_prompt_template = PromptTemplate.from_template(analysis_prompt)
        return analysis_prompt_template | self.llm | StrOutputParser()

    def generate_analysis_response(self, query: str, analysis_results: Dict) -> str:
        """Generate a comprehensive response based on the news analysis results"""
        # Convert analysis results to a format suitable for the prompt
        results_summary = []
        
        for data in analysis_results.values():
            results_summary.append(f"""
            Title: {data['title']}
            Published: {data['publish_date']}
            Summary: {data['summary']}
            Sentiment: {data['sentiment']}
            Key Entities: {', '.join([f"{entity['text']} ({entity['label']})" for entity in data['named_entities'][:5]])}
            """)
        
        formatted_results = "\n".join(results_summary)
        
        self.analysis_chain
        
        response = self.analysis_chain.invoke({
            "query": query,
            "formatted_results": formatted_results
        })
        return response.strip()