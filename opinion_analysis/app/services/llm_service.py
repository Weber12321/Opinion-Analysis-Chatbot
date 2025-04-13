# app/services/llm_service.py
import os
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class SentimentGrader(BaseModel):
    """Grade the sentiment with the given context"""
    sentiment: str = Field(description="sentiment of the comment (positive, neutral or negative")


# Pydantic
class OpinionGrader(BaseModel):
    """Grade if context is related to opinion analysis, public sentiment."""
    binary_score: str = Field(
        description="Return binary socre, true, if context is related to opinion analysis, public sentiment, otherwise return false."
    )

    
class LLMService:
    def __init__(self):
        """Initialize the LLM service with either Gemini"""

        # ==== config LLM service ====
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            timeout=60,
            google_api_key=api_key
        )
        
        structured_llm_opinion_grader = self.gemini.with_structured_output(OpinionGrader)
        structured_llm_sentiment_grader = self.gemini.with_structured_output(SentimentGrader)

        opinion_related_prompt = f"""
        Determine if the following query is related to opinion analysis, public sentiment, 
        market research, social media analysis, or news analysis. 
        
        Query: "{query}"
        
        Return only true or false.
        """
        opinion_prompt_template = PromptTemplate.from_template(opinion_related_prompt)
        self.opinion_chain = opinion_prompt_template | structured_llm_opinion_grader

        summary_prompt = f"""
        Summarize the following text in 3-5 sentences, focusing on key points:
        
        {text}
        """
        summary_prompt_template =  PromptTemplate.from_template(summary_prompt)
        self.summary_chain = summary_prompt_template | self.gemini


        reject_prompt = """
        Generate a polite message explaining that you are a specialized chatbot focused 
        on opinion analysis and can only answer questions related to public sentiment, 
        market research, social media trends, news analysis, and similar topics.
        """

        reject_prompt_template = PromptTemplate.from_template(reject_prompt)
        self.reject_chain = reject_prompt_template | self.gemini
        
        sentiment_prompt = f"""
        Analyze the sentiment of the following context.

        context: "{context}"

        Return positive, neutral or negative.
        Answer in one word.
        Answer:
        """
        sentiment_prompt_template = PromptTemplate.from_template(sentiment_prompt)
        self.sentiment_chain = sentiment_prompt_template | structured_llm_sentiment_grader


        analysis_prompt = f"""
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
        self.analysis_chain = analysis_prompt_template | self.gemini
        
    def is_opinion_related(self, query: str) -> bool:
        """Check if the query is related to opinion analysis"""
        return self.opinion_chain.invoke({"query": query})


    def generate_summary(self, text: str) -> str:
        """Generate a concise summary of the text"""
        return self.summary_chain.invoke({"text": text})

    def generate_rejection_message(self) -> str:
        """Generate a polite rejection message for non-opinion related queries"""
        return self.reject_chain.invoke({})

    def analyze_sentiment(self, context: str) -> str:
        """Analyze the sentiment of the given context"""
        return self.sentiment_chain.invoke({"context": context})

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
        return response.text.strip()