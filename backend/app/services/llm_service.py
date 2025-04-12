# app/services/llm_service.py
import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMService:
    def __init__(self):
        """Initialize the LLM service with either Gemini or OpenAI"""
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            self.gemini = genai.GenerativeModel('gemini-2.0-flash')
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.openai = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def is_opinion_related(self, query: str) -> bool:
        """Check if the query is related to opinion analysis"""
        prompt = f"""
        Determine if the following query is related to opinion analysis, public sentiment, 
        market research, social media analysis, or news analysis. 
        
        Query: "{query}"
        
        Return only true or false.
        """
        
        if self.provider == "gemini":
            response = self.gemini.generate_content(prompt)
            result = response.text.strip().lower()
        else:  # OpenAI
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result = response.choices[0].message.content.strip().lower()
        
        return result == "true"

    def generate_summary(self, text: str) -> str:
        """Generate a concise summary of the text"""
        prompt = f"""
        Summarize the following text in 3-5 sentences, focusing on key points:
        
        {text}
        """
        
        if self.provider == "gemini":
            response = self.gemini.generate_content(prompt)
            return response.text.strip()
        else:  # OpenAI
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()

    def generate_rejection_message(self) -> str:
        """Generate a polite rejection message for non-opinion related queries"""
        prompt = """
        Generate a polite message explaining that you are a specialized chatbot focused 
        on opinion analysis and can only answer questions related to public sentiment, 
        market research, social media trends, news analysis, and similar topics.
        """
        
        if self.provider == "gemini":
            response = self.gemini.generate_content(prompt)
            return response.text.strip()
        else:  # OpenAI
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()

    def generate_analysis_response(self, query: str, analysis_results: Dict) -> str:
        """Generate a comprehensive response based on the news analysis results"""
        # Convert analysis results to a format suitable for the prompt
        results_summary = []
        
        for url, data in analysis_results.items():
            results_summary.append(f"""
            Title: {data['title']}
            Published: {data['publish_date']}
            Summary: {data['summary']}
            Sentiment: {data['sentiment']}
            Key Entities: {', '.join([f"{entity['text']} ({entity['label']})" for entity in data['named_entities'][:5]])}
            """)
        
        formatted_results = "\n".join(results_summary)
        
        prompt = f"""
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
        
        if self.provider == "gemini":
            response = self.gemini.generate_content(prompt)
            return response.text.strip()
        else:  # OpenAI
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
