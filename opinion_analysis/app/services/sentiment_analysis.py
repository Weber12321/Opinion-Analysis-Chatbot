# app/services/sentiment_analysis.py
from typing import Dict
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalysisService:
    def __init__(self):
        """Initialize the sentiment analysis service"""
        # Load a pre-trained sentiment analysis model
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            # Fallback to simpler model if needed
            self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment scores and label
        """
        # For long texts, we'll split and analyze in chunks
        if len(text) > 1000:
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            results = []
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                chunk_result = self._analyze_text(chunk)
                results.append(chunk_result)
            
            # Calculate average sentiment
            positive_score = sum(r["positive"] for r in results) / len(results)
            negative_score = sum(r["negative"] for r in results) / len(results)
            
            result = {
                "positive": positive_score,
                "negative": negative_score,
                "label": "POSITIVE" if positive_score > negative_score else "NEGATIVE",
                "overall_score": positive_score - negative_score
            }
        else:
            result = self._analyze_text(text)
            
        return result
    
    def _analyze_text(self, text: str) -> Dict:
        """Analyze a single chunk of text"""
        try:
            result = self.sentiment_analyzer(text)[0]
            
            # Format the result
            scores = {item["label"].lower(): item["score"] for item in result}
            
            # Ensure we have positive and negative scores
            positive_score = scores.get("positive", 0)
            negative_score = scores.get("negative", 0)
            
            return {
                "positive": positive_score,
                "negative": negative_score,
                "label": "POSITIVE" if positive_score > negative_score else "NEGATIVE",
                "overall_score": positive_score - negative_score
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                "positive": 0.5,
                "negative": 0.5,
                "label": "NEUTRAL",
                "overall_score": 0.0
            }
