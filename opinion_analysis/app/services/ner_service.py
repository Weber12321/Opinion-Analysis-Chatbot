# app/services/ner_service.py
import spacy
from typing import List, Dict, Any
import os

class NERService:
    def __init__(self):
        """Initialize the Named Entity Recognition service"""
        # Download the model if not already available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            print("Downloading spaCy model...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: The text to analyze
            
        Returns:
            List of entities with text and label
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
        return entities
