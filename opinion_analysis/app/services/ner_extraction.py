# app/services/ner_service.py
import spacy

ner_extractor = spacy.load("zh_core_web_sm")
