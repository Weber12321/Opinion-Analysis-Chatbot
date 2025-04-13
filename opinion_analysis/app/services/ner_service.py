# app/services/ner_service.py
import spacy

ner_service = spacy.load("zh_core_web_sm")