import spacy
import re
from transformers import AutoTokenizer

class TextCleaner:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
    def clean_text(self, text):
        """Basic legal text cleaning"""
        # Remove citations and special characters
        text = re.sub(r'\d+\.\s+(\w+\s+)*\d+', '', text)  # Case citations
        text = re.sub(r'\[.*?\]', '', text)  # Bracketed content
        text = re.sub(r'\s+', ' ', text)  # Extra whitespace
        
        # SpaCy processing
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop and token.is_alpha
        ]
        
        return " ".join(tokens)
    
    def bert_preprocess(self, text):
        """Prepare text for Legal-BERT"""
        return self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='np'
        )
