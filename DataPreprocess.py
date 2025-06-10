import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Legal Text Cleaning
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_legal_text(text):
    # Remove citations and special characters
    text = re.sub(r'\d+\.\s+(\w+\s+)*\d+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # SpaCy processing
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc 
             if not token.is_stop and token.is_alpha]
    
    return ' '.join(tokens)

# Example usage
#pdf_text = extract_text_from_pdf("judgment_2020_1.pdf")
pdf_text = extract_text_from_pdf(r'C:\Users\ishan\Downloads\ML Projects\IndianLegalJudgmentDocumnet(Dataset)\supreme_court_judgments\2022\The_State_Of_Maharashtra_vs_Madhuri_Maruti_Vidhate_Since_After_on_30_September_2022_1_1.pdf')
cleaned_text = preprocess_legal_text(pdf_text)
