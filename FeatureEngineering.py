from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF for facts section
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# LegalBERT embeddings
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

def bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", 
                      truncation=True, max_length=512)
    return inputs
