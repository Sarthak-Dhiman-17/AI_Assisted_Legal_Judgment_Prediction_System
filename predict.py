import sys
import os
from pathlib import Path
from data_processing.pdf_processor import PDFProcessor

from data_processing.pdf_processor import PDFProcessor
from data_processing.text_cleaner import TextCleaner
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Predictor:
    def __init__(self, model_dir):
        self.baseline_model = joblib.load(f"{model_dir}/baseline_model.pkl")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(f"{model_dir}/legal_bert")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.cleaner = TextCleaner()
        
    def predict(self, text):
        cleaned_text = self.cleaner.clean_text(text)
        
        # Baseline prediction (raw text → pipeline handles vectorization)
        baseline_pred = self.baseline_model.predict([cleaned_text])[0]
        
        # BERT prediction
        inputs = self.bert_tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.bert_model(**inputs).logits
        bert_pred = torch.argmax(logits).item()
        
        return {
            "baseline": "allowed" if baseline_pred == 0 else "dismissed",
            "bert": "allowed" if bert_pred == 0 else "dismissed"
        }


def predict_on_folder(folder_path):
    processor = PDFProcessor()
    predictor = Predictor("saved_models")
    pdf_files = list(Path(folder_path).glob("*.pdf")) + list(Path(folder_path).glob("*.PDF"))
    if not pdf_files:
        print("No PDF files found in the folder.")
        return
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        text = processor.extract_text_from_pdf(str(pdf_path))
        prediction = predictor.predict(text)
        print(f"{pdf_path.name}: {prediction}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str, required=True)
    args = parser.parse_args()
    pdf_path = args.pdf_path

    if os.path.isdir(pdf_path):
        predict_on_folder(pdf_path)
    elif os.path.isfile(pdf_path):
        processor = PDFProcessor()
        predictor = Predictor("saved_models")
        text = processor.extract_text_from_pdf(pdf_path)
        prediction = predictor.predict(text)
        print(f"{pdf_path}: {prediction}")
    else:
        print("Provided path is neither a file nor a folder.")
