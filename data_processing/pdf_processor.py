import fitz  # PyMuPDF
from pathlib import Path
from config import config
import pandas as pd
import re

class PDFProcessor:
    def __init__(self):
        self.cache_file = config.PROCESSED_DATA / "processed_data.parquet"
        
    def extract_judgment_outcome(self, text):
        """Extract case outcome from judgment text"""
        patterns = {
            'allowed': r'(appeal|petition|application)\s+(allowed|granted)',
            'dismissed': r'(appeal|petition|application)\s+(dismissed|rejected)'
        }
        
        for label, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return label
        return 'other'
    
    def extract_text_from_pdf(self, pdf_path):
        """Extracts all text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def process_pdfs(self):
        """Process all PDFs and create labeled dataset"""
        data = []
        
        from pathlib import Path

        year_dir = Path(r"C:\Users\ishan\Downloads\ML Projects\IndianLegalJudgmentDocumnet(Dataset)\supreme_court_judgments\2022")
        for pdf_file in year_dir.glob("*.pdf"):
            try:
                with fitz.open(pdf_file) as doc:
                    text = "\n".join(page.get_text() for page in doc)
                    outcome = self.extract_judgment_outcome(text)
                    if outcome == 'other':
                        continue  # Skip unclear cases
                    data.append({
                        "text": text,
                        "label": outcome,
                        "year": 2022,
                        "source": pdf_file.name
                    })
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        '''
        for year_dir in config.RAW_DATA_PATH.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                for pdf_file in year_dir.glob("*.pdf"):
                    try:
                        with fitz.open(pdf_file) as doc:
                            text = "\n".join(page.get_text() for page in doc)
                            
                            outcome = self.extract_judgment_outcome(text)
                            if outcome == 'other': 
                                continue  # Skip unclear cases
                                
                            data.append({
                                "text": text,
                                "label": outcome,
                                "year": int(year_dir.name),
                                "source": pdf_file.name
                            })
                    except Exception as e:
                        print(f"Error processing {pdf_file}: {e}")'''
        
        df = pd.DataFrame(data)
        df.to_parquet(self.cache_file)
        return df
