import os
from pathlib import Path

# Base configuration
class Config:
    # Raw data path (update with your actual path)
    RAW_DATA_PATH = Path(r"C:\Users\ishan\Downloads\ML Projects\IndianLegalJudgmentDocumnet(Dataset)\supreme_court_judgments")
    
    # Processed data paths
    PROCESSED_DATA = Path("processed_data")
    MODELS_DIR = Path("saved_models")
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    BERT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
    
    # Ensure directories exist
    PROCESSED_DATA.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

config = Config()
