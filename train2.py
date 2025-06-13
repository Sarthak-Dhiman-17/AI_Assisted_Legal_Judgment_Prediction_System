#using this file to start from parquet itself created from Use2022Data_CreateParquet.ipynb file
#Here process starts from cleaning text then training

import pandas as pd
from config import config
from data_processing.text_cleaner import TextCleaner
from models.baseline_model import BaselineModel
from models.legal_bert import LegalBERTClassifier
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
tqdm.pandas() 


class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def main():
    # Load processed data
    df = pd.read_parquet(r'C:\Users\ishan\Downloads\ML Projects\AI_Legal\processed_data\processed_data.parquet')

    # Clean text if not already present
    cleaner = TextCleaner()
    '''if 'clean_text' not in df.columns:
        print("Cleaning text...")
        #df['clean_text'] = df['text'].progress_apply(cleaner.clean_text)
        df['clean_text'] = [cleaner.clean_text(text) for text in tqdm(df['text'], desc="Cleaning text")'''
    if 'clean_text' not in df.columns:
        print("Cleaning text...")
        df['clean_text'] = df['text'].progress_apply(cleaner.clean_text)
    df.to_parquet("processed_data_cleaned.parquet")


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=config.TEST_SIZE,
        stratify=df['label'],
        random_state=config.RANDOM_STATE
    )

    # Train baseline model
    print("Training baseline model...")
    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    baseline.save(config.MODELS_DIR / "baseline_model.pkl")

    # Prepare BERT data
    print("Preparing BERT dataset...")
    train_encodings = cleaner.bert_preprocess(X_train.tolist())
    test_encodings = cleaner.bert_preprocess(X_test.tolist())
    #train_dataset = LegalDataset(train_encodings, y_train.map({'allowed': 0, 'dismissed': 1}))
    #test_dataset = LegalDataset(test_encodings, y_test.map({'allowed': 0, 'dismissed': 1}))
    #train_labels = y_train.map({'allowed': 0, 'dismissed': 1}).to_numpy()  # Convert to numpy
    #test_labels = y_test.map({'allowed': 0, 'dismissed': 1}).to_numpy()
    train_labels = y_train.reset_index(drop=True).map({'allowed': 0, 'dismissed': 1}).to_numpy()
    test_labels = y_test.reset_index(drop=True).map({'allowed': 0, 'dismissed': 1}).to_numpy()

    train_dataset = LegalDataset(train_encodings, train_labels)
    test_dataset = LegalDataset(test_encodings, test_labels)

    # Train Legal-BERT
    print("Training Legal-BERT...")
    bert_classifier = LegalBERTClassifier()
    bert_classifier.train(train_dataset, test_dataset)
    bert_classifier.save(config.MODELS_DIR / "legal_bert")

if __name__ == "__main__":
    main()
