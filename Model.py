from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression
# Ensemble Model
base_models = [
    ('svm', SVC(kernel='linear', probability=True)),
    ('xgb', XGBClassifier())
]

meta_model = LogisticRegression()

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# LegalBERT Fine-tuning
from transformers import Trainer, TrainingArguments

legal_bert = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased", 
    num_labels=2
)
'''
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)'''
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",  # This should work after upgrade
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch"
)