from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

class LegalBERTClassifier:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=2
        )
        self.metric = evaluate.load("f1")
        
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )
        
    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir="bert_results",
            eval_strategy="epoch",  # CHANGED FROM evaluation_strategy TO eval_strategy
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=50,
            fp16=True
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        self.trainer.train()
        
    def save(self, path):
        self.model.save_pretrained(path)
