# AI Assisted Legal Judgment Prediction System

## Overview

This project is an **AI-powered system for predicting the outcome of Indian Supreme Court judgments**. Given the full text of a legal judgment, the system predicts whether the judgment is likely to be "allowed" or "dismissed." The goal is to support legal research, case analysis, and provide explainable AI assistance for the legal domain.

---

## Features

- **Dual Model Architecture:**  
  - **Baseline Model:** Random Forest with TF-IDF features for fast, interpretable predictions.
  - **Legal-BERT Model:** Fine-tuned transformer model for state-of-the-art performance on legal text.
- **PDF Judgment Processing:**  
  - Automatic extraction and cleaning of legal text from Supreme Court PDF files.
- **Explainability:**  
  - Feature importance for baseline model; ready for LIME/SHAP integration for BERT.
- **Easy Prediction:**  
  - Predict outcomes for new judgments (single PDF or batch folder).
- **Extensible:**  
  - Modular codebase for easy adaptation to other courts or legal tasks.

---

## Why This Project?

- **Legal judgments are complex, lengthy, and context-dependent.**
- Traditional models (like Random Forest) provide fast, interpretable baselines but struggle with context.
- Transformer models (like Legal-BERT) capture legal nuances and context, improving accuracy.
- By combining both, we get robust, explainable, and high-performing legal AI.

---

## Dataset

- **Source:** Indian Supreme Court judgments (1950–2025) downloaded from [Indian Kanoon](https://indiankanoon.org/).
- **Format:** PDF files organized by year.
- **Preprocessing:** PDFs are converted to text, cleaned, and labeled as "allowed" or "dismissed" based on outcome extraction.

---


## How to Use

### 1. **Setup**

git clone https://github.com/Sarthak-Dhiman-17/AI_Assisted_Legal_Judgment_Prediction_System.git

cd AI_Assisted_Legal_Judgment_Prediction_System

pip install -r requirements.txt

python -m spacy download en_core_web_sm

### 2. **Training**

python train.py


### 3. **Prediction**

- Predict the outcome of a new judgment (single PDF):

python predict.py --pdf_path "path/to/your_judgment.pdf"


- Predict outcomes for all PDFs in a folder (batch mode):

python predict.py --pdf_path "path/to/folder/"

