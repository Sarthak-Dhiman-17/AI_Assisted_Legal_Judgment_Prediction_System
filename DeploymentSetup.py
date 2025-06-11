from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CaseInput(BaseModel):
    facts: str
    petitioner: str
    respondent: str
    cited_statutes: list

@app.post("/predict")
async def predict_judgment(case: CaseInput):
    processed_text = preprocess_legal_text(case.facts)
    
    # Feature transformation
    tfidf_features = tfidf.transform([processed_text])
    bert_features = bert_embeddings(processed_text)
    
    # Prediction
    prediction = stacking_model.predict(tfidf_features)
    proba = stacking_model.predict_proba(tfidf_features)
    
    # Explanation
    explanation = explain_prediction(stacking_model, processed_text)
    
    return {
        "prediction": "Accepted" if prediction[0] else "Rejected",
        "confidence": float(proba[0][prediction[0]]),
        "explanation": explanation
    }
